"""prepare_dataset.py — convertit ``training_comments.json`` en datasets Alpaca.

============================================================================
Objectif
============================================================================

Lire ``data/training_comments.json`` (produit par ``dataset_builder.py``) et
produire **deux fichiers** prêts à charger dans Unsloth / QLoRA pour un
fine-tuning :

* ``scripts/data/generator_dataset.json`` — entraîne le modèle à **générer**
  un commentaire crédible étant donné le contexte du Reel (T-type, niche,
  caption, hashtags, audio).
* ``scripts/data/classifier_dataset.json`` — entraîne le modèle à
  **étiqueter** un commentaire avec son T-type étant donné un peu de
  contexte créateur + métriques.

Les deux datasets utilisent le **format Alpaca** (``instruction`` /
``input`` / ``output``), qui est nativement supporté par Unsloth via
``load_dataset(..., format="alpaca")`` et compatible avec la plupart des
loaders QLoRA. Pour le générateur on aurait pu sortir du ShareGPT, mais
Alpaca est plus simple à dériver à partir de notre schéma actuel et reste
convertible en ShareGPT côté training si besoin.

============================================================================
Cardinalité
============================================================================

Pour **chaque entrée** (= un Reel) on dérive **autant de paires** que de
commentaires dans ``top_comments``. Si ``top_comments`` est vide, l'entrée
est ignorée (on ne peut ni apprendre à générer, ni étiqueter).

============================================================================
CLI
============================================================================

::

    python scripts/prepare_dataset.py            # exécution normale
    python scripts/prepare_dataset.py --mock     # 10 entrées fictives
    python scripts/prepare_dataset.py --stats    # uniquement les compteurs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAINING_PATH = _PROJECT_ROOT / "data" / "training_comments.json"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "scripts" / "data"
DEFAULT_GENERATOR_PATH = DEFAULT_OUTPUT_DIR / "generator_dataset.json"
DEFAULT_CLASSIFIER_PATH = DEFAULT_OUTPUT_DIR / "classifier_dataset.json"

# Seuils minimaux pour considérer un dataset "prêt" pour un fine-tuning utile.
# Sous 200 paires, QLoRA n'a pas assez de signal pour apprendre un registre
# (cf. retours d'expérience Unsloth — 200 = plancher absolu, 1000+ = confort).
MIN_PAIRS_READY = 200

GENERATOR_INSTRUCTION = (
    "Tu es un utilisateur Instagram. Génère un commentaire naturel et "
    "humain pour ce Reel."
)
CLASSIFIER_INSTRUCTION = (
    "Classifie ce commentaire Instagram selon les types T1→T5."
)

# T-types valides côté label classifier — un commentaire sans label exploitable
# est filtré (sinon le modèle apprend à prédire ``""``).
_VALID_TTYPES: frozenset[str] = frozenset(
    {"T1", "T2", "T2b", "T3a", "T3b", "T4", "T5"}
)

_LOG = logging.getLogger("aitertainment.prepare_dataset")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _read_training(path: Path) -> list[dict[str, Any]]:
    """Charge ``training_comments.json`` ; retourne ``entries`` (jamais ``None``)."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON invalide dans {path} : {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"racine JSON doit être un objet ({type(data).__name__})")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f'"entries" doit être une liste dans {path}')
    return [e for e in entries if isinstance(e, dict)]


def _atomic_write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    """Écrit ``payload`` sur disque de façon atomique (tmp + replace).

    Écrit une **liste** au top-level (vs un dict) — c'est la convention Alpaca
    attendue par ``datasets.load_dataset("json", ...)``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        suffix=".tmp",
    ) as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        tmp_path = Path(fh.name)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Helpers de formatage
# ---------------------------------------------------------------------------


def _fmt_hashtags(hashtags: Any) -> str:
    """Liste de hashtags → ``#a #b #c`` (ou ``(aucun)``)."""
    if not hashtags:
        return "(aucun)"
    if isinstance(hashtags, str):
        return hashtags.strip() or "(aucun)"
    parts = [str(h).lstrip("#").strip() for h in hashtags if str(h).strip()]
    return " ".join(f"#{p}" for p in parts) if parts else "(aucun)"


def _fmt_optional(value: Any, *, default: str = "(inconnu)") -> str:
    """Stringifie ``value`` ou retourne ``default`` si ``None`` / vide."""
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def _fmt_ratio(value: Any) -> str:
    """Formate un ratio float en 3 décimales ; ``(inconnu)`` si ``None``."""
    if value is None:
        return "(inconnu)"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "(inconnu)"


# ---------------------------------------------------------------------------
# Construction des paires Alpaca
# ---------------------------------------------------------------------------


def build_generator_pair(
    entry: dict[str, Any], comment: dict[str, Any]
) -> dict[str, str] | None:
    """Construit une paire Alpaca **generator** ``(instruction, input, output)``.

    Retourne ``None`` si la paire est inexploitable (commentaire vide, pas
    de T-type — sans contexte le modèle apprendrait à générer du bruit).
    """
    text = str(comment.get("text") or "").strip()
    if not text:
        return None

    gen_in = entry.get("generator_input") or {}
    # Priorité au t_type du commentaire (self-contained), repli sur l'entry.
    t_type = str(comment.get("t_type") or gen_in.get("t_type") or "").strip()
    if not t_type:
        return None
    niche = str(
        comment.get("niche") or gen_in.get("niche") or ""
    ).strip() or "(non précisée)"

    caption = _fmt_optional(gen_in.get("caption"), default="(vide)")
    hashtags = _fmt_hashtags(gen_in.get("hashtags"))
    audio = _fmt_optional(gen_in.get("audio_id"), default="(aucun)")

    input_block = (
        f"T-type: {t_type}\n"
        f"Niche: {niche}\n"
        f"Caption: {caption}\n"
        f"Hashtags: {hashtags}\n"
        f"Audio: {audio}"
    )
    return {
        "instruction": GENERATOR_INSTRUCTION,
        "input": input_block,
        "output": text,
    }


def build_classifier_pair(
    entry: dict[str, Any], comment: dict[str, Any]
) -> dict[str, str] | None:
    """Construit une paire Alpaca **classifier** ``(instruction, input, output)``.

    Retourne ``None`` si :

    * texte du commentaire vide, OU
    * label T-type absent / non reconnu (cf. ``_VALID_TTYPES``).

    On filtre les T-types invalides parce que le classifier doit apprendre
    une distribution sur un set fini ; un label inconnu pollue l'objectif.
    """
    text = str(comment.get("text") or "").strip()
    if not text:
        return None

    gen_in = entry.get("generator_input") or {}
    t_type = str(comment.get("t_type") or gen_in.get("t_type") or "").strip()
    if t_type not in _VALID_TTYPES:
        return None

    niche = str(
        comment.get("niche") or gen_in.get("niche") or ""
    ).strip() or "(non précisée)"
    ctx = entry.get("classifier_context") or {}
    views = _fmt_optional(ctx.get("views"))
    ratio = _fmt_ratio(ctx.get("comment_to_like_ratio"))

    input_block = (
        f"Commentaire: {text}\n"
        f"Niche: {niche}\n"
        f"Vues: {views}\n"
        f"Ratio comments/likes: {ratio}"
    )
    return {
        "instruction": CLASSIFIER_INSTRUCTION,
        "input": input_block,
        "output": t_type,
    }


def build_datasets(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Pour chaque entrée, dérive ``len(top_comments)`` paires par dataset.

    Retourne ``(generator_pairs, classifier_pairs)``. Les paires invalides
    (texte vide, T-type manquant) sont silencieusement filtrées — c'est
    l'attendu : on ne lève pas, on garde le best-effort.
    """
    gen_pairs: list[dict[str, str]] = []
    cls_pairs: list[dict[str, str]] = []
    for entry in entries:
        comments = entry.get("top_comments") or []
        if not isinstance(comments, list):
            continue
        for c in comments:
            if not isinstance(c, dict):
                continue
            g = build_generator_pair(entry, c)
            if g is not None:
                gen_pairs.append(g)
            cl = build_classifier_pair(entry, c)
            if cl is not None:
                cls_pairs.append(cl)
    return gen_pairs, cls_pairs


# ---------------------------------------------------------------------------
# Données fictives (--mock)
# ---------------------------------------------------------------------------


def _mock_entries(count: int = 10) -> list[dict[str, Any]]:
    """Construit ``count`` entrées synthétiques au schéma exact du builder.

    Distribution approximative T2/T2b/T3b/T4/T5 pour exercer le filtre des
    T-types valides, plus 1 entrée "T1" gardée car elle reste un label
    valide pour le classifier (mais on évitera de l'utiliser pour le
    generator en prod — cf. brief Watcher).
    """
    rotation = ["T2", "T2b", "T3b", "T4", "T5", "T1"]
    niches = ["humour", "f1", "cuisine", "gaming", "tech", "lifestyle"]
    captions = [
        "moment culte",
        "tu vas adorer",
        "ne pas reproduire",
        "GG la team",
        "le passage à 0:08",
        "explication en 1 minute",
    ]
    sample_comments = [
        "mdr trop vrai",
        "j'en peux plus",
        "le passage 0:08",
        "ah ouais quand même",
        "techniquement parfait",
        "bravo le débutant",
        "très subtil",
        "on attend la suite",
        "GG",
        "no comment",
    ]
    out: list[dict[str, Any]] = []
    for i in range(count):
        t_type = rotation[i % len(rotation)]
        niche = niches[i % len(niches)]
        out.append(
            {
                "media_id": f"MOCK_{i:03d}",
                "username": f"mock_creator_{i}",
                "reel_url": f"https://www.instagram.com/reel/MOCK_{i:03d}/",
                "collected_at": "2026-05-08T12:00:00",
                "generator_input": {
                    "t_type": t_type,
                    "niche": niche,
                    "caption": captions[i % len(captions)],
                    "hashtags": [niche, f"tag{i}"],
                    "audio_id": f"AUD{i:03d}",
                },
                "classifier_context": {
                    "views": 50_000 + i * 12_345,
                    "likes": 1_000 + i * 137,
                    "comment_count": 50 + i * 9,
                    "shares": None,
                    "comment_to_like_ratio": round(0.05 + (i % 10) * 0.012, 3),
                    "share_to_like_ratio": None,
                },
                "top_comments": [
                    {
                        "text": sample_comments[(i + j) % len(sample_comments)],
                        "likes": 100 - j * 17,
                        "t_type": t_type,
                        "niche": niche,
                    }
                    for j in range(3)
                ],
            }
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_stats(
    *,
    total_entries: int,
    generator_count: int,
    classifier_count: int,
    out: Any = None,
) -> None:
    """Affiche les compteurs + indication ``Prêt pour fine-tuning si …``."""
    target = out or sys.stdout
    print(f"Total entrées training_comments.json : {total_entries}", file=target)
    print(f"→ Generator dataset : {generator_count} paires", file=target)
    print(f"→ Classifier dataset : {classifier_count} paires", file=target)
    if generator_count > MIN_PAIRS_READY and classifier_count > MIN_PAIRS_READY:
        print(
            f"Prêt pour fine-tuning (seuil minimal : {MIN_PAIRS_READY} par dataset)",
            file=target,
        )
    else:
        missing_gen = max(0, MIN_PAIRS_READY + 1 - generator_count)
        missing_cls = max(0, MIN_PAIRS_READY + 1 - classifier_count)
        print(
            "Pas encore prêt — il manque "
            f"{missing_gen} paires generator / {missing_cls} paires classifier "
            f"(seuil : > {MIN_PAIRS_READY} par dataset)",
            file=target,
        )


def run(
    *,
    mock: bool = False,
    stats_only: bool = False,
    training_path: Path | None = None,
    generator_path: Path | None = None,
    classifier_path: Path | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Point d'entrée orchestrant lecture → conversion → écriture.

    Retourne ``(generator_pairs, classifier_pairs)`` pour faciliter les
    tests. En mode ``--stats``, n'écrit rien sur disque mais affiche tout
    de même les compteurs (utile pour piloter l'avancement de la collecte).
    En mode ``--mock``, source les entrées depuis ``_mock_entries(10)`` —
    aucune lecture disque.
    """
    if mock:
        entries = _mock_entries(10)
    else:
        path = training_path or DEFAULT_TRAINING_PATH
        entries = _read_training(path)

    generator_pairs, classifier_pairs = build_datasets(entries)

    _print_stats(
        total_entries=len(entries),
        generator_count=len(generator_pairs),
        classifier_count=len(classifier_pairs),
    )

    if not stats_only:
        gen_p = generator_path or DEFAULT_GENERATOR_PATH
        cls_p = classifier_path or DEFAULT_CLASSIFIER_PATH
        _atomic_write_json(gen_p, generator_pairs)
        _atomic_write_json(cls_p, classifier_pairs)
        print(f"✓ {gen_p}", file=sys.stdout)
        print(f"✓ {cls_p}", file=sys.stdout)

    return generator_pairs, classifier_pairs


def _main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convertit data/training_comments.json en datasets Alpaca "
            "(generator + classifier) pour fine-tuning Unsloth/QLoRA."
        )
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Génère 10 entrées fictives au lieu de lire training_comments.json.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Affiche les compteurs sans rien écrire sur disque.",
    )
    args = parser.parse_args(argv)

    try:
        run(mock=args.mock, stats_only=args.stats)
    except (OSError, ValueError) as e:
        _LOG.error("prepare_dataset a échoué : %s", e)
        print(f"Erreur : {e}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "CLASSIFIER_INSTRUCTION",
    "DEFAULT_CLASSIFIER_PATH",
    "DEFAULT_GENERATOR_PATH",
    "DEFAULT_TRAINING_PATH",
    "GENERATOR_INSTRUCTION",
    "MIN_PAIRS_READY",
    "build_classifier_pair",
    "build_datasets",
    "build_generator_pair",
    "run",
]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main_cli())
