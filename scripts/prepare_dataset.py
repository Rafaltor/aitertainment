"""prepare_dataset.py — convertit ``training_comments.json`` en deux JSONL.

============================================================================
Objectif
============================================================================

Lire ``data/training_comments.json`` et produire **deux fichiers JSONL**
prêts à charger par Unsloth / QLoRA :

* ``data/dataset_classifier.jsonl`` — labellise un commentaire avec son
  T-type étant donné le contexte créateur + métriques.
* ``data/dataset_generator.jsonl`` — génère un commentaire crédible étant
  donné le T-type commentateur, niches, caption, hashtags, audio.

Chaque ligne est un objet JSON au format Alpaca
(``{instruction, input, output}``), encodé sur **une seule ligne** (pas de
pretty-print) — c'est le format consommé directement par
``datasets.load_dataset("json", ...)``.

============================================================================
Schéma d'entrée
============================================================================

Le script consomme un schéma **plat** : chaque entrée du training représente
**un commentaire** + son contexte vidéo, avec les champs au top-level :

::

    {
      "text": "mdr trop vrai",
      "t_type": "T2",                  # label du commentaire
      "t_type_profile": "T2",          # persona du commentateur (watchlist)
      "niches": ["humour", "sketch"],  # ou "niche": "humour" en rétro-compat
      "views": 500000,
      "comment_to_like_ratio": 0.283,
      "caption": "moment culte F1",
      "hashtags": ["F1", "monaco"],    # liste OU string
      "audio_id": "AUD123"
    }

La racine du fichier accepte indifféremment ``{"entries": [...]}`` (format
``dataset_builder.py``) ou une liste brute ``[...]``.

============================================================================
CLI
============================================================================

::

    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --training-path /tmp/training.json
    python scripts/prepare_dataset.py --output-dir /tmp/datasets
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import VALID_T_TYPES
from modules.generator_prompt import GENERATOR_INSTRUCTION, build_generator_input_block

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_TRAINING_PATH = _PROJECT_ROOT / "data" / "training_comments.json"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data"
VECTOR_STORE_PATH = Path("data/vector_store.json")
CLASSIFIER_FILENAME = "dataset_classifier.jsonl"
GENERATOR_FILENAME = "dataset_generator.jsonl"
# Rétro-compat notebooks Unsloth (tableau JSON unique).
CLASSIFIER_JSON_FILENAME = "classifier_dataset.json"
GENERATOR_JSON_FILENAME = "generator_dataset.json"

CLASSIFIER_INSTRUCTION = (
    "Classifie ce commentaire Instagram selon le type d'engagement."
)
_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lecture du training
# ---------------------------------------------------------------------------


def load_training(path: Path) -> list[dict]:
    """Charge ``training_comments.json`` ; retourne la liste ``entries[]``.

    Lève une exception **claire** :

    * ``FileNotFoundError`` si le fichier n'existe pas (laissé tel quel —
      le caller décide quoi faire ; ``main()`` retourne exit 1).
    * ``ValueError`` si le contenu n'est pas du JSON ou n'a pas la racine
      attendue.

    La racine peut être :

    * ``{"entries": [...]}`` (format produit par ``dataset_builder.py``).
    * ``[...]`` (liste brute — utile pour les datasets externes).

    Les éléments non-``dict`` sont silencieusement filtrés (best-effort,
    on ne casse pas le pipeline pour une ligne corrompue).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier training_comments.json introuvable : {path}"
        )
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON invalide dans {path} : {e}") from e

    if isinstance(data, dict):
        entries = data.get("entries")
        if not isinstance(entries, list):
            raise ValueError(
                f'"entries" doit être une liste dans {path} '
                f"(reçu : {type(entries).__name__})"
            )
        return [e for e in entries if isinstance(e, dict)]
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    raise ValueError(
        f"racine JSON doit être un objet ou une liste dans {path} "
        f"(reçu : {type(data).__name__})"
    )


# ---------------------------------------------------------------------------
# Helpers de formatage
# ---------------------------------------------------------------------------


def niches_str(entry: dict) -> str:
    """Retourne les niches jointes par ``", "``.

    Priorité :

    1. ``entry["niches"]`` (liste) — schéma 2026-05.
    2. ``entry["niche"]`` (string) — rétro-compat.
    3. ``"humour"`` — fallback ultime, jamais vide.

    Les items vides / non-string sont filtrés. Si après filtrage la liste
    est vide, on retombe sur l'étape suivante (puis sur ``"humour"``).
    Garantit donc une string **non vide** en sortie — le placeholder
    ``{niches}`` apparaît sinon vide dans le prompt et perturbe le LLM.
    """
    raw = entry.get("niches")
    if isinstance(raw, list):
        clean = [
            n.strip() for n in raw
            if isinstance(n, str) and n.strip()
        ]
        if clean:
            return ", ".join(clean)
    legacy = entry.get("niche")
    if isinstance(legacy, str) and legacy.strip():
        return legacy.strip()
    return "humour"


def _coerce_int(value: Any, default: int = 0) -> int:
    """Convertit en ``int`` ; ``default`` si ``None`` / invalide."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Convertit en ``float`` ; ``default`` si ``None`` / invalide."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


from modules.named_axes import NAMED_AXES


def load_vector_store(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Charge ``vector_store.json`` et indexe les entrées par username."""
    p = Path(path) if path is not None else VECTOR_STORE_PATH
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    if not p.exists():
        _LOG.info(
            "vector_store.json absent — named_axes non inclus dans le dataset"
        )
        return {}

    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        entries = [entry for entry in data if isinstance(entry, dict)]
    elif isinstance(data, dict):
        raw_entries = data.get("entries") or data.get("profiles") or []
        entries = [entry for entry in raw_entries if isinstance(entry, dict)]
    else:
        entries = []

    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        username = str(entry.get("username") or "").lstrip("@").strip().lower()
        if username:
            out[username] = entry
    return out


def _format_hashtags(value: Any) -> str:
    """Liste de hashtags → ``"a, b, c"`` ; string passée telle quelle ;
    ``""`` si absent / ``None``.

    On coerce les items non-string via ``str()`` pour éviter les
    ``TypeError`` à la jointure si le caller passe par exemple ``[1, 2]``.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(h) for h in value)
    if isinstance(value, str):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# Écriture atomique JSONL
# ---------------------------------------------------------------------------


def _atomic_write_jsonl(path: Path, lines: list[str]) -> None:
    """Écrit ``lines`` (chacune déjà JSON-encodée) en JSONL atomiquement.

    Pattern : ``tempfile`` dans le **même dossier** (donc même filesystem
    → ``os.replace`` atomique) puis ``os.replace(tmp, path)``. En cas
    d'erreur d'écriture, on tente de supprimer le tmp pour ne pas laisser
    de fichier orphelin.

    Si ``lines`` est vide, on écrit un fichier vide — c'est volontaire :
    on veut toujours produire les deux fichiers de sortie pour que les
    pipelines downstream (DVC, Make…) voient un artefact stable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line)
                fh.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Génération des datasets
# ---------------------------------------------------------------------------


def generate_classifier_dataset(
    entries: list[dict], output_path: Path
) -> int:
    """Écrit ``dataset_classifier.jsonl``. Retourne ``nb`` lignes écrites.

    Format de chaque ligne (``json.dumps``, sans pretty-print) ::

        {
          "instruction": "Classifie ce commentaire Instagram ...",
          "input": "Commentaire: ...\\nNiches du contenu: ...\\n"
                   "Vues: ...\\nRatio comments/likes: ...",
          "output": "T2"
        }

    Filtres stricts (entrée ignorée si) :

    * ``text`` absent ou vide après ``strip()``.
    * ``t_type`` absent ou pas dans ``VALID_T_TYPES``.

    Coercitions silencieuses :

    * ``views`` → ``int(entry.get("views", 0))``, ``0`` sur invalide.
    * ``ratio`` → ``float(entry.get("comment_to_like_ratio", 0.0))``,
      ``0.0`` sur invalide. Formaté avec **4 décimales** dans l'input.
    """
    lines: list[str] = []
    skipped_text = 0
    skipped_ttype = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "").strip()
        if not text:
            skipped_text += 1
            continue
        t_type = str(entry.get("t_type") or "").strip()
        if t_type not in VALID_T_TYPES:
            skipped_ttype += 1
            continue

        views = _coerce_int(entry.get("views", 0), default=0)
        ratio = _coerce_float(
            entry.get("comment_to_like_ratio", 0.0), default=0.0
        )
        n_str = niches_str(entry)

        input_block = (
            f"Commentaire: {text}\n"
            f"Niches du contenu: {n_str}\n"
            f"Vues: {views}\n"
            f"Ratio comments/likes: {ratio:.4f}"
        )
        record = {
            "instruction": CLASSIFIER_INSTRUCTION,
            "input": input_block,
            "output": t_type,
        }
        lines.append(json.dumps(record, ensure_ascii=False))

    _atomic_write_jsonl(output_path, lines)
    if skipped_text or skipped_ttype:
        _LOG.debug(
            "classifier: %d ligne(s) écrites (skip text=%d, t_type=%d)",
            len(lines), skipped_text, skipped_ttype,
        )
    return len(lines)


def _generator_input_block(
    *,
    t_type_profile: str,
    niches: str,
    caption: str,
    hashtags: str,
    audio_id: str,
    named_axes: dict[str, Any] | None,
) -> tuple[str, bool]:
    block = build_generator_input_block(
        t_type_profile=t_type_profile,
        niches=niches,
        caption=caption,
        hashtags=hashtags,
        audio_id=audio_id,
        named_axes=named_axes,
    )
    return block, bool(named_axes)


def generate_generator_dataset(
    entries: list[dict],
    output_path: Path,
    vector_store: dict[str, dict[str, Any]] | None = None,
) -> tuple[int, int]:
    """Écrit ``dataset_generator.jsonl``.

    Retourne ``(nb_lignes, nb_avec_vecteur)``.

    Format de chaque ligne ::

        {
          "instruction": "Tu es un utilisateur Instagram. Génère un ...",
          "input": "T-type commentateur: ...\\nNiches: ...\\nCaption: ...\\n"
                   "Hashtags: ...\\nAudio: ...",
          "output": "<texte du commentaire>"
        }

    Règles :

    * ``t_type_profile`` : ``entry.get("t_type_profile")`` puis
      ``entry.get("t_type")`` en fallback ; ``"(inconnu)"`` si les deux
      sont absents / non-string / vides après strip.
    * ``caption`` : ``entry.get("caption")`` ou ``""``.
    * ``hashtags`` : ``", ".join(...)`` si liste, ``str(...)`` si string,
      ``""`` si absent / ``None``.
    * ``audio_id`` : ``str(entry.get("audio_id") or "")``.
    * ``comment_text`` : ``entry.get("text") or ""`` ; entrée **ignorée**
      si vide après strip.
    """
    vector_store = vector_store or {}
    lines: list[str] = []
    skipped_text = 0
    with_vector = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        comment_text = str(entry.get("text") or "").strip()
        if not comment_text:
            skipped_text += 1
            continue

        # t_type_profile : priorité au champ explicite, fallback sur t_type.
        profile_raw = entry.get("t_type_profile")
        if isinstance(profile_raw, str) and profile_raw.strip():
            t_type_profile = profile_raw.strip()
        else:
            ttype_raw = entry.get("t_type")
            if isinstance(ttype_raw, str) and ttype_raw.strip():
                t_type_profile = ttype_raw.strip()
            else:
                t_type_profile = "(inconnu)"

        n_str = niches_str(entry)
        caption = str(entry.get("caption") or "")
        hashtags = _format_hashtags(entry.get("hashtags"))
        audio_id = str(entry.get("audio_id") or "")

        username = str(entry.get("username") or "").lstrip("@").strip().lower()
        store_entry = vector_store.get(username, {})
        named_axes = store_entry.get("named_axes")
        axes_dict = named_axes if isinstance(named_axes, dict) and named_axes else None

        input_block, has_vector = _generator_input_block(
            t_type_profile=t_type_profile,
            niches=n_str,
            caption=caption,
            hashtags=hashtags,
            audio_id=audio_id,
            named_axes=axes_dict,
        )
        if has_vector:
            with_vector += 1
        record = {
            "instruction": GENERATOR_INSTRUCTION,
            "input": input_block,
            "output": comment_text,
            "has_vector": has_vector,
        }
        lines.append(json.dumps(record, ensure_ascii=False))

    _atomic_write_jsonl(output_path, lines)
    if skipped_text:
        _LOG.debug(
            "generator: %d ligne(s) écrites (skip text=%d)",
            len(lines), skipped_text,
        )
    return len(lines), with_vector


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI. Retourne ``0`` si succès, ``1`` sur erreur.

    Erreurs catchées :

    * ``FileNotFoundError`` : training absent → log + exit 1, **aucun**
      fichier de sortie créé.
    * ``ValueError`` : JSON malformé → log + exit 1.
    * ``OSError`` : échec d'écriture (permissions, disque plein…) → log
      + exit 1. Le fichier déjà écrit avant l'erreur reste en place
      (pas de rollback inter-fichiers — on ne peut pas garantir ça
      sans sacrifier l'atomicité par-fichier).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convertit data/training_comments.json en deux JSONL "
            "(classifier + generator) prêts pour Unsloth/QLoRA."
        )
    )
    parser.add_argument(
        "--training-path",
        type=Path,
        default=DEFAULT_TRAINING_PATH,
        help=(
            f"Chemin du training_comments.json source "
            f"(défaut : {DEFAULT_TRAINING_PATH})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            f"Dossier où écrire les deux JSONL "
            f"(défaut : {DEFAULT_OUTPUT_DIR})."
        ),
    )
    args = parser.parse_args(argv)

    if not logging.getLogger().handlers:
        # Configure un handler basique uniquement si l'app cliente
        # n'en a pas déjà installé un (évite la double-log dans les
        # tests qui setUp leur propre logging).
        logging.basicConfig(
            level=logging.INFO, format="%(message)s"
        )

    try:
        entries = load_training(args.training_path)
    except FileNotFoundError as e:
        _LOG.error("%s", e)
        return 1
    except ValueError as e:
        _LOG.error("training_comments.json malformé : %s", e)
        return 1
    except OSError as e:
        _LOG.error("Erreur de lecture %s : %s", args.training_path, e)
        return 1

    classifier_path = args.output_dir / CLASSIFIER_FILENAME
    generator_path = args.output_dir / GENERATOR_FILENAME

    vector_store = load_vector_store(VECTOR_STORE_PATH)
    _LOG.info("vector_store chargé : %d comptes", len(vector_store))

    try:
        n_cls = generate_classifier_dataset(entries, classifier_path)
        n_gen, n_vec = generate_generator_dataset(
            entries, generator_path, vector_store=vector_store
        )
    except OSError as e:
        _LOG.error("Erreur d'écriture des datasets : %s", e)
        return 1

    _LOG.info("Classifier : %d entrées", n_cls)
    _LOG.info(
        "Generator : %d entrées dont %d avec vecteur 32D",
        n_gen,
        n_vec,
    )

    for jsonl_path, json_name in (
        (classifier_path, CLASSIFIER_JSON_FILENAME),
        (generator_path, GENERATOR_JSON_FILENAME),
    ):
        json_path = args.output_dir / json_name
        try:
            n_json = write_json_array_from_jsonl(jsonl_path, json_path)
            _LOG.info("Export notebook : %s (%d entrées)", json_path.name, n_json)
        except OSError as e:
            _LOG.error("Export %s échoué : %s", json_path, e)
            return 1

    return 0


def write_json_array_from_jsonl(jsonl_path: Path, json_path: Path) -> int:
    """Convertit un JSONL Alpaca en tableau JSON (notebooks ``json.load``)."""
    rows: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, json_path)
    return len(rows)


__all__ = [
    "CLASSIFIER_FILENAME",
    "CLASSIFIER_INSTRUCTION",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_TRAINING_PATH",
    "GENERATOR_FILENAME",
    "GENERATOR_INSTRUCTION",
    "VALID_T_TYPES",
    "VECTOR_STORE_PATH",
    "generate_classifier_dataset",
    "generate_generator_dataset",
    "load_training",
    "load_vector_store",
    "main",
    "niches_str",
]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
