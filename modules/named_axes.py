"""Axes nommés comparables entre créateurs (projection sur ancres sémantiques).

Chaque axe est défini par deux descriptions (pôle 0 et pôle 1), embeddées une fois
avec le même modèle que les profils. Le score d'un créateur est la position linéaire
de son embedding entre ces deux pôles — **échelle globale**, pas de min-max par compte.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable

_LOG = logging.getLogger("aitertainment.named_axes")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AXIS_ANCHORS_PATH = PROJECT_ROOT / "data" / "axis_anchors.json"

NAMED_AXES = (
    "scripted_vs_raw",
    "solo_vs_collab",
    "fictional_vs_real",
    "energy_level",
    "production_quality",
    "format_length",
    "distance_parasociale",
    "interaction_style",
    "mainstream_vs_niche",
    "safe_vs_edgy",
)

# Pôle 0.0 = premier texte, pôle 1.0 = second (descriptions Reels / créateur FR).
AXIS_ANCHOR_TEXTS: dict[str, tuple[str, str]] = {
    "scripted_vs_raw": (
        "Créateur Reels très scripté : répliques écrites, sketch monté, jeu d'acteur, structure narrative claire.",
        "Créateur Reels brut et spontané : improvisation, face caméra sans texte, réaction authentique non répétée.",
    ),
    "solo_vs_collab": (
        "Créateur seul face caméra, monologue, aucun duo ni collab visible.",
        "Créateur en duo ou collab : plusieurs personnes, voix multiples, feat ou UGC multi-personnes.",
    ),
    "fictional_vs_real": (
        "Univers fictionnel, personnages, sitcom, mise en scène — pas du quotidien réel documenté.",
        "Contenu ancré dans la vraie vie : vrai lieu, faits réels, témoignage ou documentaire court.",
    ),
    "energy_level": (
        "Ton calme et posé : voix basse, peu de cuts, rythme lent, peu d'agitation.",
        "Ton très énergique : cris, jump cuts, hype, mouvement constant, stimulation forte.",
    ),
    "production_quality": (
        "Tournage téléphone brut, son moyen, peu de montage, esthétique amateur authentique.",
        "Montage soigné, sound design, transitions, esthétique professionnelle ou semi-pro.",
    ),
    "format_length": (
        "Reels courts et denses : une punchline, format snack, peu de développement.",
        "Reels plus longs : storytelling étalé, plusieurs beats, arc narratif 30–60 secondes.",
    ),
    "distance_parasociale": (
        "Proximité audience : tutoiement, intimité, partage personnel, communauté soudée.",
        "Distance marquée : expert distant, ironie froide, persona inaccessible ou autoritaire.",
    ),
    "interaction_style": (
        "Peu d'appel à l'interaction : sketch fermé, le public observe sans être sollicité.",
        "Pousse au commentaire : questions, CTA, débat, sondage implicite dans la caption.",
    ),
    "mainstream_vs_niche": (
        "Références pop culture mainstream, trends généralistes, humour large public.",
        "Humour de niche, codes communauté restreinte, inside jokes hors cercle.",
    ),
    "safe_vs_edgy": (
        "Contenu brand-safe, famille, neutre, peu de provocation.",
        "Contenu provocateur, edgy, dark humor, clash ou sujets sensibles assumés.",
    ),
}


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm < 1e-12:
        return [float(x) for x in vector]
    return [float(x) / norm for x in vector]


def _project_on_axis(
    embedding: list[float],
    low_vec: list[float],
    high_vec: list[float],
) -> float:
    """Score 0–1 : position du créateur entre les embeddings des pôles bas / haut."""
    emb = _l2_normalize(embedding)
    low = _l2_normalize(low_vec)
    high = _l2_normalize(high_vec)
    diff = [h - lo for h, lo in zip(high, low, strict=True)]
    norm_diff = math.sqrt(sum(d * d for d in diff))
    if norm_diff < 1e-12:
        return 0.5
    direction = [d / norm_diff for d in diff]
    low_s = sum(a * b for a, b in zip(low, direction))
    high_s = sum(a * b for a, b in zip(high, direction))
    emb_s = sum(a * b for a, b in zip(emb, direction))
    span = high_s - low_s
    if abs(span) < 1e-12:
        return 0.5
    value = (emb_s - low_s) / span
    return max(0.0, min(1.0, float(value)))


def score_named_axes(
    embedding: list[float],
    anchors: dict[str, dict[str, list[float]]],
) -> dict[str, float]:
    """Calcule les 10 axes pour un embedding (échelle comparable entre comptes)."""
    if len(embedding) == 0:
        return {axis: 0.0 for axis in NAMED_AXES}
    out: dict[str, float] = {}
    for axis in NAMED_AXES:
        pole = anchors.get(axis)
        if not pole:
            out[axis] = 0.0
            continue
        low_vec = pole.get("low")
        high_vec = pole.get("high")
        if not isinstance(low_vec, list) or not isinstance(high_vec, list):
            out[axis] = 0.0
            continue
        if len(low_vec) != len(embedding) or len(high_vec) != len(embedding):
            _LOG.warning(
                "Axe %s : dimension ancres (%d/%d) ≠ embedding %d — ignoré.",
                axis,
                len(low_vec) if isinstance(low_vec, list) else 0,
                len(high_vec) if isinstance(high_vec, list) else 0,
                len(embedding),
            )
            out[axis] = 0.0
            continue
        out[axis] = _project_on_axis(embedding, low_vec, high_vec)
    return out


def build_axis_anchors(
    embed_fn: Callable[[str], list[float] | None],
    *,
    model: str,
    expected_dim: int | None = None,
    path: Path | None = None,
) -> dict[str, dict[str, list[float]]] | None:
    """Embede les 20 pôles (10 axes × 2) et persiste ``axis_anchors.json``."""
    anchors: dict[str, dict[str, list[float]]] = {}
    for axis in NAMED_AXES:
        low_text, high_text = AXIS_ANCHOR_TEXTS[axis]
        low_vec = embed_fn(low_text)
        high_vec = embed_fn(high_text)
        if low_vec is None or high_vec is None:
            _LOG.error("Impossible d'embedder les ancres pour l'axe %s.", axis)
            return None
        if expected_dim is not None and (
            len(low_vec) != expected_dim or len(high_vec) != expected_dim
        ):
            _LOG.error(
                "Dimension ancres inattendue pour %s (%d / %d, attendu %d).",
                axis,
                len(low_vec),
                len(high_vec),
                expected_dim,
            )
            return None
        anchors[axis] = {"low": [float(x) for x in low_vec], "high": [float(x) for x in high_vec]}

    out_path = path or AXIS_ANCHORS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "dim": len(next(iter(anchors.values()))["low"]),
        "anchors": anchors,
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _LOG.info("Ancres d'axes enregistrées (%s, %dD) → %s", model, payload["dim"], out_path)
    return anchors


def load_axis_anchors(
    *,
    model: str,
    expected_dim: int | None = None,
    path: Path | None = None,
) -> dict[str, dict[str, list[float]]] | None:
    """Charge les ancres si modèle (et dimension) correspondent."""
    p = path or AXIS_ANCHORS_PATH
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("axis_anchors.json illisible (%s).", exc)
        return None
    if not isinstance(data, dict):
        return None
    if str(data.get("model") or "").strip() != model.strip():
        _LOG.info(
            "axis_anchors.json : modèle %r ≠ %r — reconstruction nécessaire.",
            data.get("model"),
            model,
        )
        return None
    dim = data.get("dim")
    anchors = data.get("anchors")
    if not isinstance(anchors, dict):
        return None
    if expected_dim is not None and dim is not None and int(dim) != expected_dim:
        _LOG.info(
            "axis_anchors.json : %dD enregistré, %dD attendu — reconstruction nécessaire.",
            dim,
            expected_dim,
        )
        return None
    return anchors


def ensure_axis_anchors(
    embed_fn: Callable[[str], list[float] | None],
    *,
    model: str,
    expected_dim: int | None = None,
    force_rebuild: bool = False,
) -> dict[str, dict[str, list[float]]] | None:
    """Charge ou reconstruit les ancres d'axes."""
    if not force_rebuild:
        loaded = load_axis_anchors(model=model, expected_dim=expected_dim)
        if loaded is not None:
            return loaded
    return build_axis_anchors(
        embed_fn, model=model, expected_dim=expected_dim
    )


def refresh_named_axes_in_store(
    vector_store: list[dict[str, Any]],
    anchors: dict[str, dict[str, list[float]]],
    *,
    expected_dim: int | None = None,
) -> int:
    """Recalcule ``named_axes`` pour toutes les entrées ayant ``embedding_raw``."""
    updated = 0
    for entry in vector_store:
        raw = entry.get("embedding_raw")
        if not isinstance(raw, list) or not raw:
            continue
        if expected_dim is not None and len(raw) != expected_dim:
            continue
        entry["named_axes"] = score_named_axes(raw, anchors)
        updated += 1
    return updated
