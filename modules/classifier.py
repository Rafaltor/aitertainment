"""Génération de commentaires via Ollama (local), modèle fine-tuné (Alpaca)."""

from __future__ import annotations

from typing import Any

import requests

import config
from modules.comment_quality import (
    assess_comment_quality,
    is_incomplete_comment,
    is_repetitive_comment,
)
from modules.generator_prompt import (
    GENERATOR_INSTRUCTION,
    build_alpaca_prompt,
    build_generator_input_block,
    normalize_generator_output,
)

# Mots interdits côté générateur — listés ici pour permettre un check
# programmatique côté tests / debug (pas de filtrage automatique côté code
# de prod : on fait confiance au modèle fine-tuné, on ne fait pas de censure
# post-hoc qui dégraderait la cohérence des suggestions).
FORBIDDEN_GENERATOR_WORDS: frozenset[str] = frozenset({
    "incroyable",
    "impressionnant",
    "vraiment",
    "tellement",
    "félicitations",
    "magnifique",
    "superbe",
    "excellent",
    "contenu",
    "créateur",
    "vidéo",
    "post",
})


class ClassificationError(ValueError):
    """Réponse modèle illisible ou schéma invalide."""

    def __init__(self, message: str, *, raw_text: str | None = None) -> None:
        super().__init__(message)
        self.raw_text = raw_text


def _http_post(url: str, *, json_body: dict[str, Any], timeout: int) -> requests.Response:
    return requests.post(url, json=json_body, timeout=timeout)


def _ollama_response_text(resp: requests.Response) -> str:
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError as e:
        raise ValueError(f"corps HTTP non-JSON: {e}") from e
    return str(data.get("response", "")).strip()


def _clean_generated_comment_line(text: str) -> str:
    """Première ligne utile après ``### Response:`` (sans JSON legacy)."""
    line = str(text or "").strip().split("\n", 1)[0].strip()
    if line.startswith("{"):
        return ""
    for prefix in ("### Response:", "### Input:", "### Instruction:"):
        if line.startswith(prefix):
            line = line[len(prefix) :].strip()
    return line.strip('"').strip("'")


def _generate_comments_alpaca(
    *,
    t_type_profile: str,
    niches: list[str] | str,
    video_context: dict[str, Any] | None,
    named_axes: dict[str, Any] | None,
    model: str,
    ollama_url: str,
    num_comments: int = 3,
) -> list[str]:
    """3 commentaires via le modèle fine-tuné (un appel Alpaca par commentaire)."""
    ctx = video_context or {}
    raw_hashtags = ctx.get("hashtags") or []
    if isinstance(raw_hashtags, str):
        hashtags: str | list[Any] = raw_hashtags
    else:
        tags = [str(h).strip() for h in raw_hashtags if str(h).strip()]
        hashtags = tags

    merged_video_context = str(ctx.get("video_context") or "").strip()
    input_block = build_generator_input_block(
        t_type_profile=t_type_profile,
        niches=niches,
        caption=str(ctx.get("caption") or "").strip(),
        hashtags=hashtags,
        audio_id=str(ctx.get("audio_id") or ctx.get("audio") or "").strip(),
        video_context=merged_video_context,
        reel_id=str(ctx.get("reel_id") or ctx.get("video_id") or "").strip(),
        creator_username=str(ctx.get("creator_username") or ctx.get("username") or "").strip(),
        named_axes=named_axes if isinstance(named_axes, dict) and named_axes else None,
    )
    prompt = build_alpaca_prompt(input_block, instruction=GENERATOR_INSTRUCTION)

    out: list[str] = []
    seen: set[str] = set()
    for _ in range(max(num_comments * 4, num_comments)):
        if len(out) >= num_comments:
            break
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.15,
                "num_predict": 28,
                "stop": ["\n", "###", "@"],
            },
            "keep_alive": 0,
        }
        try:
            resp = _http_post(ollama_url, json_body=body, timeout=120)
            raw_text = _ollama_response_text(resp)
        except (requests.RequestException, ValueError) as e:
            raise ClassificationError(f"Erreur Ollama generator: {e}") from e

        comment = normalize_generator_output(_clean_generated_comment_line(raw_text))
        if not comment or not assess_comment_quality(comment).ok:
            continue
        if is_repetitive_comment(comment) or is_incomplete_comment(comment):
            continue
        key = comment.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(comment)

    while len(out) < num_comments:
        out.append("—")
    return out[:num_comments]


def generate_comments(
    classification: dict[str, Any],
    comments_sample: list[str],
    *,
    niches: list[str] | str = "",
    t_type_profile: str | None = None,
    video_context: dict[str, Any] | None = None,
    named_axes: dict[str, Any] | None = None,
) -> list[str]:
    """Produit 3 commentaires via le modèle Ollama fine-tuné (format Alpaca).

    Le pipeline repose sur ``video_context`` (dict : caption, hashtags, audio,
    ``video_context`` fusionné), ``t_type_profile`` (T-type **du commentateur**, lu dans
    ``watchlist.json`` côté caller — notre persona) et ``named_axes`` (profil
    créateur 32D). Un appel Alpaca est émis par commentaire, avec filtrage
    qualité (cf. ``_generate_comments_alpaca``).

    ``niches`` accepte ``list[str]`` (schéma 2026-05) ou ``str`` (rétro-compat).
    ``t_type_profile=None`` produit ``"(non précisé)"`` dans le prompt.

    Les paramètres ``classification`` et ``comments_sample`` sont conservés
    pour compatibilité d'appel (Watcher / generator de masse) mais ne sont
    plus utilisés par le chemin fine-tuné.

    Requiert ``OLLAMA_GENERATOR_MODEL`` dans ``.env``.
    """
    profile_tt = (t_type_profile or "").strip() or "(non précisé)"
    finetuned_model = getattr(config, "OLLAMA_GENERATOR_MODEL", "") or ""
    if not finetuned_model:
        raise ClassificationError(
            "OLLAMA_GENERATOR_MODEL non défini : le modèle fine-tuné est requis "
            "pour generate_comments."
        )
    axes = named_axes if isinstance(named_axes, dict) and named_axes else None
    return _generate_comments_alpaca(
        t_type_profile=profile_tt,
        niches=niches,
        video_context=video_context,
        named_axes=axes,
        model=finetuned_model,
        ollama_url=config.OLLAMA_URL,
    )
