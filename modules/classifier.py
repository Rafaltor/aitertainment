"""Génération de commentaires via Ollama (local), modèle fine-tuné (Alpaca)."""

from __future__ import annotations

from typing import Any

import requests

import config
from config import ORDERED_T_TYPES
from modules.comment_quality import (
    assess_comment_quality,
    is_incomplete_comment,
    is_rambly_comment,
    is_repetitive_comment,
)
from modules.generator_prompt import (
    INFERENCE_MAX_CHARS_LONG,
    INFERENCE_MAX_CHARS_SHORT,
    INFERENCE_MAX_SENTENCES_LONG,
    INFERENCE_MAX_WORDS_LONG,
    INFERENCE_MAX_WORDS_SHORT,
    LengthBucket,
    build_alpaca_prompt,
    build_generator_input_block,
    length_buckets_for_generation,
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


# Longueur par défaut à l'inférence selon le registre émotionnel du T-type.
_T_TYPE_LENGTH_BUCKET: dict[str, LengthBucket] = {
    "T1": "short",
    "T2": "short",
    "T2b": "long",
    "T3a": "short",
    "T3b": "long",
    "T4": "short",
    "T5": "short",
}
_INFERENCE_NUM_PREDICT: dict[LengthBucket, int] = {"short": 24, "long": 72}
_INFERENCE_TEMPERATURE: dict[LengthBucket, float] = {"short": 0.65, "long": 0.5}
_INFERENCE_MAX_WORDS: dict[LengthBucket, int] = {
    "short": INFERENCE_MAX_WORDS_SHORT,
    "long": INFERENCE_MAX_WORDS_LONG,
}
_INFERENCE_MAX_CHARS: dict[LengthBucket, int] = {
    "short": INFERENCE_MAX_CHARS_SHORT,
    "long": INFERENCE_MAX_CHARS_LONG,
}


def _inference_normalize(
    text: str,
    *,
    length_bucket: LengthBucket,
    max_words: int | None = None,
    max_chars: int | None = None,
    max_sentences: int | None = None,
) -> str:
    return normalize_generator_output(
        text,
        length_bucket=length_bucket,
        max_words=max_words if max_words is not None else _INFERENCE_MAX_WORDS[length_bucket],
        max_chars=max_chars if max_chars is not None else _INFERENCE_MAX_CHARS[length_bucket],
        max_sentences=(
            max_sentences
            if max_sentences is not None
            else (INFERENCE_MAX_SENTENCES_LONG if length_bucket == "long" else None)
        ),
    )


def _prepare_generator_context(
    video_context: dict[str, Any] | None,
) -> tuple[str | list[Any], str]:
    ctx = video_context or {}
    raw_hashtags = ctx.get("hashtags") or []
    if isinstance(raw_hashtags, str):
        hashtags: str | list[Any] = raw_hashtags
    else:
        tags = [str(h).strip() for h in raw_hashtags if str(h).strip()]
        hashtags = tags
    merged_video_context = str(ctx.get("video_context") or "").strip()
    return hashtags, merged_video_context


def _generate_single_comment(
    *,
    t_type_profile: str,
    niches: list[str] | str,
    video_context: dict[str, Any] | None,
    model: str,
    ollama_url: str,
    length_bucket: LengthBucket,
    seen: set[str] | None = None,
    max_attempts: int = 4,
    instruction: str | None = None,
    options_overrides: dict[str, Any] | None = None,
    normalize_overrides: dict[str, Any] | None = None,
    extra_accept: Any | None = None,
) -> str:
    """Un commentaire Alpaca ; ``—`` si aucune sortie valide après les essais."""
    ctx = video_context or {}
    hashtags, merged_video_context = _prepare_generator_context(video_context)
    dedupe = seen if seen is not None else set()

    for _ in range(max_attempts):
        input_block = build_generator_input_block(
            t_type_profile=t_type_profile,
            niches=niches,
            caption=str(ctx.get("caption") or "").strip(),
            hashtags=hashtags,
            audio_id=str(ctx.get("audio_id") or ctx.get("audio") or "").strip(),
            video_context=merged_video_context,
            transcript=str(ctx.get("transcript") or "").strip(),
            visual_description=str(ctx.get("visual_description") or "").strip(),
            reel_id=str(ctx.get("reel_id") or ctx.get("video_id") or "").strip(),
            creator_username=str(
                ctx.get("creator_username") or ctx.get("username") or ""
            ).strip(),
            length_bucket=length_bucket,
        )
        prompt = build_alpaca_prompt(
            input_block,
            instruction,
            length_bucket=length_bucket,
        )
        num_predict = _INFERENCE_NUM_PREDICT[length_bucket]
        options: dict[str, Any] = {
            "temperature": _INFERENCE_TEMPERATURE[length_bucket],
            "top_p": 0.85,
            "repeat_penalty": 1.2,
            "num_predict": num_predict,
            "stop": (
                ["\n\n", "###", "@", "\n###"]
                if length_bucket == "long"
                else ["\n", "###", "@", "\n###"]
            ),
        }
        if options_overrides:
            options.update(options_overrides)
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
            "keep_alive": getattr(config, "OLLAMA_KEEP_ALIVE", "5m"),
        }
        try:
            resp = _http_post(
                ollama_url,
                json_body=body,
                timeout=int(getattr(config, "OLLAMA_REQUEST_TIMEOUT_S", 60)),
            )
            raw_text = _ollama_response_text(resp)
        except (requests.RequestException, ValueError) as e:
            raise ClassificationError(f"Erreur Ollama generator: {e}") from e

        comment = _inference_normalize(
            _clean_generated_comment_line(raw_text),
            length_bucket=length_bucket,
            **(normalize_overrides or {}),
        )
        if not comment or not assess_comment_quality(
            comment, length_bucket=length_bucket
        ).ok:
            continue
        if (
            is_repetitive_comment(comment)
            or is_incomplete_comment(comment)
            or is_rambly_comment(
                comment,
                max_sentences=INFERENCE_MAX_SENTENCES_LONG
                if length_bucket == "long"
                else 2,
            )
        ):
            continue
        if extra_accept is not None and not extra_accept(comment):
            continue
        key = comment.lower()
        if key in dedupe:
            continue
        dedupe.add(key)
        return comment
    return "—"


def _generate_comments_alpaca(
    *,
    t_type_profile: str,
    niches: list[str] | str,
    video_context: dict[str, Any] | None,
    model: str,
    ollama_url: str,
    num_comments: int = 3,
) -> list[str]:
    """N commentaires via le modèle fine-tuné (un appel Alpaca par commentaire)."""
    target_buckets = length_buckets_for_generation(num_comments)
    bucket_queue: list[LengthBucket] = list(target_buckets)

    out: list[str] = []
    seen: set[str] = set()
    for _ in range(max(num_comments * 4, num_comments)):
        if len(out) >= num_comments:
            break
        length_bucket = (
            bucket_queue[len(out)]
            if len(out) < len(bucket_queue)
            else "long"
        )
        comment = _generate_single_comment(
            t_type_profile=t_type_profile,
            niches=niches,
            video_context=video_context,
            model=model,
            ollama_url=ollama_url,
            length_bucket=length_bucket,
            seen=seen,
        )
        if comment == "—":
            continue
        out.append(comment)

    while len(out) < num_comments:
        out.append("—")
    return out[:num_comments]


def generate_comments_per_category(
    *,
    niches: list[str] | str = "",
    video_context: dict[str, Any] | None = None,
    t_types: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Un commentaire par T-type pour le même prompt (caption / contexte vidéo).

    Requiert ``OLLAMA_GENERATOR_MODEL`` dans ``.env``.
    """
    finetuned_model = getattr(config, "OLLAMA_GENERATOR_MODEL", "") or ""
    if not finetuned_model:
        raise ClassificationError(
            "OLLAMA_GENERATOR_MODEL non défini : le modèle fine-tuné est requis "
            "pour generate_comments_per_category."
        )
    categories = t_types or ORDERED_T_TYPES
    seen: set[str] = set()
    out: dict[str, str] = {}
    for t_type in categories:
        length_bucket = _T_TYPE_LENGTH_BUCKET.get(t_type, "short")
        out[t_type] = _generate_single_comment(
            t_type_profile=t_type,
            niches=niches,
            video_context=video_context,
            model=finetuned_model,
            ollama_url=config.OLLAMA_URL,
            length_bucket=length_bucket,
            seen=seen,
        )
    return out


def generate_comments(
    classification: dict[str, Any],
    comments_sample: list[str],
    *,
    niches: list[str] | str = "",
    t_type_profile: str | None = None,
    video_context: dict[str, Any] | None = None,
) -> list[str]:
    """Produit 3 commentaires via le modèle Ollama fine-tuné (format Alpaca).

    Le pipeline repose sur ``video_context`` (dict : caption, hashtags, audio,
    ``video_context`` fusionné) et ``t_type_profile`` (T-type **du commentateur**,
    lu dans ``watchlist.json`` côté caller — notre persona). Un appel Alpaca est
    émis par commentaire, avec filtrage qualité (cf. ``_generate_comments_alpaca``).

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
    return _generate_comments_alpaca(
        t_type_profile=profile_tt,
        niches=niches,
        video_context=video_context,
        model=finetuned_model,
        ollama_url=config.OLLAMA_URL,
    )
