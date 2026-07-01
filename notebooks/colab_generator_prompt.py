"""Standalone copy of ``modules/generator_prompt.py`` for Colab / RunPod.

Upload ce fichier dans ``/content/colab_generator_prompt.py`` à côté de
``generator_dataset.json`` — pas besoin de cloner le repo (privé).
"""

from __future__ import annotations

import re
from typing import Any, Literal

NAMED_AXES = ()  # conservé vide pour rétro-compat notebooks anciens

LengthBucket = Literal["short", "long"]

GENERATOR_LENGTH_SHORT_MAX_WORDS = 10
MAX_GENERATOR_OUTPUT_WORDS_SHORT = 10
MAX_GENERATOR_OUTPUT_WORDS_LONG = 80
MAX_TRAINING_COMMENT_WORDS = MAX_GENERATOR_OUTPUT_WORDS_LONG
MAX_GENERATOR_OUTPUT_CHARS_SHORT = 72
MAX_GENERATOR_OUTPUT_CHARS_LONG = 480
MAX_GENERATOR_CAPTION_CHARS = 300

_MENTION_RE = re.compile(r"@\w[\w.]*")
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "\U0001F3FB-\U0001F3FF"
    "\U0000200D"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)


def comment_length_bucket(text: str) -> LengthBucket:
    n = len(str(text or "").split())
    return "short" if n <= GENERATOR_LENGTH_SHORT_MAX_WORDS else "long"


def generator_length_target_label(bucket: LengthBucket) -> str:
    if bucket == "long":
        return "développé (11 à 60 mots)"
    return "court (3 à 10 mots)"


def build_generator_instruction(length_bucket: LengthBucket | None = None) -> str:
    base = (
        "Tu es un utilisateur Instagram. Écris UN commentaire en français. "
        "Pas d'emoji. Pas de @mention. "
    )
    if length_bucket == "short":
        length = "Court : 3 à 10 mots, une phrase percutante, jamais coupée."
    elif length_bucket == "long":
        length = (
            "Développé : 11 à 60 mots, une ou deux phrases complètes, "
            "jamais coupées — pour une punchline qui demande du contexte."
        )
    else:
        length = (
            "Respecte la longueur cible indiquée dans l'Input "
            "(court ou développé)."
        )
    context = (
        "Si Transcript et/ou Visuel sont fournis, utilise-les pour rendre le commentaire "
        "spécifique au contenu réel de la vidéo (dialogues, scène, personnages, ton). "
        "Les champs Creator et Reel identifient le post cible — ne commente "
        "que le contenu de ce Reel (pas un autre créateur)."
    )
    return f"{base}{length} {context}"


GENERATOR_INSTRUCTION = build_generator_instruction()


def normalize_generator_output(
    text: str,
    *,
    length_bucket: LengthBucket | None = None,
) -> str:
    s = _EMOJI_RE.sub("", str(text or ""))
    s = _MENTION_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    words = s.split()
    if not words:
        return ""
    bucket = length_bucket or comment_length_bucket(s)
    max_words = (
        MAX_GENERATOR_OUTPUT_WORDS_LONG
        if bucket == "long"
        else MAX_GENERATOR_OUTPUT_WORDS_SHORT
    )
    max_chars = (
        MAX_GENERATOR_OUTPUT_CHARS_LONG
        if bucket == "long"
        else MAX_GENERATOR_OUTPUT_CHARS_SHORT
    )
    out = " ".join(words[:max_words])
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0]
    if bucket == "short":
        out = re.split(r"[.!?…]+", out, maxsplit=1)[0].strip()
    return out.strip()


def normalize_generator_caption(text: str) -> str:
    cap = str(text or "").strip()
    if len(cap) <= MAX_GENERATOR_CAPTION_CHARS:
        return cap
    return cap[:MAX_GENERATOR_CAPTION_CHARS].rsplit(" ", 1)[0]


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_niches(niches: list[str] | str | None) -> str:
    if niches is None:
        return "(non précisée)"
    if isinstance(niches, str):
        s = niches.strip()
        return s or "(non précisée)"
    if isinstance(niches, list):
        clean = [str(n).strip() for n in niches if str(n).strip()]
        return ", ".join(clean) if clean else "(non précisée)"
    return "(non précisée)"


def format_hashtags(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(h).strip() for h in value if str(h).strip())
    return str(value).strip()


def build_generator_input_block(
    *,
    t_type_profile: str,
    niches: list[str] | str,
    caption: str = "",
    hashtags: str | list[Any] | None = None,
    audio_id: str = "",
    video_context: str = "",
    transcript: str = "",
    visual_description: str = "",
    reel_id: str = "",
    creator_username: str = "",
    length_bucket: LengthBucket | None = None,
) -> str:
    lines = [
        f"T-type commentateur: {t_type_profile}",
        f"Niches: {format_niches(niches)}",
    ]
    creator = str(creator_username or "").lstrip("@").strip()
    if creator:
        lines.append(f"Creator: @{creator}")
    rid = str(reel_id or "").strip()
    if rid:
        lines.append(f"Reel: {rid}")
    if length_bucket:
        lines.append(f"Longueur cible: {generator_length_target_label(length_bucket)}")
    lines.extend(
        [
            f"Caption: {normalize_generator_caption(caption)}",
            f"Hashtags: {format_hashtags(hashtags)}",
            f"Audio: {audio_id}",
        ]
    )
    vc = str(video_context or "").strip()
    if vc:
        lines.append(f"Contexte vidéo: {vc[:600]}")
    tr = str(transcript or "").strip()
    if tr:
        lines.append(f"Transcript: {tr[:500]}")
    vis = str(visual_description or "").strip()
    if vis:
        lines.append(f"Visuel: {vis[:400]}")
    return "\n".join(lines)


def build_alpaca_prompt(
    input_block: str,
    instruction: str | None = None,
    *,
    length_bucket: LengthBucket | None = None,
) -> str:
    instr = instruction or build_generator_instruction(length_bucket)
    return (
        f"### Instruction:\n{instr}\n\n"
        f"### Input:\n{input_block}\n\n"
        "### Response:\n"
    )


def length_buckets_for_generation(num_comments: int) -> list[LengthBucket]:
    n = max(1, int(num_comments))
    short_count = max(1, n // 3) if n > 1 else 1
    if n == 1:
        return ["short"]
    return ["short"] * short_count + ["long"] * (n - short_count)
