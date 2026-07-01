"""Prompts et format Alpaca pour le générateur de commentaires (fine-tune + inférence)."""

from __future__ import annotations

import re
from typing import Any, Literal

LengthBucket = Literal["short", "long"]

# Seuil dataset : ≤10 mots = court, >10 = développé (aligné médiane virale ~8 mots).
GENERATOR_LENGTH_SHORT_MAX_WORDS = 10
MAX_GENERATOR_OUTPUT_WORDS_SHORT = 10
MAX_GENERATOR_OUTPUT_WORDS_LONG = 80
MAX_TRAINING_COMMENT_WORDS = MAX_GENERATOR_OUTPUT_WORDS_LONG
MAX_GENERATOR_OUTPUT_CHARS_SHORT = 72
MAX_GENERATOR_OUTPUT_CHARS_LONG = 480
# Plafonds inférence (plus stricts que le training pour éviter les pavés).
INFERENCE_MAX_WORDS_SHORT = 10
INFERENCE_MAX_WORDS_LONG = 40
INFERENCE_MAX_CHARS_SHORT = 72
INFERENCE_MAX_CHARS_LONG = 280
INFERENCE_MAX_SENTENCES_LONG = 2
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
    """Classe un commentaire viral : court (≤10 mots) ou développé (>10)."""
    n = len(str(text or "").split())
    return "short" if n <= GENERATOR_LENGTH_SHORT_MAX_WORDS else "long"


def generator_length_target_label(bucket: LengthBucket) -> str:
    if bucket == "long":
        return "développé (11 à 40 mots)"
    return "court (3 à 10 mots)"


def build_generator_instruction(length_bucket: LengthBucket | None = None) -> str:
    """Instruction Alpaca — précise court vs développé si ``length_bucket`` est fourni."""
    base = (
        "Tu es un utilisateur Instagram. Écris UN commentaire en français. "
        "Pas d'emoji. Pas de @mention. "
    )
    if length_bucket == "short":
        length = (
            "Court : 3 à 10 mots, une phrase percutante, jamais coupée."
        )
    elif length_bucket == "long":
        length = (
            "Développé : 11 à 40 mots, une ou deux phrases complètes, "
            "jamais coupées — ancré sur la Caption, pas de digression."
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


def clamp_sentences(text: str, max_sentences: int = 2) -> str:
    """Garde au plus ``max_sentences`` phrases (séparateurs . ! ? …)."""
    s = str(text or "").strip()
    if not s or max_sentences < 1:
        return s
    parts = re.split(r"(?<=[.!?…])\s+", s)
    kept = [p.strip() for p in parts if p.strip()]
    if len(kept) <= max_sentences:
        return s
    return " ".join(kept[:max_sentences]).strip()


def normalize_generator_output(
    text: str,
    *,
    length_bucket: LengthBucket | None = None,
    max_words: int | None = None,
    max_chars: int | None = None,
    max_sentences: int | None = None,
) -> str:
    """Nettoie emoji/@ ; tronque selon le bucket court ou développé."""
    s = _EMOJI_RE.sub("", str(text or ""))
    s = _MENTION_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    words = s.split()
    if not words:
        return ""
    bucket = length_bucket or comment_length_bucket(s)
    word_cap = max_words if max_words is not None else (
        MAX_GENERATOR_OUTPUT_WORDS_LONG
        if bucket == "long"
        else MAX_GENERATOR_OUTPUT_WORDS_SHORT
    )
    char_cap = max_chars if max_chars is not None else (
        MAX_GENERATOR_OUTPUT_CHARS_LONG
        if bucket == "long"
        else MAX_GENERATOR_OUTPUT_CHARS_SHORT
    )
    out = " ".join(words[:word_cap])
    if len(out) > char_cap:
        out = out[:char_cap].rsplit(" ", 1)[0]
    if bucket == "short":
        out = re.split(r"[.!?…]+", out, maxsplit=1)[0].strip()
    elif max_sentences:
        out = clamp_sentences(out, max_sentences=max_sentences)
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
    """Bloc ``input`` identique à ``prepare_dataset.py`` (entraînement = inférence)."""
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
    """Prompt brut Alpaca — doit matcher ``finetune_generator.ipynb`` cellule 4."""
    instr = instruction or build_generator_instruction(length_bucket)
    return (
        f"### Instruction:\n{instr}\n\n"
        f"### Input:\n{input_block}\n\n"
        "### Response:\n"
    )


def length_buckets_for_generation(num_comments: int) -> list[LengthBucket]:
    """Mix court/développé à l'inférence (~1/3 court, 2/3 développé)."""
    n = max(1, int(num_comments))
    short_count = max(1, n // 3) if n > 1 else 1
    if n == 1:
        return ["short"]
    return ["short"] * short_count + ["long"] * (n - short_count)
