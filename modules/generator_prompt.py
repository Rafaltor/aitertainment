"""Prompts et format Alpaca pour le générateur de commentaires (fine-tune + inférence)."""

from __future__ import annotations

import re
from typing import Any

from modules.named_axes import NAMED_AXES

GENERATOR_INSTRUCTION = (
    "Tu es un utilisateur Instagram. Écris UN commentaire court (5 à 10 mots). "
    "Pas d'emoji. Pas de @mention. Une seule phrase complète, jamais coupée."
)

# Médiane viral_comments ~8 mots ; cap inférence légèrement au-dessus.
MAX_GENERATOR_OUTPUT_WORDS = 10
MAX_GENERATOR_OUTPUT_CHARS = 72
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


def normalize_generator_output(text: str) -> str:
    """Nettoie et tronque (emoji, @, longueur) — aligné inférence + export dataset."""
    s = _EMOJI_RE.sub("", str(text or ""))
    s = _MENTION_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    words = s.split()
    if not words:
        return ""
    out = " ".join(words[:MAX_GENERATOR_OUTPUT_WORDS])
    if len(out) > MAX_GENERATOR_OUTPUT_CHARS:
        out = out[:MAX_GENERATOR_OUTPUT_CHARS].rsplit(" ", 1)[0]
    # Première phrase seulement (évite les enchaînements bavards).
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


def named_axes_block(named_axes: dict[str, Any]) -> str:
    values = [f"{axis}={_coerce_float(named_axes.get(axis)):.2f}" for axis in NAMED_AXES]
    return (
        "Profil créateur:\n"
        f"  {values[0]} {values[1]}\n"
        f"  {values[2]} {values[3]}\n"
        f"  {values[4]} {values[5]}\n"
        f"  {values[6]} {values[7]}\n"
        f"  {values[8]} {values[9]}"
    )


def build_generator_input_block(
    *,
    t_type_profile: str,
    niches: list[str] | str,
    caption: str = "",
    hashtags: str | list[Any] | None = None,
    audio_id: str = "",
    named_axes: dict[str, Any] | None = None,
) -> str:
    """Bloc ``input`` identique à ``prepare_dataset.py`` (entraînement = inférence)."""
    lines = [
        f"T-type commentateur: {t_type_profile}",
        f"Niches: {format_niches(niches)}",
    ]
    if named_axes:
        lines.append(named_axes_block(named_axes))
    lines.extend(
        [
            f"Caption: {normalize_generator_caption(caption)}",
            f"Hashtags: {format_hashtags(hashtags)}",
            f"Audio: {audio_id}",
        ]
    )
    return "\n".join(lines)


def build_alpaca_prompt(input_block: str, instruction: str = GENERATOR_INSTRUCTION) -> str:
    """Prompt brut Alpaca — doit matcher ``finetune_generator.ipynb`` cellule 4."""
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_block}\n\n"
        "### Response:\n"
    )
