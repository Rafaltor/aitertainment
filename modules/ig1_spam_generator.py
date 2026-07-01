"""Génération IG1 : 1 base filtrée (Ollama) → 1 substitution lowtaper67."""

from __future__ import annotations

import logging
import re
from typing import Any

import config
from modules.classifier import ClassificationError, _generate_single_comment
from modules.comment_quality import (
    assess_comment_quality,
    is_incomplete_comment,
    is_repetitive_comment,
    is_rambly_comment,
)
from modules.generator_prompt import build_generator_instruction
from modules.ollama_text import call_ollama_text
from modules.reel_enrichment import _clean_fused_context

_LOG = logging.getLogger("aitertainment.ig1_spam_generator")

_SPAM_BASE_MAX_ATTEMPTS = int(
    getattr(config, "SPAM_GENERATOR_BASE_MAX_ATTEMPTS", 6) or 6
)

# Mots à ne pas remplacer (adverbes, déterminants, prépositions, etc.)
_NON_SUBSTITUTABLE = frozenset(
    {
        "pas", "plus", "tres", "très", "trop", "bien", "mal", "peu", "tout", "tous",
        "toute", "correctement", "vraiment", "juste", "aussi", "encore", "deja", "déjà",
        "jamais", "comme", "pour", "dans", "avec", "sans", "sur", "sous", "chez", "mais",
        "donc", "pov", "oral", "ton", "ta", "mon", "son", "leur", "les", "des", "une",
        "un", "le", "la", "de", "du", "en", "et", "ou", "ni", "car", "que", "qui", "quoi",
        "tu", "te", "me", "se", "ce", "ca", "ça", "lui", "elle", "ils", "nous", "vous",
        "je", "on", "y", "ne", "au", "aux", "du", "des",
    }
)


def spam_keyword() -> str:
    return (config.SPAM_COMMENT_KEYWORD or config.SPAM_COMMENT_TEXT or "lowtaper67").strip()


def comment_contains_keyword(comment: str, keyword: str | None = None) -> bool:
    kw = (keyword or spam_keyword()).strip()
    if not kw:
        return False
    return kw.lower() in str(comment or "").lower()


def keyword_occurrence_count(comment: str, keyword: str | None = None) -> int:
    kw = (keyword or spam_keyword()).strip()
    if not kw:
        return 0
    return len(re.findall(re.escape(kw), str(comment or ""), flags=re.IGNORECASE))


def keyword_well_integrated(comment: str, keyword: str | None = None) -> bool:
    kw = (keyword or spam_keyword()).strip()
    if not kw or keyword_occurrence_count(comment, kw) != 1:
        return False
    if str(comment or "").strip().lower() == kw.lower():
        return False
    return True


def is_publishable_spam_comment(comment: str, keyword: str | None = None) -> bool:
    kw = (keyword or spam_keyword()).strip()
    text = str(comment or "").strip()
    return bool(text and kw and comment_contains_keyword(text, kw))


def is_publishable_base_comment(comment: str) -> bool:
    """Filtre qualité sur la passe 1 (commentaire naturel avant weave)."""
    text = str(comment or "").strip()
    if not text:
        return False
    if not assess_comment_quality(text, length_bucket="short").ok:
        return False
    if is_incomplete_comment(text):
        return False
    if is_repetitive_comment(text):
        return False
    if is_rambly_comment(text, max_sentences=1):
        return False
    words = text.split()
    if len(words) < 3 or len(words) > 10:
        return False
    low = text.lower()
    if re.search(r"\bmais\s+tu\.?$", low):
        return False
    if low.startswith("pov") and not text.rstrip().endswith((".", "!", "?", "…")):
        return False
    return True


def _word_core(word: str) -> str:
    return re.sub(r"[^\w']", "", str(word or "")).lower()


def _is_substitutable_word(word: str) -> bool:
    """Verbe, sujet ou nom commun — pas adverbe / fonctionnel."""
    core = _word_core(word)
    if len(core) < 3:
        return False
    if core in _NON_SUBSTITUTABLE:
        return False
    if core.endswith("ment"):
        return False
    return True


def _replaced_word(base: str, woven: str) -> str | None:
    bw = base.split()
    ww = woven.split()
    if len(bw) != len(ww):
        return None
    changed = [
        bw[i]
        for i, (a, b) in enumerate(zip(bw, ww))
        if _word_core(a) != _word_core(b)
    ]
    if len(changed) != 1:
        return None
    return changed[0]


def _clean_weave_output(raw: str) -> str:
    line = str(raw or "").strip().split("\n", 1)[0].strip()
    for prefix in ("### Response:", "Commentaire:", "Réponse:"):
        if line.lower().startswith(prefix.lower()):
            line = line[len(prefix) :].strip()
    if (line.startswith("«") and line.endswith("»")) or (
        len(line) >= 2 and line[0] == line[-1] and line[0] in "\"'"
    ):
        line = line[1:-1].strip()
    return re.sub(r"\s+", " ", line).strip()


def _weave_substitute_prompt(base: str, kw: str) -> str:
    n = len(base.split())
    return f"""Remplace UN SEUL mot du commentaire par «{kw}» (orthographe exacte).

Commentaire ({n} mots) : {base}

RÈGLES STRICTES :
- Choisis un VERBE, un SUJET (nom/pronom) ou un NOM COMMUN à remplacer — pas un adverbe (-ment), pas une préposition, pas «pas/trop/très/juste»
- Remplace ce mot par «{kw}» — ne rajoute AUCUN mot
- La réponse doit faire exactement {n} mots
- Ne recopie pas de caption ni de contexte externe
- Réponds uniquement la phrase finale

Exemples :
«le prof est mort» → «le lowtaper67 est mort» (sujet)
«il gère trop bien» → «il lowtaper67 trop bien» (verbe)
«trop vrai mdrr» → «trop lowtaper67 mdrr» (nom/adj)"""


def _is_valid_substitution_weave(base: str, woven: str, kw: str) -> bool:
    base_s = str(base or "").strip()
    woven_s = str(woven or "").strip()
    kw_l = kw.lower()
    if not woven_s or kw_l not in woven_s.lower():
        return False
    if keyword_occurrence_count(woven_s, kw) != 1:
        return False
    if base_s:
        if len(woven_s.split()) != len(base_s.split()):
            return False
        if woven_s.lower() == base_s.lower():
            return False
        replaced = _replaced_word(base_s, woven_s)
        if not replaced or not _is_substitutable_word(replaced):
            return False
    return True


def _fallback_substitute_word(base: str, kw: str) -> str:
    """Repli : remplace le meilleur verbe/nom substituable."""
    words = base.split()
    if not words:
        return kw
    best_i = -1
    best_score = -1
    for i, w in enumerate(words):
        if not _is_substitutable_word(w):
            continue
        core = _word_core(w)
        score = len(core)
        if re.search(r"(er|ir|re|é|ée|és|ait|ais)$", core):
            score += 3
        if len(core) >= 4:
            score += 1
        if score > best_score:
            best_score = score
            best_i = i
    if best_i < 0:
        return ""
    w = words[best_i]
    punct = ""
    if w and not w[-1].isalnum():
        punct = w[-1]
        w = w[:-1]
    words[best_i] = kw + punct
    return " ".join(words)


def weave_keyword_into_comment(
    base_comment: str,
    keyword: str,
    *,
    caption: str = "",
    video_context: str = "",
) -> str:
    """Passe 2 : remplace 1 verbe/sujet/nom par le mot-clé."""
    del caption, video_context
    base = str(base_comment or "").strip()
    kw = str(keyword or "").strip()
    if not base or not kw:
        return ""
    model = (
        getattr(config, "SPAM_WEAVE_OLLAMA_MODEL", None)
        or config.OLLAMA_GENERATOR_MODEL
        or config.OLLAMA_MODEL
        or ""
    ).strip()
    if not model:
        return _fallback_substitute_word(base, kw)

    prompt = _weave_substitute_prompt(base, kw)
    for attempt in range(2):
        raw = call_ollama_text(
            prompt,
            model=model,
            num_predict=64,
            temperature=0.2 if attempt else 0.35,
        )
        woven = _clean_weave_output(raw)
        if _is_valid_substitution_weave(base, woven, kw):
            return woven
        _LOG.debug("weave tentative %d rejetée : %r", attempt + 1, woven[:80])

    fallback = _fallback_substitute_word(base, kw)
    if _is_valid_substitution_weave(base, fallback, kw):
        _LOG.info("weave repli : %r → %r", base[:60], fallback[:60])
        return fallback
    return ""


def _fuse_ig1_context(
    transcript: str,
    visual_description: str,
    caption: str = "",
) -> str:
    if not (transcript or visual_description):
        return ""
    model = (getattr(config, "SPAM_FUSION_OLLAMA_MODEL", None) or config.OLLAMA_MODEL or "").strip()
    if not model:
        return ""
    prompt = f"""Fusionne en 2-4 phrases le transcript et la description visuelle de ce reel Instagram :

Caption : {str(caption or "")[:300]}
Transcript : {str(transcript or "")[:1000]}
Visuel : {str(visual_description or "")[:600]}

Réponds uniquement la fusion en français."""
    raw = call_ollama_text(prompt, model=model, num_predict=200, temperature=0.3)
    return _clean_fused_context(raw) if raw else ""


def _build_video_context(
    *,
    caption: str,
    transcript: str,
    visual_description: str,
    media_id: str,
    username: str,
) -> tuple[dict[str, Any], str]:
    caption_s = str(caption or "").strip()
    merged = _fuse_ig1_context(transcript, visual_description, caption_s)
    return (
        {
            "caption": caption_s,
            "hashtags": [],
            "audio_id": "",
            "video_context": merged,
            "video_id": str(media_id or "").strip(),
            "username": str(username or "").lstrip("@").strip(),
        },
        merged,
    )


def _generate_base_comment(
    *,
    video_context: dict[str, Any],
    t_type: str,
    model: str,
    seen: set[str] | None,
) -> str:
    return _generate_single_comment(
        t_type_profile=t_type,
        niches=["humour"],
        video_context=video_context,
        model=model,
        ollama_url=config.OLLAMA_URL,
        length_bucket="short",
        seen=seen,
        max_attempts=4,
        instruction=build_generator_instruction("short"),
        extra_accept=is_publishable_base_comment,
    )


def generate_ig1_spam_comments(
    *,
    caption: str = "",
    transcript: str = "",
    visual_description: str = "",
    media_id: str = "",
    username: str = "",
    keyword: str | None = None,
    seen: set[str] | None = None,
    base_count: int | None = None,
) -> list[tuple[str, str]]:
    """1 base filtrée + 1 weave. ``base_count`` ignoré (rétro-compat, toujours 1)."""
    del base_count
    kw = (keyword or spam_keyword()).strip() or "lowtaper67"
    model = (config.OLLAMA_GENERATOR_MODEL or "").strip()
    if not model:
        raise ClassificationError("OLLAMA_GENERATOR_MODEL non défini.")

    t_type = (getattr(config, "SPAM_GENERATOR_T_TYPE", None) or "T3a").strip()
    video_context, merged = _build_video_context(
        caption=caption,
        transcript=transcript,
        visual_description=visual_description,
        media_id=media_id,
        username=username,
    )

    for attempt in range(max(1, _SPAM_BASE_MAX_ATTEMPTS)):
        base = _generate_base_comment(
            video_context=video_context,
            t_type=t_type,
            model=model,
            seen=seen,
        )
        if not base or base == "—":
            _LOG.debug("passe 1 tentative %d : pas de base qualité", attempt + 1)
            continue
        woven = weave_keyword_into_comment(base, kw)
        if woven:
            _LOG.info("IG1 OK base=%r → woven=%r", base[:80], woven[:80])
            return [(base, woven)]

    _LOG.warning("IG1 : échec après %d essais base+weave", _SPAM_BASE_MAX_ATTEMPTS)
    return []


def generate_ig1_spam_comment(
    *,
    caption: str = "",
    transcript: str = "",
    visual_description: str = "",
    media_id: str = "",
    username: str = "",
    keyword: str | None = None,
    seen: set[str] | None = None,
    max_attempts: int | None = None,
) -> str:
    """Retourne le commentaire tissé (1 base + 1 weave)."""
    pairs = generate_ig1_spam_comments(
        caption=caption,
        transcript=transcript,
        visual_description=visual_description,
        media_id=media_id,
        username=username,
        keyword=keyword,
        seen=seen,
    )
    kw = (keyword or spam_keyword()).strip()
    for _base, woven in pairs:
        text = normalize_for_instagram(woven, keyword=kw)
        if text:
            if seen is not None:
                seen.add(text.lower())
            return text
    return ""


def normalize_for_instagram(comment: str, *, keyword: str | None = None) -> str:
    text = re.sub(r"\s+", " ", str(comment or "").strip())
    if not text:
        return ""
    kw = (keyword or spam_keyword()).strip()
    if kw and not comment_contains_keyword(text, kw):
        return ""
    if len(text) > 280:
        text = text[:277].rsplit(" ", 1)[0]
    return text
