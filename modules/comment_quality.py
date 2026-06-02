"""Heuristiques de qualité pour commentaires Instagram (training + raw)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from modules.generator_prompt import MAX_GENERATOR_OUTPUT_WORDS

# Motifs promo / spam évidents (insensible à la casse).
_PROMO_RE = re.compile(
    r"https?://|www\.|link in bio|lien en bio|follow me|follow @|whatsapp|"
    r"telegram @|\.com\b|code promo|promo code|\bsponsor",
    re.IGNORECASE,
)
_TAG_SPAM_RE = re.compile(r"(?:#\w+\s*){3,}")
_REPEAT_EMOJI_RE = re.compile(
    r"([\U0001F300-\U0001FAFF\U00002600-\U000027BF])\1{4,}"
)
_REPEAT_CHAR_RE = re.compile(r"(.)\1{7,}")
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
# Suppression large (drapeaux, pictos, dingbats, ZWJ, sélecteurs de variation).
_STRIP_EMOJI_RE = re.compile(
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
_EMOJI_ONLY_RE = re.compile(r"^[\s\U0001F300-\U0001FAFF\U00002600-\U000027BF]+$")
_LETTER_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9]")
_MENTION_ONLY_RE = re.compile(r"^@[\w.]{2,}$")
_FR_ACCENT_RE = re.compile(r"[àâäéèêëïîôùûüç]", re.IGNORECASE)
_EN_STRONG_RE = re.compile(
    r"\b(the|and|you|your|yours|this|that|those|these|bro|dude|literally|"
    r"when|what|how|why|she|her|his|they|them|just|about|would|could|should|"
    r"don't|doesn't|didn't|i'm|it's|that's|you're|we're|english|speak)\b",
    re.IGNORECASE,
)
_FR_HINT_WORDS = frozenset(
    {
        "mdr", "ptdr", "jpp", "lol", "trop", "chez", "avec", "pour", "dans",
        "sans", "mais", "donc", "quoi", "comment", "pourquoi", "parce", "bce",
        "ouf", "chelou", "genant", "gênant", "grave", "wesh", "frr", "frero",
        "frère", "meuf", "gros", "les", "des", "une", "pas", "plus", "même",
        "meme", "c'est", "cest", "j'ai", "jai", "t'es", "tes", "n'ai", "qu'",
        "la", "le", "du", "de", "je", "tu", "il", "elle", "on", "nous", "vous",
        "ils", "elles", "mon", "ton", "son", "notre", "votre", "leur",
    }
)
_HAS_MENTION_RE = re.compile(r"@\w")
_INCOMPLETE_TAIL = frozenset(
    {
        "de", "du", "des", "le", "la", "les", "un", "une",
        "à", "au", "aux", "en", "pour", "par", "sur", "avec", "sans",
        "que", "qui", "dont", "depuis", "dans", "comme", "mais", "car",
        "si", "et", "ou", "ne", "pas", "plus", "moins", "très", "trop",
        "faut", "faudrait", "besoin", "ton", "ta", "mon", "ma", "son", "sa",
        "te", "me", "se", "lui", "leur", "y", "en",
        "fait", "fais", "dit", "dis", "est", "suis", "es", "sont", "ai", "as",
        "a", "ont", "été", "ete", "tout", "toute", "tous", "toutes",
        "même", "meme", "type", "premier", "première", "premiere",
    }
)

REJECT_TOO_SHORT = "too_short"
REJECT_TOO_LONG = "too_long"
REJECT_EMOJI_ONLY = "emoji_only"
REJECT_EMOJI_SPAM = "emoji_spam"
REJECT_NO_TEXT = "no_text"
REJECT_PROMO_LINK = "promo_link"
REJECT_TAG_SPAM = "tag_spam"
REJECT_REPEAT_CHAR = "repeat_char"
REJECT_MENTION_ONLY = "mention_only"
REJECT_HAS_MENTION = "has_mention"
REJECT_INCOMPLETE = "incomplete"
REJECT_NOT_FRENCH = "not_french"
REJECT_EMPTY = "empty"

ALL_REJECT_REASONS = frozenset(
    {
        REJECT_EMPTY,
        REJECT_TOO_SHORT,
        REJECT_TOO_LONG,
        REJECT_EMOJI_ONLY,
        REJECT_EMOJI_SPAM,
        REJECT_NO_TEXT,
        REJECT_PROMO_LINK,
        REJECT_TAG_SPAM,
        REJECT_REPEAT_CHAR,
        REJECT_MENTION_ONLY,
        REJECT_HAS_MENTION,
        REJECT_INCOMPLETE,
        REJECT_NOT_FRENCH,
    }
)


@dataclass(frozen=True)
class CommentQuality:
    """Résultat d'évaluation d'un commentaire."""

    text: str
    ok: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def primary_reason(self) -> str | None:
        return self.reasons[0] if self.reasons else None


def assess_comment_quality(text: str) -> CommentQuality:
    """Évalue si un commentaire convient à l'entraînement générateur/classifier.

    Règles (alignées sur ``MAX_GENERATOR_OUTPUT_WORDS`` / usage Instagram réel) :

    * vide, trop court (< 3 car.), trop long (> 5 mots)
    * emoji seul, spam emoji (répétition ou ratio élevé)
    * sans lettre/chiffre (ponctuation seule)
    * promo / lien, spam hashtags (≥ 3), répétition de caractères
    * mention seule (@user) ou @mention dans le texte
    """
    raw = str(text or "")
    stripped = raw.strip()
    reasons: list[str] = []

    if not stripped:
        reasons.append(REJECT_EMPTY)
    else:
        if len(stripped) < 3:
            reasons.append(REJECT_TOO_SHORT)
        if len(stripped.split()) > MAX_GENERATOR_OUTPUT_WORDS:
            reasons.append(REJECT_TOO_LONG)
        if _EMOJI_ONLY_RE.fullmatch(stripped):
            reasons.append(REJECT_EMOJI_ONLY)
        emoji_count = len(_EMOJI_RE.findall(stripped))
        if _REPEAT_EMOJI_RE.search(stripped):
            reasons.append(REJECT_EMOJI_SPAM)
        elif emoji_count >= 5 and emoji_count / max(len(stripped), 1) > 0.35:
            reasons.append(REJECT_EMOJI_SPAM)
        if not _LETTER_RE.search(stripped):
            reasons.append(REJECT_NO_TEXT)
        if _PROMO_RE.search(stripped):
            reasons.append(REJECT_PROMO_LINK)
        if _TAG_SPAM_RE.search(stripped):
            reasons.append(REJECT_TAG_SPAM)
        if _REPEAT_CHAR_RE.search(stripped):
            reasons.append(REJECT_REPEAT_CHAR)
        if _MENTION_ONLY_RE.match(stripped):
            reasons.append(REJECT_MENTION_ONLY)
        if _HAS_MENTION_RE.search(stripped):
            reasons.append(REJECT_HAS_MENTION)
        if is_incomplete_comment(stripped):
            reasons.append(REJECT_INCOMPLETE)

    return CommentQuality(
        text=stripped,
        ok=not reasons,
        reasons=tuple(reasons),
    )


def is_french_comment(text: str) -> bool:
    """Heuristique FR : accents, mots FR, rejet anglais évident."""
    s = str(text or "").strip()
    if not s:
        return False
    # Réactions emoji — neutres, très fréquentes sur reels FR
    if _EMOJI_ONLY_RE.fullmatch(s) or (
        _EMOJI_RE.search(s) and not re.findall(r"[\w']+", s.lower())
    ):
        return len(s) >= 2
    if len(s) < 3:
        return False
    if _FR_ACCENT_RE.search(s):
        return True
    if _EN_STRONG_RE.search(s):
        words = re.findall(r"[\w']+", s.lower())
        fr_hits = sum(1 for w in words if w in _FR_HINT_WORDS)
        if fr_hits == 0:
            return False
    words = re.findall(r"[\w']+", s.lower())
    if not words:
        return False
    fr_hits = sum(1 for w in words if w in _FR_HINT_WORDS)
    en_hits = len(_EN_STRONG_RE.findall(s))
    if en_hits >= 2 and fr_hits == 0:
        return False
    if en_hits > fr_hits and len(words) >= 3:
        return False
    # Phrase ASCII longue sans indice FR → probablement EN
    if len(words) >= 5 and fr_hits == 0 and not _FR_ACCENT_RE.search(s):
        return False
    return True


def is_incomplete_comment(text: str) -> bool:
    """Phrase coupée (finit sur une préposition / conjonction / mot tronqué)."""
    raw = str(text or "").strip()
    if re.search(r"\b(?:n'|j'|l'|d'|qu'|s'|c'|m'|t')\w*$", raw, re.IGNORECASE):
        return True
    words = re.findall(r"[\w']+", raw.lower())
    if len(words) < 4:
        return False
    tail = words[-1].strip(".,!?…\"'")
    if len(tail) == 1 and tail.isalpha():
        return True
    return tail in _INCOMPLETE_TAIL


def is_repetitive_comment(text: str) -> bool:
    """Détecte les boucles de mots (ex. « bravo bravo bravo … ») hors training set."""
    words = re.findall(r"[\w']+", str(text or "").lower())
    if len(words) < 4:
        return False
    counts: dict[str, int] = {}
    for word in words:
        if len(word) < 3:
            continue
        counts[word] = counts.get(word, 0) + 1
    if counts:
        top = max(counts.values())
        if top >= 2 and top / len(words) >= 0.22:
            return True
    return len(set(words)) / len(words) < 0.55


def strip_emojis(text: str) -> str:
    """Retire les emojis et normalise les espaces."""
    cleaned = _STRIP_EMOJI_RE.sub("", str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


__all__ = [
    "ALL_REJECT_REASONS",
    "CommentQuality",
    "REJECT_EMPTY",
    "REJECT_EMOJI_ONLY",
    "REJECT_EMOJI_SPAM",
    "REJECT_MENTION_ONLY",
    "REJECT_NO_TEXT",
    "REJECT_PROMO_LINK",
    "REJECT_REPEAT_CHAR",
    "REJECT_TAG_SPAM",
    "REJECT_TOO_LONG",
    "REJECT_TOO_SHORT",
    "assess_comment_quality",
    "is_french_comment",
    "is_incomplete_comment",
    "is_repetitive_comment",
    "strip_emojis",
]
