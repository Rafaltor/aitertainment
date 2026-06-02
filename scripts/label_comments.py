"""label_comments.py — labellisation T-type du pool viral.

Lit ``viral_comments.json`` (non labellisé), appelle Qwen3-35B via LM Studio
(``LABEL_LLM_*``) pour T-type + ``video_context`` en un seul appel,
écrit ``training_comments_viral.json`` et retire du pool viral les entrées
déjà labellisées (vase communicant).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_PROJECT_ROOT))

from config import LABEL_LLM_MODEL, LABEL_LLM_URL, VALID_T_TYPES
from database import load_db, merge_profile_pipeline, save_db
from modules.named_axes import NAMED_AXES
from modules.pipeline_state import build_pipeline_patch, comment_dedup_key

VALID_TTYPES = set(VALID_T_TYPES)
VIRAL_COMMENTS_PATH = Path("data/viral_comments.json")
TRAINING_COMMENTS_PATH = Path("data/training_comments_viral.json")
WATCHLIST_PATH = Path("data/watchlist.json")
DATABASE_PATH = Path("data/database.json")
VECTOR_STORE_PATH = Path("data/vector_store.json")

SYSTEM_PROMPT = """Tu es un expert en analyse de commentaires Instagram français.

Tu reçois pour chaque commentaire :
- Le texte du commentaire
- Les niches du contenu  
- Le profil du créateur (T-type dominant, axes de style)
- La caption du reel
- Le transcript audio (ce qui est dit dans la vidéo)
- La description visuelle (ce qu'on voit dans la vidéo)

Tu dois retourner UNIQUEMENT un JSON valide avec exactement ces deux clés :
{
  "t_type": "T1|T2|T2b|T3a|T3b|T4|T5",
  "video_context": "Description fusionnée et chronologique de la vidéo en 2-4 phrases, qui intègre l'audio ET le visuel. Ex: 'Sketch parodique où le personnage central, ressemblant à un dirigeant politique, débat avec un commentateur dans une rue parisienne entouré de gardes du corps masqués. Le ton est ironique, avec des dialogues percutants.'"
}

T-types :
T1 = spam, bot, commentaire sans valeur
T2 = engagement basique court (lol, mdr, top)
T2b = engagement émotionnel fort (je suis mort, trop drôle sérieux)
T3a = question ou curiosité  
T3b = partage d'expérience personnelle
T4 = référence communautaire, inside joke, mention d'amis
T5 = contenu généré (suite, collab, repost demandé)

Réponds UNIQUEMENT le JSON. Pas d'explication, pas de markdown."""
_LOG = logging.getLogger("aitertainment.label_comments")
_TOKEN_RE = re.compile(r"\S+")
# Ordre long → court pour ne pas couper T2b / T3b en T2 / T3.
_TTYPE_RE = re.compile(r"\b(T5|T2b|T3b|T3a|T4|T2|T1)\b")
_NUMBERED_CLASS_LINE_RE = re.compile(r"^([1-5])(a|b)?\s*:", re.IGNORECASE)
_VERDICT_RE = re.compile(
    r"(?:label|classifi(?:cation|é)?|réponse|answer|verdict|conclusion|therefore|thus|donc|→)\s*:?\s*"
    r"(T5|T2b|T3b|T3a|T4|T2|T1)\b",
    re.IGNORECASE,
)
_REASONING_ANALYSIS_MARKERS = (
    "analyze the comment",
    "analyse du commentaire",
    "analyse le commentaire",
)


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dedup_key(media_id: str, text: str) -> str:
    """Clé de déduplication pour ``training_comments_viral.json``."""
    return comment_dedup_key(media_id, text)


def load_viral_comments(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Charge ``viral_comments.json`` (liste ou ``{"entries": [...]}``)."""
    p = _resolve_path(Path(path) if path is not None else VIRAL_COMMENTS_PATH)
    if not p.exists():
        return []

    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    if isinstance(data, dict):
        entries = data.get("entries") or data.get("comments") or []
        if isinstance(entries, list):
            return [entry for entry in entries if isinstance(entry, dict)]
    return []


def load_training_comments(
    path: Path | str | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Charge ``training_comments_viral.json`` et retourne entrées + clés de dédup."""
    p = _resolve_path(Path(path) if path is not None else TRAINING_COMMENTS_PATH)
    if not p.exists():
        return [], set()

    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        entries = [entry for entry in data if isinstance(entry, dict)]
    elif isinstance(data, dict):
        raw_entries = data.get("entries") or data.get("comments") or []
        entries = [entry for entry in raw_entries if isinstance(entry, dict)]
    else:
        entries = []

    keys = {
        dedup_key(str(entry.get("media_id") or ""), str(entry.get("text") or ""))
        for entry in entries
        if entry.get("media_id") and entry.get("text")
    }
    return entries, keys


def load_database_profiles(
    path: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Index ``database.json`` profiles par username (t_type_final, niches)."""
    p = _resolve_path(Path(path) if path is not None else DATABASE_PATH)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for username, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        key = str(username).lstrip("@").strip().lower()
        if key:
            out[key] = profile
    return out


def load_watchlist(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Charge ``watchlist.json`` et indexe les créateurs par username."""
    p = _resolve_path(Path(path) if path is not None else WATCHLIST_PATH)
    if not p.exists():
        return {}

    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("creators") or data.get("watchlist") or []
    else:
        return {}

    out: dict[str, dict[str, Any]] = {}
    if not isinstance(entries, list):
        return out

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        username = str(entry.get("username") or "").lstrip("@").strip().lower()
        if username:
            out[username] = entry
    return out


def load_vector_store(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Charge ``vector_store.json`` indexé par username."""
    p = _resolve_path(Path(path) if path is not None else VECTOR_STORE_PATH)
    if not p.exists():
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


def _truncate_field(text: str, limit: int = 500) -> str:
    t = (text or "").strip()
    if not t:
        return "(vide)"
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


def _format_niches(niches: list[str] | str | None) -> str:
    if niches is None:
        return "(non précisée)"
    if isinstance(niches, str):
        s = niches.strip()
        return s or "(non précisée)"
    if isinstance(niches, list):
        clean = [str(n).strip() for n in niches if str(n).strip()]
        return ", ".join(clean) if clean else "(non précisée)"
    return "(non précisée)"


def _format_hashtags(value: Any) -> str:
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return ", ".join(parts) if parts else "(vide)"
    if isinstance(value, str):
        return value.strip() or "(vide)"
    return "(vide)"


def build_creator_context_for_label(
    raw_entry: dict[str, Any],
    creator: dict[str, Any],
    vs_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble le contexte créateur + reel pour le prompt de classification."""
    username = str(raw_entry.get("username") or "").lstrip("@").strip()
    history = creator.get("scores_history") or []
    last_score = (
        history[-1] if history and isinstance(history[-1], dict) else {}
    )
    named_axes: dict[str, Any] = {}
    if vs_entry and isinstance(vs_entry.get("named_axes"), dict):
        named_axes = vs_entry["named_axes"]

    return {
        "username": username,
        "t_type_profile": str(
            creator.get("t_type_final")
            or creator.get("t_type")
            or raw_entry.get("t_type_profile")
            or ""
        ),
        "niches": raw_entry.get("niches") or creator.get("niches") or ["humour"],
        "tier": str(creator.get("tier") or ""),
        "followers": int(creator.get("followers") or 0),
        "discovery_score": float(last_score.get("score") or 0),
        "reel_engagement_median": float(
            last_score.get("reel_engagement_median") or 0
        ),
        "named_axes": named_axes,
        "caption": str(raw_entry.get("caption") or ""),
        "hashtags": raw_entry.get("hashtags") or [],
        "comment_likes": int(raw_entry.get("comment_likes") or 0),
        "views": int(raw_entry.get("views") or 0),
        "media_id": str(raw_entry.get("media_id") or ""),
        "comment_to_like_ratio": float(
            raw_entry.get("comment_to_like_ratio") or 0
        ),
        "has_vector_profile": bool(named_axes),
    }


def _build_user_prompt(
    text: str,
    niches: list[str],
    creator_context: dict[str, Any] | None,
) -> str:
    if not creator_context:
        return (
            "Classifie ce commentaire Instagram.\n"
            f"Commentaire: {text}\n"
            f"Niches: {_format_niches(niches)}\n"
            "Réponds avec exactement un de ces labels : T1 T2 T2b T3a T3b T4 T5"
        )

    lines = [
        "Classifie ce commentaire Instagram en un seul T-type.",
        "",
        "=== Commentaire ===",
        f"Texte: {_truncate_field(text, 600)}",
        f"Likes sur ce commentaire: {int(creator_context.get('comment_likes') or 0)}",
    ]
    ratio = float(creator_context.get("comment_to_like_ratio") or 0)
    if ratio > 0:
        lines.append(f"Ratio commentaires/likes du reel: {ratio:.4f}")

    lines.extend(
        [
            "",
            "=== Reel source ===",
            f"Media ID: {creator_context.get('media_id') or '?'}",
            f"Vues du reel (approx.): {int(creator_context.get('views') or 0)}",
            f"Caption: {_truncate_field(str(creator_context.get('caption') or ''), 500)}",
            f"Hashtags: {_format_hashtags(creator_context.get('hashtags'))}",
            "",
            "=== Créateur ===",
            f"Username: @{creator_context.get('username') or '?'}",
            f"Niches: {_format_niches(creator_context.get('niches') or niches)}",
        ]
    )
    tier = str(creator_context.get("tier") or "").strip()
    if tier:
        lines.append(f"Tier: {tier}")
    followers = int(creator_context.get("followers") or 0)
    if followers > 0:
        lines.append(f"Followers: {followers}")
    t_type_profile = str(creator_context.get("t_type_profile") or "").strip()
    if t_type_profile:
        lines.append(f"T-type dominant (profil créateur): {t_type_profile}")
    score = float(creator_context.get("discovery_score") or 0)
    if score > 0:
        lines.append(f"Score discovery: {score:.0f}")
    eng = float(creator_context.get("reel_engagement_median") or 0)
    if eng > 0:
        lines.append(f"Engagement reel médian: {eng:.3f}")

    named_axes = creator_context.get("named_axes") or {}
    if isinstance(named_axes, dict) and named_axes:
        lines.append(
            "Profil style (named_axes 0-1, échelle globale entre créateurs):"
        )
        for axis in NAMED_AXES:
            if axis in named_axes:
                lines.append(f"  - {axis}={float(named_axes[axis]):.2f}")
    else:
        lines.append(
            "Profil style: non disponible (créateur absent du vector_store)"
        )

    lines.extend(
        [
            "",
            "Label (un seul, pas T1 par défaut si vanne/moquerie): T1, T2, T2b, T3a, T3b, T4 ou T5",
        ]
    )
    return "\n".join(lines)


def _ttype_from_number_prefix(num: str, suffix: str | None) -> str | None:
    """Mappe une ligne « 2b : … » (suite du prefill « T ») vers T2b, etc."""
    key = (num, (suffix or "").lower() or None)
    mapping: dict[tuple[str, str | None], str] = {
        ("1", None): "T1",
        ("2", None): "T2",
        ("2", "b"): "T2b",
        ("3", "a"): "T3a",
        ("3", "b"): "T3b",
        ("3", None): "T3a",
        ("4", None): "T4",
        ("5", None): "T5",
    }
    return mapping.get(key)


def _extract_from_numbered_class_line(content: str) -> str | None:
    """Réponse LM Studio après prefill « T » → « 1 : Admiration… »."""
    m = _NUMBERED_CLASS_LINE_RE.match(str(content).strip())
    if not m:
        return None
    return _ttype_from_number_prefix(m.group(1), m.group(2))


def _extract_t_type_from_content(content: str) -> str | None:
    """Extrait le T-type (dernière occurrence = label final si CoT court)."""
    if not content or not str(content).strip():
        return None
    numbered = _extract_from_numbered_class_line(content)
    if numbered:
        return numbered
    matches = _TTYPE_RE.findall(str(content))
    if matches:
        label = matches[-1]
        if label in VALID_TTYPES:
            return label
    for token in _TOKEN_RE.findall(content):
        if token in VALID_TTYPES:
            return token
    return None


def _reasoning_analysis_tail(reasoning: str, comment_text: str) -> str:
    """Limite l'extraction au raisonnement après l'analyse du commentaire."""
    lower = reasoning.lower()
    for marker in _REASONING_ANALYSIS_MARKERS:
        idx = lower.find(marker)
        if idx >= 0:
            return reasoning[idx:]
    snippet = str(comment_text).strip()[:60]
    if snippet:
        pos = reasoning.find(snippet)
        if pos >= 0:
            return reasoning[pos:]
    return reasoning


def _extract_t_type_from_reasoning(reasoning: str, comment_text: str = "") -> str | None:
    """Extrait le label depuis le CoT sans confondre avec la liste des définitions."""
    if not reasoning or not str(reasoning).strip():
        return None
    tail = _reasoning_analysis_tail(str(reasoning), comment_text)

    for line in reversed(tail.splitlines()):
        stripped = line.strip().strip("`*.,;")
        if stripped in VALID_TTYPES:
            return stripped
        solo = re.match(r"^(T5|T2b|T3b|T3a|T4|T2|T1)\s*[.:!]?\s*$", stripped)
        if solo and solo.group(1) in VALID_TTYPES:
            return solo.group(1)

    verdicts = _VERDICT_RE.findall(tail)
    if verdicts:
        label = verdicts[-1]
        return label if label in VALID_TTYPES else None

    matches = _TTYPE_RE.findall(tail)
    if matches:
        label = matches[-1]
        return label if label in VALID_TTYPES else None
    return None


def _extract_t_type_from_message(
    message: dict[str, Any], comment_text: str = ""
) -> str | None:
    """Priorité : content (réponse courte), puis reasoning ciblé."""
    content = str(message.get("content") or "").strip()
    if content:
        label = _extract_t_type_from_content(content)
        if label:
            return label

    reasoning = str(message.get("reasoning_content") or "").strip()
    if reasoning:
        return _extract_t_type_from_reasoning(reasoning, comment_text)
    return None


def _t_type_from_raw_entry(raw_entry: dict[str, Any]) -> str | None:
    """Réutilise un T-type déjà posé par Discovery (évite un appel LLM)."""
    label = str(raw_entry.get("t_type") or "").strip()
    if label in VALID_TTYPES:
        return label
    return None


def _extract_json_block(text: str) -> dict[str, Any] | None:
    """Extrait le premier bloc JSON valide en comptant les accolades."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _parse_label_fuse_response(content: str) -> tuple[str | None, str]:
    """Extrait ``(t_type, video_context)`` depuis la réponse LLM."""
    content = (content or "").strip()
    if not content:
        return None, ""

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL | re.IGNORECASE)
    if fenced:
        content = fenced.group(1).strip()

    data = _extract_json_block(content)
    if not data:
        ttype_m = re.search(r"\b(T1|T2b|T2|T3a|T3b|T4|T5)\b", content)
        if ttype_m and ttype_m.group(1) in VALID_TTYPES:
            return ttype_m.group(1), ""
        return None, ""

    t_type_raw = data.get("t_type")
    t_type = str(t_type_raw or "").strip()
    if t_type not in VALID_TTYPES:
        t_type = _extract_t_type_from_content(str(t_type_raw or "")) or ""
    video_context = str(data.get("video_context") or "").strip()
    if t_type in VALID_TTYPES:
        return t_type, video_context

    ttype_m = re.search(r"\b(T1|T2b|T2|T3a|T3b|T4|T5)\b", content)
    if ttype_m and ttype_m.group(1) in VALID_TTYPES:
        return ttype_m.group(1), ""
    return None, ""


def label_and_fuse(
    text: str,
    niches: list[str],
    creator_context: dict[str, Any] | None,
    *,
    caption: str = "",
    transcript: str = "",
    visual_description: str = "",
    model: str | None = None,
    url: str | None = None,
) -> tuple[str | None, str]:
    """Labélise le commentaire et fusionne transcript + visuel en ``video_context``."""
    ctx = creator_context or {}
    user_prompt = f"""Commentaire à classifier : {text}

Niches du contenu : {', '.join(niches) if niches else 'humour'}

Profil créateur :
- T-type dominant : {ctx.get('t_type_profile') or '(inconnu)'}
- Axes : {ctx.get('named_axes', {})}

Caption du reel : {caption[:300] if caption else '(vide)'}

Transcript audio : {transcript[:1000] if transcript else '(non disponible)'}

Description visuelle : {visual_description[:600] if visual_description else '(non disponible)'}

Retourne le JSON."""

    api_url = (url or LABEL_LLM_URL or "").strip().rstrip("/")
    api_model = (model or LABEL_LLM_MODEL or "").strip()
    if not api_url or not api_model:
        _LOG.warning("label_and_fuse : LABEL_LLM_URL ou LABEL_LLM_MODEL absent.")
        return None, ""

    try:
        resp = requests.post(
            f"{api_url}/chat/completions",
            json={
                "model": api_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 2000,
                "temperature": 0.2,
            },
            timeout=120,
        )
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
        content = str(message.get("content") or "").strip()
        if not content:
            content = str(message.get("reasoning_content") or "").strip()
        t_type, video_context = _parse_label_fuse_response(content)
        if t_type is None and content:
            _LOG.warning(
                "label_and_fuse : réponse illisible (extrait=%r).",
                content.replace("\n", " ")[:120],
            )
        return t_type, video_context
    except Exception as exc:
        _LOG.warning("label_and_fuse : appel LLM échoué (%s).", exc)
        return None, ""


def build_training_entry(
    raw_entry: dict[str, Any],
    t_type: str,
    creators: dict[str, dict[str, Any]],
    *,
    label_source: str = "llm",
    video_context: str = "",
) -> dict[str, Any]:
    """Construit une entrée ``training_comments_viral.json`` depuis le brut."""
    username = str(raw_entry.get("username") or "").lstrip("@").strip()
    creator = creators.get(username.lower(), {})
    t_type_profile = (
        str(creator.get("t_type_final") or "")
        or str(creator.get("t_type") or "")
        or str(creator.get("t_type_original") or "")
    )
    return {
        "media_id": raw_entry["media_id"],
        "username": username,
        "t_type": t_type,
        "t_type_profile": t_type_profile,
        "niches": raw_entry.get("niches") or creator.get("niches") or ["humour"],
        "text": raw_entry["text"],
        "comment_likes": raw_entry.get("comment_likes", 0),
        "views": raw_entry.get("views", 0),
        "comment_to_like_ratio": raw_entry.get("comment_to_like_ratio", 0.0),
        "caption": raw_entry.get("caption", ""),
        "hashtags": raw_entry.get("hashtags", []),
        "audio_id": raw_entry.get("audio_id", ""),
        "transcript": raw_entry.get("transcript", ""),
        "visual_description": raw_entry.get("visual_description", ""),
        "video_context": video_context,
        "collected_at": raw_entry.get("collected_at", ""),
        "labelled_at": _utc_now_iso(),
        "llm_model": str(raw_entry.get("llm_model") or LABEL_LLM_MODEL),
        "llm_validated": label_source == "llm",
        "label_source": label_source,
    }


def _merge_creator_sources(
    watchlist: dict[str, dict[str, Any]],
    database: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Watchlist + database (database enrichit t_type / niches)."""
    merged = dict(watchlist)
    for username, profile in database.items():
        base = dict(merged.get(username, {}))
        base.update(profile)
        merged[username] = base
    return merged


_LABEL_FIELDS_ON_RAW = (
    "t_type",
    "t_type_profile",
    "llm_validated",
    "label_source",
    "labelled_at",
    "llm_model",
)


def save_viral_comments(
    entries: list[dict[str, Any]],
    path: Path | str | None = None,
    *,
    allow_shrink: bool = False,
    allow_empty: bool = False,
) -> None:
    """Écrit ``viral_comments.json`` (verrou + garde-fou anti-écrasement)."""
    from scripts.instagram_browser import save_viral_comments_file

    p = _resolve_path(Path(path) if path is not None else VIRAL_COMMENTS_PATH)
    save_viral_comments_file(
        entries,
        p,
        merge=False,
        allow_shrink=allow_shrink,
        allow_empty=allow_empty,
    )


def purge_labeled_from_viral_pool(
    *,
    viral_path: Path | str | None = None,
    training_path: Path | str | None = None,
) -> tuple[int, int]:
    """Retire du pool viral les entrées déjà présentes dans le training.

    Retourne ``(removed_count, remaining_count)``.
    """
    viral_entries = load_viral_comments(viral_path)
    _, labeled_keys = load_training_comments(training_path)
    if not labeled_keys:
        return 0, len(viral_entries)

    remaining: list[dict[str, Any]] = []
    for entry in viral_entries:
        media_id = str(entry.get("media_id") or "")
        text = str(entry.get("text") or "")
        if not media_id or not text:
            remaining.append(entry)
            continue
        if dedup_key(media_id, text) in labeled_keys:
            continue
        remaining.append(entry)

    removed = len(viral_entries) - len(remaining)
    if removed:
        save_viral_comments(
            remaining,
            viral_path,
            allow_shrink=True,
            allow_empty=True,
        )
    return removed, len(remaining)


def reset_all_comment_labels(
    *,
    viral_path: Path | str | None = None,
    training_path: Path | str | None = None,
) -> tuple[int, int]:
    """Vide ``training_comments_viral.json`` (le pool viral n'est pas modifié).

    Retourne ``(previous_training_count, viral_pool_size)``.
    """
    viral_entries = load_viral_comments(viral_path)
    training_entries, _ = load_training_comments(training_path)
    prev_training = len(training_entries)
    save_training_comments([], training_path)
    return prev_training, len(viral_entries)


def _sync_label_pipeline_to_database(
    training_by_key: dict[str, dict[str, Any]],
) -> None:
    """Met à jour ``pipeline.labeled_*`` dans database.json par créateur."""
    if not training_by_key:
        return
    from collections import Counter

    per_user: Counter[str] = Counter()
    for entry in training_by_key.values():
        u = str(entry.get("username") or "").lstrip("@").strip().lower()
        if u:
            per_user[u] += 1
    if not per_user:
        return
    try:
        db = load_db()
    except Exception as exc:
        _LOG.debug("sync pipeline database ignoré (%s).", exc)
        return
    now = _utc_now_iso()
    changed = False
    for username, count in per_user.items():
        patch = build_pipeline_patch(labeled_count=count, labeled_at=now)
        if merge_profile_pipeline(db, username, patch) is not None:
            changed = True
    if changed:
        save_db(db)


def save_training_comments(
    entries: list[dict[str, Any]],
    path: Path | str | None = None,
) -> None:
    """Écrit ``training_comments_viral.json`` de façon atomique."""
    p = _resolve_path(Path(path) if path is not None else TRAINING_COMMENTS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, p)


def _ensure_llm_available() -> bool:
    if LABEL_LLM_URL and LABEL_LLM_MODEL:
        return True
    _LOG.error(
        "LABEL_LLM_URL / LABEL_LLM_MODEL non configurés (labelliseur Qwen3-35B)."
    )
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Labellise les commentaires bruts.")
    parser.add_argument("--account", help="Labéliser uniquement ce compte (@username).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche le plan sans écrire ni appeler le LLM.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-labélise les entrées déjà présentes dans training_comments_viral.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre max de commentaires à traiter (utile pour un batch test).",
    )
    parser.add_argument(
        "--reset-all-labels",
        action="store_true",
        help="Vide training_comments_viral.json (le pool viral reste inchangé).",
    )
    parser.add_argument(
        "--viral-path",
        type=Path,
        default=None,
        help="Pool viral source (défaut : data/viral_comments.json).",
    )
    parser.add_argument(
        "--training-path",
        type=Path,
        default=None,
        help="Fichier training de sortie (défaut : data/training_comments_viral.json).",
    )
    parser.add_argument(
        "--purge-only",
        action="store_true",
        help="Retire du pool viral les entrées déjà labellisées, sans appeler le LLM.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    viral_path = args.viral_path or VIRAL_COMMENTS_PATH
    training_path = args.training_path or TRAINING_COMMENTS_PATH

    if args.reset_all_labels:
        prev_training, viral_size = reset_all_comment_labels(
            viral_path=viral_path,
            training_path=training_path,
        )
        _LOG.info(
            "=== Reset labels : training vidé (%d entrée(s)), "
            "pool viral inchangé (%d entrée(s)) ===",
            prev_training,
            viral_size,
        )
        return 0

    if args.purge_only:
        removed, remaining = purge_labeled_from_viral_pool(
            viral_path=viral_path,
            training_path=training_path,
        )
        _LOG.info(
            "=== Purge pool viral : %d retiré(s), %d restant(s) dans %s ===",
            removed,
            remaining,
            viral_path,
        )
        return 0

    raw_entries = load_viral_comments(viral_path)
    training_entries, existing_keys = load_training_comments(training_path)
    watchlist = load_watchlist(WATCHLIST_PATH)
    database = load_database_profiles(DATABASE_PATH)
    creators = _merge_creator_sources(watchlist, database)
    vector_store = load_vector_store(VECTOR_STORE_PATH)

    pending = []
    skipped_in_training = 0
    for raw_entry in raw_entries:
        media_id = str(raw_entry.get("media_id") or "")
        text = str(raw_entry.get("text") or "")
        if not media_id or not text:
            continue
        key = dedup_key(media_id, text)
        if not args.force and key in existing_keys:
            skipped_in_training += 1
            continue
        pending.append(raw_entry)

    if args.account:
        target = args.account.lstrip("@").strip().lower()
        pending = [
            entry
            for entry in pending
            if str(entry.get("username") or "").lstrip("@").strip().lower() == target
        ]

    if args.limit is not None and args.limit > 0:
        pending = pending[: args.limit]

    if not pending:
        _LOG.info("Rien à labéliser")
        return 0

    _LOG.info(
        "%d commentaire(s) à labéliser via LLM (raw=%d, déjà en training=%d, "
        "créateurs=%d)",
        len(pending),
        len(raw_entries),
        len(training_entries),
        len(creators),
    )
    if skipped_in_training:
        _LOG.info(
            "%d déjà labellisés dans training — ignorés "
            "(nouveau discover : seuls les commentaires absents seront traités ; --force pour tout refaire).",
            skipped_in_training,
        )

    if args.dry_run:
        for raw_entry in pending:
            preview = str(raw_entry.get("text") or "")[:40]
            username = str(raw_entry.get("username") or "")
            _LOG.info("DRY-RUN : labelliserait @%s — %s...", username, preview)
        _LOG.info(
            "=== Labélisation : %d nouveaux commentaires → training ===",
            0,
        )
        return 0

    if not _ensure_llm_available():
        return 1

    training_by_key = {
        dedup_key(str(entry.get("media_id") or ""), str(entry.get("text") or "")): entry
        for entry in training_entries
        if entry.get("media_id") and entry.get("text")
    }
    added = 0
    failed = 0

    for raw_entry in pending:
        text = str(raw_entry.get("text") or "")
        niches = list(raw_entry.get("niches") or ["humour"])
        username = str(raw_entry.get("username") or "").lstrip("@").strip()
        username_key = username.lower()

        creator = creators.get(username_key, {})
        creator_context = build_creator_context_for_label(
            raw_entry,
            creator,
            vector_store.get(username_key),
        )

        t_type, video_context = label_and_fuse(
            text,
            niches,
            creator_context,
            caption=str(raw_entry.get("caption") or ""),
            transcript=str(raw_entry.get("transcript") or ""),
            visual_description=str(raw_entry.get("visual_description") or ""),
        )

        if t_type is None:
            _LOG.warning(
                "@%s commentaire non classifié, skip — %s",
                username,
                _truncate_field(text, 80),
            )
            failed += 1
            continue

        entry = build_training_entry(
            raw_entry,
            t_type,
            creators,
            label_source="llm",
            video_context=video_context,
        )
        key = dedup_key(str(entry["media_id"]), str(entry["text"]))
        training_by_key[key] = entry
        added += 1
        preview = text[:40]
        _LOG.info("✓ [%s] (llm) %s...", t_type, preview)

    save_training_comments(list(training_by_key.values()), training_path)
    removed, remaining = purge_labeled_from_viral_pool(
        viral_path=viral_path,
        training_path=training_path,
    )
    _sync_label_pipeline_to_database(training_by_key)
    _LOG.info(
        "=== Labélisation : %d nouveaux via LLM (%d échecs) → %s "
        "(total %d) ; pool viral : %d retiré(s), %d restant(s) ===",
        added,
        failed,
        training_path,
        len(training_by_key),
        removed,
        remaining,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
