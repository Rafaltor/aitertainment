"""label_comments.py — labellisation T-type des commentaires bruts.

Script autonome : lit ``raw_comments.json``, appelle Ollama (Qwen2.5:7b)
et écrit ``training_comments.json``.
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

from config import LM_STUDIO_URL, OLLAMA_MODEL, VALID_T_TYPES
from database import load_db, merge_profile_pipeline, save_db
from modules.named_axes import NAMED_AXES
from modules.pipeline_state import build_pipeline_patch, comment_dedup_key

VALID_TTYPES = set(VALID_T_TYPES)
RAW_COMMENTS_PATH = Path("data/raw_comments.json")
TRAINING_COMMENTS_PATH = Path("data/training_comments.json")
WATCHLIST_PATH = Path("data/watchlist.json")
DATABASE_PATH = Path("data/database.json")
VECTOR_STORE_PATH = Path("data/vector_store.json")

_CLASSIFY_SYSTEM_PROMPT = """Tu classes des commentaires Instagram en UN seul T-type.

Types :
- T1 : admiration/encouragement sincère, hype positive (sans vanne ni moquerie)
- T2 : humour tribal, meme, catchphrase, inside joke, jeu de mots sur le créateur
- T2b : le commentaire prolonge ou complète la blague / le sketch du reel
- T3a : insulte ou haine explicite
- T3b : moquerie, second degré, foutage de gueule, sarcasme (même si le ton semble léger)
- T4 : rituel ou slogan d'identité de niche
- T5 : créateur provocateur, commentaire alimente la polémique

Règles importantes :
- Ne mets PAS T1 par défaut. T1 seulement si le registre est clairement admiratif/sincère.
- Vannes, « mdr » ironique, références implicites, moquerie → T2, T2b ou T3b (pas T1).
- « je te déteste mdr », « c'est pas toi », emoji moqueur → plutôt T3b que T1.

Exemples :
- « t'es trop fort fréro » → T1
- « you're jude or you're not but I doooo » → T2
- « c'est pas toi ça s'entend 😂 » → T3b
- « je te déteste mdr très bon acting » → T3b

Réponds par EXACTEMENT un token : T1, T2, T2b, T3a, T3b, T4 ou T5."""
_LM_STUDIO_LABEL_PREFILL = "Label: "
_LM_STUDIO_TEMPERATURE = 0.15
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
    """Clé de déduplication pour ``training_comments.json``."""
    return comment_dedup_key(media_id, text)


def load_raw_comments(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Charge ``raw_comments.json`` (liste brute ou ``{"entries": [...]}``)."""
    p = _resolve_path(Path(path) if path is not None else RAW_COMMENTS_PATH)
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
    """Charge ``training_comments.json`` et retourne entrées + clés de dédup."""
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


def classify_comment(
    text: str,
    niches: list[str],
    ollama_model: str,
    *,
    creator_context: dict[str, Any] | None = None,
) -> str | None:
    """Classifie un commentaire via LM Studio (distant) ou Ollama (local)."""
    user_prompt = _build_user_prompt(text, niches, creator_context)

    message: dict[str, Any] = {}
    content = ""

    if LM_STUDIO_URL:
        lm_model = (
            os.environ.get("LM_STUDIO_MODEL")
            or os.environ.get("LM_STUDIO_CHAT_MODEL")
            or "qwen/qwen3.6-35b-a3b"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            # Évite le prefill « T » qui pousse le modèle vers « 1 : Admiration » → T1.
            {"role": "assistant", "content": _LM_STUDIO_LABEL_PREFILL},
        ]
        payload = {
            "model": lm_model,
            "messages": messages,
            "max_tokens": 12,
            "temperature": _LM_STUDIO_TEMPERATURE,
            "thinking": {"type": "disabled"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            resp = requests.post(
                f"{LM_STUDIO_URL.rstrip('/')}/chat/completions",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            message = data["choices"][0]["message"]
            label = _extract_t_type_from_message(message, text)
        except Exception as exc:
            _LOG.warning("classify_comment : appel LM Studio échoué (%s).", exc)
            return None
    else:
        try:
            import ollama
        except ImportError as exc:
            _LOG.warning("ollama indisponible (%s).", exc)
            return None

        try:
            response = ollama.chat(
                model=ollama_model,
                messages=[
                    {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            _LOG.warning("classify_comment : appel Ollama échoué (%s).", exc)
            return None

        if isinstance(response, dict):
            message = response.get("message") or {}
            if isinstance(message, dict):
                content = str(message.get("content") or "")
            else:
                content = str(getattr(message, "content", "") or "")
        else:
            message = getattr(response, "message", None)
            content = str(getattr(message, "content", "") or "")
        label = _extract_t_type_from_content(content)

    if label is None:
        snippet = ""
        if LM_STUDIO_URL and message:
            snippet = (
                str(message.get("content") or "")[:80]
                or str(message.get("reasoning_content") or "")[-120:]
            )
        elif content.strip():
            snippet = content[:120]
        if snippet:
            _LOG.warning(
                "classify_comment : réponse illisible (extrait=%r).",
                snippet.replace("\n", " "),
            )
    return label


def build_training_entry(
    raw_entry: dict[str, Any],
    t_type: str,
    creators: dict[str, dict[str, Any]],
    *,
    label_source: str = "llm",
) -> dict[str, Any]:
    """Construit une entrée ``training_comments.json`` depuis le brut."""
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
        "collected_at": raw_entry.get("collected_at", ""),
        "labelled_at": _utc_now_iso(),
        "llm_model": str(raw_entry.get("llm_model") or OLLAMA_MODEL),
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


def save_raw_comments(
    entries: list[dict[str, Any]],
    path: Path | str | None = None,
) -> None:
    """Écrit ``raw_comments.json`` de façon atomique."""
    p = _resolve_path(Path(path) if path is not None else RAW_COMMENTS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, p)


def reset_all_comment_labels(
    *,
    raw_path: Path | str | None = None,
    training_path: Path | str | None = None,
) -> tuple[int, int, int]:
    """Efface tous les labels (raw + training) pour repartir de zéro.

    Retourne ``(cleared_in_raw, total_raw, previous_training_count)``.
    """
    raw_entries = load_raw_comments(raw_path)
    training_entries, _ = load_training_comments(training_path)
    prev_training = len(training_entries)

    cleared = 0
    for entry in raw_entries:
        had_label = _t_type_from_raw_entry(entry) or entry.get("llm_validated")
        if not had_label:
            continue
        for key in _LABEL_FIELDS_ON_RAW:
            entry.pop(key, None)
        cleared += 1

    save_raw_comments(raw_entries, raw_path)
    save_training_comments([], training_path)
    return cleared, len(raw_entries), prev_training


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
    """Écrit ``training_comments.json`` de façon atomique."""
    p = _resolve_path(Path(path) if path is not None else TRAINING_COMMENTS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, p)


def _ensure_llm_available() -> bool:
    if LM_STUDIO_URL:
        return True
    try:
        import ollama  # noqa: F401
    except ImportError as exc:
        _LOG.error("Ollama indisponible au démarrage (%s).", exc)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Labellise les commentaires bruts.")
    parser.add_argument("--account", help="Labéliser uniquement ce compte (@username).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche le plan sans écrire ni appeler Ollama.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-labélise les entrées déjà présentes dans training_comments.json.",
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
        help="Efface t_type / llm_validated dans raw_comments.json et vide training_comments.json.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.reset_all_labels:
        cleared, total_raw, prev_training = reset_all_comment_labels()
        _LOG.info(
            "=== Reset labels : %d/%d entrées raw délabellisées, "
            "training_comments vidé (%d entrée(s) supprimée(s)) ===",
            cleared,
            total_raw,
            prev_training,
        )
        return 0

    raw_entries = load_raw_comments()
    training_entries, existing_keys = load_training_comments()
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
            "%d déjà labellisés dans training_comments.json — ignorés "
            "(nouveau discover : seuls les commentaires absents seront traités ; --force pour tout refaire).",
            skipped_in_training,
        )

    if args.dry_run:
        for raw_entry in pending:
            preview = str(raw_entry.get("text") or "")[:40]
            username = str(raw_entry.get("username") or "")
            _LOG.info("DRY-RUN : labelliserait @%s — %s...", username, preview)
        _LOG.info(
            "=== Labélisation : %d nouveaux commentaires → training_comments.json ===",
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

        t_type = classify_comment(
            text,
            niches,
            OLLAMA_MODEL,
            creator_context=creator_context,
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
            raw_entry, t_type, creators, label_source="llm"
        )
        key = dedup_key(str(entry["media_id"]), str(entry["text"]))
        training_by_key[key] = entry
        added += 1
        preview = text[:40]
        _LOG.info("✓ [%s] (llm) %s...", t_type, preview)

    save_training_comments(list(training_by_key.values()))
    _sync_label_pipeline_to_database(training_by_key)
    _LOG.info(
        "=== Labélisation : %d nouveaux via LLM (%d échecs) → training_comments.json "
        "(total %d) ===",
        added,
        failed,
        len(training_by_key),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
