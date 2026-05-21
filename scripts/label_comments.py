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

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "")
OLLAMA_MODEL = "qwen2.5:7b"
VALID_TTYPES = {"T1", "T2", "T2b", "T3a", "T3b", "T4", "T5"}
RAW_COMMENTS_PATH = Path("data/raw_comments.json")
TRAINING_COMMENTS_PATH = Path("data/training_comments.json")
WATCHLIST_PATH = Path("data/watchlist.json")
VECTOR_STORE_PATH = Path("data/vector_store.json")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG = logging.getLogger("aitertainment.label_comments")
_TOKEN_RE = re.compile(r"\S+")


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dedup_key(media_id: str, text: str) -> str:
    """Clé de déduplication pour ``training_comments.json``."""
    return f"{media_id}||{text.strip().lower()}"


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


def _build_user_prompt(
    text: str,
    niches: list[str],
    creator_context: dict[str, Any] | None,
) -> str:
    if not creator_context:
        return (
            "Classifie ce commentaire Instagram.\n"
            f"Commentaire: {text}\n"
            f"Niches: {', '.join(niches)}\n"
            "Réponds avec exactement un de ces labels : T1 T2 T2b T3a T3b T4 T5"
        )

    ctx_niches = creator_context.get("niches") or niches
    if isinstance(ctx_niches, list):
        niches_str = ", ".join(str(n) for n in ctx_niches)
    else:
        niches_str = str(ctx_niches)

    t_type_profile = creator_context.get("t_type_profile")
    lines = [
        f"Commentaire: {text}",
        f"Niches du contenu: {niches_str}",
        "Profil du créateur:",
        f"  - T-type dominant: {t_type_profile or ''}",
    ]

    named_axes = creator_context.get("named_axes") or {}
    if isinstance(named_axes, dict) and named_axes:
        scripted = float(named_axes.get("scripted_vs_raw") or 0)
        energy = float(named_axes.get("energy_level") or 0)
        mainstream = float(named_axes.get("mainstream_vs_niche") or 0)
        lines.append(
            f"  - Style: scripted={scripted:.2f}, energie={energy:.2f}, "
            f"mainstream={mainstream:.2f}"
        )

    lines.append("Label (T1/T2/T2b/T3a/T3b/T4/T5):")
    return "\n".join(lines)


def _extract_t_type_from_content(content: str) -> str | None:
    for token in _TOKEN_RE.findall(content):
        if token in VALID_TTYPES:
            return token
    return None


def classify_comment(
    text: str,
    niches: list[str],
    ollama_model: str,
    *,
    creator_context: dict[str, Any] | None = None,
) -> str | None:
    """Classifie un commentaire via LM Studio (distant) ou Ollama (local)."""
    system_prompt = (
        "Tu es un expert en analyse de commentaires Instagram.\n"
        "Réponds UNIQUEMENT avec le T-type, rien d'autre.\n"
        "T1=spam/bot, T2=engagement basique, T2b=engagement émotionnel,\n"
        "T3a=question/curiosité, T3b=partage expérience, T4=référence\n"
        "communautaire, T5=contenu généré (suite/collab)"
    )
    user_prompt = _build_user_prompt(text, niches, creator_context)

    content = ""

    if LM_STUDIO_URL:
        lm_model = os.environ.get("LM_STUDIO_MODEL", "qwen/qwen3.6-35b-a3b")
        payload = {
            "model": lm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 50,
            "temperature": 0.0,
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
            content = str(message.get("content") or "")
            if not content:
                content = str(message.get("reasoning_content") or "")
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
                    {"role": "system", "content": system_prompt},
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

    return _extract_t_type_from_content(content)


def build_training_entry(
    raw_entry: dict[str, Any],
    t_type: str,
    watchlist: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Construit une entrée ``training_comments.json`` depuis le brut."""
    username = str(raw_entry.get("username") or "").lstrip("@").strip()
    creator = watchlist.get(username.lower(), {})
    return {
        "media_id": raw_entry["media_id"],
        "username": username,
        "t_type": t_type,
        "t_type_profile": str(creator.get("t_type") or ""),
        "niches": raw_entry.get("niches") or ["humour"],
        "text": raw_entry["text"],
        "comment_likes": raw_entry.get("comment_likes", 0),
        "views": raw_entry.get("views", 0),
        "comment_to_like_ratio": raw_entry.get("comment_to_like_ratio", 0.0),
        "caption": raw_entry.get("caption", ""),
        "hashtags": raw_entry.get("hashtags", []),
        "audio_id": raw_entry.get("audio_id", ""),
        "collected_at": raw_entry.get("collected_at", ""),
        "labelled_at": _utc_now_iso(),
        "llm_model": OLLAMA_MODEL,
    }


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
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw_entries = load_raw_comments()
    training_entries, existing_keys = load_training_comments()
    watchlist = load_watchlist(WATCHLIST_PATH)
    vector_store = load_vector_store(VECTOR_STORE_PATH)

    pending = []
    for raw_entry in raw_entries:
        media_id = str(raw_entry.get("media_id") or "")
        text = str(raw_entry.get("text") or "")
        if not media_id or not text:
            continue
        key = dedup_key(media_id, text)
        if not args.force and key in existing_keys:
            continue
        pending.append(raw_entry)

    if args.account:
        target = args.account.lstrip("@").strip().lower()
        pending = [
            entry
            for entry in pending
            if str(entry.get("username") or "").lstrip("@").strip().lower() == target
        ]

    if not pending:
        _LOG.info("Rien à labéliser")
        return 0

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

    for raw_entry in pending:
        text = str(raw_entry.get("text") or "")
        niches = list(raw_entry.get("niches") or ["humour"])
        username = str(raw_entry.get("username") or "").lstrip("@").strip()
        username_key = username.lower()

        creator_context = {
            "t_type_profile": watchlist.get(username_key, {}).get("t_type"),
            "niches": raw_entry.get("niches") or ["humour"],
            "named_axes": vector_store.get(username_key, {}).get("named_axes", {}),
        }
        t_type = classify_comment(
            text,
            niches,
            OLLAMA_MODEL,
            creator_context=creator_context,
        )
        if t_type is None:
            _LOG.warning("@%s commentaire non classifié, skip", username)
            continue

        entry = build_training_entry(raw_entry, t_type, watchlist)
        key = dedup_key(str(entry["media_id"]), str(entry["text"]))
        training_by_key[key] = entry
        added += 1
        preview = text[:40]
        _LOG.info("✓ [%s] %s...", t_type, preview)

    save_training_comments(list(training_by_key.values()))
    _LOG.info(
        "=== Labélisation : %d nouveaux commentaires → training_comments.json ===",
        added,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
