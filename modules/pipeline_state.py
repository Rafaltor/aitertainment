"""État incrémental du pipeline commentaires (viral → label).

Évite de re-scraper ou re-labéliser ce qui est déjà traité.
La clé de dédup commentaire est partagée partout : ``media_id||text.lower()``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def comment_dedup_key(media_id: str, text: str) -> str:
    """Clé stable pour viral / training et empreintes."""
    return f"{media_id}||{text.strip().lower()}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_username(username: str) -> str:
    return str(username or "").lstrip("@").strip().lower()


def comments_for_username(
    username: str, raw_entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Filtre les entrées raw pour un créateur."""
    uname = _normalize_username(username)
    if not uname:
        return []
    out: list[dict[str, Any]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        if _normalize_username(str(entry.get("username") or "")) != uname:
            continue
        media_id = str(entry.get("media_id") or "").strip()
        text = str(entry.get("text") or "").strip()
        if media_id and text:
            out.append(entry)
    return out


def comments_fingerprint_for_account(
    username: str, raw_entries: list[dict[str, Any]]
) -> tuple[str, int]:
    """Empreinte SHA-256 des clés commentaire (ordre trié) + nombre de lignes."""
    rows = comments_for_username(username, raw_entries)
    keys = sorted(
        comment_dedup_key(str(r.get("media_id") or ""), str(r.get("text") or ""))
        for r in rows
    )
    digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()
    return digest, len(keys)


def load_training_labeled_keys(
    path: Path | str | None = None,
) -> set[str]:
    """Clés déjà présentes dans ``training_comments_viral.json``."""
    p = Path(path) if path is not None else _PROJECT_ROOT / "data" / "training_comments_viral.json"
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    if not p.exists():
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("entries") or data.get("comments") or []
    else:
        entries = []
    keys: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        media_id = str(entry.get("media_id") or "").strip()
        text = str(entry.get("text") or "").strip()
        if media_id and text:
            keys.add(comment_dedup_key(media_id, text))
    return keys


def should_skip_playwright_collect(
    username: str,
    raw_entries: list[dict[str, Any]],
    *,
    force_scrape: bool = False,
) -> bool:
    """True si des commentaires existent déjà en raw (discover / run précédent)."""
    if force_scrape:
        return False
    return len(comments_for_username(username, raw_entries)) > 0


def build_pipeline_patch(
    *,
    comments_count: int | None = None,
    comments_fingerprint: str | None = None,
    labeled_count: int | None = None,
    labeled_at: str | None = None,
) -> dict[str, Any]:
    """Sous-dict ``pipeline`` à fusionner dans un profil database."""
    patch: dict[str, Any] = {}
    if comments_count is not None:
        patch["comments_count"] = int(comments_count)
    if comments_fingerprint is not None:
        patch["comments_fingerprint"] = comments_fingerprint
    if labeled_count is not None:
        patch["labeled_count"] = int(labeled_count)
    if labeled_at is not None:
        patch["labeled_at"] = labeled_at
    return patch
