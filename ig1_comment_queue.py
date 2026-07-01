"""File d'attente IG1 — reels à commenter (watcher + spam feed).

Le watcher (IG2) pousse les nouveaux posts détectés ; ``scripts/ig1_spam_reels.py``
consomme la file en priorité avant de continuer le scroll fil Reels.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modules.atomic_json import atomic_write_json, json_lock

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_QUEUE_PATH = _PROJECT_ROOT / "data" / "ig1_comment_queue.json"
DEFAULT_COMMENTED_PATH = _PROJECT_ROOT / "data" / "ig1_commented_reels.json"
COMMENTED_TTL_S = 7 * 24 * 3600
MAX_QUEUE_ITEMS = 500


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return dict(default)
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else dict(default)
    except (OSError, ValueError):
        return dict(default)


def _save_json(path: Path, data: dict[str, Any], *, use_lock: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, data, use_lock=use_lock)


def _prune_queue(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cutoff = time.time() - 48 * 3600
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            ts = datetime.fromisoformat(
                str(it.get("queued_at") or "").replace("Z", "+00:00")
            ).timestamp()
        except (TypeError, ValueError):
            ts = 0.0
        if ts >= cutoff:
            out.append(it)
    return out[-MAX_QUEUE_ITEMS:]


def enqueue_ig1_comment(
    media_id: str,
    *,
    username: str = "",
    source: str = "watcher",
    path: Path | None = None,
) -> bool:
    """Ajoute un reel à commenter sur IG1. Retourne False si déjà en file ou commenté."""
    mid = str(media_id or "").strip()
    if not mid:
        return False
    if was_recently_commented(mid, path=DEFAULT_COMMENTED_PATH):
        return False

    qpath = path or DEFAULT_QUEUE_PATH
    with json_lock(qpath):
        data = _load_json(qpath, {"items": []})
        items = data.get("items")
        if not isinstance(items, list):
            items = []
        if any(str(it.get("media_id") or "") == mid for it in items if isinstance(it, dict)):
            return False
        items.append(
            {
                "media_id": mid,
                "username": str(username or "").lstrip("@"),
                "source": str(source or "watcher"),
                "queued_at": _now_iso(),
            }
        )
        data["items"] = _prune_queue(items)
        _save_json(qpath, data, use_lock=False)
    return True


def pop_ig1_comment(*, path: Path | None = None) -> dict[str, Any] | None:
    """Retire le prochain reel à commenter (FIFO)."""
    qpath = path or DEFAULT_QUEUE_PATH
    with json_lock(qpath):
        data = _load_json(qpath, {"items": []})
        items = data.get("items")
        if not isinstance(items, list) or not items:
            return None
        item = items.pop(0)
        data["items"] = items
        _save_json(qpath, data, use_lock=False)
    return item if isinstance(item, dict) else None


def queue_length(*, path: Path | None = None) -> int:
    qpath = path or DEFAULT_QUEUE_PATH
    data = _load_json(qpath, {"items": []})
    items = data.get("items")
    return len(items) if isinstance(items, list) else 0


def _prune_commented(commented: dict[str, Any]) -> dict[str, Any]:
    cutoff = time.time() - COMMENTED_TTL_S
    reels = commented.get("reels")
    if not isinstance(reels, dict):
        return {"reels": {}}
    kept = {
        k: v
        for k, v in reels.items()
        if isinstance(v, (int, float)) and float(v) >= cutoff
    }
    return {"reels": kept}


def was_recently_commented(
    media_id: str,
    *,
    path: Path | None = None,
) -> bool:
    mid = str(media_id or "").strip()
    if not mid:
        return True
    cpath = path or DEFAULT_COMMENTED_PATH
    data = _prune_commented(_load_json(cpath, {"reels": {}}))
    reels = data.get("reels") or {}
    return mid in reels


def mark_commented(
    media_id: str,
    *,
    path: Path | None = None,
) -> None:
    mid = str(media_id or "").strip()
    if not mid:
        return
    cpath = path or DEFAULT_COMMENTED_PATH
    with json_lock(cpath):
        data = _prune_commented(_load_json(cpath, {"reels": {}}))
        reels = data.setdefault("reels", {})
        if not isinstance(reels, dict):
            reels = {}
        reels[mid] = time.time()
        data["reels"] = reels
        _save_json(cpath, data, use_lock=False)
