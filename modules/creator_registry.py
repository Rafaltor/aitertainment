"""Lookup créateurs (database + watchlist + vector_store) pour le pipeline commentaires."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = _PROJECT_ROOT / "data" / "database.json"
WATCHLIST_PATH = _PROJECT_ROOT / "data" / "watchlist.json"
VECTOR_STORE_PATH = _PROJECT_ROOT / "data" / "vector_store.json"


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_creator_index() -> dict[str, dict[str, Any]]:
    """Index username → niches, t_type, présence vector_store."""
    index: dict[str, dict[str, Any]] = {}

    db = _read_json(DATABASE_PATH)
    profiles = db.get("profiles") if isinstance(db, dict) else {}
    if isinstance(profiles, dict):
        for username, profile in profiles.items():
            if not isinstance(profile, dict):
                continue
            key = str(username).lstrip("@").strip().lower()
            if not key:
                continue
            index[key] = {
                "username": key,
                "niches": profile.get("niches") or [],
                "t_type": profile.get("t_type_final") or profile.get("t_type"),
                "has_vector": False,
            }

    wl = _read_json(WATCHLIST_PATH)
    for creator in wl.get("creators") or []:
        if not isinstance(creator, dict):
            continue
        key = str(creator.get("username") or "").lstrip("@").strip().lower()
        if not key:
            continue
        base = dict(index.get(key, {}))
        base["username"] = key
        if creator.get("niches"):
            base["niches"] = creator.get("niches")
        if creator.get("t_type"):
            base["t_type"] = creator.get("t_type")
        index[key] = base

    vs = _read_json(VECTOR_STORE_PATH)
    if isinstance(vs, list):
        raw_entries = vs
    elif isinstance(vs, dict):
        raw_entries = vs.get("entries") or vs.get("profiles") or []
    else:
        raw_entries = []
    if isinstance(raw_entries, list):
        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("username") or "").lstrip("@").strip().lower()
            if not key:
                continue
            base = dict(index.get(key, {"username": key, "niches": []}))
            base["has_vector"] = True
            if entry.get("niches") and not base.get("niches"):
                base["niches"] = entry.get("niches")
            index[key] = base

    return index


def resolve_creator_fields(
    username: str,
    *,
    index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Résout niches / t_type / needs_embed pour un créateur de reel."""
    key = str(username or "").lstrip("@").strip().lower()
    if not key or key == "unknown":
        return {
            "username": "",
            "niches": ["humour"],
            "t_type_profile": None,
            "needs_embed": True,
            "known_creator": False,
        }

    idx = index if index is not None else build_creator_index()
    hit = idx.get(key, {})
    niches_raw = hit.get("niches") or ["humour"]
    niches = niches_raw if isinstance(niches_raw, list) else [str(niches_raw)]
    niches = [str(n).strip() for n in niches if str(n).strip()] or ["humour"]

    return {
        "username": key,
        "niches": niches,
        "t_type_profile": hit.get("t_type"),
        "needs_embed": not bool(hit.get("has_vector")),
        "known_creator": bool(hit),
    }
