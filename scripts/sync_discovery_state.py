#!/usr/bin/env python3
"""Nettoie candidates.json depuis watchlist / database.

``database.json`` est la source de vérité pour les profils déjà scorés ;
les candidats déjà validés (watchlist ou ``validated`` en base) sont retirés.

Usage :
    python scripts/sync_discovery_state.py           # applique les changements
    python scripts/sync_discovery_state.py --dry-run # affiche sans écrire
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DATA_DIR = _ROOT / "data"
WATCHLIST_PATH = DATA_DIR / "watchlist.json"
DATABASE_PATH = DATA_DIR / "database.json"
CANDIDATES_PATH = DATA_DIR / "candidates.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_watchlist_entries() -> list[dict[str, Any]]:
    if not WATCHLIST_PATH.exists():
        return []
    data = _read_json(WATCHLIST_PATH)
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in data.get("creators") or []:
        if not isinstance(item, dict):
            continue
        u = str(item.get("username") or "").lstrip("@").strip().lower()
        if not u or u in seen:
            continue
        seen.add(u)
        entries.append(item)
    return entries


def _processed_usernames() -> set[str]:
    """Profils déjà traités : watchlist + validés en database."""
    processed: set[str] = set()
    for entry in _load_watchlist_entries():
        u = str(entry.get("username") or "").lstrip("@").strip().lower()
        if u:
            processed.add(u)

    db = _read_json(DATABASE_PATH)
    profiles = db.get("profiles")
    if isinstance(profiles, dict):
        for key, profile in profiles.items():
            u = str(key).lstrip("@").strip().lower()
            if not u:
                continue
            if isinstance(profile, dict) and profile.get("validated"):
                processed.add(u)
    return processed


def prune_candidates(*, dry_run: bool) -> tuple[int, int]:
    data = _read_json(CANDIDATES_PATH) if CANDIDATES_PATH.exists() else {"candidates": []}
    candidates = list(data.get("candidates") or [])

    processed = _processed_usernames()

    kept: list[dict[str, Any]] = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        u = str(c.get("username") or "").lstrip("@").strip().lower()
        if u and u not in processed:
            kept.append(c)

    removed = len(candidates) - len(kept)
    if dry_run:
        print(
            f"[dry-run] candidates : {len(candidates)} → {len(kept)} "
            f"({removed} retiré(s), déjà validés/watchlist)"
        )
        return len(candidates), len(kept)

    _atomic_write_json(CANDIDATES_PATH, {"candidates": kept})
    print(
        f"candidates nettoyés : {len(candidates)} → {len(kept)} "
        f"({removed} retiré(s)) → {CANDIDATES_PATH}"
    )
    return len(candidates), len(kept)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les changements sans écrire sur disque",
    )
    args = parser.parse_args()
    prune_candidates(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
