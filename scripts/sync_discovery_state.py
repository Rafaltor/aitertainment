#!/usr/bin/env python3
"""Synchronise watchlist, blacklist et candidates avec l'état réel du pipeline.

Usage :
    python scripts/sync_discovery_state.py           # applique les changements
    python scripts/sync_discovery_state.py --dry-run # affiche sans écrire
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from database import DB_PATH, load_db  # noqa: E402

DATA_DIR = _ROOT / "data"
WATCHLIST_PATH = DATA_DIR / "watchlist.json"
ROOT_WATCHLIST_PATH = _ROOT / "watchlist.json"
VALIDATIONS_PATH = DATA_DIR / "validations.json"
BLACKLIST_PATH = DATA_DIR / "blacklist.json"
CANDIDATES_PATH = DATA_DIR / "candidates.json"
NOTIFY_THRESHOLD = float(config.DISCOVERY_NOTIFY_THRESHOLD)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_watchlist_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in (ROOT_WATCHLIST_PATH, WATCHLIST_PATH):
        if not path.exists():
            continue
        data = _read_json(path)
        for key in ("creators", "watchlist"):
            for item in data.get(key) or []:
                if not isinstance(item, dict):
                    continue
                u = str(item.get("username") or "").lstrip("@").strip().lower()
                if not u or u in seen:
                    continue
                seen.add(u)
                entries.append(item)
    return entries


def merge_watchlist(*, dry_run: bool) -> int:
    entries = _load_watchlist_entries()
    if dry_run:
        print(f"[dry-run] watchlist → {WATCHLIST_PATH} : {len(entries)} créateur(s)")
        return len(entries)
    _atomic_write_json(WATCHLIST_PATH, {"creators": entries})
    if ROOT_WATCHLIST_PATH.exists():
        ROOT_WATCHLIST_PATH.unlink()
        print(f"watchlist racine supprimée : {ROOT_WATCHLIST_PATH}")
    print(f"watchlist unifiée : {len(entries)} créateur(s) → {WATCHLIST_PATH}")
    return len(entries)


def _latest_validation_actions() -> dict[str, str]:
    data = _read_json(VALIDATIONS_PATH)
    out: dict[str, str] = {}
    for v in data.get("validations") or []:
        if not isinstance(v, dict):
            continue
        u = str(v.get("username") or "").lstrip("@").strip().lower()
        action = str(v.get("action") or "").strip().lower()
        if u and action:
            out[u] = action
    return out


def _domain_for_username(
    username: str,
    *,
    validation_domains: dict[str, str],
    candidate_domains: dict[str, str],
    default_domain: str,
) -> str:
    return (
        validation_domains.get(username)
        or candidate_domains.get(username)
        or default_domain
    )


def _latest_score(profile: dict[str, Any]) -> float | None:
    hist = profile.get("scores_history") or []
    if not hist:
        return None
    raw = hist[-1].get("score")
    return float(raw) if raw is not None else None


def _blacklist_outcome(
    username: str,
    profile: dict[str, Any],
    *,
    validation_action: str | None,
    notify_threshold: float,
) -> str:
    if validation_action == "rejected":
        return "rejected"
    score = _latest_score(profile)
    if score is not None and score > notify_threshold:
        return "candidate"
    if profile.get("archived") or profile.get("tier") == "C":
        return "rejected"
    return "rejected" if score is not None else "ineligible"


def rebuild_blacklist(*, dry_run: bool, notify_threshold: float) -> int:
    db = load_db(path=DB_PATH)
    profiles = db.get("profiles") or {}
    validation_actions = _latest_validation_actions()
    validations_data = _read_json(VALIDATIONS_PATH)
    validation_domains: dict[str, str] = {}
    for v in validations_data.get("validations") or []:
        if isinstance(v, dict):
            u = str(v.get("username") or "").lstrip("@").strip().lower()
            d = str(v.get("domain") or "").strip()
            if u and d:
                validation_domains[u] = d

    candidate_domains: dict[str, str] = {}
    cands = _read_json(CANDIDATES_PATH) if CANDIDATES_PATH.exists() else {"candidates": []}
    for c in cands.get("candidates") or []:
        if isinstance(c, dict):
            u = str(c.get("username") or "").lstrip("@").strip().lower()
            d = str(c.get("domain") or "").strip()
            if u and d:
                candidate_domains[u] = d

    default_domain = "unknown"
    seeds = _read_json(_ROOT / "seeds.json") or _read_json(DATA_DIR / "seeds.json")
    domains = seeds.get("domains") or []
    if domains and isinstance(domains[0], dict):
        default_domain = str(domains[0].get("name") or default_domain)

    entries: list[dict[str, Any]] = []
    for username, profile in sorted(profiles.items()):
        if not isinstance(profile, dict):
            continue
        u = username.lstrip("@").strip().lower()
        score = _latest_score(profile)
        added_at = str(profile.get("last_scored_at") or profile.get("added_at") or _now_iso())
        entries.append(
            {
                "username": u,
                "platform": str(profile.get("platform") or "instagram"),
                "domain": _domain_for_username(
                    u,
                    validation_domains=validation_domains,
                    candidate_domains=candidate_domains,
                    default_domain=default_domain,
                ),
                "outcome": _blacklist_outcome(
                    u,
                    profile,
                    validation_action=validation_actions.get(u),
                    notify_threshold=notify_threshold,
                ),
                "score": score,
                "added_at": added_at,
            }
        )

    if dry_run:
        print(f"[dry-run] blacklist : {len(entries)} entrée(s) depuis database")
        return len(entries)

    _atomic_write_json(BLACKLIST_PATH, {"profiles": entries})
    print(f"blacklist reconstruite : {len(entries)} entrée(s) → {BLACKLIST_PATH}")
    return len(entries)


def prune_candidates(*, dry_run: bool) -> tuple[int, int]:
    data = _read_json(CANDIDATES_PATH) if CANDIDATES_PATH.exists() else {"candidates": []}
    candidates = list(data.get("candidates") or [])

    processed: set[str] = set(_latest_validation_actions())
    for entry in _load_watchlist_entries():
        u = str(entry.get("username") or "").lstrip("@").strip().lower()
        if u:
            processed.add(u)

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
    parser.add_argument(
        "--notify-threshold",
        type=float,
        default=NOTIFY_THRESHOLD,
        help="Seuil score pour outcome=candidate (défaut: config)",
    )
    args = parser.parse_args()

    merge_watchlist(dry_run=args.dry_run)
    rebuild_blacklist(dry_run=args.dry_run, notify_threshold=args.notify_threshold)
    prune_candidates(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
