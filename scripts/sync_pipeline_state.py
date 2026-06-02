"""sync_pipeline_state.py — aligne database.pipeline avec embedder / vector_store / training viral.

À lancer une fois après validation du pipeline pour backfiller les empreintes
sans re-embedder ni re-labéliser.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from database import load_db, merge_profile_pipeline, save_db
from modules.pipeline_state import (
    build_pipeline_patch,
    comment_dedup_key,
    load_training_labeled_keys,
)
from scripts.embedder import load_vector_store, save_vector_store

_LOG = logging.getLogger("aitertainment.sync_pipeline_state")


def _training_counts_by_user(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else data.get("entries", data.get("comments", []))
    counts: Counter[str] = Counter()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        u = str(entry.get("username") or "").lstrip("@").strip().lower()
        if u and entry.get("t_type"):
            counts[u] += 1
    return dict(counts)


def _latest_labeled_at(path: Path) -> str | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else data.get("entries", data.get("comments", []))
    latest: str | None = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ts = str(entry.get("labelled_at") or "").strip()
        if ts and (latest is None or ts > latest):
            latest = ts
    return latest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill pipeline.* dans database.json et empreintes vector_store."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les mises à jour sans écrire.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    store = load_vector_store()
    training_path = _PROJECT_ROOT / "data/training_comments_viral.json"
    labeled_by_user = _training_counts_by_user(training_path)
    latest_label = _latest_labeled_at(training_path)
    labeled_keys = load_training_labeled_keys(training_path)

    usernames: set[str] = set()
    for entry in store:
        u = str(entry.get("username") or "").lstrip("@").strip().lower()
        if u:
            usernames.add(u)
    usernames.update(labeled_by_user.keys())

    db = load_db()
    profiles = db.get("profiles") or {}
    db_updates = 0
    vs_updates = 0

    for username in sorted(usernames):
        vs_entry = next(
            (
                e
                for e in store
                if str(e.get("username") or "").lstrip("@").strip().lower() == username
            ),
            None,
        )
        fp = ""
        n = 0
        if isinstance(vs_entry, dict):
            fp = str(vs_entry.get("comments_fingerprint") or "")
            sources = vs_entry.get("sources")
            if not fp and isinstance(sources, dict):
                fp = str(sources.get("comments_fingerprint") or "")
            n = int(vs_entry.get("comments_count") or 0)
            if not n and isinstance(sources, dict):
                n = int(sources.get("comments_count") or 0)
        patch: dict[str, Any] = build_pipeline_patch(
            comments_count=n,
            comments_fingerprint=fp or None,
        )
        if username in labeled_by_user:
            patch["labeled_count"] = labeled_by_user[username]
        if latest_label:
            patch["labeled_at"] = latest_label

        for entry in store:
            if str(entry.get("username") or "").lstrip("@").strip().lower() != username:
                continue
            if entry.get("embedding_raw"):
                patch["embedded_at"] = str(entry.get("updated_at") or "")
            if not entry.get("comments_fingerprint") and fp:
                entry["comments_fingerprint"] = fp
                entry["comments_count"] = n
                sources = entry.get("sources")
                if isinstance(sources, dict):
                    sources["comments_fingerprint"] = fp
                    sources["comments_count"] = n
                vs_updates += 1

        if username in profiles:
            if args.dry_run:
                _LOG.info("DRY-RUN database @%s pipeline ← %s", username, patch)
            elif merge_profile_pipeline(db, username, patch) is not None:
                db_updates += 1

    if args.dry_run:
        _LOG.info(
            "DRY-RUN : %d profil(s) database, %d entrée(s) vector_store",
            db_updates,
            vs_updates,
        )
        return 0

    if db_updates:
        save_db(db)
    if vs_updates:
        save_vector_store(store)

    _LOG.info(
        "=== Sync pipeline : %d profil(s) database, %d vector_store, "
        "%d clés training ===",
        db_updates,
        vs_updates,
        len(labeled_keys),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
