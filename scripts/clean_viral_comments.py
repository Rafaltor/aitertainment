#!/usr/bin/env python3
"""Nettoie viral_comments.json : français uniquement + métadonnées créateur."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules.comment_quality import is_french_comment
from modules.creator_registry import build_creator_index, resolve_creator_fields
from scripts.instagram_browser import (
    VIRAL_COMMENTS_PATH,
    load_raw_comments_file,
    save_raw_comments_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Filtre FR + enrichit viral_comments.")
    parser.add_argument("--input", type=Path, default=VIRAL_COMMENTS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    entries, _ = load_raw_comments_file(args.input)
    idx = build_creator_index()
    kept: list[dict] = []
    dropped_en = dropped_unknown = 0

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "").strip()
        user = str(entry.get("username") or "").lstrip("@").strip().lower()
        if not text or not user or user == "unknown":
            dropped_unknown += 1
            continue
        if not is_french_comment(text):
            dropped_en += 1
            continue
        meta = resolve_creator_fields(user, index=idx)
        row = dict(entry)
        row["username"] = meta["username"]
        row["niches"] = meta["niches"]
        row["t_type_profile"] = meta.get("t_type_profile")
        row["needs_embed"] = meta["needs_embed"]
        kept.append(row)

    out = args.output or args.input
    save_raw_comments_file(kept, out)
    print(
        f"OK : {len(kept)} gardés, {dropped_en} anglais, "
        f"{dropped_unknown} sans créateur → {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
