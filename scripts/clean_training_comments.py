"""Nettoie training_comments_viral.json — strip emojis + filtre qualité automatique.

Usage::

    python scripts/clean_training_comments.py --dry-run
    python scripts/clean_training_comments.py
    python scripts/clean_training_comments.py --in-place --backup

Puis régénérer les datasets :

    python scripts/prepare_dataset.py
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules.comment_quality import assess_comment_quality, strip_emojis

DEFAULT_INPUT = _PROJECT_ROOT / "data" / "training_comments_viral.json"
_LOG = logging.getLogger("aitertainment.clean_training_comments")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_entries(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)], None
    if isinstance(data, dict):
        entries = data.get("entries") or data.get("comments") or []
        meta = {k: v for k, v in data.items() if k not in ("entries", "comments")}
        return [e for e in entries if isinstance(e, dict)], meta
    raise ValueError(f"JSON inattendu dans {path}")


def save_entries(
    path: Path,
    entries: list[dict[str, Any]],
    wrapper_meta: dict[str, Any] | None,
) -> None:
    if wrapper_meta is not None:
        payload: dict[str, Any] = {**wrapper_meta, "entries": entries}
        payload["emoji_stripped_at"] = _utc_now_iso()
    else:
        payload = entries
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_entries(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    kept: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()
    stripped_count = 0

    for entry in entries:
        original = str(entry.get("text") or "")
        cleaned = strip_emojis(original)
        if cleaned != original.strip():
            stripped_count += 1
        quality = assess_comment_quality(cleaned)
        if not quality.ok:
            reject_counts[quality.primary_reason or "unknown"] += 1
            continue
        new_entry = dict(entry)
        new_entry["text"] = cleaned
        if cleaned != original:
            new_entry["emoji_stripped"] = True
        kept.append(new_entry)

    return kept, reject_counts, stripped_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strip emojis + filtre qualité sur training_comments_viral.json.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--in-place", action="store_true", help="Écrase --input.")
    parser.add_argument("--backup", action="store_true", help="Backup .bak avant écriture.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = args.input if args.input.is_absolute() else _PROJECT_ROOT / args.input
    if not path.exists():
        _LOG.error("Fichier introuvable : %s", path)
        return 1

    entries, meta = load_entries(path)
    kept, rejects, stripped = clean_entries(entries)

    _LOG.info("Entrées initiales : %d", len(entries))
    _LOG.info("Emojis retirés sur : %d commentaires", stripped)
    _LOG.info("Conservées : %d", len(kept))
    _LOG.info("Rejetées après nettoyage : %d", len(entries) - len(kept))
    if rejects:
        _LOG.info("Raisons de rejet :")
        for reason, count in rejects.most_common():
            _LOG.info("  - %s : %d", reason, count)

    if args.dry_run:
        _LOG.info("(dry-run — aucun fichier écrit)")
        return 0

    out = path if args.in_place else path
    if args.in_place and args.backup:
        backup = path.with_suffix(path.suffix + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(path, backup)
        _LOG.info("Backup : %s", backup.name)

    save_entries(out, kept, meta)
    _LOG.info("Écrit : %s (%d entrées)", out, len(kept))
    _LOG.info("Prochaine étape : python scripts/prepare_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
