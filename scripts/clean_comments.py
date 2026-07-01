#!/usr/bin/env python3
"""Nettoie les pools de commentaires viral et/ou training.

**viral** (``viral_comments.json``, avant labélisation) :
  - transcript ET description visuelle requis
  - français uniquement, créateur connu
  - enrichit ``niches``, ``t_type_profile`` depuis watchlist / database

**training** (``training_comments_viral.json``, après labélisation) :
  - retire les emojis du texte
  - filtre qualité (``assess_comment_quality``)

Usage::

    python scripts/clean_comments.py --viral
    python scripts/clean_comments.py --training
    python scripts/clean_comments.py              # viral puis training
    python scripts/clean_comments.py --viral --dry-run
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

from modules.comment_quality import assess_comment_quality, is_french_comment, strip_emojis
from modules.creator_registry import build_creator_index, resolve_creator_fields
from scripts.instagram_browser import (
    VIRAL_COMMENTS_PATH,
    load_viral_comments_file,
    save_viral_comments_file,
)

DEFAULT_VIRAL = VIRAL_COMMENTS_PATH
DEFAULT_TRAINING = _PROJECT_ROOT / "data" / "training_comments_viral.json"
_LOG = logging.getLogger("aitertainment.clean_comments")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def strip_emojis_viral_pool(
    path: Path,
    *,
    dry_run: bool,
    backup: bool = False,
) -> tuple[int, int, int, int]:
    """Retire les emojis du champ ``text`` dans ``viral_comments.json``.

    Retourne ``(initial, kept, stripped, dropped_empty)``.
    """
    entries, _ = load_viral_comments_file(path)
    initial = len(entries)
    kept: list[dict[str, Any]] = []
    stripped_count = 0
    dropped_empty = 0

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        original = str(entry.get("text") or "")
        cleaned = strip_emojis(original)
        if not cleaned:
            dropped_empty += 1
            continue
        row = dict(entry)
        if cleaned != original.strip():
            stripped_count += 1
            row["text"] = cleaned
            row["emoji_stripped"] = True
        else:
            row["text"] = cleaned
        kept.append(row)

    if not dry_run:
        if backup:
            backup_path = path.with_suffix(
                path.suffix + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy2(path, backup_path)
            _LOG.info("Backup viral : %s", backup_path.name)
        save_viral_comments_file(
            kept, path, merge=False, allow_shrink=True
        )
    return initial, len(kept), stripped_count, dropped_empty


def clean_viral_pool(
    path: Path,
    *,
    dry_run: bool,
    backup: bool = False,
    force: bool = False,
) -> tuple[int, int, int, int, int]:
    """Retourne ``(initial, kept, dropped_en, dropped_unknown, dropped_no_enrichment)``."""
    entries, _ = load_viral_comments_file(path)
    initial = len(entries)
    idx = build_creator_index()
    kept: list[dict[str, Any]] = []
    dropped_en = dropped_unknown = dropped_no_enrichment = 0

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "").strip()
        user = str(entry.get("username") or "").lstrip("@").strip().lower()
        transcript = str(entry.get("transcript") or "").strip()
        visual = str(entry.get("visual_description") or "").strip()
        if not transcript or not visual:
            dropped_no_enrichment += 1
            continue
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
        kept.append(row)

    if not dry_run:
        if backup:
            backup_path = path.with_suffix(
                path.suffix + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy2(path, backup_path)
            _LOG.info("Backup viral : %s", backup_path.name)
        save_viral_comments_file(
            kept, path, merge=False, allow_shrink=force
        )
    return initial, len(kept), dropped_en, dropped_unknown, dropped_no_enrichment


def _load_training_entries(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)], None
    if isinstance(data, dict):
        entries = data.get("entries") or data.get("comments") or []
        meta = {k: v for k, v in data.items() if k not in ("entries", "comments")}
        return [e for e in entries if isinstance(e, dict)], meta
    raise ValueError(f"JSON inattendu dans {path}")


def _save_training_entries(
    path: Path,
    entries: list[dict[str, Any]],
    wrapper_meta: dict[str, Any] | None,
) -> None:
    if wrapper_meta is not None:
        payload: dict[str, Any] = {**wrapper_meta, "entries": entries}
        payload["cleaned_at"] = _utc_now_iso()
    else:
        payload = entries
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def clean_training_pool(
    path: Path,
    *,
    dry_run: bool,
    backup: bool,
) -> tuple[int, int, int, Counter[str]]:
    """Retourne ``(initial, kept, stripped, reject_counts)``."""
    entries, meta = _load_training_entries(path)
    initial = len(entries)
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

    if not dry_run:
        if backup:
            backup_path = path.with_suffix(
                path.suffix + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy2(path, backup_path)
            _LOG.info("Backup training : %s", backup_path.name)
        _save_training_entries(path, kept, meta)

    return initial, len(kept), stripped_count, reject_counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viral",
        action="store_true",
        help="Nettoie viral_comments.json uniquement.",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Nettoie training_comments_viral.json uniquement.",
    )
    parser.add_argument("--viral-path", type=Path, default=DEFAULT_VIRAL)
    parser.add_argument("--training-path", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup horodaté avant écriture (viral et/ou training).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Désactive le backup automatique avant écriture.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Autorise clean viral même si >50%% des entrées seraient supprimées.",
    )
    parser.add_argument(
        "--strip-emojis-viral",
        action="store_true",
        help="Retire les emojis du texte dans viral_comments.json (sans autres filtres).",
    )
    args = parser.parse_args(argv)
    do_backup = args.backup or not args.no_backup

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_viral = bool(args.viral)
    run_training = bool(args.training)
    run_strip_viral = bool(args.strip_emojis_viral)
    if not run_viral and not run_training and not run_strip_viral:
        parser.error(
            "Indiquez --viral, --training et/ou --strip-emojis-viral "
            "(plus de mode par défaut)."
        )

    if run_strip_viral:
        viral_path = args.viral_path
        if not viral_path.exists() and not args.dry_run:
            _LOG.error("Fichier viral introuvable : %s", viral_path)
            return 1
        if viral_path.exists():
            i, k, stripped, dropped = strip_emojis_viral_pool(
                viral_path,
                dry_run=args.dry_run,
                backup=do_backup and not args.dry_run,
            )
            _LOG.info(
                "Viral emojis %s : %d → %d gardés (%d strip, %d vides après strip)",
                viral_path.name,
                i,
                k,
                stripped,
                dropped,
            )
        else:
            _LOG.info("Viral : fichier absent — skip.")

    if run_viral:
        viral_path = args.viral_path
        if not viral_path.exists() and not args.dry_run:
            _LOG.error("Fichier viral introuvable : %s", viral_path)
            return 1
        if viral_path.exists():
            i, k, en, unk, no_enrich = clean_viral_pool(
                viral_path,
                dry_run=args.dry_run,
                backup=do_backup and not args.dry_run,
                force=args.force,
            )
            _LOG.info(
                "Viral %s : %d → %d gardés (%d sans enrichissement, %d EN, %d sans créateur)",
                viral_path.name,
                i,
                k,
                no_enrich,
                en,
                unk,
            )
        else:
            _LOG.info("Viral : fichier absent — skip.")

    if run_training:
        training_path = args.training_path
        if not training_path.exists():
            _LOG.error("Fichier training introuvable : %s", training_path)
            return 1
        i, k, stripped, rejects = clean_training_pool(
            training_path,
            dry_run=args.dry_run,
            backup=do_backup and not args.dry_run,
        )
        _LOG.info(
            "Training %s : %d → %d gardés (%d emojis strip)",
            training_path.name,
            i,
            k,
            stripped,
        )
        if rejects:
            _LOG.info("Raisons de rejet training :")
            for reason, count in rejects.most_common():
                _LOG.info("  - %s : %d", reason, count)

    if args.dry_run:
        _LOG.info("(dry-run — aucun fichier écrit)")
    elif run_training:
        _LOG.info("Prochaine étape : python scripts/prepare_dataset.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
