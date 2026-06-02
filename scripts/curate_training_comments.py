"""Filtre les commentaires peu adaptés à l'entraînement depuis training_comments_viral.json.

Usage::

    python scripts/curate_training_comments.py --dry-run
    python scripts/curate_training_comments.py
    python scripts/curate_training_comments.py --in-place --backup

Produit par défaut :

* ``data/training_comments_curated.json`` — entrées conservées
* ``data/training_rejected.json`` — entrées rejetées + ``reject_reasons``

Ensuite régénérer les datasets :

    python scripts/prepare_dataset.py \\
        --training-path data/training_comments_curated.json
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

from modules.comment_quality import CommentQuality, assess_comment_quality

DEFAULT_INPUT = _PROJECT_ROOT / "data" / "training_comments_viral.json"
DEFAULT_CURATED = _PROJECT_ROOT / "data" / "training_comments_curated.json"
DEFAULT_REJECTED = _PROJECT_ROOT / "data" / "training_rejected.json"
_LOG = logging.getLogger("aitertainment.curate_training_comments")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_entries(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Charge les entrées ; retourne ``(entries, wrapped)``."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return [], True
    data = json.loads(raw)
    if isinstance(data, dict):
        entries = data.get("entries") or data.get("comments") or []
        if not isinstance(entries, list):
            raise ValueError(f'"entries" invalide dans {path}')
        return [e for e in entries if isinstance(e, dict)], True
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)], False
    raise ValueError(f"racine JSON inattendue dans {path}")


def save_entries(
    path: Path,
    entries: list[dict[str, Any]],
    *,
    wrapped: bool,
    meta: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if wrapped:
        payload: dict[str, Any] = {"entries": entries}
        if meta:
            payload.update(meta)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        text = json.dumps(entries, ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def curate_entries(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for entry in entries:
        text = str(entry.get("text") or "")
        quality = assess_comment_quality(text)
        if quality.ok:
            kept.append(entry)
            continue
        reason_counts[quality.primary_reason or "unknown"] += 1
        rejected.append(
            {
                **entry,
                "reject_reasons": list(quality.reasons),
                "reject_primary": quality.primary_reason,
            }
        )

    return kept, rejected, reason_counts


def _print_summary(
    *,
    total: int,
    kept: int,
    rejected: int,
    reason_counts: Counter[str],
    examples: dict[str, list[str]],
) -> None:
    _LOG.info("Total : %d", total)
    _LOG.info("Conservés : %d (%.1f%%)", kept, 100.0 * kept / max(total, 1))
    _LOG.info("Rejetés : %d (%.1f%%)", rejected, 100.0 * rejected / max(total, 1))
    if reason_counts:
        _LOG.info("Raisons (première cause) :")
        for reason, count in reason_counts.most_common():
            _LOG.info("  - %s : %d", reason, count)
            for ex in examples.get(reason, [])[:2]:
                _LOG.info("      ex. %r", ex[:90])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Filtre training_comments_viral.json pour l'entraînement.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Fichier source (défaut : {DEFAULT_INPUT.name}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CURATED,
        help=f"Sortie conservée (défaut : {DEFAULT_CURATED.name}).",
    )
    parser.add_argument(
        "--rejected-output",
        type=Path,
        default=DEFAULT_REJECTED,
        help=f"Sortie rejetée (défaut : {DEFAULT_REJECTED.name}).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Écrase --input avec les entrées conservées.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Avec --in-place, sauvegarde l'original en .bak.<timestamp>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les stats sans écrire de fichier.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    input_path = args.input if args.input.is_absolute() else _PROJECT_ROOT / args.input
    if not input_path.exists():
        _LOG.error("Fichier introuvable : %s", input_path)
        return 1

    try:
        entries, wrapped = load_entries(input_path)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        _LOG.error("Lecture impossible : %s", e)
        return 1

    kept, rejected, reason_counts = curate_entries(entries)

    examples: dict[str, list[str]] = {}
    for entry in rejected:
        primary = str(entry.get("reject_primary") or "unknown")
        examples.setdefault(primary, [])
        if len(examples[primary]) < 3:
            examples[primary].append(str(entry.get("text") or ""))

    _print_summary(
        total=len(entries),
        kept=len(kept),
        rejected=len(rejected),
        reason_counts=reason_counts,
        examples=examples,
    )

    if args.dry_run:
        _LOG.info("(dry-run — aucun fichier écrit)")
        return 0

    meta = {
        "curated_at": _utc_now_iso(),
        "source": str(input_path.name),
        "kept_count": len(kept),
        "rejected_count": len(rejected),
    }
    rejected_meta = {**meta, "entries": rejected}

    out_path = input_path if args.in_place else (
        args.output if args.output.is_absolute() else _PROJECT_ROOT / args.output
    )
    rej_path = (
        args.rejected_output
        if args.rejected_output.is_absolute()
        else _PROJECT_ROOT / args.rejected_output
    )

    if args.in_place and args.backup:
        backup = input_path.with_suffix(
            input_path.suffix + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.copy2(input_path, backup)
        _LOG.info("Backup : %s", backup.name)

    try:
        save_entries(out_path, kept, wrapped=wrapped, meta=meta if wrapped else None)
        save_entries(
            rej_path,
            rejected,
            wrapped=True,
            meta={k: v for k, v in rejected_meta.items() if k != "entries"},
        )
    except OSError as e:
        _LOG.error("Écriture échouée : %s", e)
        return 1

    _LOG.info("Écrit : %s", out_path)
    _LOG.info("Écrit : %s", rej_path)
    if not args.in_place:
        _LOG.info(
            "Prochaine étape : python scripts/prepare_dataset.py "
            "--training-path %s",
            out_path.relative_to(_PROJECT_ROOT)
            if out_path.is_relative_to(_PROJECT_ROOT)
            else out_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
