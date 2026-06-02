#!/usr/bin/env python3
"""Pipeline entraînement à partir de viral_comments.json uniquement.

Étapes :
  1. clean_viral_comments (FR + métadonnées créateur)
  2. label_comments → training_comments_viral.json (LM Studio / Ollama)
  3. prepare_dataset → data/generator_dataset.json + dataset_generator.jsonl
  4. (optionnel) indique la commande notebook / GPU pour le fine-tune

Usage::

    .venv/bin/python scripts/train_from_viral.py --dry-run
    .venv/bin/python scripts/train_from_viral.py --label-limit 50
    .venv/bin/python scripts/train_from_viral.py
    .venv/bin/python scripts/train_from_viral.py --skip-label --skip-clean
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.instagram_browser import VIRAL_COMMENTS_PATH, load_raw_comments_file

_LOG = logging.getLogger(__name__)

DEFAULT_VIRAL = VIRAL_COMMENTS_PATH
DEFAULT_TRAINING = _PROJECT_ROOT / "data" / "training_comments_viral.json"
DATA_DIR = _PROJECT_ROOT / "data"
PYTHON = sys.executable


def _run(cmd: list[str], *, dry_run: bool) -> int:
    _LOG.info("$ %s", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=_PROJECT_ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prépare l'entraînement generator depuis viral_comments.json."
    )
    parser.add_argument("--viral-path", type=Path, default=DEFAULT_VIRAL)
    parser.add_argument("--training-path", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Répertoire de sortie prepare_dataset (défaut: data/).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-label", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument(
        "--label-limit",
        type=int,
        default=0,
        help="Max commentaires à labelliser (0 = tout).",
    )
    parser.add_argument(
        "--force-label",
        action="store_true",
        help="Re-labellise même si déjà dans training_comments_viral.json.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.viral_path.exists():
        _LOG.error("Fichier viral introuvable : %s", args.viral_path)
        return 1

    entries, _ = load_raw_comments_file(args.viral_path)
    _LOG.info("Viral : %d entrée(s) dans %s", len(entries), args.viral_path)

    if not args.skip_clean:
        code = _run(
            [PYTHON, "scripts/clean_viral_comments.py", "--input", str(args.viral_path)],
            dry_run=args.dry_run,
        )
        if code != 0:
            return code

    if not args.skip_label:
        cmd = [
            PYTHON,
            "scripts/label_comments.py",
            "--viral-path",
            str(args.viral_path),
            "--training-path",
            str(args.training_path),
        ]
        if args.force_label:
            cmd.append("--force")
        if args.label_limit > 0:
            cmd.extend(["--limit", str(args.label_limit)])
        code = _run(cmd, dry_run=args.dry_run)
        if code != 0:
            return code

    if not args.skip_prepare:
        if not args.training_path.exists() and not args.dry_run:
            _LOG.error(
                "Training absent : %s — lancez d'abord la labélisation.",
                args.training_path,
            )
            return 1
        code = _run(
            [
                PYTHON,
                "scripts/prepare_dataset.py",
                "--training-path",
                str(args.training_path),
                "--output-dir",
                str(args.output_dir),
            ],
            dry_run=args.dry_run,
        )
        if code != 0:
            return code

    gen_json = args.output_dir / "generator_dataset.json"

    if not args.dry_run and gen_json.exists():
        data = json.loads(gen_json.read_text(encoding="utf-8"))
        n = len(data) if isinstance(data, list) else 0
        _LOG.info("Generator dataset : %d paires → %s", n, gen_json)
    elif args.dry_run:
        _LOG.info("Generator dataset → %s", gen_json)

    _LOG.info(
        "\n=== Fine-tune (GPU CUDA requis — Colab / RunPod) ===\n"
        "Notebook : DATASET_PATH = data/generator_dataset.json\n"
        "\n"
        "Sur Mac (MPS seulement) : Unsloth/QLoRA n'est pas supporté localement ;\n"
        "utilisez le notebook sur GPU cloud.\n",
        gen_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
