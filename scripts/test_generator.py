#!/usr/bin/env python3
"""Test local du générateur (OLLAMA_GENERATOR_MODEL requis)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from config import ORDERED_T_TYPES
from modules.classifier import generate_comments, generate_comments_per_category


def main() -> int:
    parser = argparse.ArgumentParser(description="Test generate_comments (Ollama).")
    parser.add_argument(
        "--t-type",
        default=None,
        help="Une seule catégorie (ex. T3b). Par défaut : toutes les catégories.",
    )
    parser.add_argument("--niche", default="humour", help="Niche(s), séparées par des virgules")
    parser.add_argument(
        "--caption",
        default="quand tu rates ton exam mais que t'assumes",
    )
    args = parser.parse_args()

    model = config.OLLAMA_GENERATOR_MODEL or config.OLLAMA_MODEL
    mode = f"Ollama ({config.OLLAMA_GENERATOR_MODEL})"
    print(f"Mode: {mode}")
    print(f"Modèle: {model}")
    print(f"URL: {config.OLLAMA_URL}\n")

    niches = [n.strip() for n in args.niche.split(",") if n.strip()]
    video_context = {
        "caption": args.caption,
        "hashtags": ["humour", "etudiant"],
        "audio_id": "",
    }

    if args.t_type:
        classification = {"type": args.t_type, "confidence": 0.9}
        comments = generate_comments(
            classification,
            [],
            niches=niches,
            t_type_profile=args.t_type,
            video_context=video_context,
        )
        for i, c in enumerate(comments, 1):
            print(f"  {i}. {c}")
        return 0

    by_type = generate_comments_per_category(
        niches=niches,
        video_context=video_context,
        t_types=ORDERED_T_TYPES,
    )
    for t_type in ORDERED_T_TYPES:
        print(f"  [{t_type}] {by_type.get(t_type, '—')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
