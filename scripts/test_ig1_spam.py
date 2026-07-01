#!/usr/bin/env python3
"""Test IG1 : 2 bases Ollama + weave lowtaper67 (sans filtre qualité)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from modules.ig1_spam_generator import (
    _fuse_ig1_context,
    generate_ig1_spam_comments,
    spam_keyword,
)


def _sample_reel() -> dict[str, str]:
    return {
        "caption": "pov : tu rates ton oral mais tu restes confiant",
        "transcript": (
            "Non mais attendez les gars, j'ai révisé trois heures, "
            "le prof il me regarde comme si j'avais volé sa voiture, "
            "et moi je sors un sourire de winner quand même."
        ),
        "visual_description": (
            "Un jeune en hoodie dans une salle de classe, caméra selfie. "
            "Il hausse les épaules avec un air satisfait alors que ses potes "
            "se cachent le visage. Ton comique, lumière néon."
        ),
        "media_id": "TEST_REEL_001",
        "username": "sketch_humour",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Test IG1 2-pass sans filtre.")
    parser.add_argument(
        "--bases",
        type=int,
        default=1,
        help="(ignoré) — toujours 1 base + 1 weave",
    )
    args = parser.parse_args()

    reel = _sample_reel()
    kw = spam_keyword()

    print("=== Config ===")
    print(f"Generator : {config.OLLAMA_GENERATOR_MODEL}")
    print(f"Weave     : {getattr(config, 'SPAM_WEAVE_OLLAMA_MODEL', config.OLLAMA_MODEL)}")
    print(f"Mot-clé   : {kw}")
    print()

    merged = _fuse_ig1_context(
        reel["transcript"], reel["visual_description"], reel["caption"]
    )
    print(f"Contexte fusionné : {merged[:300]}{'…' if len(merged) > 300 else ''}")
    print()

    pairs = generate_ig1_spam_comments(
        caption=reel["caption"],
        transcript=reel["transcript"],
        visual_description=reel["visual_description"],
        media_id=reel["media_id"],
        username=reel["username"],
        base_count=args.bases,
    )

    print(f"=== {len(pairs)} passe(s) base → weave ===")
    for i, (base, woven) in enumerate(pairs, 1):
        print(f"\n  [{i}] BASE  : {base or '—'}")
        print(f"      WOVEN : {woven or '—'}")
        if woven:
            print(f"      kw?   : {'oui' if kw.lower() in woven.lower() else 'NON'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
