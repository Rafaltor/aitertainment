"""Tests scripts/clean_comments.py (pools viral + training)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import clean_comments as clean


class CleanViralPoolTest(unittest.TestCase):
    def test_drops_english_and_unknown_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "viral.json"
            path.write_text(
                json.dumps(
                    [
                        {"text": "super vidéo", "username": "creator_fr"},
                        {"text": "this is clearly english content", "username": "creator_en"},
                        {"text": "ok", "username": "unknown"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            idx = {
                "creator_fr": {
                    "username": "creator_fr",
                    "niches": ["humour"],
                    "t_type": "T2",
                    "has_vector": True,
                }
            }
            with patch.object(clean, "build_creator_index", return_value=idx):
                i, k, en, unk = clean.clean_viral_pool(path, dry_run=True)
            self.assertEqual(i, 3)
            self.assertEqual(k, 1)
            self.assertGreaterEqual(en, 1)
            self.assertEqual(unk, 1)


class CleanTrainingPoolTest(unittest.TestCase):
    def test_strips_emojis_and_rejects_low_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.json"
            path.write_text(
                json.dumps(
                    [
                        {"text": "🔥 super punchline ici", "t_type": "T2"},
                        {"text": "x", "t_type": "T2"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            i, k, stripped, rejects = clean.clean_training_pool(
                path, dry_run=True, backup=False
            )
            self.assertEqual(i, 2)
            self.assertEqual(k, 1)
            self.assertGreaterEqual(stripped, 1)
            self.assertTrue(rejects)


if __name__ == "__main__":
    unittest.main()
