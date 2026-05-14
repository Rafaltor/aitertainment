"""Tests pour les utilitaires I/O de ``dataset_builder.py``."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dataset_builder as db_mod  # noqa: E402


class DatasetBuilderIOTest(unittest.TestCase):
    def test_load_training_missing_file_returns_empty_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_comments.json"
            self.assertEqual(db_mod._load_training(path), {"entries": []})

    def test_save_and_load_training_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_comments.json"
            payload = {"entries": [{"text": "mdr", "t_type": "T2"}]}
            db_mod._save_training(payload, path=path)
            loaded = db_mod._load_training(path)
            self.assertEqual(loaded, payload)

    def test_read_json_rejects_invalid_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
            with self.assertRaises(db_mod.DatasetIOError):
                db_mod._read_json(path)

    def test_load_training_rejects_non_list_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_comments.json"
            path.write_text(json.dumps({"entries": "bad"}), encoding="utf-8")
            with self.assertRaises(db_mod.DatasetIOError):
                db_mod._load_training(path)


if __name__ == "__main__":
    unittest.main()
