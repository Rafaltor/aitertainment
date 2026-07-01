"""Tests pour ``scripts.prepare_dataset`` (refonte JSONL 2026-05).

Couvre les 4 surfaces publiques exposées par le module :

* ``niches_str(entry)``           — résolution de la string niches.
* ``generate_generator_dataset``  — sortie JSONL du generator.
* ``main([...])``                 — orchestration CLI / exit codes.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.generator_prompt import build_generator_instruction
from scripts.prepare_dataset import (
    GENERATOR_FILENAME,
    generate_generator_dataset,
    main,
    niches_str,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    """Lit un JSONL (1 objet par ligne, lignes vides ignorées)."""
    if not path.exists():
        return []
    out: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            out.append(json.loads(raw))
    return out


def _valid_classifier_entry(**overrides) -> dict:
    """Entrée plate avec tous les champs requis pour le classifier."""
    base = {
        "text": "mdr trop vrai",
        "t_type": "T2",
        "niches": ["humour", "sketch"],
        "views": 500_000,
        "comment_to_like_ratio": 0.2834,
    }
    base.update(overrides)
    return base


def _valid_generator_entry(**overrides) -> dict:
    """Entrée plate avec tous les champs requis pour le generator."""
    base = {
        "text": "le passage 0:08",
        "t_type": "T2",
        "t_type_profile": "T3b",
        "niches": ["humour", "sketch"],
        "caption": "moment culte F1",
        "hashtags": ["F1", "monaco"],
        "audio_id": "AUD123",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# niches_str
# ---------------------------------------------------------------------------


class NichesStrTest(unittest.TestCase):
    """Cas couverts par le brief (4 cas) + edge cases dérivés."""

    def test_list_joined_with_comma(self) -> None:
        # Cas 1 : liste ["humour", "sketch"] → "humour, sketch".
        self.assertEqual(
            niches_str({"niches": ["humour", "sketch"]}),
            "humour, sketch",
        )

    def test_string_passed_through(self) -> None:
        # Cas 2 : string "humour" → "humour" (rétro-compat ancien schéma).
        self.assertEqual(niches_str({"niche": "humour"}), "humour")

    def test_empty_entry_falls_back_to_humour(self) -> None:
        # Cas 3 : entrée vide → fallback "humour".
        self.assertEqual(niches_str({}), "humour")

    def test_list_with_empty_items_filtered(self) -> None:
        # Cas 4 : liste avec items vides → filtrés avant jointure.
        self.assertEqual(
            niches_str({"niches": ["humour", "", "sketch"]}),
            "humour, sketch",
        )


# ---------------------------------------------------------------------------
# generate_generator_dataset
# ---------------------------------------------------------------------------


class GeneratorDatasetTest(unittest.TestCase):
    """5 cas du brief — vérifie l'input prompt et le fallback t_type."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.out_path = self.tmpdir / "dataset_generator.jsonl"
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def test_t_type_profile_is_used_when_present(self) -> None:
        # Cas 1 : t_type_profile présent → priorité absolue.
        entry = _valid_generator_entry(t_type_profile="T3b", t_type="T2")
        n = generate_generator_dataset([entry], self.out_path)
        self.assertEqual(n, 1)
        rows = _read_jsonl(self.out_path)
        self.assertEqual(
            rows[0]["instruction"], build_generator_instruction("short")
        )
        self.assertIn("Longueur cible: court", rows[0]["input"])
        self.assertEqual(rows[0]["output"], "le passage 0:08")
        self.assertIn("T-type commentateur: T3b", rows[0]["input"])
        # ``t_type`` ne doit PAS écraser ``t_type_profile``.
        self.assertNotIn("T-type commentateur: T2", rows[0]["input"])

    def test_t_type_profile_missing_falls_back_to_t_type(self) -> None:
        # Cas 2 : pas de t_type_profile → fallback sur t_type.
        entry = _valid_generator_entry(t_type="T4")
        del entry["t_type_profile"]
        generate_generator_dataset([entry], self.out_path)
        rows = _read_jsonl(self.out_path)
        self.assertIn("T-type commentateur: T4", rows[0]["input"])

    def test_hashtags_list_joined_by_comma(self) -> None:
        # Cas 3 : hashtags liste → jointure par virgule (pas de '#').
        entry = _valid_generator_entry(hashtags=["F1", "monaco"])
        generate_generator_dataset([entry], self.out_path)
        rows = _read_jsonl(self.out_path)
        self.assertIn("Hashtags: F1, monaco", rows[0]["input"])

    def test_hashtags_string_passed_through(self) -> None:
        # Cas 4 : hashtags string → tel quel (pas de retraitement).
        entry = _valid_generator_entry(hashtags="#F1 #monaco")
        generate_generator_dataset([entry], self.out_path)
        rows = _read_jsonl(self.out_path)
        self.assertIn("Hashtags: #F1 #monaco", rows[0]["input"])

    def test_empty_text_is_skipped(self) -> None:
        # Cas 5 : commentaire vide → entrée ignorée.
        entry = _valid_generator_entry(text="   ")
        n = generate_generator_dataset([entry], self.out_path)
        self.assertEqual(n, 0)
        self.assertEqual(_read_jsonl(self.out_path), [])

    def test_video_context_included_in_input_block(self) -> None:
        entry = _valid_generator_entry(
            video_context="Deux potes en cuisine, ton ironique, drop vendredi."
        )
        generate_generator_dataset([entry], self.out_path)
        rows = _read_jsonl(self.out_path)
        self.assertIn(
            "Contexte vidéo: Deux potes en cuisine, ton ironique, drop vendredi.",
            rows[0]["input"],
        )


# ---------------------------------------------------------------------------
# main() — CLI / exit codes
# ---------------------------------------------------------------------------


class MainTest(unittest.TestCase):
    """3 cas du brief — orchestration bout-en-bout via la CLI."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.training_path = self.tmpdir / "training_comments_viral.json"
        self.output_dir = self.tmpdir / "out"
        self.generator_out = self.output_dir / GENERATOR_FILENAME
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def _write_training(self, entries: list[dict]) -> None:
        self.training_path.write_text(
            json.dumps({"entries": entries}, ensure_ascii=False),
            encoding="utf-8",
        )

    def _argv(self) -> list[str]:
        return [
            "--training-path", str(self.training_path),
            "--output-dir", str(self.output_dir),
        ]

    def test_empty_training_yields_zero_lines_exit_zero(self) -> None:
        self._write_training([])
        rc = main(self._argv())
        self.assertEqual(rc, 0)
        self.assertTrue(self.generator_out.exists())
        self.assertEqual(_read_jsonl(self.generator_out), [])

    def test_two_valid_entries_produce_two_generator_lines(self) -> None:
        # Schéma plat unifié : chaque entrée a tous les champs des deux datasets.
        entries = [
            {
                "text": "mdr trop vrai",
                "t_type": "T2",
                "t_type_profile": "T2",
                "niches": ["humour"],
                "views": 1000,
                "comment_to_like_ratio": 0.1,
                "caption": "cap 1",
                "hashtags": ["a"],
                "audio_id": "AUD1",
            },
            {
                "text": "le passage 0:08",
                "t_type": "T3b",
                "t_type_profile": "T3b",
                "niches": ["humour", "sketch"],
                "views": 2000,
                "comment_to_like_ratio": 0.2,
                "caption": "cap 2",
                "hashtags": ["b", "c"],
                "audio_id": "AUD2",
            },
        ]
        self._write_training(entries)
        rc = main(self._argv())
        self.assertEqual(rc, 0)

        gen_rows = _read_jsonl(self.generator_out)
        self.assertEqual(len(gen_rows), 2)
        self.assertEqual(
            [r["output"] for r in gen_rows],
            ["mdr trop vrai", "le passage 0:08"],
        )

    def test_missing_training_file_returns_exit_one_no_files_created(self) -> None:
        # On ne crée PAS self.training_path — il doit être absent.
        self.assertFalse(self.training_path.exists())
        rc = main(self._argv())
        self.assertEqual(rc, 1)
        self.assertFalse(self.generator_out.exists())


if __name__ == "__main__":
    unittest.main()
