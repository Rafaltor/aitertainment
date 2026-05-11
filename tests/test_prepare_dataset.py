"""Tests pour ``scripts.prepare_dataset`` (refonte JSONL 2026-05).

Couvre les 4 surfaces publiques exposées par le module :

* ``niches_str(entry)``           — résolution de la string niches.
* ``generate_classifier_dataset`` — sortie JSONL du classifier.
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

from scripts.prepare_dataset import (
    CLASSIFIER_FILENAME,
    CLASSIFIER_INSTRUCTION,
    GENERATOR_FILENAME,
    GENERATOR_INSTRUCTION,
    generate_classifier_dataset,
    generate_generator_dataset,
    load_vector_store,
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
# generate_classifier_dataset
# ---------------------------------------------------------------------------


class ClassifierDatasetTest(unittest.TestCase):
    """5 cas du brief — chaque cas écrit un fichier réel et le relit."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.out_path = self.tmpdir / "dataset_classifier.jsonl"
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def test_valid_entry_produces_complete_jsonl_line(self) -> None:
        entry = _valid_classifier_entry()
        n = generate_classifier_dataset([entry], self.out_path)
        self.assertEqual(n, 1)
        rows = _read_jsonl(self.out_path)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["instruction"], CLASSIFIER_INSTRUCTION)
        self.assertEqual(row["output"], "T2")
        # Tous les champs sont présents dans l'input.
        self.assertIn("Commentaire: mdr trop vrai", row["input"])
        self.assertIn("Niches du contenu: humour, sketch", row["input"])
        self.assertIn("Vues: 500000", row["input"])
        # Ratio formaté en 4 décimales (cf. brief).
        self.assertIn("Ratio comments/likes: 0.2834", row["input"])

    def test_empty_text_is_skipped(self) -> None:
        entry = _valid_classifier_entry(text="   ")
        n = generate_classifier_dataset([entry], self.out_path)
        self.assertEqual(n, 0)
        # Le fichier est créé (vide) — pour que les pipelines downstream
        # voient un artefact stable.
        self.assertEqual(_read_jsonl(self.out_path), [])

    def test_invalid_ttype_t9_is_skipped(self) -> None:
        entry = _valid_classifier_entry(t_type="T9")
        n = generate_classifier_dataset([entry], self.out_path)
        self.assertEqual(n, 0)
        self.assertEqual(_read_jsonl(self.out_path), [])

    def test_missing_views_and_ratio_default_to_zero(self) -> None:
        entry = _valid_classifier_entry()
        del entry["views"]
        del entry["comment_to_like_ratio"]
        n = generate_classifier_dataset([entry], self.out_path)
        self.assertEqual(n, 1)
        rows = _read_jsonl(self.out_path)
        self.assertIn("Vues: 0", rows[0]["input"])
        # 0.0 formaté en 4 décimales.
        self.assertIn("Ratio comments/likes: 0.0000", rows[0]["input"])

    def test_three_entries_one_invalid_writes_two_lines(self) -> None:
        entries = [
            _valid_classifier_entry(text="ligne 1"),
            _valid_classifier_entry(text="", t_type="T2"),  # invalide : text vide
            _valid_classifier_entry(text="ligne 3"),
        ]
        n = generate_classifier_dataset(entries, self.out_path)
        self.assertEqual(n, 2)
        rows = _read_jsonl(self.out_path)
        outputs = [r["output"] for r in rows]
        self.assertEqual(outputs, ["T2", "T2"])
        captures = [r["input"] for r in rows]
        self.assertIn("ligne 1", captures[0])
        self.assertIn("ligne 3", captures[1])


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
        n, with_vector = generate_generator_dataset([entry], self.out_path)
        self.assertEqual(n, 1)
        self.assertEqual(with_vector, 0)
        self.assertFalse(_read_jsonl(self.out_path)[0]["has_vector"])
        rows = _read_jsonl(self.out_path)
        self.assertEqual(rows[0]["instruction"], GENERATOR_INSTRUCTION)
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
        n, with_vector = generate_generator_dataset([entry], self.out_path)
        self.assertEqual(n, 0)
        self.assertEqual(with_vector, 0)
        self.assertEqual(_read_jsonl(self.out_path), [])


# ---------------------------------------------------------------------------
# load_vector_store
# ---------------------------------------------------------------------------


class LoadVectorStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def test_present_file_indexes_username_to_entry(self) -> None:
        store_path = self.tmpdir / "vector_store.json"
        store_path.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "username": "creator1",
                            "named_axes": {"scripted_vs_raw": 0.5},
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        loaded = load_vector_store(store_path)
        self.assertEqual(set(loaded), {"creator1"})
        self.assertEqual(loaded["creator1"]["username"], "creator1")

    def test_missing_file_returns_empty_dict_without_error(self) -> None:
        missing = self.tmpdir / "absent.json"
        with self.assertLogs("scripts.prepare_dataset", level="INFO") as logs:
            loaded = load_vector_store(missing)
        self.assertEqual(loaded, {})
        self.assertTrue(
            any(
                "vector_store.json absent — named_axes non inclus dans le dataset"
                in msg
                for msg in logs.output
            )
        )


# ---------------------------------------------------------------------------
# generate_generator_dataset — named_axes / vector_store
# ---------------------------------------------------------------------------


def _sample_named_axes() -> dict[str, float]:
    return {
        "scripted_vs_raw": 0.11,
        "solo_vs_collab": 0.22,
        "fictional_vs_real": 0.33,
        "energy_level": 0.44,
        "production_quality": 0.55,
        "format_length": 0.66,
        "distance_parasociale": 0.77,
        "interaction_style": 0.88,
        "mainstream_vs_niche": 0.99,
        "safe_vs_edgy": 0.12,
    }


class GeneratorWithVectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.out_path = self.tmpdir / "dataset_generator.jsonl"
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def test_account_in_vector_store_includes_creator_profile(self) -> None:
        entry = _valid_generator_entry(username="creator1")
        vector_store = {
            "creator1": {
                "username": "creator1",
                "named_axes": _sample_named_axes(),
            }
        }
        n, with_vector = generate_generator_dataset(
            [entry], self.out_path, vector_store=vector_store
        )
        self.assertEqual(n, 1)
        self.assertEqual(with_vector, 1)
        row = _read_jsonl(self.out_path)[0]
        self.assertTrue(row["has_vector"])
        self.assertIn("Profil créateur:", row["input"])
        self.assertIn("scripted_vs_raw=0.11", row["input"])

    def test_account_missing_from_vector_store_omits_creator_profile(self) -> None:
        entry = _valid_generator_entry(username="unknown_creator")
        n, with_vector = generate_generator_dataset(
            [entry], self.out_path, vector_store={}
        )
        self.assertEqual(n, 1)
        self.assertEqual(with_vector, 0)
        row = _read_jsonl(self.out_path)[0]
        self.assertFalse(row["has_vector"])
        self.assertNotIn("Profil créateur:", row["input"])

    def test_empty_named_axes_falls_back_without_vector(self) -> None:
        entry = _valid_generator_entry(username="creator1")
        vector_store = {"creator1": {"username": "creator1", "named_axes": {}}}
        n, with_vector = generate_generator_dataset(
            [entry], self.out_path, vector_store=vector_store
        )
        self.assertEqual(n, 1)
        self.assertEqual(with_vector, 0)
        row = _read_jsonl(self.out_path)[0]
        self.assertFalse(row["has_vector"])
        self.assertNotIn("Profil créateur:", row["input"])


# ---------------------------------------------------------------------------
# main() — CLI / exit codes
# ---------------------------------------------------------------------------


class MainTest(unittest.TestCase):
    """3 cas du brief — orchestration bout-en-bout via la CLI."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.training_path = self.tmpdir / "training_comments.json"
        self.output_dir = self.tmpdir / "out"
        self.classifier_out = self.output_dir / CLASSIFIER_FILENAME
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

    def test_empty_training_yields_zero_lines_in_both_files_exit_zero(self) -> None:
        self._write_training([])
        rc = main(self._argv())
        self.assertEqual(rc, 0)
        # Les deux fichiers sont créés (vides) — artefact stable downstream.
        self.assertTrue(self.classifier_out.exists())
        self.assertTrue(self.generator_out.exists())
        self.assertEqual(_read_jsonl(self.classifier_out), [])
        self.assertEqual(_read_jsonl(self.generator_out), [])

    def test_two_valid_entries_produce_two_lines_in_each_file(self) -> None:
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

        cls_rows = _read_jsonl(self.classifier_out)
        gen_rows = _read_jsonl(self.generator_out)
        self.assertEqual(len(cls_rows), 2)
        self.assertEqual(len(gen_rows), 2)
        # Sanity check : labels classifier corrects.
        self.assertEqual([r["output"] for r in cls_rows], ["T2", "T3b"])
        # Sanity check : outputs generator = textes des commentaires.
        self.assertEqual(
            [r["output"] for r in gen_rows],
            ["mdr trop vrai", "le passage 0:08"],
        )

    def test_missing_training_file_returns_exit_one_no_files_created(self) -> None:
        # On ne crée PAS self.training_path — il doit être absent.
        self.assertFalse(self.training_path.exists())
        rc = main(self._argv())
        self.assertEqual(rc, 1)
        # Aucun fichier de sortie ne doit avoir été créé : on bail-out
        # avant l'étape d'écriture.
        self.assertFalse(self.classifier_out.exists())
        self.assertFalse(self.generator_out.exists())


class MainWithVectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.training_path = self.tmpdir / "training_comments.json"
        self.output_dir = self.tmpdir / "out"
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def test_main_logs_generator_vector_count(self) -> None:
        self.training_path.write_text(
            json.dumps(
                {
                    "entries": [
                        _valid_generator_entry(username="creator1"),
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        vector_store = {
            "creator1": {
                "username": "creator1",
                "named_axes": _sample_named_axes(),
            }
        }
        argv = [
            "--training-path",
            str(self.training_path),
            "--output-dir",
            str(self.output_dir),
        ]
        with patch(
            "scripts.prepare_dataset.load_vector_store",
            return_value=vector_store,
        ):
            with self.assertLogs("scripts.prepare_dataset", level="INFO") as logs:
                rc = main(argv)
        self.assertEqual(rc, 0)
        self.assertTrue(
            any("Generator : 1 entrées dont 1 avec vecteur 32D" in msg for msg in logs.output)
        )


if __name__ == "__main__":
    unittest.main()
