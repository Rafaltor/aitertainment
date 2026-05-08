"""Tests pour ``scripts.prepare_dataset``."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_dataset import (
    CLASSIFIER_INSTRUCTION,
    GENERATOR_INSTRUCTION,
    MIN_PAIRS_READY,
    _atomic_write_json,
    _fmt_hashtags,
    _fmt_optional,
    _fmt_ratio,
    _mock_entries,
    _print_stats,
    _read_training,
    build_classifier_pair,
    build_datasets,
    build_generator_pair,
    run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _entry(**overrides) -> dict:
    """Construit une entrée valide complète (le builder produit ce schéma)."""
    base = {
        "media_id": "ABC123",
        "username": "raikkonenaf",
        "reel_url": "https://www.instagram.com/reel/ABC123/",
        "collected_at": "2026-05-08T12:00:00",
        "generator_input": {
            "t_type": "T2",
            "niche": "humour",
            "caption": "moment culte F1",
            "hashtags": ["F1", "monaco"],
            "audio_id": "AUD123",
        },
        "classifier_context": {
            "views": 500000,
            "likes": 12000,
            "comment_count": 3400,
            "shares": None,
            "comment_to_like_ratio": 0.283,
            "share_to_like_ratio": None,
        },
        "top_comments": [
            {"text": "mdr trop vrai", "likes": 42, "t_type": "T2", "niche": "humour"},
            {"text": "le passage 0:08", "likes": 30, "t_type": "T2", "niche": "humour"},
            {"text": "no comment", "likes": 18, "t_type": "T2", "niche": "humour"},
        ],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Helpers de formatage
# ---------------------------------------------------------------------------


class FormatHelpersTest(unittest.TestCase):
    def test_fmt_hashtags_list(self) -> None:
        self.assertEqual(_fmt_hashtags(["F1", "monaco"]), "#F1 #monaco")

    def test_fmt_hashtags_strips_existing_hash(self) -> None:
        self.assertEqual(_fmt_hashtags(["#F1", "  monaco "]), "#F1 #monaco")

    def test_fmt_hashtags_empty_or_none(self) -> None:
        self.assertEqual(_fmt_hashtags([]), "(aucun)")
        self.assertEqual(_fmt_hashtags(None), "(aucun)")
        self.assertEqual(_fmt_hashtags(["", "  "]), "(aucun)")

    def test_fmt_hashtags_string_passthrough(self) -> None:
        self.assertEqual(_fmt_hashtags("#a #b"), "#a #b")

    def test_fmt_optional(self) -> None:
        self.assertEqual(_fmt_optional("AUD42"), "AUD42")
        self.assertEqual(_fmt_optional(None), "(inconnu)")
        self.assertEqual(_fmt_optional("", default="(vide)"), "(vide)")
        self.assertEqual(_fmt_optional(0), "0")

    def test_fmt_ratio(self) -> None:
        self.assertEqual(_fmt_ratio(0.283), "0.283")
        self.assertEqual(_fmt_ratio(None), "(inconnu)")
        self.assertEqual(_fmt_ratio("not_a_number"), "(inconnu)")
        self.assertEqual(_fmt_ratio(0), "0.000")


# ---------------------------------------------------------------------------
# build_generator_pair
# ---------------------------------------------------------------------------


class BuildGeneratorPairTest(unittest.TestCase):
    def test_full_entry_produces_complete_input(self) -> None:
        entry = _entry()
        pair = build_generator_pair(entry, entry["top_comments"][0])
        self.assertIsNotNone(pair)
        self.assertEqual(pair["instruction"], GENERATOR_INSTRUCTION)
        self.assertEqual(pair["output"], "mdr trop vrai")
        # Toutes les features sont injectées dans l'input
        self.assertIn("T-type: T2", pair["input"])
        self.assertIn("Niche: humour", pair["input"])
        self.assertIn("Caption: moment culte F1", pair["input"])
        self.assertIn("Hashtags: #F1 #monaco", pair["input"])
        self.assertIn("Audio: AUD123", pair["input"])

    def test_missing_caption_audio_falls_back_to_placeholders(self) -> None:
        entry = _entry()
        entry["generator_input"]["caption"] = None
        entry["generator_input"]["audio_id"] = None
        entry["generator_input"]["hashtags"] = []
        pair = build_generator_pair(entry, entry["top_comments"][0])
        self.assertIsNotNone(pair)
        self.assertIn("Caption: (vide)", pair["input"])
        self.assertIn("Hashtags: (aucun)", pair["input"])
        self.assertIn("Audio: (aucun)", pair["input"])

    def test_empty_text_returns_none(self) -> None:
        entry = _entry()
        comment = {"text": "  ", "likes": 0, "t_type": "T2", "niche": "humour"}
        self.assertIsNone(build_generator_pair(entry, comment))

    def test_missing_ttype_returns_none(self) -> None:
        entry = _entry()
        entry["generator_input"]["t_type"] = ""
        comment = {"text": "ok", "likes": 1, "t_type": "", "niche": "humour"}
        self.assertIsNone(build_generator_pair(entry, comment))

    def test_comment_ttype_overrides_entry_ttype(self) -> None:
        """Le t_type du commentaire prime — chaque commentaire est self-contained."""
        entry = _entry()
        entry["generator_input"]["t_type"] = "T2"
        # Manually labelled differently (cas humain validateur)
        comment = {"text": "ok", "likes": 5, "t_type": "T3b", "niche": "humour"}
        pair = build_generator_pair(entry, comment)
        self.assertIsNotNone(pair)
        self.assertIn("T-type: T3b", pair["input"])


# ---------------------------------------------------------------------------
# build_classifier_pair
# ---------------------------------------------------------------------------


class BuildClassifierPairTest(unittest.TestCase):
    def test_full_entry_produces_complete_pair(self) -> None:
        entry = _entry()
        pair = build_classifier_pair(entry, entry["top_comments"][0])
        self.assertIsNotNone(pair)
        self.assertEqual(pair["instruction"], CLASSIFIER_INSTRUCTION)
        self.assertEqual(pair["output"], "T2")  # le label
        self.assertIn("Commentaire: mdr trop vrai", pair["input"])
        self.assertIn("Niche: humour", pair["input"])
        self.assertIn("Vues: 500000", pair["input"])
        self.assertIn("Ratio comments/likes: 0.283", pair["input"])

    def test_invalid_ttype_filtered(self) -> None:
        entry = _entry()
        comment = {"text": "ok", "likes": 5, "t_type": "TX", "niche": "humour"}
        self.assertIsNone(build_classifier_pair(entry, comment))

    def test_missing_metrics_use_unknown_placeholder(self) -> None:
        entry = _entry()
        entry["classifier_context"] = {
            "views": None,
            "likes": None,
            "comment_count": None,
            "shares": None,
            "comment_to_like_ratio": None,
            "share_to_like_ratio": None,
        }
        pair = build_classifier_pair(entry, entry["top_comments"][0])
        self.assertIsNotNone(pair)
        self.assertIn("Vues: (inconnu)", pair["input"])
        self.assertIn("Ratio comments/likes: (inconnu)", pair["input"])

    def test_empty_text_returns_none(self) -> None:
        self.assertIsNone(
            build_classifier_pair(_entry(), {"text": "", "t_type": "T2", "niche": "x"})
        )


# ---------------------------------------------------------------------------
# build_datasets : agrégation
# ---------------------------------------------------------------------------


class BuildDatasetsTest(unittest.TestCase):
    def test_three_comments_produce_three_pairs_in_each(self) -> None:
        entries = [_entry()]
        gen, cls = build_datasets(entries)
        self.assertEqual(len(gen), 3)
        self.assertEqual(len(cls), 3)
        # Toutes les paires generator partagent la même instruction.
        for p in gen:
            self.assertEqual(p["instruction"], GENERATOR_INSTRUCTION)
        for p in cls:
            self.assertEqual(p["instruction"], CLASSIFIER_INSTRUCTION)

    def test_empty_top_comments_skipped(self) -> None:
        entry = _entry()
        entry["top_comments"] = []
        gen, cls = build_datasets([entry])
        self.assertEqual(gen, [])
        self.assertEqual(cls, [])

    def test_invalid_ttype_kept_in_generator_dropped_in_classifier(self) -> None:
        """Asymétrie voulue : un T-type non standard reste exploitable pour le
        generator (on connaît au moins le label fourni à l'inférence) mais pas
        pour le classifier (label hors set fini)."""
        entry = _entry()
        entry["generator_input"]["t_type"] = "TX"
        for c in entry["top_comments"]:
            c["t_type"] = "TX"
        gen, cls = build_datasets([entry])
        self.assertEqual(len(gen), 3)
        self.assertEqual(cls, [])

    def test_non_dict_comments_filtered(self) -> None:
        entry = _entry()
        entry["top_comments"] = ["pas un dict", None, {"text": "ok", "t_type": "T2", "niche": "x"}]
        gen, cls = build_datasets([entry])
        self.assertEqual(len(gen), 1)
        self.assertEqual(len(cls), 1)


# ---------------------------------------------------------------------------
# IO + run() bout en bout
# ---------------------------------------------------------------------------


class RunIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.training_path = self.tmpdir / "training_comments.json"
        self.gen_path = self.tmpdir / "generator_dataset.json"
        self.cls_path = self.tmpdir / "classifier_dataset.json"

    def _write_training(self, entries: list[dict]) -> None:
        self.training_path.write_text(
            json.dumps({"entries": entries}, ensure_ascii=False),
            encoding="utf-8",
        )

    def test_run_writes_two_files(self) -> None:
        self._write_training([_entry()])
        gen, cls = run(
            training_path=self.training_path,
            generator_path=self.gen_path,
            classifier_path=self.cls_path,
        )
        self.assertEqual(len(gen), 3)
        self.assertEqual(len(cls), 3)
        self.assertTrue(self.gen_path.exists())
        self.assertTrue(self.cls_path.exists())
        # Le top-level est une LISTE (Alpaca convention pour datasets HF).
        gen_disk = json.loads(self.gen_path.read_text(encoding="utf-8"))
        cls_disk = json.loads(self.cls_path.read_text(encoding="utf-8"))
        self.assertIsInstance(gen_disk, list)
        self.assertIsInstance(cls_disk, list)
        self.assertEqual(gen_disk[0]["output"], "mdr trop vrai")
        self.assertEqual(cls_disk[0]["output"], "T2")

    def test_stats_only_does_not_write(self) -> None:
        self._write_training([_entry()])
        run(
            stats_only=True,
            training_path=self.training_path,
            generator_path=self.gen_path,
            classifier_path=self.cls_path,
        )
        self.assertFalse(self.gen_path.exists())
        self.assertFalse(self.cls_path.exists())

    def test_run_mock_does_not_read_disk(self) -> None:
        """``--mock`` doit produire 30 paires (10 entrées × 3 commentaires).

        Une seule entrée mock est de t_type ``T1`` (i=5 dans la rotation) ;
        ``T1`` est dans le set valide → le classifier garde tout aussi.
        """
        # Pas de fichier d'entrée — _read_training ne doit pas être appelé.
        gen, cls = run(
            mock=True,
            stats_only=True,
            training_path=self.tmpdir / "does_not_exist.json",
        )
        self.assertEqual(len(gen), 30)
        self.assertEqual(len(cls), 30)

    def test_run_missing_input_file_returns_empty(self) -> None:
        # Pas de fichier → entries=[] silencieusement (best-effort).
        gen, cls = run(
            training_path=self.tmpdir / "absent.json",
            generator_path=self.gen_path,
            classifier_path=self.cls_path,
        )
        self.assertEqual(gen, [])
        self.assertEqual(cls, [])
        # Les deux fichiers de sortie sont quand même créés (vides).
        self.assertEqual(json.loads(self.gen_path.read_text(encoding="utf-8")), [])
        self.assertEqual(json.loads(self.cls_path.read_text(encoding="utf-8")), [])

    def test_atomic_write_creates_parent_dir(self) -> None:
        nested = self.tmpdir / "a" / "b" / "c.json"
        _atomic_write_json(nested, [{"k": "v"}])
        self.assertTrue(nested.exists())
        self.assertEqual(json.loads(nested.read_text(encoding="utf-8")), [{"k": "v"}])

    def test_read_training_invalid_json_raises(self) -> None:
        self.training_path.write_text("not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            _read_training(self.training_path)

    def test_read_training_root_not_dict_raises(self) -> None:
        self.training_path.write_text("[]", encoding="utf-8")
        with self.assertRaises(ValueError):
            _read_training(self.training_path)


# ---------------------------------------------------------------------------
# Stats / seuil de readiness
# ---------------------------------------------------------------------------


class PrintStatsTest(unittest.TestCase):
    def test_below_threshold_says_not_ready(self) -> None:
        buf = io.StringIO()
        _print_stats(total_entries=5, generator_count=10, classifier_count=10, out=buf)
        msg = buf.getvalue()
        self.assertIn("Pas encore prêt", msg)
        self.assertIn("Total entrées training_comments.json : 5", msg)
        self.assertIn("Generator dataset : 10 paires", msg)

    def test_above_threshold_says_ready(self) -> None:
        buf = io.StringIO()
        _print_stats(
            total_entries=300,
            generator_count=MIN_PAIRS_READY + 1,
            classifier_count=MIN_PAIRS_READY + 1,
            out=buf,
        )
        msg = buf.getvalue()
        self.assertIn("Prêt pour fine-tuning", msg)


# ---------------------------------------------------------------------------
# Mock entries : structure cohérente avec le builder
# ---------------------------------------------------------------------------


class MockEntriesTest(unittest.TestCase):
    def test_count_and_structure(self) -> None:
        entries = _mock_entries(10)
        self.assertEqual(len(entries), 10)
        for e in entries:
            self.assertIn("generator_input", e)
            self.assertIn("classifier_context", e)
            self.assertIn("top_comments", e)
            self.assertEqual(len(e["top_comments"]), 3)
            for c in e["top_comments"]:
                self.assertIn("text", c)
                self.assertIn("t_type", c)
                self.assertIn("niche", c)


if __name__ == "__main__":
    unittest.main()
