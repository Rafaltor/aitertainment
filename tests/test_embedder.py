"""Tests pour ``scripts.embedder``."""

from __future__ import annotations

import json
import math
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts import embedder


class BuildInputTextTest(unittest.TestCase):
    def test_all_sections_present(self) -> None:
        text = embedder.build_input_text(
            "caption ici",
            ["humour", "sketch"],
            "transcript ici",
            ["comment un", "comment deux"],
        )
        self.assertIn("[CAPTION] caption ici", text)
        self.assertIn("[HASHTAGS] humour sketch", text)
        self.assertIn("[TRANSCRIPT] transcript ici", text)
        self.assertIn("[COMMENTS_RECEIVED] comment un\ncomment deux", text)

    def test_empty_caption_placeholder(self) -> None:
        text = embedder.build_input_text("", [], "", [])
        self.assertIn("[CAPTION] (vide)", text)

    def test_hashtags_list_joined_with_space(self) -> None:
        text = embedder.build_input_text("x", ["alpha", "beta"], "t", ["c"])
        self.assertIn("[HASHTAGS] alpha beta", text)

    def test_empty_comments_placeholder(self) -> None:
        text = embedder.build_input_text("x", ["tag"], "t", [])
        self.assertIn("[COMMENTS_RECEIVED] (vide)", text)


class EmbedTextTest(unittest.TestCase):
    def test_valid_embedding_returns_float_list(self) -> None:
        fake_ollama = MagicMock()
        fake_ollama.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            result = embedder.embed_text("bonjour")
        self.assertEqual(result, [0.1, 0.2, 0.3])

    def test_nan_embedding_returns_none_and_logs_warning(self) -> None:
        fake_ollama = MagicMock()
        fake_ollama.embed.return_value = {"embeddings": [[float("nan"), 0.2]]}
        with patch.dict(sys.modules, {"ollama": fake_ollama}), self.assertLogs(
            "aitertainment.embedder", level="WARNING"
        ) as cm:
            result = embedder.embed_text("bonjour")
        self.assertIsNone(result)
        self.assertTrue(any("NaN" in message for message in cm.output))

    def test_ollama_exception_returns_none(self) -> None:
        fake_ollama = MagicMock()
        fake_ollama.embed.side_effect = RuntimeError("ollama down")
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            result = embedder.embed_text("bonjour")
        self.assertIsNone(result)


class ProjectToNamedAxesTest(unittest.TestCase):
    def test_pca_none_returns_zeros_with_warning(self) -> None:
        with self.assertLogs("aitertainment.embedder", level="WARNING") as cm:
            axes = embedder.project_to_named_axes([0.1] * 8, None)
        self.assertEqual(set(axes), set(embedder.NAMED_AXES))
        self.assertTrue(all(value == 0.0 for value in axes.values()))
        self.assertTrue(any("PCA non disponible" in message for message in cm.output))

    def test_valid_pca_returns_named_axes_between_zero_and_one(self) -> None:
        from sklearn.decomposition import PCA

        vectors = [[float(i + j) for j in range(12)] for i in range(10)]
        model = PCA(n_components=10)
        model.fit(vectors)
        axes = embedder.project_to_named_axes(vectors[0], model)
        self.assertEqual(set(axes), set(embedder.NAMED_AXES))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in axes.values()))

    def test_min_max_normalization_spans_zero_to_one(self) -> None:
        from sklearn.decomposition import PCA

        vectors = [[float(i + j) for j in range(12)] for i in range(10)]
        model = PCA(n_components=10)
        model.fit(vectors)
        axes = embedder.project_to_named_axes(vectors[3], model)
        values = list(axes.values())
        self.assertAlmostEqual(min(values), 0.0, places=6)
        self.assertAlmostEqual(max(values), 1.0, places=6)


class FitPcaTest(unittest.TestCase):
    def test_less_than_ten_entries_returns_none(self) -> None:
        store = [
            {"username": f"u{i}", "embedding_raw": [float(i)] * 12}
            for i in range(9)
        ]
        self.assertIsNone(embedder.fit_pca(store))

    def test_ten_or_more_entries_fits_and_writes_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pca_path = Path(tmp) / "pca_model.pkl"
            store = [
                {"username": f"u{i}", "embedding_raw": [float(i + j) for j in range(12)]}
                for i in range(10)
            ]
            with patch.object(embedder, "PCA_MODEL_PATH", Path("pca_model.pkl")), patch.object(
                embedder, "_resolve_path", return_value=pca_path
            ):
                model = embedder.fit_pca(store)
            self.assertIsNotNone(model)
            self.assertTrue(pca_path.exists())
            with pca_path.open("rb") as fh:
                loaded = pickle.load(fh)
            self.assertEqual(loaded.n_components, 10)


class SaveVectorStoreTest(unittest.TestCase):
    def test_atomic_write_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vector_store.json"
            entries = [{"username": "alpha", "embedding_raw": [0.1, 0.2]}]
            embedder.save_vector_store(entries, path)
            self.assertEqual(embedder.load_vector_store(path), entries)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload, entries)


class MainTest(unittest.TestCase):
    def test_dry_run_skips_ollama_whisper_and_instagram(self) -> None:
        creators = [{"username": "alpha", "action": "validated"}]
        with patch.object(embedder, "load_watchlist", return_value=creators), patch.object(
            embedder, "load_vector_store", return_value=[]
        ), patch.object(embedder, "load_pca", return_value=None), patch(
            "instagram_client.get_client"
        ) as mock_client, patch.object(embedder, "embed_text") as mock_embed, patch.object(
            embedder, "transcribe_audio"
        ) as mock_transcribe:
            code = embedder.main(["--dry-run"])
        self.assertEqual(code, 0)
        mock_client.assert_not_called()
        mock_embed.assert_not_called()
        mock_transcribe.assert_not_called()

    def test_unknown_account_exits_with_clear_message(self) -> None:
        with patch.object(
            embedder,
            "load_watchlist",
            return_value=[{"username": "known", "action": "validated"}],
        ), self.assertLogs("aitertainment.embedder", level="ERROR") as cm:
            code = embedder.main(["--account", "@missing"])
        self.assertEqual(code, 1)
        self.assertTrue(
            any("Compte @missing introuvable dans la watchlist." in message for message in cm.output)
        )


if __name__ == "__main__":
    unittest.main()
