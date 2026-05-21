"""Tests pour ``scripts.embedder``."""

from __future__ import annotations

import json
import math
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from scripts import embedder


class BuildInputTextTest(unittest.TestCase):
    def test_all_sections_present(self) -> None:
        text = embedder.build_input_text(
            "caption ici",
            ["humour", "sketch"],
            "transcript ici",
            ["comment un", "comment deux"],
            biography="Bio du créateur",
            niches=["humour", "sketch"],
        )
        self.assertIn("[NICHES] humour, sketch", text)
        self.assertIn("[BIOGRAPHY] Bio du créateur", text)
        self.assertIn("[CAPTION] caption ici", text)
        self.assertIn("[HASHTAGS] humour sketch", text)
        self.assertIn("[TRANSCRIPT] transcript ici", text)
        self.assertIn("[COMMENTS_RECEIVED] comment un\ncomment deux", text)

    def test_empty_biography_omits_section(self) -> None:
        text = embedder.build_input_text("x", [], "t", [], biography="")
        self.assertNotIn("[BIOGRAPHY]", text)
        self.assertIn("[NICHES] humour", text)

    def test_empty_caption_placeholder(self) -> None:
        text = embedder.build_input_text("", [], "", [])
        self.assertIn("[CAPTION] (vide)", text)
        self.assertIn("[NICHES] humour", text)

    def test_hashtags_list_joined_with_space(self) -> None:
        text = embedder.build_input_text("x", ["alpha", "beta"], "t", ["c"])
        self.assertIn("[HASHTAGS] alpha beta", text)

    def test_empty_comments_placeholder(self) -> None:
        text = embedder.build_input_text("x", ["tag"], "t", [])
        self.assertIn("[COMMENTS_RECEIVED] (vide)", text)


class EmbedTextTest(unittest.TestCase):
    @patch("scripts.embedder.requests.post")
    def test_lm_studio_embedding_returns_vector(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_post.return_value = mock_resp
        with patch.object(embedder, "LM_STUDIO_URL", "http://127.0.0.1:1234/v1"), patch.object(
            embedder, "LM_STUDIO_EMBED_MODEL", "bge-m3"
        ):
            result = embedder.embed_text("bonjour")
        self.assertEqual(result, [0.1, 0.2, 0.3])
        mock_post.assert_called_once_with(
            "http://127.0.0.1:1234/v1/embeddings",
            json={"model": "bge-m3", "input": "bonjour"},
            timeout=60,
        )

    @patch("scripts.embedder.requests.post")
    def test_lm_studio_nan_returns_none(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"data": [{"embedding": [float("nan"), 0.2]}]}
        mock_post.return_value = mock_resp
        with patch.object(embedder, "LM_STUDIO_URL", "http://127.0.0.1:1234/v1"), patch.object(
            embedder, "LM_STUDIO_EMBED_MODEL", "bge-m3"
        ), self.assertLogs("aitertainment.embedder", level="WARNING") as cm:
            result = embedder.embed_text("bonjour")
        self.assertIsNone(result)
        self.assertTrue(any("NaN" in message for message in cm.output))

    def test_valid_embedding_returns_float_list(self) -> None:
        fake_ollama = MagicMock()
        fake_ollama.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        with patch.object(embedder, "LM_STUDIO_URL", ""), patch.object(
            embedder, "LM_STUDIO_EMBED_MODEL", ""
        ), patch.dict(sys.modules, {"ollama": fake_ollama}):
            result = embedder.embed_text("bonjour")
        self.assertEqual(result, [0.1, 0.2, 0.3])

    def test_nan_embedding_returns_none_and_logs_warning(self) -> None:
        fake_ollama = MagicMock()
        fake_ollama.embed.return_value = {"embeddings": [[float("nan"), 0.2]]}
        with patch.object(embedder, "LM_STUDIO_URL", ""), patch.object(
            embedder, "LM_STUDIO_EMBED_MODEL", ""
        ), patch.dict(sys.modules, {"ollama": fake_ollama}), self.assertLogs(
            "aitertainment.embedder", level="WARNING"
        ) as cm:
            result = embedder.embed_text("bonjour")
        self.assertIsNone(result)
        self.assertTrue(any("NaN" in message for message in cm.output))

    def test_ollama_exception_returns_none(self) -> None:
        fake_ollama = MagicMock()
        fake_ollama.embed.side_effect = RuntimeError("ollama down")
        with patch.object(embedder, "LM_STUDIO_URL", ""), patch.object(
            embedder, "LM_STUDIO_EMBED_MODEL", ""
        ), patch.dict(sys.modules, {"ollama": fake_ollama}):
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


class LoadCreatorsFromDatabaseTest(unittest.TestCase):
    def test_skips_archived_and_builds_creator_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "database.json"
            db_path.write_text(
                json.dumps(
                    {
                        "profiles": {
                            "active_one": {
                                "archived": False,
                                "niches": ["humour"],
                                "t_type_original": "T2",
                                "t_type_final": "T3",
                                "followers": 1200,
                                "tier": "B",
                            },
                            "archived_one": {
                                "archived": True,
                                "niches": [],
                                "tier": "C",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            creators = embedder.load_creators_from_database(db_path)
        self.assertEqual(len(creators), 1)
        entry = creators[0]
        self.assertEqual(entry["username"], "active_one")
        self.assertEqual(entry["action"], "validated")
        self.assertEqual(entry["niches"], ["humour"])
        self.assertEqual(entry["t_type"], "T3")
        self.assertEqual(entry["followers"], 1200)
        self.assertEqual(entry["tier"], "B")

    def test_tier_filter_keeps_matching_profiles_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "database.json"
            db_path.write_text(
                json.dumps(
                    {
                        "profiles": {
                            "tier_a": {"archived": False, "tier": "A", "niches": []},
                            "tier_b": {"archived": False, "tier": "B", "niches": []},
                            "tier_c_archived": {"archived": True, "tier": "C", "niches": []},
                        }
                    }
                ),
                encoding="utf-8",
            )
            creators = embedder.load_creators_from_database(db_path, tier="B")
        self.assertEqual([c["username"] for c in creators], ["tier_b"])


class ExtractCaptionFromOgDescriptionTest(unittest.TestCase):
    def test_real_instagram_format_with_closing_quote(self) -> None:
        og = (
            '7,117 likes, 33 comments - raikkonenaf le  May 19, 2026: '
            '"🎵@juldetp - POW POW"'
        )
        self.assertEqual(
            embedder._extract_caption_from_og_description(og),
            "🎵@juldetp - POW POW",
        )

    def test_og_without_closing_quote(self) -> None:
        og = '42 likes - user le 1 jan 2026: "caption sans guillemet fermant'
        self.assertEqual(
            embedder._extract_caption_from_og_description(og),
            "caption sans guillemet fermant",
        )

    def test_empty_og_returns_empty(self) -> None:
        self.assertEqual(embedder._extract_caption_from_og_description(""), "")

    def test_get_caption_strips_trailing_quotes_and_dots(self) -> None:
        mock_page = MagicMock()
        mock_page.wait_for_load_state.return_value = None
        mock_page.wait_for_timeout.return_value = None
        mock_page.get_attribute.return_value = (
            '10 likes - user le May 19, 2026: "caption avec résidu".'
        )
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page

        caption = embedder._get_caption_from_reel_page("ABC123", mock_context)
        self.assertEqual(caption, "caption avec résidu")


class GetCaptionFromReelPageTest(unittest.TestCase):
    def test_prefers_og_description_over_graphql(self) -> None:
        media_id = "DYcbkPMM1cR"
        mock_page = MagicMock()
        mock_page.wait_for_load_state.return_value = None
        mock_page.wait_for_timeout.return_value = None
        mock_page.get_attribute.return_value = (
            '42 likes, 3 comments - creator le 1 janvier 2026: "Caption depuis og meta"'
        )
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page

        caption = embedder._get_caption_from_reel_page(media_id, mock_context)
        self.assertEqual(caption, "Caption depuis og meta")
        mock_page.on.assert_called_once()

    def test_extracts_caption_matching_media_code(self) -> None:
        media_id = "DYcbkPMM1cR"
        payload = {
            "data": {
                "xdt_shortcode_media": {
                    "code": media_id,
                    "caption": {"text": "Premier reel avec assez de mots"},
                }
            }
        }

        mock_response = MagicMock()
        mock_response.url = "https://www.instagram.com/graphql/query"
        mock_response.text.return_value = json.dumps(payload)

        mock_page = MagicMock()
        mock_page.wait_for_load_state.return_value = None
        mock_page.wait_for_timeout.return_value = None
        mock_page.get_attribute.return_value = ""

        def register_handler(event: str, handler: Any) -> None:
            if event == "response":
                handler(mock_response)

        mock_page.on.side_effect = register_handler
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page

        caption = embedder._get_caption_from_reel_page(media_id, mock_context)
        self.assertEqual(caption, "Premier reel avec assez de mots")
        mock_page.goto.assert_called_once_with(
            f"https://www.instagram.com/reel/{media_id}/"
        )
        mock_page.close.assert_called_once()

    def test_get_attribute_timeout_falls_back_to_graphql(self) -> None:
        media_id = "DYcbkPMM1cR"
        payload = {
            "data": {
                "xdt_shortcode_media": {
                    "code": media_id,
                    "caption": {"text": "Caption depuis graphql fallback"},
                }
            }
        }

        mock_response = MagicMock()
        mock_response.url = "https://www.instagram.com/graphql/query"
        mock_response.text.return_value = json.dumps(payload)

        mock_page = MagicMock()
        mock_page.wait_for_load_state.return_value = None
        mock_page.wait_for_timeout.return_value = None
        mock_page.get_attribute.side_effect = TimeoutError("og:description timeout")

        def register_handler(event: str, handler: Any) -> None:
            if event == "response":
                handler(mock_response)

        mock_page.on.side_effect = register_handler
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page

        caption = embedder._get_caption_from_reel_page(media_id, mock_context)
        self.assertEqual(caption, "Caption depuis graphql fallback")
        mock_page.get_attribute.assert_called_once_with(
            'meta[property="og:description"]',
            "content",
            timeout=5000,
        )

    def test_returns_empty_when_no_graphql_match(self) -> None:
        mock_page = MagicMock()
        mock_page.on.return_value = None
        mock_page.get_attribute.return_value = ""
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page

        caption = embedder._get_caption_from_reel_page("UNKNOWN", mock_context)
        self.assertEqual(caption, "")
        mock_page.close.assert_called_once()


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
    def test_dry_run_skips_ollama_whisper_and_playwright(self) -> None:
        creators = [{"username": "alpha", "action": "validated"}]
        with patch.object(embedder, "load_creators", return_value=creators), patch.object(
            embedder, "load_vector_store", return_value=[]
        ), patch.object(embedder, "load_pca", return_value=None), patch.object(
            embedder, "sync_playwright"
        ) as mock_pw, patch.object(embedder, "get_browser_context") as mock_ctx, patch.object(
            embedder, "process_account"
        ) as mock_process, patch.object(embedder, "embed_text") as mock_embed, patch.object(
            embedder, "transcribe_audio"
        ) as mock_transcribe:
            code = embedder.main(["--dry-run"])
        self.assertEqual(code, 0)
        mock_pw.assert_not_called()
        mock_ctx.assert_not_called()
        mock_process.assert_not_called()
        mock_embed.assert_not_called()
        mock_transcribe.assert_not_called()

    def test_unknown_account_exits_with_clear_message(self) -> None:
        with patch.object(
            embedder,
            "load_creators",
            return_value=[{"username": "known", "action": "validated"}],
        ), self.assertLogs("aitertainment.embedder", level="ERROR") as cm:
            code = embedder.main(["--account", "@missing"])
        self.assertEqual(code, 1)
        self.assertTrue(
            any("Compte @missing introuvable dans la watchlist." in message for message in cm.output)
        )

    def test_dry_run_database_source_logs_count(self) -> None:
        creators = [{"username": "alpha", "action": "validated"}]
        with patch.object(
            embedder, "load_creators", return_value=creators
        ) as mock_load, patch.object(embedder, "load_vector_store", return_value=[]), patch.object(
            embedder, "load_pca", return_value=None
        ), self.assertLogs("aitertainment.embedder", level="INFO") as cm:
            code = embedder.main(["--dry-run", "--source", "database"])
        self.assertEqual(code, 0)
        mock_load.assert_called_once_with("database", tier=None)
        self.assertTrue(
            any("Source : database.json — 1 profils tier tous" in m for m in cm.output)
        )

    def test_dry_run_database_tier_b_logs_tier_label(self) -> None:
        creators = [{"username": "alpha", "action": "validated", "tier": "B"}]
        with patch.object(
            embedder, "load_creators", return_value=creators
        ) as mock_load, patch.object(embedder, "load_vector_store", return_value=[]), patch.object(
            embedder, "load_pca", return_value=None
        ), self.assertLogs("aitertainment.embedder", level="INFO") as cm:
            code = embedder.main(["--dry-run", "--source", "database", "--tier", "B"])
        self.assertEqual(code, 0)
        mock_load.assert_called_once_with("database", tier="B")
        self.assertTrue(
            any("Source : database.json — 1 profils tier B" in m for m in cm.output)
        )

    def test_tier_without_database_source_exits(self) -> None:
        with self.assertLogs("aitertainment.embedder", level="ERROR") as cm:
            code = embedder.main(["--dry-run", "--tier", "A"])
        self.assertEqual(code, 1)
        self.assertTrue(
            any("--tier n'est utilisable qu'avec --source database" in m for m in cm.output)
        )


if __name__ == "__main__":
    unittest.main()
