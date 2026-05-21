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
from contextlib import contextmanager
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

    def test_comments_include_like_counts(self) -> None:
        text = embedder.build_input_text(
            "x",
            [],
            "t",
            [{"text": "super reel", "comment_likes": 42}],
        )
        self.assertIn("[COMMENTS_RECEIVED] [42 likes] super reel", text)


class EnsureCommentsForAccountTest(unittest.TestCase):
    def test_collects_and_reloads_all_comments(self) -> None:
        reels = [{"media_id": "ABC", "comment_count": 10}]
        after = (
            [{"text": "hi", "comment_likes": 1, "media_id": "ABC"}],
            {"ABC": [{"text": "hi", "comment_likes": 1, "media_id": "ABC"}]},
        )
        with patch.object(
            embedder,
            "comments_for_account_from_raw",
            side_effect=[([], {}), after],
        ), patch.object(embedder, "collect_top_comments", return_value=2) as mock_collect:
            comments, _, source = embedder.ensure_comments_for_account(
                "user", reels, MagicMock(), ["humour"]
            )
        mock_collect.assert_called_once()
        self.assertEqual(len(comments), 1)
        self.assertEqual(source, "raw_comments.json+playwright")

    def test_uses_raw_only_when_collect_adds_nothing(self) -> None:
        existing = (
            [{"text": "x", "comment_likes": 2, "media_id": "Z"}],
            {"Z": [{"text": "x", "comment_likes": 2, "media_id": "Z"}]},
        )
        with patch.object(
            embedder, "comments_for_account_from_raw", return_value=existing
        ), patch.object(embedder, "collect_top_comments", return_value=0):
            comments, _, source = embedder.ensure_comments_for_account(
                "user", [], MagicMock(), "humour"
            )
        self.assertEqual(source, "raw_comments.json")
        self.assertEqual(comments, existing[0])


class CommentsFromRawTest(unittest.TestCase):
    def test_returns_all_stored_comments_for_account(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw_comments.json"
            entries = [
                {
                    "username": "creator",
                    "media_id": "A",
                    "text": "c1",
                    "comment_likes": 1,
                },
                {
                    "username": "creator",
                    "media_id": "B",
                    "text": "c2",
                    "comment_likes": 9,
                },
                {
                    "username": "other",
                    "media_id": "Z",
                    "text": "nope",
                    "comment_likes": 99,
                },
            ]
            path.write_text(json.dumps(entries), encoding="utf-8")
            rows, by_media = embedder.comments_for_account_from_raw("creator", path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["text"], "c2")
        self.assertEqual(rows[0]["comment_likes"], 9)
        self.assertEqual(len(by_media["A"]), 1)
        self.assertEqual(len(by_media["B"]), 1)

    def test_keeps_same_text_on_different_reels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw_comments.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "username": "u",
                            "media_id": "X",
                            "text": "dup",
                            "comment_likes": 5,
                        },
                        {
                            "username": "u",
                            "media_id": "Y",
                            "text": "dup",
                            "comment_likes": 3,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            rows, by_media = embedder.comments_for_account_from_raw("u", path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(by_media["X"]), 1)
        self.assertEqual(len(by_media["Y"]), 1)


class EmbedTextTest(unittest.TestCase):
    def test_embedding_config_ok(self) -> None:
        with patch.object(embedder, "LM_STUDIO_URL", "http://127.0.0.1:1234/v1"), patch.object(
            embedder, "LM_STUDIO_EMBED_MODEL", "my-embed-model"
        ):
            self.assertTrue(embedder.embedding_config_ok())

    def test_missing_config_returns_none(self) -> None:
        with patch.object(embedder, "LM_STUDIO_URL", ""), patch.object(
            embedder, "LM_STUDIO_EMBED_MODEL", ""
        ), self.assertLogs("aitertainment.embedder", level="ERROR") as cm:
            result = embedder.embed_text("bonjour")
        self.assertIsNone(result)
        self.assertTrue(any("LM_STUDIO_EMBED_MODEL" in message for message in cm.output))

    @patch("scripts.embedder.requests.post")
    def test_lm_studio_embedding_returns_vector(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_post.return_value = mock_resp
        with patch.object(embedder, "LM_STUDIO_URL", "http://127.0.0.1:1234/v1"), patch.object(
            embedder, "LM_STUDIO_EMBED_MODEL", "my-embed-model"
        ):
            result = embedder.embed_text("bonjour")
        self.assertEqual(result, [0.1, 0.2, 0.3])
        mock_post.assert_called_once_with(
            "http://127.0.0.1:1234/v1/embeddings",
            json={"model": "my-embed-model", "input": "bonjour"},
            timeout=120,
        )

    @patch("scripts.embedder.requests.post")
    def test_lm_studio_nan_returns_none(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"data": [{"embedding": [float("nan"), 0.2]}]}
        mock_post.return_value = mock_resp
        with patch.object(embedder, "LM_STUDIO_URL", "http://127.0.0.1:1234/v1"), patch.object(
            embedder, "LM_STUDIO_EMBED_MODEL", "my-embed-model"
        ), self.assertLogs("aitertainment.embedder", level="WARNING") as cm:
            result = embedder.embed_text("bonjour")
        self.assertIsNone(result)
        self.assertTrue(any("NaN" in message for message in cm.output))


class ProjectToNamedAxesTest(unittest.TestCase):
    def test_pca_none_returns_zeros_with_warning(self) -> None:
        with self.assertLogs("aitertainment.embedder", level="WARNING") as cm:
            axes = embedder.project_to_named_axes([0.1] * 8, None)
        self.assertEqual(set(axes), set(embedder.NAMED_AXES))
        self.assertTrue(all(value == 0.0 for value in axes.values()))
        self.assertTrue(any("PCA non disponible" in message for message in cm.output))

    def test_pca_dim_mismatch_returns_zeros_without_crash(self) -> None:
        from sklearn.decomposition import PCA

        vectors = [[float(i + j) for j in range(12)] for i in range(10)]
        model = PCA(n_components=10)
        model.fit(vectors)
        with self.assertLogs("aitertainment.embedder", level="WARNING") as cm:
            axes = embedder.project_to_named_axes([0.1] * 1024, model)
        self.assertTrue(all(value == 0.0 for value in axes.values()))
        self.assertTrue(any("PCA ignorée" in message for message in cm.output))

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

    def test_fit_pca_ignores_mixed_dimensions(self) -> None:
        store = [
            {
                "username": f"u{i}",
                "embedding_raw": [float(i + j) for j in range(12)],
            }
            for i in range(10)
        ]
        store.append({"username": "small", "embedding_raw": [1.0] * 8})
        with self.assertLogs("aitertainment.embedder", level="WARNING"):
            model = embedder.fit_pca(store, target_dim=12)
        self.assertIsNotNone(model)
        self.assertEqual(embedder._pca_input_dim(model), 12)

    def test_fit_pca_target_dim_requires_ten_at_that_dim(self) -> None:
        store = [
            {"username": f"u{i}", "embedding_raw": [float(i)] * 1024}
            for i in range(3)
        ]
        with self.assertLogs("aitertainment.embedder", level="WARNING") as cm:
            self.assertIsNone(embedder.fit_pca(store, target_dim=1024))
        self.assertTrue(any("10 minimum" in message for message in cm.output))

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

    def test_og_without_quotes_after_colon(self) -> None:
        og = "120 likes, 2 comments - user le May 19, 2026: POW POW sans guillemets"
        self.assertEqual(
            embedder._extract_caption_from_og_description(og),
            "POW POW sans guillemets",
        )

    def test_graphql_text_window_extracts_caption(self) -> None:
        mid = "DYcbkPMM1cR"
        blob = (
            '{"code":"DYcbkPMM1cR","caption":{"text":"Audio only track name"},'
            '"caption_text":"ignored"}'
        )
        self.assertEqual(
            embedder._caption_from_graphql_text_window(blob, mid),
            "Audio only track name",
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
        with _mock_reel_page_no_comments():
            mock_context = MagicMock()
            mock_context.new_page.return_value = mock_page
            caption = embedder._get_caption_from_reel_page("ABC123", mock_context)
        self.assertEqual(caption, "caption avec résidu")


def _mock_reel_page_no_comments():
    """Pas de panneau commentaires / DOM caption pendant les tests caption."""
    return patch.multiple(
        embedder,
        navigate_to_reel_page=MagicMock(return_value=True),
        click_reel_comment_button=MagicMock(return_value=None),
        extract_reel_caption_from_dom=MagicMock(return_value=""),
    )


class GetCaptionFromReelPageTest(unittest.TestCase):
    def test_prefers_og_description_over_graphql(self) -> None:
        media_id = "DYcbkPMM1cR"
        mock_page = MagicMock()
        mock_page.wait_for_load_state.return_value = None
        mock_page.wait_for_timeout.return_value = None
        mock_page.get_attribute.return_value = (
            '42 likes, 3 comments - creator le 1 janvier 2026: "Caption depuis og meta"'
        )
        with _mock_reel_page_no_comments():
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
        with _mock_reel_page_no_comments():
            mock_context = MagicMock()
            mock_context.new_page.return_value = mock_page
            caption = embedder._get_caption_from_reel_page(media_id, mock_context)
        self.assertEqual(caption, "Premier reel avec assez de mots")
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
        with _mock_reel_page_no_comments():
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
        with _mock_reel_page_no_comments(), patch.object(
            embedder, "navigate_to_reel_page", return_value=True
        ):
            mock_context = MagicMock()
            mock_context.new_page.return_value = mock_page
            caption = embedder._get_caption_from_reel_page("UNKNOWN", mock_context)
        self.assertEqual(caption, "")
        mock_page.close.assert_called_once()


class ParseCommentsDomFlexibleTest(unittest.TestCase):
    def test_reel_dialog_panel_layout(self) -> None:
        from scripts.instagram_browser import parse_comments_from_dom_text

        panel = (
            "judemgmt\n"
            "@noahpdillon about @lucamadar in berlin for church electronic\n"
            "\n"
            "luca@judemgmt.com\n"
            "7 sem\n"
            "Voir la traduction\n"
            "loevasoltani\n"
            "🙏🙏🙏🙏🙏\n"
            "7 semRépondre\n"
            "shadrinsky\n"
            "@lucamadar 🥹🥹🥹\n"
            "6 sem1 J'aimeRépondre\n"
        )
        parsed = parse_comments_from_dom_text(panel)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["text"], "🙏🙏🙏🙏🙏")
        self.assertEqual(parsed[1]["text"], "@lucamadar 🥹🥹🥹")
        self.assertEqual(parsed[1]["like_count"], 1)


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
        ), patch.object(embedder, "_infer_embedding_dim", return_value=4096), patch.object(
            embedder, "embedding_config_ok", return_value=True
        ), patch.object(
            embedder, "load_pca", return_value=None), patch.object(
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
