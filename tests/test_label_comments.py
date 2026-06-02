"""Tests pour ``scripts.label_comments``."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts import label_comments


class ExtractTTypeTest(unittest.TestCase):
    def test_extracts_last_ttype_from_chain_of_thought(self) -> None:
        text = "maybe T1 or T5 but finally the label is T2b for this comment"
        self.assertEqual(label_comments._extract_t_type_from_content(text), "T2b")

    def test_prefers_t2b_over_t2(self) -> None:
        self.assertEqual(label_comments._extract_t_type_from_content("answer: T2b"), "T2b")

    def test_numbered_class_line_after_prefill(self) -> None:
        self.assertEqual(
            label_comments._extract_from_numbered_class_line(
                "1 : Admiration sincère, encouragements"
            ),
            "T1",
        )
        self.assertEqual(
            label_comments._extract_from_numbered_class_line("2b : Humour participatif"),
            "T2b",
        )

    def test_reasoning_ignores_definition_list_before_analysis(self) -> None:
        reasoning = (
            "T1: admiration\nT5: hate\n"
            "3. **Analyze the Comment:**\n"
            "Text: hello\n"
            "Conclusion label: T2b"
        )
        self.assertEqual(
            label_comments._extract_t_type_from_reasoning(reasoning, "hello"),
            "T2b",
        )


class BuildUserPromptTest(unittest.TestCase):
    def test_without_creator_context_uses_legacy_format(self) -> None:
        prompt = label_comments._build_user_prompt("hello", ["humour"], None)
        self.assertIn("Classifie ce commentaire Instagram", prompt)
        self.assertIn("Niches: humour", prompt)
        self.assertNotIn("Profil du créateur", prompt)

    def test_with_full_context_includes_likes_caption_and_axes(self) -> None:
        prompt = label_comments._build_user_prompt(
            "super comment",
            ["humour"],
            {
                "username": "alpha",
                "t_type_profile": "T2",
                "niches": ["humour", "sketch"],
                "tier": "B",
                "followers": 12000,
                "comment_likes": 34,
                "views": 50000,
                "caption": "Ma punchline du jour",
                "hashtags": ["humour", "reels"],
                "media_id": "ABC123",
                "named_axes": {
                    "scripted_vs_raw": 0.11,
                    "energy_level": 0.44,
                    "mainstream_vs_niche": 0.99,
                },
            },
        )
        self.assertIn("Likes sur ce commentaire: 34", prompt)
        self.assertIn("Caption: Ma punchline du jour", prompt)
        self.assertIn("Vues du reel (approx.): 50000", prompt)
        self.assertIn("Niches: humour, sketch", prompt)
        self.assertIn("T-type dominant (profil créateur): T2", prompt)
        self.assertIn("scripted_vs_raw=0.11", prompt)
        self.assertIn("energy_level=0.44", prompt)
        self.assertIn("pas T1 par défaut", prompt)

    def test_empty_named_axes_shows_unavailable(self) -> None:
        prompt = label_comments._build_user_prompt(
            "super",
            ["humour"],
            {
                "username": "beta",
                "t_type_profile": "T3b",
                "niches": ["humour"],
                "named_axes": {},
                "comment_likes": 2,
            },
        )
        self.assertIn("absent du vector_store", prompt)
        self.assertIn("Likes sur ce commentaire: 2", prompt)


class BuildCreatorContextTest(unittest.TestCase):
    def test_merges_database_scores_and_raw_entry(self) -> None:
        raw = {
            "username": "alpha",
            "text": "mdr",
            "media_id": "m1",
            "caption": "cap",
            "comment_likes": 10,
            "views": 1000,
            "niches": ["humour"],
        }
        creator = {
            "tier": "C",
            "followers": 5000,
            "scores_history": [{"score": 420, "reel_engagement_median": 0.12}],
        }
        vs = {"named_axes": {"energy_level": 0.8}}
        ctx = label_comments.build_creator_context_for_label(raw, creator, vs)
        self.assertEqual(ctx["comment_likes"], 10)
        self.assertEqual(ctx["caption"], "cap")
        self.assertEqual(ctx["tier"], "C")
        self.assertEqual(ctx["discovery_score"], 420)
        self.assertEqual(ctx["named_axes"]["energy_level"], 0.8)
        self.assertTrue(ctx["has_vector_profile"])


class ParseLabelFuseResponseTest(unittest.TestCase):
    def test_parses_json_object(self) -> None:
        content = (
            '{"t_type": "T2b", "video_context": "Sketch de rue avec deux potes."}'
        )
        t_type, ctx = label_comments._parse_label_fuse_response(content)
        self.assertEqual(t_type, "T2b")
        self.assertIn("Sketch", ctx)

    def test_parses_json_with_apostrophe_in_video_context(self) -> None:
        # Apostrophe non échappée (JSON valide) — cas fréquent en français.
        content = (
            '{"t_type": "T2", "video_context": "il dit qu\'il part en cuisine."}'
        )
        t_type, ctx = label_comments._parse_label_fuse_response(content)
        self.assertEqual(t_type, "T2")
        self.assertIn("qu'il part", ctx)

    def test_extract_json_block_ignores_braces_inside_strings(self) -> None:
        raw = (
            '{"t_type": "T3b", "video_context": "arc {ironique} et ton moqueur"}'
        )
        data = label_comments._extract_json_block(raw)
        self.assertIsNotNone(data)
        self.assertEqual(data["t_type"], "T3b")
        self.assertIn("{ironique}", data["video_context"])

    def test_fallback_ttype_without_json(self) -> None:
        t_type, ctx = label_comments._parse_label_fuse_response("Verdict final : T3b")
        self.assertEqual(t_type, "T3b")
        self.assertEqual(ctx, "")


class LabelAndFuseTest(unittest.TestCase):
    @patch("scripts.label_comments.requests.post")
    def test_returns_t_type_and_video_context(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"t_type": "T2", "video_context": "Parodie en cuisine."}'
                        )
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp
        t_type, video_context = label_comments.label_and_fuse(
            "mdr trop fort",
            ["humour"],
            {"t_type_profile": "T2", "named_axes": {}},
            caption="cap",
            transcript="dialogue audio",
            visual_description="deux personnes",
            url="http://127.0.0.1:1234/v1",
            model="qwen/qwen3.6-35b-a3b",
        )
        self.assertEqual(t_type, "T2")
        self.assertEqual(video_context, "Parodie en cuisine.")
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "qwen/qwen3.6-35b-a3b")
        self.assertEqual(payload["max_tokens"], 2000)
        user_msg = payload["messages"][1]["content"]
        self.assertIn("mdr trop fort", user_msg)
        self.assertIn("dialogue audio", user_msg)
        self.assertIn("deux personnes", user_msg)

    @patch("scripts.label_comments.requests.post")
    def test_uses_reasoning_content_when_content_empty(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": (
                            '{"t_type": "T3b", "video_context": "Moment relatable."}'
                        ),
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp
        t_type, video_context = label_comments.label_and_fuse(
            "c'est moi",
            ["humour"],
            None,
            url="http://127.0.0.1:1234/v1",
            model="qwen/qwen3.6-35b-a3b",
        )
        self.assertEqual(t_type, "T3b")
        self.assertEqual(video_context, "Moment relatable.")

    @patch("scripts.label_comments.requests.post")
    def test_request_failure_returns_none_empty(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = RuntimeError("connection refused")
        with self.assertLogs("aitertainment.label_comments", level="WARNING"):
            t_type, video_context = label_comments.label_and_fuse(
                "hello",
                ["humour"],
                None,
                url="http://127.0.0.1:1234/v1",
                model="qwen/qwen3.6-35b-a3b",
            )
        self.assertIsNone(t_type)
        self.assertEqual(video_context, "")

    def test_missing_url_returns_none(self) -> None:
        with patch.object(label_comments, "LABEL_LLM_URL", ""):
            t_type, video_context = label_comments.label_and_fuse(
                "hello", ["humour"], None
            )
        self.assertIsNone(t_type)
        self.assertEqual(video_context, "")


class BuildTrainingEntryTest(unittest.TestCase):
    def test_t_type_profile_from_watchlist(self) -> None:
        raw_entry = {
            "media_id": "m1",
            "username": "alpha",
            "niches": ["humour"],
            "text": "hello",
            "comment_likes": 3,
            "views": 10,
            "comment_to_like_ratio": 0.1,
            "caption": "cap",
            "hashtags": ["humour"],
            "audio_id": "a1",
            "collected_at": "2026-05-11T00:00:00",
        }
        watchlist = {"alpha": {"username": "alpha", "t_type": "T2"}}
        entry = label_comments.build_training_entry(raw_entry, "T3b", watchlist)
        self.assertEqual(entry["t_type_profile"], "T2")

    def test_t_type_profile_empty_when_username_missing(self) -> None:
        raw_entry = {
            "media_id": "m1",
            "username": "ghost",
            "text": "hello",
        }
        entry = label_comments.build_training_entry(raw_entry, "T2", {})
        self.assertEqual(entry["t_type_profile"], "")

    def test_raw_fields_propagated(self) -> None:
        raw_entry = {
            "media_id": "m1",
            "username": "alpha",
            "niches": ["humour", "sketch"],
            "text": "hello world",
            "comment_likes": 7,
            "views": 42,
            "comment_to_like_ratio": 0.2,
            "caption": "cap",
            "hashtags": ["humour"],
            "audio_id": "a1",
            "collected_at": "2026-05-11T00:00:00",
        }
        entry = label_comments.build_training_entry(raw_entry, "T4", {"alpha": {"t_type": "T2"}})
        self.assertEqual(entry["media_id"], "m1")
        self.assertEqual(entry["username"], "alpha")
        self.assertEqual(entry["t_type"], "T4")
        self.assertEqual(entry["niches"], ["humour", "sketch"])
        self.assertEqual(entry["text"], "hello world")
        self.assertEqual(entry["comment_likes"], 7)
        self.assertEqual(entry["views"], 42)
        self.assertAlmostEqual(entry["comment_to_like_ratio"], 0.2)
        self.assertEqual(entry["caption"], "cap")
        self.assertEqual(entry["hashtags"], ["humour"])
        self.assertEqual(entry["audio_id"], "a1")
        self.assertEqual(entry["collected_at"], "2026-05-11T00:00:00")
        self.assertEqual(entry["llm_model"], label_comments.LABEL_LLM_MODEL)
        self.assertIn("labelled_at", entry)

    def test_video_context_stored_in_entry(self) -> None:
        raw_entry = {
            "media_id": "m1",
            "username": "alpha",
            "niches": ["humour"],
            "text": "hello",
            "transcript": "audio ici",
            "visual_description": "visuel ici",
        }
        entry = label_comments.build_training_entry(
            raw_entry,
            "T2",
            {"alpha": {"t_type": "T2"}},
            video_context="Fusion audio + visuel.",
        )
        self.assertEqual(entry["video_context"], "Fusion audio + visuel.")
        self.assertEqual(entry["transcript"], "audio ici")
        self.assertEqual(entry["visual_description"], "visuel ici")


class ResetAllLabelsTest(unittest.TestCase):
    def test_reset_empties_training_and_keeps_viral_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            viral_path = Path(tmp) / "viral_comments.json"
            train_path = Path(tmp) / "training_comments_viral.json"
            viral_path.write_text(
                json.dumps(
                    [
                        {"media_id": "m1", "username": "alpha", "text": "hello"},
                        {"media_id": "m2", "username": "beta", "text": "world"},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            train_path.write_text(
                json.dumps([{"media_id": "m1", "text": "hello", "t_type": "T2"}])
                + "\n",
                encoding="utf-8",
            )
            prev, viral_size = label_comments.reset_all_comment_labels(
                viral_path=viral_path, training_path=train_path
            )
            self.assertEqual(prev, 1)
            self.assertEqual(viral_size, 2)
            self.assertEqual(len(json.loads(viral_path.read_text())), 2)
            self.assertEqual(json.loads(train_path.read_text()), [])


class PurgeViralPoolTest(unittest.TestCase):
    def test_purge_removes_labeled_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            viral_path = Path(tmp) / "viral_comments.json"
            train_path = Path(tmp) / "training_comments_viral.json"
            viral_path.write_text(
                json.dumps(
                    [
                        {"media_id": "m1", "text": "hello"},
                        {"media_id": "m2", "text": "world"},
                    ]
                ),
                encoding="utf-8",
            )
            train_path.write_text(
                json.dumps([{"media_id": "m1", "text": "hello", "t_type": "T2"}]),
                encoding="utf-8",
            )
            removed, remaining = label_comments.purge_labeled_from_viral_pool(
                viral_path=viral_path, training_path=train_path
            )
            self.assertEqual(removed, 1)
            self.assertEqual(remaining, 1)
            left = json.loads(viral_path.read_text(encoding="utf-8"))
            self.assertEqual(len(left), 1)
            self.assertEqual(left[0]["media_id"], "m2")


class MainTest(unittest.TestCase):
    def _raw_entry(self, *, username: str = "alpha", text: str = "hello world") -> dict[str, object]:
        return {
            "media_id": "m1",
            "username": username,
            "niches": ["humour"],
            "text": text,
            "comment_likes": 1,
            "views": 10,
            "comment_to_like_ratio": 0.1,
            "caption": "cap",
            "hashtags": [],
            "audio_id": "",
            "collected_at": "2026-05-11T00:00:00",
        }

    def test_dry_run_does_not_write_or_call_llm(self) -> None:
        raw = [self._raw_entry()]
        with patch.object(label_comments, "load_viral_comments", return_value=raw), patch.object(
            label_comments, "load_training_comments", return_value=([], set())
        ), patch.object(label_comments, "load_watchlist", return_value={}), patch.object(
            label_comments, "save_training_comments"
        ) as save_mock, patch.object(
            label_comments, "label_and_fuse"
        ) as fuse_mock:
            code = label_comments.main(["--dry-run"])
        self.assertEqual(code, 0)
        save_mock.assert_not_called()
        fuse_mock.assert_not_called()

    def test_existing_training_entry_is_skipped_without_force(self) -> None:
        raw = [self._raw_entry()]
        key = label_comments.dedup_key("m1", "hello world")
        existing = [
            {
                "media_id": "m1",
                "text": "hello world",
                "t_type": "T2",
            }
        ]
        with patch.object(label_comments, "load_viral_comments", return_value=raw), patch.object(
            label_comments, "load_training_comments", return_value=(existing, {key})
        ), patch.object(label_comments, "load_watchlist", return_value={}), patch.object(
            label_comments, "label_and_fuse"
        ) as fuse_mock, patch.object(
            label_comments, "save_training_comments"
        ) as save_mock:
            code = label_comments.main([])
        self.assertEqual(code, 0)
        fuse_mock.assert_not_called()
        save_mock.assert_not_called()

    def test_force_relabels_existing_entry(self) -> None:
        raw = [self._raw_entry()]
        key = label_comments.dedup_key("m1", "hello world")
        existing = [
            {
                "media_id": "m1",
                "text": "hello world",
                "t_type": "T2",
            }
        ]
        watchlist = {"alpha": {"username": "alpha", "t_type": "T2"}}
        vector_store = {
            "alpha": {
                "username": "alpha",
                "named_axes": {
                    "scripted_vs_raw": 0.1,
                    "energy_level": 0.2,
                    "mainstream_vs_niche": 0.3,
                },
            }
        }
        with patch.object(label_comments, "load_viral_comments", return_value=raw), patch.object(
            label_comments, "load_training_comments", return_value=(existing, {key})
        ), patch.object(label_comments, "load_watchlist", return_value=watchlist), patch.object(
            label_comments, "load_vector_store", return_value=vector_store
        ), patch.object(
            label_comments, "label_and_fuse", return_value=("T4", "Contexte vidéo fusionné.")
        ) as fuse_mock, patch.object(
            label_comments, "save_training_comments"
        ) as save_mock, patch.object(
            label_comments, "_ensure_llm_available", return_value=True
        ):
            code = label_comments.main(["--force"])
        self.assertEqual(code, 0)
        fuse_mock.assert_called_once()
        ctx = fuse_mock.call_args.args[2]
        self.assertEqual(ctx["t_type_profile"], "T2")
        self.assertEqual(ctx["named_axes"]["energy_level"], 0.2)
        self.assertEqual(ctx["comment_likes"], 1)
        save_mock.assert_called_once()
        saved = save_mock.call_args.args[0]
        self.assertEqual(saved[0]["t_type"], "T4")
        self.assertEqual(saved[0]["video_context"], "Contexte vidéo fusionné.")

    def test_account_filter_limits_processing(self) -> None:
        raw = [
            self._raw_entry(username="alpha", text="alpha text"),
            self._raw_entry(username="beta", text="beta text"),
        ]
        with patch.object(label_comments, "load_viral_comments", return_value=raw), patch.object(
            label_comments, "load_training_comments", return_value=([], set())
        ), patch.object(label_comments, "load_watchlist", return_value={}), patch.object(
            label_comments, "load_vector_store", return_value={}
        ), patch.object(
            label_comments, "label_and_fuse", return_value=("T2", "")
        ) as fuse_mock, patch.object(
            label_comments, "save_training_comments"
        ), patch.object(label_comments, "_ensure_llm_available", return_value=True):
            code = label_comments.main(["--account", "@beta"])
        self.assertEqual(code, 0)
        self.assertEqual(fuse_mock.call_count, 1)
        self.assertEqual(fuse_mock.call_args.args[0], "beta text")
        self.assertIsInstance(fuse_mock.call_args.args[2], dict)


class SaveTrainingCommentsTest(unittest.TestCase):
    def test_atomic_write_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_comments_viral.json"
            entries = [{"media_id": "m1", "text": "hello", "t_type": "T2"}]
            label_comments.save_training_comments(entries, path)
            loaded, keys = label_comments.load_training_comments(path)
            self.assertEqual(loaded, entries)
            self.assertEqual(keys, {label_comments.dedup_key("m1", "hello")})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload, entries)


if __name__ == "__main__":
    unittest.main()
