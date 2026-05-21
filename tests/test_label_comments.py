"""Tests pour ``scripts.label_comments``."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts import label_comments


class BuildUserPromptTest(unittest.TestCase):
    def test_without_creator_context_uses_legacy_format(self) -> None:
        prompt = label_comments._build_user_prompt("hello", ["humour"], None)
        self.assertIn("Classifie ce commentaire Instagram", prompt)
        self.assertIn("Niches: humour", prompt)
        self.assertNotIn("Profil du créateur", prompt)

    def test_with_named_axes_includes_style_section(self) -> None:
        prompt = label_comments._build_user_prompt(
            "super",
            ["humour"],
            {
                "t_type_profile": "T2",
                "niches": ["humour", "sketch"],
                "named_axes": {
                    "scripted_vs_raw": 0.11,
                    "energy_level": 0.44,
                    "mainstream_vs_niche": 0.99,
                },
            },
        )
        self.assertIn("Niches du contenu: humour, sketch", prompt)
        self.assertIn("T-type dominant: T2", prompt)
        self.assertIn("Style: scripted=0.11, energie=0.44, mainstream=0.99", prompt)
        self.assertIn("Label (T1/T2/T2b/T3a/T3b/T4/T5):", prompt)

    def test_empty_named_axes_omits_style_section(self) -> None:
        prompt = label_comments._build_user_prompt(
            "super",
            ["humour"],
            {"t_type_profile": "T3b", "niches": ["humour"], "named_axes": {}},
        )
        self.assertNotIn("Style:", prompt)


class ClassifyCommentTest(unittest.TestCase):
    def _chat_response(self, content: str) -> dict[str, object]:
        return {"message": {"content": content}}

    def test_returns_t2_when_ollama_replies_t2(self) -> None:
        fake_ollama = MagicMock()
        fake_ollama.chat.return_value = self._chat_response("T2")
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            result = label_comments.classify_comment("super commentaire", ["humour"], "qwen2.5:7b")
        self.assertEqual(result, "T2")

    def test_extracts_ttype_from_surrounding_text(self) -> None:
        fake_ollama = MagicMock()
        fake_ollama.chat.return_value = self._chat_response("Je pense que c'est T3b")
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            result = label_comments.classify_comment("super commentaire", ["humour"], "qwen2.5:7b")
        self.assertEqual(result, "T3b")

    def test_invalid_ttype_returns_none(self) -> None:
        fake_ollama = MagicMock()
        fake_ollama.chat.return_value = self._chat_response("T9")
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            result = label_comments.classify_comment("super commentaire", ["humour"], "qwen2.5:7b")
        self.assertIsNone(result)

    def test_ollama_exception_returns_none_with_warning(self) -> None:
        fake_ollama = MagicMock()
        fake_ollama.chat.side_effect = RuntimeError("down")
        with patch.dict(sys.modules, {"ollama": fake_ollama}), self.assertLogs(
            "aitertainment.label_comments", level="WARNING"
        ) as cm:
            result = label_comments.classify_comment("super commentaire", ["humour"], "qwen2.5:7b")
        self.assertIsNone(result)
        self.assertTrue(any("Ollama" in message for message in cm.output))

    def test_empty_response_returns_none(self) -> None:
        fake_ollama = MagicMock()
        fake_ollama.chat.return_value = self._chat_response("")
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            result = label_comments.classify_comment("super commentaire", ["humour"], "qwen2.5:7b")
        self.assertIsNone(result)

    def test_creator_context_passed_to_ollama_messages(self) -> None:
        fake_ollama = MagicMock()
        fake_ollama.chat.return_value = {"message": {"content": "T4"}}
        ctx = {
            "t_type_profile": "T2",
            "niches": ["humour"],
            "named_axes": {"scripted_vs_raw": 0.5, "energy_level": 0.6, "mainstream_vs_niche": 0.7},
        }
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            result = label_comments.classify_comment(
                "comment test",
                ["humour"],
                "qwen2.5:7b",
                creator_context=ctx,
            )
        self.assertEqual(result, "T4")
        user_msg = fake_ollama.chat.call_args.kwargs["messages"][1]["content"]
        self.assertIn("T-type dominant: T2", user_msg)
        self.assertIn("Style: scripted=0.50", user_msg)

    @patch("scripts.label_comments.requests.post")
    def test_lm_studio_returns_t2(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "T2"}}]
        }
        mock_post.return_value = mock_resp
        with patch.object(label_comments, "LM_STUDIO_URL", "http://127.0.0.1:1234"):
            result = label_comments.classify_comment("super commentaire", ["humour"], "qwen2.5:7b")
        self.assertEqual(result, "T2")
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        self.assertEqual(call_kwargs["json"]["temperature"], 0.0)
        self.assertEqual(call_kwargs["json"]["max_tokens"], 50)
        self.assertEqual(call_kwargs["json"]["thinking"], {"type": "disabled"})
        self.assertEqual(
            call_kwargs["json"]["chat_template_kwargs"], {"enable_thinking": False}
        )

    @patch("scripts.label_comments.requests.post")
    def test_lm_studio_uses_reasoning_content_when_content_empty(
        self, mock_post: MagicMock
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "", "reasoning_content": "T3b"}}]
        }
        mock_post.return_value = mock_resp
        with patch.object(label_comments, "LM_STUDIO_URL", "http://127.0.0.1:1234"):
            result = label_comments.classify_comment("super commentaire", ["humour"], "qwen2.5:7b")
        self.assertEqual(result, "T3b")

    @patch("scripts.label_comments.requests.post")
    def test_lm_studio_request_failure_returns_none(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = RuntimeError("connection refused")
        with patch.object(label_comments, "LM_STUDIO_URL", "http://127.0.0.1:1234"), self.assertLogs(
            "aitertainment.label_comments", level="WARNING"
        ) as cm:
            result = label_comments.classify_comment("super commentaire", ["humour"], "qwen2.5:7b")
        self.assertIsNone(result)
        self.assertTrue(any("LM Studio" in message for message in cm.output))


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
        self.assertEqual(entry["llm_model"], label_comments.OLLAMA_MODEL)
        self.assertIn("labelled_at", entry)


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

    def test_dry_run_does_not_write_or_call_ollama(self) -> None:
        raw = [self._raw_entry()]
        with patch.object(label_comments, "load_raw_comments", return_value=raw), patch.object(
            label_comments, "load_training_comments", return_value=([], set())
        ), patch.object(label_comments, "load_watchlist", return_value={}), patch.object(
            label_comments, "save_training_comments"
        ) as save_mock, patch.object(
            label_comments, "classify_comment"
        ) as classify_mock:
            code = label_comments.main(["--dry-run"])
        self.assertEqual(code, 0)
        save_mock.assert_not_called()
        classify_mock.assert_not_called()

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
        with patch.object(label_comments, "load_raw_comments", return_value=raw), patch.object(
            label_comments, "load_training_comments", return_value=(existing, {key})
        ), patch.object(label_comments, "load_watchlist", return_value={}), patch.object(
            label_comments, "classify_comment"
        ) as classify_mock, patch.object(
            label_comments, "save_training_comments"
        ) as save_mock:
            code = label_comments.main([])
        self.assertEqual(code, 0)
        classify_mock.assert_not_called()
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
        with patch.object(label_comments, "load_raw_comments", return_value=raw), patch.object(
            label_comments, "load_training_comments", return_value=(existing, {key})
        ), patch.object(label_comments, "load_watchlist", return_value=watchlist), patch.object(
            label_comments, "load_vector_store", return_value=vector_store
        ), patch.object(
            label_comments, "classify_comment", return_value="T4"
        ) as classify_mock, patch.object(
            label_comments, "save_training_comments"
        ) as save_mock, patch.object(
            label_comments, "_ensure_llm_available", return_value=True
        ):
            code = label_comments.main(["--force"])
        self.assertEqual(code, 0)
        classify_mock.assert_called_once()
        ctx = classify_mock.call_args.kwargs["creator_context"]
        self.assertEqual(ctx["t_type_profile"], "T2")
        self.assertEqual(ctx["named_axes"]["energy_level"], 0.2)
        save_mock.assert_called_once()
        saved = save_mock.call_args.args[0]
        self.assertEqual(saved[0]["t_type"], "T4")

    def test_account_filter_limits_processing(self) -> None:
        raw = [
            self._raw_entry(username="alpha", text="alpha text"),
            self._raw_entry(username="beta", text="beta text"),
        ]
        with patch.object(label_comments, "load_raw_comments", return_value=raw), patch.object(
            label_comments, "load_training_comments", return_value=([], set())
        ), patch.object(label_comments, "load_watchlist", return_value={}), patch.object(
            label_comments, "load_vector_store", return_value={}
        ), patch.object(
            label_comments, "classify_comment", return_value="T2"
        ) as classify_mock, patch.object(
            label_comments, "save_training_comments"
        ), patch.object(label_comments, "_ensure_llm_available", return_value=True):
            code = label_comments.main(["--account", "@beta"])
        self.assertEqual(code, 0)
        self.assertEqual(classify_mock.call_count, 1)
        self.assertEqual(classify_mock.call_args.args[0], "beta text")
        self.assertIn("creator_context", classify_mock.call_args.kwargs)


class SaveTrainingCommentsTest(unittest.TestCase):
    def test_atomic_write_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_comments.json"
            entries = [{"media_id": "m1", "text": "hello", "t_type": "T2"}]
            label_comments.save_training_comments(entries, path)
            loaded, keys = label_comments.load_training_comments(path)
            self.assertEqual(loaded, entries)
            self.assertEqual(keys, {label_comments.dedup_key("m1", "hello")})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload, entries)


if __name__ == "__main__":
    unittest.main()
