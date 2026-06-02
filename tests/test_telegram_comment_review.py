"""Tests du bot de review commentaires training."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts import telegram_comment_review as tcr


class CommentReviewLogicTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.training = Path(self.tmp.name) / "training_comments_viral.json"
        self.state = Path(self.tmp.name) / "state.json"
        self.entries = [
            {
                "media_id": "ABC",
                "username": "creator",
                "text": "mdr trop vrai",
                "t_type": "T2",
                "t_type_profile": "T2",
            },
            {
                "media_id": "DEF",
                "username": "other",
                "text": "spam emoji",
                "t_type": "T1",
                "t_type_profile": "T1",
            },
        ]
        self.training.write_text(json.dumps(self.entries), encoding="utf-8")
        self.cache = tcr.TrainingCache(self.training)

    def test_parse_callback(self) -> None:
        self.assertEqual(tcr._parse_callback("tc:k:abc123"), ("k", "abc123"))
        self.assertEqual(tcr._parse_callback("tc:d:xyz"), ("d", "xyz"))
        self.assertIsNone(tcr._parse_callback("v:user"))

    def test_apply_keep_marks_human_validated(self) -> None:
        entry = self.entries[0]
        sid = tcr.entry_short_id(entry)
        state = tcr.default_state()
        state["pending"][sid] = {
            "dedup_key": tcr.entry_dedup_key(entry),
            "message_id": 1,
            "chat_id": "123",
        }
        item, note, _ = tcr.apply_keep(cache=self.cache, state=state, short_id=sid)
        self.assertIsNotNone(item)
        self.assertIn("gardé", note)
        saved = json.loads(self.training.read_text(encoding="utf-8"))
        self.assertTrue(saved[0]["human_validated"])
        self.assertEqual(state["reviewed"][tcr.entry_dedup_key(entry)], "kept")

    def test_apply_delete_removes_entry(self) -> None:
        entry = self.entries[1]
        sid = tcr.entry_short_id(entry)
        state = tcr.default_state()
        state["pending"][sid] = {
            "dedup_key": tcr.entry_dedup_key(entry),
            "message_id": 2,
            "chat_id": "123",
        }
        _, note, _ = tcr.apply_delete(cache=self.cache, state=state, short_id=sid)
        self.assertIn("supprimé", note)
        saved = json.loads(self.training.read_text(encoding="utf-8"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["text"], "mdr trop vrai")

    def test_unreviewed_skips_human_validated(self) -> None:
        self.entries[0]["human_validated"] = True
        state = tcr.default_state()
        queue = tcr._unreviewed_entries(self.entries, state)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["text"], "spam emoji")

    @patch("scripts.telegram_comment_review._ack_callback")
    @patch("scripts.telegram_comment_review._telegram_post")
    def test_handle_callback_keep(
        self, mock_post: MagicMock, mock_ack: MagicMock
    ) -> None:
        entry = self.entries[0]
        sid = tcr.entry_short_id(entry)
        tcr.save_state(
            self.state,
            {
                **tcr.default_state(),
                "pending": {
                    sid: {
                        "dedup_key": tcr.entry_dedup_key(entry),
                        "message_id": 99,
                        "chat_id": "8526951936",
                    }
                },
            },
        )
        state = tcr.load_state(self.state)
        mock_post.return_value = {"ok": True}
        result = tcr.handle_callback(
            {
                "id": "cq1",
                "data": f"tc:k:{sid}",
                "message": {
                    "message_id": 99,
                    "chat": {"id": "8526951936"},
                },
            },
            cache=self.cache,
            state=state,
            state_path=self.state,
            token="tok",
            expected_chat_id="8526951936",
        )
        self.assertIn("gardé", result)
        mock_ack.assert_called_once()
        saved = json.loads(self.training.read_text(encoding="utf-8"))
        self.assertTrue(saved[0]["human_validated"])

    @patch("scripts.telegram_comment_review._ack_callback")
    def test_stale_callback_acks_already_done(self, mock_ack: MagicMock) -> None:
        state = tcr.default_state()
        state["resolved"]["stale1"] = {
            "dedup_key": "x||y",
            "status": "kept",
            "message_id": 1,
            "chat_id": "123",
        }
        result = tcr._handle_stale_callback(
            short_id="stale1",
            cq_id="cq",
            chat_id="123",
            message_id=1,
            state=state,
            token="tok",
        )
        self.assertEqual(result, "déjà traité")
        mock_ack.assert_called_once()
        self.assertEqual(mock_ack.call_args.kwargs.get("text"), "Déjà traité")


if __name__ == "__main__":
    unittest.main()
