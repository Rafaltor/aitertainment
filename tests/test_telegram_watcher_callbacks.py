"""Tests telegram_watcher_callbacks — boutons publier commentaire."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from config import ORDERED_T_TYPES
from telegram_watcher_callbacks import (
    build_comment_keyboard,
    handle_callback,
    register_pending_post,
)


class TelegramWatcherCallbacksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pending = Path(self.tmp.name) / "pending.json"

    def test_build_keyboard_one_row_per_t_type(self) -> None:
        comments = {t: f"comment-{t}" for t in ORDERED_T_TYPES}
        kb = build_comment_keyboard("tok123", comments)
        rows = kb["inline_keyboard"]
        self.assertEqual(len(rows), len(ORDERED_T_TYPES))
        self.assertTrue(rows[0][0]["callback_data"].startswith("w:tok123:"))

    def test_register_and_callback_posts(self) -> None:
        import telegram_watcher_callbacks as twc

        old_path = twc.PENDING_PATH
        twc.PENDING_PATH = self.pending
        self.addCleanup(setattr, twc, "PENDING_PATH", old_path)

        comments = {"T2": "mdr trop vrai", "T3a": "nul"}
        token = register_pending_post(
            creator={"username": "creator", "t_type": "T2"},
            post={"video_id": "REEL1", "username": "creator", "url": "http://x"},
            comments=comments,
        )

        post_fn = MagicMock(return_value=(True, ""))
        with patch.object(twc, "_ack_callback"), patch.object(
            twc, "_edit_message_keyboard"
        ):
            result = handle_callback(
                {
                    "id": "cq1",
                    "data": f"w:{token}:T2",
                    "message": {"message_id": 99, "chat": {"id": "123"}},
                },
                token="bot",
                expected_chat_id="123",
                post_fn=post_fn,
            )

        self.assertIn("post OK", result)
        post_fn.assert_called_once_with("REEL1", "mdr trop vrai")

        data = json.loads(self.pending.read_text(encoding="utf-8"))
        self.assertTrue(data["posts"][token]["posted"]["T2"])

    def test_callback_rejects_wrong_chat(self) -> None:
        import telegram_watcher_callbacks as twc

        with patch.object(twc, "_ack_callback") as mock_ack:
            out = handle_callback(
                {
                    "id": "cq1",
                    "data": "w:abc:T2",
                    "message": {"chat": {"id": "999"}},
                },
                token="bot",
                expected_chat_id="123",
            )
        self.assertIn("ignoré", out)
        mock_ack.assert_called_once()


if __name__ == "__main__":
    unittest.main()
