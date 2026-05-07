"""Tests pour TelegramNotifier."""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from modules.detector import CreatorStats
from modules.notifier import TelegramNotifier


class TelegramNotifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.posted = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        self.creator = CreatorStats(
            creator_id="c1",
            platform="instagram",
            username="test_user",
            followers=12_000,
            recent_videos=[
                {
                    "video_id": "r1",
                    "views": 50_000,
                    "likes": 1000,
                    "comments": 50,
                    "shares": 10,
                    "saves": 2,
                    "posted_at": self.posted,
                    "url": "https://instagram.com/reel/r1",
                },
            ],
        )
        self.signal = {
            "score_viral": 0.88,
            "alert": "CRITICAL",
            "groups": {
                "g1": {"score": 0.9},
                "g2": {"score": 0.85},
                "g3": {"score": 0.7},
                "g4": {"score": 0.8},
            },
        }
        self.classification = {
            "type": "T2",
            "tone": "Humour de niche",
            "patterns": ["pattern A", "pattern B"],
        }
        self.suggestions = ["Un", "Deux", "Trois"]

    @patch("modules.notifier.requests.post")
    def test_send_alert_payload(self, mock_post: MagicMock) -> None:
        mock_post.return_value.json.return_value = {"ok": True, "result": {"message_id": 1}}
        mock_post.return_value.raise_for_status = MagicMock()

        fixed_now = self.posted + timedelta(minutes=45)
        with patch("modules.notifier.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            n = TelegramNotifier(bot_token="T", chat_id="123")
            n.send_alert(
                self.creator,
                self.signal,
                self.classification,
                self.suggestions,
            )

        mock_post.assert_called_once()
        url = mock_post.call_args[0][0]
        self.assertIn("botT/sendMessage", url)
        body = mock_post.call_args[1]["json"]
        self.assertEqual(body["chat_id"], "123")
        self.assertEqual(body["parse_mode"], "Markdown")
        text = body["text"]
        self.assertIn("ALERTE VIRALE", text)
        self.assertIn("CRITICAL", text)
        self.assertIn("@test\\_user", text)
        self.assertIn("instagram", text)
        self.assertIn("12000", text)
        self.assertIn("0.88", text)
        self.assertIn("T2", text)
        self.assertIn("45", text)
        self.assertIn("Commentaires suggérés", text)
        self.assertIn("instagram.com/reel/r1", text)

    @patch("modules.notifier.requests.post")
    def test_test_connection(self, mock_post: MagicMock) -> None:
        mock_post.return_value.json.return_value = {"ok": True, "result": {}}
        mock_post.return_value.raise_for_status = MagicMock()
        n = TelegramNotifier(bot_token="tok", chat_id="99")
        out = n.test_connection()
        self.assertTrue(out.get("ok"))
        sent = mock_post.call_args[1]["json"]["text"]
        self.assertIn("AItertainment", sent)


if __name__ == "__main__":
    unittest.main()
