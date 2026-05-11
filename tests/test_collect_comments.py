"""Tests pour ``scripts.collect_comments``."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scripts import collect_comments


class IsValidCommentTest(unittest.TestCase):
    def test_valid_comment_returns_true(self) -> None:
        self.assertTrue(
            collect_comments.is_valid_comment(
                "trop vrai frère vraiment oui",
                12,
            )
        )

    def test_like_count_none_returns_false(self) -> None:
        self.assertFalse(
            collect_comments.is_valid_comment("trop vrai frère vraiment", None)
        )

    def test_missing_like_count_key_returns_false(self) -> None:
        comment: dict[str, object] = {}
        self.assertFalse(
            collect_comments.is_valid_comment(
                "trop vrai frère vraiment oui",
                comment.get("like_count"),
            )
        )

    def test_too_few_words_returns_false(self) -> None:
        self.assertFalse(collect_comments.is_valid_comment("trop vrai frère", 3))

    def test_http_link_returns_false(self) -> None:
        self.assertFalse(
            collect_comments.is_valid_comment(
                "regarde http://spam.com maintenant",
                4,
            )
        )

    def test_emoji_only_returns_false(self) -> None:
        self.assertFalse(
            collect_comments.is_valid_comment("😂 😂 😂 😂 😂", 4)
        )


class ExtractHashtagsTest(unittest.TestCase):
    def test_extracts_hashtags_from_caption(self) -> None:
        self.assertEqual(
            collect_comments.extract_hashtags("super vidéo #humour #sketch"),
            ["humour", "sketch"],
        )

    def test_empty_caption_returns_empty_list(self) -> None:
        self.assertEqual(collect_comments.extract_hashtags(""), [])

    def test_inline_hash_without_boundary_is_ignored(self) -> None:
        self.assertEqual(collect_comments.extract_hashtags("anti#humour"), [])


class BuildDedupKeyTest(unittest.TestCase):
    def test_same_media_and_text_case_insensitive(self) -> None:
        key_a = collect_comments.build_dedup_key("m1", "Trop Vrai")
        key_b = collect_comments.build_dedup_key("m1", "trop vrai")
        self.assertEqual(key_a, key_b)

    def test_different_texts_produce_different_keys(self) -> None:
        key_a = collect_comments.build_dedup_key("m1", "alpha")
        key_b = collect_comments.build_dedup_key("m1", "beta")
        self.assertNotEqual(key_a, key_b)


class CollectForAccountTest(unittest.TestCase):
    def setUp(self) -> None:
        self._sleep_patch = patch(
            "instagram_client.polite_sleep", lambda *a, **k: None
        )
        self._sleep_patch.start()
        self.addCleanup(self._sleep_patch.stop)

    def _creator(self) -> dict[str, object]:
        return {"username": "alpha", "niches": ["humour", "sketch"]}

    def test_no_reels_returns_empty_list(self) -> None:
        client = MagicMock()
        client.user_id_from_username.return_value = "111"
        client.user_medias.return_value = [
            SimpleNamespace(pk="p1", media_type="feed", product_type="feed")
        ]
        result = collect_comments.collect_for_account(
            "alpha",
            self._creator(),
            client,
            3,
            50,
            set(),
        )
        self.assertEqual(result, [])
        client.media_comments.assert_not_called()

    def test_invalid_comment_like_count_none_is_filtered(self) -> None:
        client = MagicMock()
        client.user_id_from_username.return_value = "111"
        client.user_medias.return_value = [
            SimpleNamespace(
                pk="r1",
                media_type="clip",
                caption_text="POV test #humour",
                view_count=1000,
                like_count=100,
                comment_count=10,
                audio="aud1",
            )
        ]
        client.media_comments.return_value = [
            SimpleNamespace(text="trop vrai frère vraiment", like_count=None)
        ]
        result = collect_comments.collect_for_account(
            "alpha",
            self._creator(),
            client,
            3,
            50,
            set(),
        )
        self.assertEqual(result, [])

    def test_duplicate_comment_is_ignored(self) -> None:
        client = MagicMock()
        client.user_id_from_username.return_value = "111"
        client.user_medias.return_value = [
            SimpleNamespace(
                pk="r1",
                media_type="clip",
                caption_text="POV test #humour",
                view_count=1000,
                like_count=100,
                comment_count=10,
                audio="aud1",
            )
        ]
        client.media_comments.return_value = [
            SimpleNamespace(text="trop vrai frère vraiment oui", like_count=5)
        ]
        dedup_keys = {
            collect_comments.build_dedup_key("r1", "trop vrai frère vraiment oui")
        }
        result = collect_comments.collect_for_account(
            "alpha",
            self._creator(),
            client,
            3,
            50,
            dedup_keys,
        )
        self.assertEqual(result, [])

    def test_valid_comment_returns_full_entry(self) -> None:
        client = MagicMock()
        client.user_id_from_username.return_value = "111"
        client.user_medias.return_value = [
            SimpleNamespace(
                pk="r1",
                media_type="clip",
                caption_text="POV test #humour #sketch",
                view_count=1500,
                like_count=100,
                comment_count=10,
                audio="aud1",
            )
        ]
        client.media_comments.return_value = [
            SimpleNamespace(text="  trop vrai frère vraiment oui  ", like_count=42)
        ]
        result = collect_comments.collect_for_account(
            "alpha",
            self._creator(),
            client,
            3,
            50,
            set(),
        )
        self.assertEqual(len(result), 1)
        entry = result[0]
        self.assertEqual(entry["media_id"], "r1")
        self.assertEqual(entry["username"], "alpha")
        self.assertEqual(entry["niches"], ["humour", "sketch"])
        self.assertEqual(entry["text"], "trop vrai frère vraiment oui")
        self.assertEqual(entry["comment_likes"], 42)
        self.assertEqual(entry["views"], 1500)
        self.assertAlmostEqual(entry["comment_to_like_ratio"], 0.1, places=4)
        self.assertEqual(entry["caption"], "POV test #humour #sketch")
        self.assertEqual(entry["hashtags"], ["humour", "sketch"])
        self.assertEqual(entry["audio_id"], "aud1")
        self.assertIn("collected_at", entry)


class MainTest(unittest.TestCase):
    def test_dry_run_does_not_write_to_disk(self) -> None:
        creators = [{"username": "alpha", "action": "validated"}]
        with patch.object(collect_comments, "load_watchlist", return_value=creators), patch.object(
            collect_comments, "load_raw_comments", return_value=([], set())
        ), patch.object(collect_comments, "save_raw_comments") as save_mock, patch(
            "instagram_client.get_client"
        ) as client_mock:
            code = collect_comments.main(["--dry-run"])
        self.assertEqual(code, 0)
        save_mock.assert_not_called()
        client_mock.assert_not_called()

    def test_unknown_account_exits_with_code_one(self) -> None:
        creators = [{"username": "known", "action": "validated"}]
        with patch.object(collect_comments, "load_watchlist", return_value=creators), patch.object(
            collect_comments, "load_raw_comments", return_value=([], set())
        ), self.assertLogs("aitertainment.collect_comments", level="ERROR") as cm:
            code = collect_comments.main(["--account", "@missing"])
        self.assertEqual(code, 1)
        self.assertTrue(
            any("Compte @missing introuvable dans la watchlist." in message for message in cm.output)
        )

    def test_empty_watchlist_exits_with_code_two(self) -> None:
        with patch.object(collect_comments, "load_watchlist", return_value=[]), self.assertLogs(
            "aitertainment.collect_comments", level="ERROR"
        ) as cm:
            code = collect_comments.main([])
        self.assertEqual(code, 2)
        self.assertTrue(any("Watchlist vide" in message for message in cm.output))


class SaveRawCommentsTest(unittest.TestCase):
    def test_atomic_write_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw_comments.json"
            entries = [{"media_id": "m1", "text": "hello world"}]
            collect_comments.save_raw_comments(entries, path)
            loaded, keys = collect_comments.load_raw_comments(path)
            self.assertEqual(loaded, entries)
            self.assertEqual(keys, {collect_comments.build_dedup_key("m1", "hello world")})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload, entries)


if __name__ == "__main__":
    unittest.main()
