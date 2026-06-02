"""Tests pour scrape_viral_comments (helpers sans Playwright)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.scrape_viral_comments import _accounts_from_csv


class ScrapeViralCommentsHelpersTest(unittest.TestCase):
    def test_accounts_from_csv(self) -> None:
        rows = _accounts_from_csv("@foo, bar , ,baz")
        self.assertEqual([r[0] for r in rows], ["foo", "bar", "baz"])

    @patch("scripts.scrape_viral_comments.load_watchlist")
    def test_accounts_from_watchlist_limit(self, mock_load: unittest.mock.MagicMock) -> None:
        from scripts.scrape_viral_comments import _accounts_from_watchlist

        mock_load.return_value = [
            {"username": "a", "niches": ["humour"]},
            {"username": "b", "niches": ["sketch"]},
        ]
        rows = _accounts_from_watchlist(1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "a")


class ScrapeViralCommentsDryRunTest(unittest.TestCase):
    def test_dry_run_feed_mode(self) -> None:
        from scripts.scrape_viral_comments import main

        self.assertEqual(main(["--dry-run", "--mode", "feed", "--feed-scrolls", "10"]), 0)


class CountEntriesTest(unittest.TestCase):
    def test_count_entries(self) -> None:
        import tempfile
        from scripts.scrape_viral_comments import _count_entries

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "viral.json"
            path.write_text(json.dumps([{"media_id": "a", "text": "x"}]))
            self.assertEqual(_count_entries(path), 1)

    def test_count_entries_wrapped_entries_key(self) -> None:
        from scripts.instagram_browser import load_viral_comments_file

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "viral.json"
            path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {"media_id": "a", "text": "x"},
                            {"media_id": "b", "text": "y"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            entries, keys = load_viral_comments_file(path)
            self.assertEqual(len(entries), 2)
            self.assertEqual(len(keys), 2)

    def test_load_raises_on_corrupt_json(self) -> None:
        from scripts.instagram_browser import ViralCommentsIOError, load_viral_comments_file

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "viral.json"
            path.write_text("[{not valid json", encoding="utf-8")
            with self.assertRaises(ViralCommentsIOError):
                load_viral_comments_file(path, retries=1, retry_delay_s=0)

    def test_save_merge_keeps_existing(self) -> None:
        from scripts.instagram_browser import load_viral_comments_file, save_viral_comments_file

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "viral.json"
            save_viral_comments_file(
                [{"media_id": "a", "text": "un", "username": "u1"}],
                path,
                merge=False,
                allow_shrink=True,
            )
            save_viral_comments_file(
                [{"media_id": "b", "text": "deux", "username": "u2"}],
                path,
                merge=True,
            )
            entries, _ = load_viral_comments_file(path)
            self.assertEqual(len(entries), 2)

    def test_save_refuses_empty_pool(self) -> None:
        from scripts.instagram_browser import load_viral_comments_file, save_viral_comments_file

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "viral.json"
            save_viral_comments_file(
                [{"media_id": "a", "text": "un", "username": "u"}],
                path,
                merge=False,
                allow_shrink=True,
            )
            save_viral_comments_file([], path, merge=True, allow_empty=False)
            entries, _ = load_viral_comments_file(path)
            self.assertEqual(len(entries), 1)


class FilterReelsForScrapeTest(unittest.TestCase):
    def test_filter_by_comment_count_and_sort(self) -> None:
        from scripts.instagram_browser import _filter_reels_for_comment_scrape

        candidates = [
            {"media_id": "a", "comment_count": 5},
            {"media_id": "b", "comment_count": 100},
            {"media_id": "c", "comment_count": 50},
            {"media_id": "d", "comment_count": 0},
        ]
        out = _filter_reels_for_comment_scrape(candidates, max_reels=2, min_comment_count=20)
        self.assertEqual([r["media_id"] for r in out], ["b", "c"])


if __name__ == "__main__":
    unittest.main()
