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


class EnrichReelTest(unittest.TestCase):
    @patch("scripts.scrape_viral_comments.config")
    @patch("scripts.scrape_viral_comments._describe_grid", return_value="scène cuisine")
    @patch("scripts.scrape_viral_comments._make_frames_grid")
    @patch("scripts.scrape_viral_comments.transcribe_audio", return_value="bonjour le monde")
    @patch("scripts.scrape_viral_comments.extract_wav_from_video")
    @patch("scripts.scrape_viral_comments.download_reel_video")
    def test_enrich_returns_transcript_and_visual(
        self,
        mock_dl: unittest.mock.MagicMock,
        mock_wav: unittest.mock.MagicMock,
        mock_tr: unittest.mock.MagicMock,
        mock_grid: unittest.mock.MagicMock,
        mock_desc: unittest.mock.MagicMock,
        mock_cfg: unittest.mock.MagicMock,
    ) -> None:
        from scripts.scrape_viral_comments import _enrich_reel_with_transcript_and_visual

        mock_cfg.LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
        mock_cfg.LM_STUDIO_VISION_MODEL = "openbmb/minicpm-v-2_6"
        mp4 = Path("/tmp/fake.mp4")
        mock_dl.return_value = mp4
        mock_wav.return_value = Path("/tmp/fake.wav")
        mock_grid.return_value = Path("/tmp/grid.jpg")

        tr, vis = _enrich_reel_with_transcript_and_visual(
            "reel1",
            unittest.mock.MagicMock(),
        )
        self.assertEqual(tr, "bonjour le monde")
        self.assertEqual(vis, "scène cuisine")

    @patch("scripts.scrape_viral_comments.download_reel_video")
    def test_enrich_skips_when_flags(self, mock_dl: unittest.mock.MagicMock) -> None:
        from scripts.scrape_viral_comments import _enrich_reel_with_transcript_and_visual

        tr, vis = _enrich_reel_with_transcript_and_visual(
            "reel1",
            unittest.mock.MagicMock(),
            skip_transcript=True,
            skip_visual=True,
        )
        self.assertEqual((tr, vis), ("", ""))
        mock_dl.assert_not_called()


if __name__ == "__main__":
    unittest.main()
