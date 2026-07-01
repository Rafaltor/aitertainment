"""Tests pour ``scripts.reel_media``."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.reel_media import download_reel_video


class DownloadReelVideoTest(unittest.TestCase):
    @patch("scripts.reel_media.subprocess.run")
    def test_no_video_formats_logs_debug_not_warning(
        self, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="ERROR: No video formats found",
        )
        with self.assertLogs("aitertainment.reel_media", level="DEBUG") as logs:
            out = download_reel_video("carousel_id", MagicMock(), Path(tempfile.mkdtemp()))
        self.assertIsNone(out)
        self.assertTrue(
            any("pas de vidéo (carousel/photo)" in msg for msg in logs.output)
        )
        self.assertFalse(any("yt-dlp vidéo échoué" in msg for msg in logs.output))
