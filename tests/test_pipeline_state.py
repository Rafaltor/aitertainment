"""Tests pour ``modules.pipeline_state``."""

from __future__ import annotations

import unittest

from modules import pipeline_state


class PipelineStateTest(unittest.TestCase):
    def test_comment_dedup_key_normalizes_text(self) -> None:
        self.assertEqual(
            pipeline_state.comment_dedup_key("mid1", "  Hello  "),
            "mid1||hello",
        )

    def test_should_skip_playwright_when_comments_exist(self) -> None:
        raw = [
            {
                "username": "alpha",
                "media_id": "m1",
                "text": "salut",
            }
        ]
        self.assertTrue(
            pipeline_state.should_skip_playwright_collect("alpha", raw, force_scrape=False)
        )
        self.assertFalse(
            pipeline_state.should_skip_playwright_collect("alpha", raw, force_scrape=True)
        )

    def test_build_pipeline_patch(self) -> None:
        patch = pipeline_state.build_pipeline_patch(
            comments_count=3,
            comments_fingerprint="abc",
            labeled_count=2,
            labeled_at="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual(patch["comments_count"], 3)
        self.assertEqual(patch["comments_fingerprint"], "abc")
        self.assertEqual(patch["labeled_count"], 2)
        self.assertNotIn("embedded_at", patch)
