"""Tests pour ``modules.pipeline_state``."""

from __future__ import annotations

import unittest

from modules import pipeline_state


class PipelineStateTest(unittest.TestCase):
    def test_comment_dedup_key_normalizes_text(self) -> None:
        self.assertEqual(
            pipeline_state.comment_dedup_key("ABC", "  Hello  "),
            "ABC||hello",
        )

    def test_fingerprint_stable_and_changes_with_new_comment(self) -> None:
        raw = [
            {
                "username": "alpha",
                "media_id": "m1",
                "text": "first",
            },
        ]
        fp1, n1 = pipeline_state.comments_fingerprint_for_account("alpha", raw)
        self.assertEqual(n1, 1)
        raw.append(
            {
                "username": "alpha",
                "media_id": "m2",
                "text": "second",
            }
        )
        fp2, n2 = pipeline_state.comments_fingerprint_for_account("alpha", raw)
        self.assertEqual(n2, 2)
        self.assertNotEqual(fp1, fp2)

    def test_should_skip_embed_when_fingerprint_matches(self) -> None:
        entry = {
            "username": "alpha",
            "embedding_raw": [0.1, 0.2],
            "comments_fingerprint": "abc",
        }
        self.assertTrue(
            pipeline_state.should_skip_embed("alpha", "abc", entry, force=False)
        )
        self.assertFalse(
            pipeline_state.should_skip_embed("alpha", "abc", entry, force=True)
        )
        self.assertFalse(
            pipeline_state.should_skip_embed("alpha", "xyz", entry, force=False)
        )

    def test_should_skip_playwright_when_raw_has_comments(self) -> None:
        raw = [{"username": "beta", "media_id": "m1", "text": "yo"}]
        self.assertTrue(
            pipeline_state.should_skip_playwright_collect("beta", raw, force_scrape=False)
        )
        self.assertFalse(
            pipeline_state.should_skip_playwright_collect("beta", raw, force_scrape=True)
        )
        self.assertFalse(
            pipeline_state.should_skip_playwright_collect("gamma", raw, force_scrape=False)
        )


if __name__ == "__main__":
    unittest.main()
