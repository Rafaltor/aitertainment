"""Tests engagement fil Reels (caption FR, sans Playwright)."""

import random
import unittest

from scripts.instagram_browser import feed_watch_duration_s, should_boost_french_reel_on_feed


class ShouldBoostFrenchReelOnFeedTest(unittest.TestCase):
    def test_empty_caption_no_boost(self) -> None:
        self.assertFalse(should_boost_french_reel_on_feed("", french_only=True))

    def test_english_caption_no_boost(self) -> None:
        self.assertFalse(
            should_boost_french_reel_on_feed(
                "Just cooked the hardest beat of the year.",
                french_only=True,
            )
        )

    def test_french_caption_boost(self) -> None:
        self.assertTrue(
            should_boost_french_reel_on_feed(
                "5 voix de la TV. 1 Saucisse Vegan. 0 coupure 🤣",
                french_only=True,
            )
        )

    def test_include_english_disables_boost(self) -> None:
        self.assertFalse(
            should_boost_french_reel_on_feed(
                "mdr trop vrai",
                french_only=False,
            )
        )


class FeedWatchDurationTest(unittest.TestCase):
    def test_respects_low_fr_watch_s(self) -> None:
        rng = random.Random(0)
        for _ in range(20):
            d = feed_watch_duration_s(5.0, rng=rng)
            self.assertLessEqual(d, 7.0)
            self.assertGreaterEqual(d, 3.5)

    def test_not_clamped_to_15s_minimum(self) -> None:
        rng = random.Random(1)
        durations = [feed_watch_duration_s(5.0, rng=rng) for _ in range(30)]
        self.assertTrue(all(d < 12.0 for d in durations))


if __name__ == "__main__":
    unittest.main()
