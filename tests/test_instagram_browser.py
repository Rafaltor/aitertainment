"""Tests unitaires pour scripts/instagram_browser.py (parsing GraphQL/DOM)."""

from __future__ import annotations

import json
import unittest

from scripts.instagram_browser import (
    _reel_comment_post_urls,
    _filter_reels_for_comment_scrape,
    _ingest_metrics_from_graphql_text,
    _ingest_suggestion_usernames_from_graphql_text,
    _is_clips_media_bucket,
    _merge_profile_graphql_text,
    _metrics_for_dom_media_id,
    _parse_count,
    _reel_from_clips_api_media,
    _reels_grid_rows_needed,
    _walk_graphql_metrics,
    _walk_profile_graphql,
    parse_comments_from_dom_text,
)


class ParseCountTest(unittest.TestCase):
    def test_compact_french_numbers(self) -> None:
        cases: list[tuple[str, int]] = [
            ("73,7 k", 73_700),
            ("1,2 M", 1_200_000),
            ("974 k", 974_000),
            ("27,7 k", 27_700),
            ("304,4 k", 304_400),
            ("1 234", 1_234),
            ("29,3 k", 29_300),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(_parse_count(raw), expected)


class ParseCommentsDomTest(unittest.TestCase):
    def test_legacy_blocks_and_spam_filter(self) -> None:
        sample = (
            "alice\n"
            "\xa0\n"
            "2 j\n"
            "Super sketch de fou rire\n"
            "1\u202f234\xa0J\u2019aime\n"
            "Répondre\n"
            "bob\n"
            "\xa0\n"
            "1 sem\n"
            "ok\n"
            "12\xa0J'aime\n"
            "Répondre\n"
            "spam\n"
            "\xa0\n"
            "1 j\n"
            "http://evil.com scam\n"
            "99\xa0J'aime\n"
            "Répondre\n"
        )
        parsed = parse_comments_from_dom_text(sample)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["like_count"], 1234)


class SuggestionsGraphqlTest(unittest.TestCase):
    def test_ingest_usernames(self) -> None:
        sample = json.dumps(
            {
                "data": {
                    "user": {
                        "username": "seed_one",
                        "edge_suggested_users": {
                            "edges": [
                                {"node": {"username": "suggest_a"}},
                                {"node": {"username": "suggest_b"}},
                            ]
                        },
                    }
                }
            }
        )
        suggested: list[str] = []
        seen: set[str] = set()
        _ingest_suggestion_usernames_from_graphql_text(
            sample, "seed_one", seen, suggested
        )
        self.assertEqual(suggested, ["suggest_a", "suggest_b"])


class ProfileGraphqlTest(unittest.TestCase):
    def test_merge_profile_fields(self) -> None:
        sample = json.dumps(
            {
                "data": {
                    "user": {
                        "username": "recrutestagiaire",
                        "full_name": "Test User",
                        "biography": "Bio test",
                        "follower_count": 48000,
                        "following_count": 120,
                        "media_count": 42,
                        "is_private": False,
                    }
                }
            }
        )
        profile_data: dict = {}
        _merge_profile_graphql_text(sample, "recrutestagiaire", profile_data)
        _walk_profile_graphql(json.loads(sample), "recrutestagiaire", profile_data)
        self.assertEqual(profile_data.get("followers"), 48000)
        self.assertEqual(profile_data.get("posts_count"), 42)


class ReelsGridScrollTest(unittest.TestCase):
    def test_rows_needed(self) -> None:
        self.assertEqual(_reels_grid_rows_needed(20), 4)
        self.assertEqual(_reels_grid_rows_needed(5), 1)


class GraphqlMetricsTest(unittest.TestCase):
    def test_metrics_by_shortcode(self) -> None:
        sample = json.dumps(
            {
                "items": [
                    {
                        "pk": "3893326453836395926",
                        "code": "DYcbkPMM1cR",
                        "view_count": 87300,
                        "like_count": 1063,
                        "comment_count": 12,
                        "caption": {"text": "Premier reel caption"},
                    },
                    {
                        "pk": "3893326453836395927",
                        "code": "DYaMNJqMckX",
                        "play_count": 171000,
                        "like_count": 748,
                        "caption_text": "Deuxième via caption_text",
                    },
                ]
            }
        )
        by_pk: dict = {}
        by_code: dict = {}
        _ingest_metrics_from_graphql_text(sample, by_pk, by_code)
        _walk_graphql_metrics(json.loads(sample), by_pk, by_code)

        m1 = _metrics_for_dom_media_id("DYcbkPMM1cR", by_pk, by_code)
        m2 = _metrics_for_dom_media_id("DYaMNJqMckX", by_pk, by_code)
        self.assertEqual(m1["view_count"], 87300)
        self.assertEqual(m1["like_count"], 1063)
        self.assertEqual(m2["view_count"], 171000)
        self.assertEqual(m2["like_count"], 748)
        self.assertEqual(
            by_code.get("DYcbkPMM1cR", {}).get("caption"), "Premier reel caption"
        )
        self.assertEqual(
            by_code.get("DYaMNJqMckX", {}).get("caption"),
            "Deuxième via caption_text",
        )


class GraphqlPinnedTest(unittest.TestCase):
    def test_pinned_flag(self) -> None:
        sample = json.dumps(
            {
                "items": [
                    {
                        "code": "DSDvH57CDnT",
                        "view_count": 12000,
                        "clips_tab_pinned_user_ids": ["17841400000000000"],
                    },
                    {
                        "code": "UNPINNED01",
                        "view_count": 5000,
                        "clips_tab_pinned_user_ids": [],
                    },
                ]
            }
        )
        by_pk: dict = {}
        by_code: dict = {}
        _ingest_metrics_from_graphql_text(sample, by_pk, by_code)
        _walk_graphql_metrics(json.loads(sample), by_pk, by_code)

        self.assertTrue(by_code.get("DSDvH57CDnT", {}).get("is_pinned"))
        self.assertFalse(by_code.get("UNPINNED01", {}).get("is_pinned"))


class ReelCommentPostUrlsTest(unittest.TestCase):
    def test_prefers_classic_post_url(self) -> None:
        urls = _reel_comment_post_urls("ABC123")
        self.assertEqual(urls[0], "https://www.instagram.com/p/ABC123/")
        self.assertIn("/reel/ABC123/", urls[1])


class FilterReelsForScrapeTest(unittest.TestCase):
    def test_filter_by_comment_count_and_sort(self) -> None:
        candidates = [
            {"media_id": "a", "comment_count": 5},
            {"media_id": "b", "comment_count": 100},
            {"media_id": "c", "comment_count": 50},
            {"media_id": "d", "comment_count": 0},
        ]
        out = _filter_reels_for_comment_scrape(
            candidates, max_reels=2, min_comment_count=20
        )
        self.assertEqual([r["media_id"] for r in out], ["b", "c"])


class ClipsMediaBucketTest(unittest.TestCase):
    def test_product_type_and_view_count(self) -> None:
        self.assertTrue(_is_clips_media_bucket({"product_type": "clips"}))
        self.assertFalse(
            _is_clips_media_bucket({"product_type": "carousel_container"})
        )
        self.assertFalse(_is_clips_media_bucket({"product_type": "feed"}))
        self.assertTrue(_is_clips_media_bucket({"view_count": 1200}))
        self.assertFalse(_is_clips_media_bucket({}))


class ClipsApiMediaTest(unittest.TestCase):
    def test_reel_from_clips_api_media(self) -> None:
        media = {
            "code": "DZSt2YhtEJE",
            "product_type": "clips",
            "play_count": 12000,
            "like_count": 900,
            "comment_count": 12,
            "taken_at": 1780852042,
            "clips_tab_pinned_user_ids": [],
            "caption": {"text": "hello reel"},
            "user": {"username": "marrant_club"},
            "image_versions2": {"candidates": [{"url": "https://example.com/t.jpg"}]},
        }
        reel = _reel_from_clips_api_media(media, expected_owner="marrant_club")
        assert reel is not None
        self.assertEqual(reel["media_id"], "DZSt2YhtEJE")
        self.assertEqual(reel["owner_username"], "marrant_club")
        self.assertEqual(reel["view_count"], 12000)
        self.assertEqual(reel["row_source"], "api_clips")


if __name__ == "__main__":
    unittest.main()
