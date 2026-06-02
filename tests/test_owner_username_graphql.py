"""Tests extraction username créateur depuis GraphQL."""

import unittest

from scripts.instagram_browser import (
    _extract_owner_username_from_graphql_window,
    _merge_owner_username_into_bucket,
)


class OwnerUsernameGraphqlTest(unittest.TestCase):
    def test_extracts_owner_from_user_block(self) -> None:
        window = (
            '"code":"ABC12345","user":{"username":"real_creator","id":"123"},'
            '"like_count":5000'
        )
        self.assertEqual(_extract_owner_username_from_graphql_window(window), "real_creator")

    def test_merge_skips_reserved(self) -> None:
        bucket: dict = {}
        _merge_owner_username_into_bucket(bucket, "reels")
        self.assertNotIn("username", bucket)
        _merge_owner_username_into_bucket(bucket, "humour_page")
        self.assertEqual(bucket["username"], "humour_page")


if __name__ == "__main__":
    unittest.main()
