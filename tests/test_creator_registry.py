"""Tests pour ``modules.creator_registry``."""

from __future__ import annotations

import unittest

from modules.creator_registry import resolve_creator_fields


class ResolveCreatorFieldsTest(unittest.TestCase):
    def test_unknown_creator_defaults(self) -> None:
        meta = resolve_creator_fields("unknown", index={})
        self.assertEqual(meta["niches"], ["humour"])
        self.assertFalse(meta["known_creator"])

    def test_known_creator_from_index(self) -> None:
        index = {
            "foo": {
                "username": "foo",
                "niches": ["humour"],
                "t_type": "T2b",
            }
        }
        meta = resolve_creator_fields("foo", index=index)
        self.assertEqual(meta["t_type_profile"], "T2b")
        self.assertTrue(meta["known_creator"])
