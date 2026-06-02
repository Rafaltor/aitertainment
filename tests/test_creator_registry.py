"""Tests creator_registry."""

import unittest

from modules.creator_registry import resolve_creator_fields


class CreatorRegistryTest(unittest.TestCase):
    def test_unknown_creator_needs_embed(self) -> None:
        meta = resolve_creator_fields("compte_inconnu_xyz", index={})
        self.assertTrue(meta["needs_embed"])
        self.assertEqual(meta["niches"], ["humour"])

    def test_vector_store_list_format(self) -> None:
        idx = {
            "foo": {
                "username": "foo",
                "niches": ["humour"],
                "t_type": "T2b",
                "has_vector": False,
            }
        }
        # Simule merge vector_store (liste) comme dans build_creator_index
        for entry in [{"username": "foo"}, {"username": "bar"}]:
            key = str(entry["username"]).lower()
            base = dict(idx.get(key, {"username": key, "niches": []}))
            base["has_vector"] = True
            idx[key] = base
        meta = resolve_creator_fields("foo", index=idx)
        self.assertFalse(meta["needs_embed"])
        meta_bar = resolve_creator_fields("bar", index=idx)
        self.assertFalse(meta_bar["needs_embed"])


if __name__ == "__main__":
    unittest.main()
