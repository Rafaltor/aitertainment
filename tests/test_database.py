"""Tests database.py — persistance + tier + rescore (pas de réseau)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import database


def _score_result(
    *,
    username: str = "creator_a",
    score: float = 500.0,
    score_reels: float = 600.0,
    score_posts: float = 350.0,
    domain: str = "humour",
    niches: list[str] | None = None,
    followers: int = 48_000,
    t_type_dominant: str | None = "T2",
    scored_at: str | None = "2026-05-08T12:00:00",
    reel_ratio_median: float = 6.57,
    reel_ratio_p90: float = 32.7,
    reel_engagement_median: float = 0.068,
    reel_trend: str = "stable",
    posting_rhythm: float = 1.59,
    t_type_distribution: dict[str, float] | None = None,
    platform: str = "instagram",
) -> dict[str, Any]:
    # Schéma 2026-05 : ``score_profile`` produit ``niches`` (liste) — on
    # aligne la fixture pour refléter la production. Si un test veut
    # explicitement tester le fallback string ``niche`` ou l'absence de
    # niches, il peut passer ``niches=[]`` puis ajouter le champ voulu.
    out: dict[str, Any] = {
        "username": username,
        "domain": domain,
        "niches": list(niches) if niches is not None else ["humour"],
        "platform": platform,
        "followers": followers,
        "score": score,
        "score_reels": score_reels,
        "score_posts": score_posts,
        "reel_ratio_median": reel_ratio_median,
        "reel_ratio_p90": reel_ratio_p90,
        "reel_engagement_median": reel_engagement_median,
        "reel_trend": reel_trend,
        "posting_rhythm": posting_rhythm,
        "t_type_dominant": t_type_dominant,
        "t_type_distribution": t_type_distribution or {"T2": 0.7, "T3b": 0.3},
        "scored_at": scored_at,
    }
    return out


class ComputeTierTest(unittest.TestCase):
    def test_score_above_700_is_A(self) -> None:
        self.assertEqual(database.compute_tier(701), "A")
        self.assertEqual(database.compute_tier(925), "A")

    def test_score_400_to_700_inclusive_is_B(self) -> None:
        self.assertEqual(database.compute_tier(400), "B")
        self.assertEqual(database.compute_tier(550), "B")
        self.assertEqual(database.compute_tier(700), "B")

    def test_score_below_400_is_C(self) -> None:
        self.assertEqual(database.compute_tier(399), "C")
        self.assertEqual(database.compute_tier(0), "C")
        self.assertEqual(database.compute_tier(-10), "C")


class ComputeNextRescoreTest(unittest.TestCase):
    def test_tier_A_in_seven_days(self) -> None:
        anchor = datetime(2026, 5, 8, 12, 0, 0)
        out = database.compute_next_rescore_at("A", anchor=anchor)
        self.assertEqual(out, "2026-05-15T12:00:00")

    def test_tier_B_in_thirty_days(self) -> None:
        anchor = datetime(2026, 5, 8, 12, 0, 0)
        out = database.compute_next_rescore_at("B", anchor=anchor)
        self.assertEqual(out, "2026-06-07T12:00:00")

    def test_tier_C_returns_none(self) -> None:
        self.assertIsNone(database.compute_next_rescore_at("C"))

    def test_invalid_tier_raises(self) -> None:
        with self.assertRaises(database.DatabaseIOError):
            database.compute_next_rescore_at("D")


class LoadSaveTest(unittest.TestCase):
    def test_load_missing_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "missing.json"
            data = database.load_db(path=p)
            self.assertEqual(data, {"profiles": {}})

    def test_save_then_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "database.json"
            db = {"profiles": {"alice": {"tier": "B", "scores_history": []}}}
            database.save_db(db, path=p)
            self.assertTrue(p.exists())
            re = database.load_db(path=p)
            self.assertEqual(re["profiles"]["alice"]["tier"], "B")

    def test_save_writes_indented_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "database.json"
            database.save_db({"profiles": {}}, path=p)
            raw = p.read_text(encoding="utf-8")
            self.assertIn("\n", raw)

    def test_load_rejects_non_dict_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "database.json"
            p.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaises(database.DatabaseIOError):
                database.load_db(path=p)

    def test_load_rejects_missing_profiles_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "database.json"
            p.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
            with self.assertRaises(database.DatabaseIOError):
                database.load_db(path=p)

    def test_load_rejects_profiles_not_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "database.json"
            p.write_text(json.dumps({"profiles": []}), encoding="utf-8")
            with self.assertRaises(database.DatabaseIOError):
                database.load_db(path=p)

    def test_save_rejects_invalid_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "database.json"
            with self.assertRaises(database.DatabaseIOError):
                database.save_db({"foo": "bar"}, path=p)  # type: ignore[arg-type]


class UpsertProfileTest(unittest.TestCase):
    def test_insert_tier_B_sets_all_fields(self) -> None:
        db: dict[str, Any] = {"profiles": {}}
        result = _score_result(score=500.0)
        profile = database.upsert_profile(db, result, added_via="discovery")

        self.assertIn("creator_a", db["profiles"])
        self.assertEqual(profile["platform"], "instagram")
        self.assertEqual(profile["followers"], 48_000)
        # Schéma 2026-05 : ``niches`` (liste) à la place de ``niche`` (string).
        # Avec un score_result legacy (juste ``domain``), la cascade de
        # fallback produit ``["humour"]``.
        self.assertEqual(profile["niches"], ["humour"])
        self.assertNotIn("niche", profile)
        self.assertEqual(profile["tier"], "B")
        self.assertFalse(profile["validated"])
        self.assertEqual(profile["t_type_original"], "T2")
        self.assertIsNone(profile["t_type_final"])
        self.assertEqual(profile["added_via"], "discovery")
        self.assertEqual(profile["added_at"], "2026-05-08T12:00:00")
        self.assertEqual(profile["last_scored_at"], "2026-05-08T12:00:00")
        # Tier B → +30 jours
        self.assertEqual(profile["next_rescore_at"], "2026-06-07T12:00:00")
        self.assertFalse(profile["archived"])
        self.assertEqual(len(profile["scores_history"]), 1)

    def test_insert_tier_A_when_score_above_700(self) -> None:
        db: dict[str, Any] = {"profiles": {}}
        profile = database.upsert_profile(
            db,
            _score_result(score=850.0),
            added_via="discovery",
        )
        self.assertEqual(profile["tier"], "A")
        self.assertEqual(profile["next_rescore_at"], "2026-05-15T12:00:00")
        self.assertFalse(profile["archived"])

    def test_insert_tier_C_archives_and_no_next_rescore(self) -> None:
        db: dict[str, Any] = {"profiles": {}}
        profile = database.upsert_profile(
            db,
            _score_result(score=350.0),
            added_via="discovery",
        )
        self.assertEqual(profile["tier"], "C")
        self.assertTrue(profile["archived"])
        self.assertIsNone(profile["next_rescore_at"])

    def test_history_entry_contains_all_metric_fields(self) -> None:
        db: dict[str, Any] = {"profiles": {}}
        profile = database.upsert_profile(
            db, _score_result(score=500.0), added_via="seed"
        )
        entry = profile["scores_history"][0]
        self.assertEqual(entry["date"], "2026-05-08T12:00:00")
        self.assertEqual(entry["score"], 500.0)
        self.assertEqual(entry["score_reels"], 600.0)
        self.assertEqual(entry["score_posts"], 350.0)
        self.assertEqual(entry["reel_ratio_median"], 6.57)
        self.assertEqual(entry["reel_ratio_p90"], 32.7)
        self.assertEqual(entry["reel_engagement_median"], 0.068)
        self.assertEqual(entry["reel_trend"], "stable")
        self.assertEqual(entry["posting_rhythm"], 1.59)
        self.assertEqual(entry["t_type_dominant"], "T2")
        self.assertEqual(entry["t_type_distribution"], {"T2": 0.7, "T3b": 0.3})

    def test_update_appends_history_and_preserves_added_metadata(self) -> None:
        db: dict[str, Any] = {"profiles": {}}
        database.upsert_profile(
            db,
            _score_result(score=500.0, scored_at="2026-05-01T12:00:00"),
            added_via="seed",
        )
        # Validation humaine entre les deux scores.
        database.validate_profile(db, "creator_a", t_type_final="T2")

        profile = database.upsert_profile(
            db,
            _score_result(score=750.0, scored_at="2026-05-08T12:00:00"),
            added_via="discovery",  # Doit être ignoré (déjà ajouté via "seed").
        )

        self.assertEqual(profile["added_at"], "2026-05-01T12:00:00")
        self.assertEqual(profile["added_via"], "seed")
        self.assertTrue(profile["validated"])
        self.assertEqual(profile["t_type_final"], "T2")
        self.assertEqual(profile["t_type_original"], "T2")
        self.assertEqual(profile["last_scored_at"], "2026-05-08T12:00:00")
        self.assertEqual(profile["tier"], "A")
        self.assertEqual(profile["next_rescore_at"], "2026-05-15T12:00:00")
        self.assertEqual(len(profile["scores_history"]), 2)
        self.assertEqual(profile["scores_history"][0]["date"], "2026-05-01T12:00:00")
        self.assertEqual(profile["scores_history"][1]["date"], "2026-05-08T12:00:00")

    def test_username_normalization_strips_at_and_lowercases(self) -> None:
        db: dict[str, Any] = {"profiles": {}}
        database.upsert_profile(
            db, _score_result(username="@Creator_A"), added_via="manual"
        )
        self.assertIn("creator_a", db["profiles"])
        self.assertNotIn("@Creator_A", db["profiles"])

    def test_missing_score_raises(self) -> None:
        db: dict[str, Any] = {"profiles": {}}
        bad = _score_result()
        del bad["score"]
        with self.assertRaises(database.DatabaseIOError):
            database.upsert_profile(db, bad, added_via="discovery")

    # ------------------------------------------------------------------
    # Schéma niches : ``niches`` (liste) seule source de vérité
    # ------------------------------------------------------------------

    def test_rule1_niches_list_from_score_result_is_persisted_as_is(self) -> None:
        """``score_result["niches"]`` (liste) → écrit tel quel."""
        db: dict[str, Any] = {"profiles": {}}
        result = _score_result(niches=["humour", "sketch", "imitation"])
        profile = database.upsert_profile(db, result, added_via="discovery")
        self.assertEqual(profile["niches"], ["humour", "sketch", "imitation"])
        self.assertNotIn("niche", profile)

    def test_existing_profile_niches_refreshed_from_score_result(self) -> None:
        """Profil existant : ``niches`` rafraîchi par le scoring courant, métadonnées préservées."""
        db: dict[str, Any] = {
            "profiles": {
                "creator_a": {
                    "platform": "instagram",
                    "followers": 48_000,
                    "niches": ["humour"],
                    "tier": "B",
                    "validated": True,
                    "t_type_original": "T2",
                    "t_type_final": "T2",
                    "added_via": "seed",
                    "added_at": "2026-04-01T00:00:00",
                    "last_scored_at": "2026-04-01T00:00:00",
                    "next_rescore_at": "2026-05-01T00:00:00",
                    "archived": False,
                    "scores_history": [],
                }
            }
        }
        result = _score_result(niches=["humour", "sketch"])
        profile = database.upsert_profile(db, result, added_via="rescore")
        self.assertEqual(profile["niches"], ["humour", "sketch"])
        # Métadonnées préservées (validated, t_type_final, added_at, ...).
        self.assertTrue(profile["validated"])
        self.assertEqual(profile["t_type_final"], "T2")
        self.assertEqual(profile["added_at"], "2026-04-01T00:00:00")

    def test_incoming_niches_refresh_existing_profile(self) -> None:
        """Si le scoring le plus récent porte des niches différentes, on rafraîchit."""
        db: dict[str, Any] = {"profiles": {}}
        database.upsert_profile(
            db, _score_result(niches=["humour"]), added_via="discovery"
        )
        profile = database.upsert_profile(
            db,
            _score_result(
                niches=["humour", "sketch", "réaction"],
                scored_at="2026-05-15T12:00:00",
            ),
            added_via="rescore",
        )
        self.assertEqual(profile["niches"], ["humour", "sketch", "réaction"])

    def test_empty_incoming_niches_does_not_overwrite_existing(self) -> None:
        """Si ``incoming`` est vide on **conserve** les niches existantes —
        un upsert sans niches ne doit pas effacer la donnée précédente.
        """
        db: dict[str, Any] = {"profiles": {}}
        database.upsert_profile(
            db, _score_result(niches=["humour", "sketch"]), added_via="discovery"
        )

        # Score_result minimal sans aucun champ niche.
        result = _score_result(scored_at="2026-05-15T12:00:00")
        del result["niches"]
        profile = database.upsert_profile(db, result, added_via="rescore")
        self.assertEqual(profile["niches"], ["humour", "sketch"])

    def test_clean_niches_list_strips_and_filters_garbage(self) -> None:
        """Les items non-string / vides / whitespace dans ``niches`` sont filtrés."""
        db: dict[str, Any] = {"profiles": {}}
        # On bypass la fixture pour passer des niches volontairement bruyantes.
        result = _score_result()
        result["niches"] = ["  humour  ", "", None, 42, "sketch"]
        profile = database.upsert_profile(db, result, added_via="discovery")
        self.assertEqual(profile["niches"], ["humour", "sketch"])

    def test_missing_username_raises(self) -> None:
        db: dict[str, Any] = {"profiles": {}}
        bad = _score_result()
        bad["username"] = ""
        with self.assertRaises(database.DatabaseIOError):
            database.upsert_profile(db, bad, added_via="discovery")


class GetProfilesDueForRescoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 6, 1, 12, 0, 0)
        self.db: dict[str, Any] = {
            "profiles": {
                "due_a": {
                    "tier": "A",
                    "archived": False,
                    "next_rescore_at": "2026-05-30T12:00:00",
                    "scores_history": [],
                },
                "future_b": {
                    "tier": "B",
                    "archived": False,
                    "next_rescore_at": "2026-06-30T12:00:00",
                    "scores_history": [],
                },
                "archived_c": {
                    "tier": "C",
                    "archived": True,
                    "next_rescore_at": None,
                    "scores_history": [],
                },
                "due_b_exactly_now": {
                    "tier": "B",
                    "archived": False,
                    "next_rescore_at": "2026-06-01T12:00:00",
                    "scores_history": [],
                },
                "no_next_rescore": {
                    "tier": "B",
                    "archived": False,
                    "next_rescore_at": None,
                    "scores_history": [],
                },
            }
        }

    def test_returns_only_due_and_active(self) -> None:
        out = database.get_profiles_due_for_rescore(self.db, now=self.now)
        usernames = {p["username"] for p in out}
        self.assertEqual(usernames, {"due_a", "due_b_exactly_now"})

    def test_returned_dicts_include_username(self) -> None:
        out = database.get_profiles_due_for_rescore(self.db, now=self.now)
        for p in out:
            self.assertIn("username", p)
            self.assertIsInstance(p["username"], str)

    def test_returned_dicts_are_copies(self) -> None:
        """Muter le retour ne doit pas altérer la db."""
        out = database.get_profiles_due_for_rescore(self.db, now=self.now)
        for p in out:
            p["tier"] = "Z"
        for username, profile in self.db["profiles"].items():
            self.assertNotEqual(profile.get("tier"), "Z", msg=username)

    def test_empty_db(self) -> None:
        self.assertEqual(
            database.get_profiles_due_for_rescore({"profiles": {}}, now=self.now),
            [],
        )


class PromoteTierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db: dict[str, Any] = {"profiles": {}}
        database.upsert_profile(
            self.db,
            _score_result(score=350.0, scored_at="2026-05-08T12:00:00"),
            added_via="manual",
        )

    def test_promote_C_to_A_unarchives_and_sets_next_rescore(self) -> None:
        profile = database.promote_tier(self.db, "creator_a", "A")
        self.assertEqual(profile["tier"], "A")
        self.assertFalse(profile["archived"])
        self.assertEqual(profile["next_rescore_at"], "2026-05-15T12:00:00")

    def test_promote_to_B_unarchives_and_uses_30d(self) -> None:
        profile = database.promote_tier(self.db, "creator_a", "B")
        self.assertEqual(profile["tier"], "B")
        self.assertFalse(profile["archived"])
        self.assertEqual(profile["next_rescore_at"], "2026-06-07T12:00:00")

    def test_promote_to_C_archives_and_clears_next_rescore(self) -> None:
        # On part d'un tier B forcé.
        database.promote_tier(self.db, "creator_a", "B")
        profile = database.promote_tier(self.db, "creator_a", "C")
        self.assertEqual(profile["tier"], "C")
        self.assertTrue(profile["archived"])
        self.assertIsNone(profile["next_rescore_at"])

    def test_invalid_tier_raises(self) -> None:
        with self.assertRaises(database.DatabaseIOError):
            database.promote_tier(self.db, "creator_a", "Z")

    def test_unknown_username_raises(self) -> None:
        with self.assertRaises(database.DatabaseIOError):
            database.promote_tier(self.db, "ghost", "A")

    def test_username_normalization(self) -> None:
        profile = database.promote_tier(self.db, "@Creator_A", "A")
        self.assertEqual(profile["tier"], "A")


class ArchiveProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db: dict[str, Any] = {"profiles": {}}
        database.upsert_profile(
            self.db,
            _score_result(score=850.0),
            added_via="discovery",
        )

    def test_archive_sets_flags(self) -> None:
        profile = database.archive_profile(self.db, "creator_a")
        self.assertTrue(profile["archived"])
        self.assertEqual(profile["tier"], "C")
        self.assertIsNone(profile["next_rescore_at"])

    def test_archive_unknown_raises(self) -> None:
        with self.assertRaises(database.DatabaseIOError):
            database.archive_profile(self.db, "ghost")

    def test_archive_excludes_from_due(self) -> None:
        # Sans archivage le profil A est due à j+7. On l'archive → exclu.
        database.archive_profile(self.db, "creator_a")
        far_future = datetime(2099, 1, 1)
        self.assertEqual(
            database.get_profiles_due_for_rescore(self.db, now=far_future),
            [],
        )


class ValidateProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db: dict[str, Any] = {"profiles": {}}
        database.upsert_profile(
            self.db,
            _score_result(score=500.0),
            added_via="discovery",
        )

    def test_validate_sets_validated_and_t_type_final(self) -> None:
        profile = database.validate_profile(self.db, "creator_a", "T3b")
        self.assertTrue(profile["validated"])
        self.assertEqual(profile["t_type_final"], "T3b")
        self.assertEqual(profile["t_type_original"], "T2")

    def test_validate_unknown_raises(self) -> None:
        with self.assertRaises(database.DatabaseIOError):
            database.validate_profile(self.db, "ghost", "T2")

    def test_validate_empty_t_type_raises(self) -> None:
        with self.assertRaises(database.DatabaseIOError):
            database.validate_profile(self.db, "creator_a", "")

    def test_validate_survives_rescore(self) -> None:
        """``upsert_profile`` ne doit pas écraser ``validated`` / ``t_type_final``."""
        database.validate_profile(self.db, "creator_a", "T3b")
        database.upsert_profile(
            self.db,
            _score_result(score=750.0, scored_at="2026-05-15T12:00:00"),
            added_via="discovery",
        )
        profile = self.db["profiles"]["creator_a"]
        self.assertTrue(profile["validated"])
        self.assertEqual(profile["t_type_final"], "T3b")


class RebuildWatchlistTest(unittest.TestCase):
    """``rebuild_watchlist`` : database = vérité, runtime préservé."""

    def setUp(self) -> None:
        self.db: dict[str, Any] = {"profiles": {}}
        database.upsert_profile(
            self.db,
            _score_result(username="scored", score=800.0, niches=["humour", "sketch"]),
            added_via="discovery",
        )
        database.validate_profile(self.db, "scored", "T3b")

    def test_refreshes_niches_and_ttype_from_db(self) -> None:
        # L'entrée watchlist porte des valeurs périmées + un curseur runtime.
        existing = [
            {
                "username": "scored",
                "platform": "instagram",
                "niches": ["stale"],
                "t_type": "T2",
                "engagement_baseline": 0.05,
                "last_post_id": "abc123",
                "added_at": "2026-05-01T00:00:00",
            }
        ]
        out = database.rebuild_watchlist(self.db, existing)
        self.assertEqual(len(out), 1)
        entry = out[0]
        # Métadonnées re-dérivées depuis database.json.
        self.assertEqual(entry["niches"], ["humour", "sketch"])
        self.assertEqual(entry["t_type"], "T3b")
        # Champs runtime préservés.
        self.assertEqual(entry["last_post_id"], "abc123")
        self.assertEqual(entry["engagement_baseline"], 0.05)
        self.assertEqual(entry["added_at"], "2026-05-01T00:00:00")

    def test_drops_archived_creator(self) -> None:
        # Re-score tier C → archived True ; la validation humaine survit mais le
        # créateur sort de la watchlist.
        database.upsert_profile(
            self.db,
            _score_result(username="scored", score=100.0, scored_at="2026-05-20T00:00:00"),
            added_via="discovery",
        )
        self.assertTrue(self.db["profiles"]["scored"]["archived"])
        out = database.rebuild_watchlist(self.db, [{"username": "scored"}])
        self.assertEqual(out, [])

    def test_keeps_creator_absent_from_db(self) -> None:
        # Validation manuelle d'un profil jamais scoré → conservé tel quel.
        existing = [
            {"username": "never_scored", "niches": ["humour"], "t_type": "T2",
             "last_post_id": "z9"}
        ]
        out = database.rebuild_watchlist(self.db, existing)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["niches"], ["humour"])
        self.assertEqual(out[0]["last_post_id"], "z9")

    def test_idempotent(self) -> None:
        existing = [{"username": "scored", "last_post_id": "k"}]
        once = database.rebuild_watchlist(self.db, existing)
        twice = database.rebuild_watchlist(self.db, once)
        self.assertEqual(once, twice)


class IntegrationTest(unittest.TestCase):
    """Pipeline réaliste : upsert → save → load → mutations → save → load."""

    def test_round_trip_with_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "database.json"
            db: dict[str, Any] = {"profiles": {}}

            database.upsert_profile(
                db,
                _score_result(
                    username="alice",
                    score=850.0,
                    scored_at="2026-05-08T12:00:00",
                ),
                added_via="discovery",
            )
            database.upsert_profile(
                db,
                _score_result(
                    username="bob",
                    score=300.0,
                    scored_at="2026-05-08T12:00:00",
                ),
                added_via="manual",
            )
            database.validate_profile(db, "alice", "T2")
            database.save_db(db, path=p)

            reloaded = database.load_db(path=p)
            self.assertIn("alice", reloaded["profiles"])
            self.assertIn("bob", reloaded["profiles"])
            self.assertTrue(reloaded["profiles"]["alice"]["validated"])
            self.assertEqual(reloaded["profiles"]["alice"]["tier"], "A")
            self.assertEqual(reloaded["profiles"]["bob"]["tier"], "C")
            self.assertTrue(reloaded["profiles"]["bob"]["archived"])

            now = datetime(2026, 5, 16, 12, 0, 0)
            due = database.get_profiles_due_for_rescore(reloaded, now=now)
            self.assertEqual([p["username"] for p in due], ["alice"])


if __name__ == "__main__":
    unittest.main()
