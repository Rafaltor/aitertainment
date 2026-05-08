"""Tests pour ``dataset_builder.py`` (collecte commentaires + pending).

Patterns utilisés :
- ``SimpleNamespace`` pour mocker les Media instagrapi (attributs).
- ``MagicMock`` pour le client (``user_id_from_username``, ``user_medias``,
  ``media_comments``).
- Fichiers JSON dans des ``tempfile.TemporaryDirectory`` pour vérifier la
  persistance atomique.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Permet d'importer le module à la racine sans dépendre du PWD du test runner.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dataset_builder as db_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc(offset_days: float = 0.0) -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
        days=offset_days
    )


def _reel(
    *,
    pk: str,
    days_ago: float,
    code: str | None = None,
    caption: str = "",
    play_count: int = 0,
    likes: int = 0,
    comments: int = 0,
    shares: int | None = None,
    is_pinned: bool = False,
    product_type: str = "clips",
    audio_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        pk=pk,
        id=pk,
        code=code or f"C{pk}",
        caption_text=caption,
        play_count=play_count,
        view_count=0,
        like_count=likes,
        comment_count=comments,
        share_count=shares,
        is_pinned=is_pinned,
        product_type=product_type,
        taken_at=_now_utc(-days_ago),
        audio_id=audio_id,
    )


def _comment(text: str, likes: int) -> SimpleNamespace:
    return SimpleNamespace(text=text, like_count=likes)


def _profile(
    *,
    followers: int = 50_000,
    niche: str = "humour",
    t_type_final: str | None = "T2",
    t_type_original: str | None = "T2",
) -> dict:
    return {
        "platform": "instagram",
        "followers": followers,
        "niche": niche,
        "tier": "B",
        "validated": True,
        "t_type_original": t_type_original,
        "t_type_final": t_type_final,
        "added_via": "discovery",
    }


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------


class HelpersTest(unittest.TestCase):
    def test_extract_hashtags(self) -> None:
        self.assertEqual(
            db_mod._extract_hashtags("Test #F1 #Monaco2026 et #_underscore"),
            ["f1", "monaco2026", "_underscore"],
        )
        self.assertEqual(db_mod._extract_hashtags(""), [])
        self.assertEqual(db_mod._extract_hashtags("aucun hashtag ici"), [])

    def test_top_comments_filters_zero_likes_and_sorts(self) -> None:
        comments = [
            _comment("a", 1),
            _comment("b", 0),  # filtré
            _comment("c", 100),
            _comment("d", 50),
            _comment("", 999),  # filtré (texte vide)
            _comment("e", 10),
        ]
        top = db_mod._top_comments(comments, t_type="T2", niche="humour", n=3)
        self.assertEqual([c["text"] for c in top], ["c", "d", "e"])
        self.assertEqual([c["likes"] for c in top], [100, 50, 10])
        # Chaque commentaire dénormalise t_type/niche pour le classifier.
        self.assertTrue(all(c["t_type"] == "T2" for c in top))
        self.assertTrue(all(c["niche"] == "humour" for c in top))

    def test_top_comments_handles_empty(self) -> None:
        self.assertEqual(db_mod._top_comments([], t_type="T1", niche="x"), [])
        self.assertEqual(
            db_mod._top_comments(None, t_type="T1", niche="x"),  # type: ignore[arg-type]
            [],
        )

    def test_reel_url_format(self) -> None:
        reel = SimpleNamespace(code="ABCxyz")
        self.assertEqual(
            db_mod._reel_url(reel), "https://www.instagram.com/reel/ABCxyz/"
        )
        self.assertIsNone(db_mod._reel_url(SimpleNamespace(code="")))

    def test_select_eligible_reels_filters_pinned_recent_and_non_clips(self) -> None:
        medias = [
            _reel(pk="ok1", days_ago=10),
            _reel(pk="too_recent", days_ago=2),  # < 7j
            _reel(pk="pinned", days_ago=30, is_pinned=True),
            _reel(pk="post", days_ago=15, product_type="feed"),
            _reel(pk="ok2", days_ago=8),
        ]
        out = db_mod._select_eligible_reels(medias)
        self.assertEqual([m.pk for m in out], ["ok1", "ok2"])

    def test_build_classifier_context_full(self) -> None:
        m = _reel(pk="x", days_ago=10, play_count=100_000,
                  likes=10_000, comments=500, shares=300)
        out = db_mod._build_classifier_context(m)
        self.assertEqual(out["views"], 100_000)
        self.assertEqual(out["likes"], 10_000)
        self.assertEqual(out["comment_count"], 500)
        self.assertEqual(out["shares"], 300)
        self.assertAlmostEqual(out["comment_to_like_ratio"], 0.05, places=4)
        self.assertAlmostEqual(out["share_to_like_ratio"], 0.03, places=4)

    def test_build_classifier_context_likes_zero_returns_none_ratios(self) -> None:
        m = _reel(pk="x", days_ago=10, likes=0, comments=10, shares=5)
        out = db_mod._build_classifier_context(m)
        self.assertEqual(out["likes"], 0)
        self.assertIsNone(out["comment_to_like_ratio"])
        # likes=0 → on ne calcule rien, share_to_like_ratio aussi None
        self.assertIsNone(out["share_to_like_ratio"])

    def test_build_classifier_context_shares_none_keeps_none(self) -> None:
        m = _reel(pk="x", days_ago=10, likes=100, comments=10, shares=None)
        out = db_mod._build_classifier_context(m)
        self.assertIsNone(out["shares"])  # None préservé (pas forcé à 0)
        self.assertIsNone(out["share_to_like_ratio"])
        # comment_to_like_ratio reste calculé puisque likes > 0.
        self.assertAlmostEqual(out["comment_to_like_ratio"], 0.1, places=4)

    def test_build_classifier_context_shares_zero_returns_none_ratio(self) -> None:
        m = _reel(pk="x", days_ago=10, likes=100, comments=10, shares=0)
        out = db_mod._build_classifier_context(m)
        self.assertEqual(out["shares"], 0)
        # shares falsy (0) → share_to_like_ratio None (cohérent avec "shares
        # falsy" du brief : un share=0 n'est pas un signal exploitable).
        self.assertIsNone(out["share_to_like_ratio"])
        self.assertAlmostEqual(out["comment_to_like_ratio"], 0.1, places=4)

    def test_build_generator_input_isolates_prod_features_only(self) -> None:
        m = _reel(pk="x", days_ago=10, caption="Top moment #F1 #monaco",
                  play_count=999_999, likes=10_000, comments=500, shares=300,
                  audio_id="AUD42")
        out = db_mod._build_generator_input(m, t_type="T2", niche="humour")
        # Schéma exact attendu — aucune feature de réception ne doit fuiter.
        self.assertEqual(set(out), {"t_type", "niche", "caption", "hashtags", "audio_id"})
        self.assertEqual(out["t_type"], "T2")
        self.assertEqual(out["niche"], "humour")
        self.assertEqual(out["caption"], "Top moment #F1 #monaco")
        self.assertEqual(out["hashtags"], ["f1", "monaco"])
        self.assertEqual(out["audio_id"], "AUD42")

    def test_media_audio_id_falls_back_through_chains(self) -> None:
        direct = SimpleNamespace(audio_id="DIR")
        self.assertEqual(db_mod._media_audio_id(direct), "DIR")

        clip = SimpleNamespace(
            audio_id=None,
            music_metadata=None,
            clips_metadata=SimpleNamespace(
                original_sound_info=SimpleNamespace(audio_asset_id="CLIP_AUD")
            ),
        )
        self.assertEqual(db_mod._media_audio_id(clip), "CLIP_AUD")

        none_path = SimpleNamespace(
            audio_id=None, music_metadata=None, clips_metadata=None
        )
        self.assertIsNone(db_mod._media_audio_id(none_path))


# ---------------------------------------------------------------------------
# collect_training_data
# ---------------------------------------------------------------------------


class CollectTrainingDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.training = self.tmp / "training_comments.json"
        self.pending = self.tmp / "pending_collection.json"
        self.addCleanup(self.tmpdir.cleanup)

    def _client(self, medias: list, comments_by_id: dict[str, list]) -> MagicMock:
        client = MagicMock()
        client.user_id_from_username.return_value = "1234567"
        client.user_medias.return_value = medias

        def _comments(media_id, amount=50):
            return comments_by_id.get(str(media_id), [])

        client.media_comments.side_effect = _comments
        return client

    def test_mock_mode_is_no_op(self) -> None:
        client = MagicMock()
        out = db_mod.collect_training_data(
            "alice", _profile(), client=client, mock=True,
            training_path=self.training,
        )
        self.assertEqual(out, [])
        client.user_id_from_username.assert_not_called()
        self.assertFalse(self.training.exists())

    def test_no_eligible_reels_returns_empty_no_write(self) -> None:
        # Tous trop récents (< 7 jours)
        medias = [_reel(pk=f"r{i}", days_ago=2 + i) for i in range(3)]
        client = self._client(medias, {})
        out = db_mod.collect_training_data(
            "alice", _profile(), client=client,
            training_path=self.training,
        )
        self.assertEqual(out, [])
        client.media_comments.assert_not_called()
        self.assertFalse(self.training.exists())

    def test_collects_eligible_reels_and_persists(self) -> None:
        medias = [
            _reel(pk="r1", days_ago=10, caption="Top moment #F1 #monaco",
                  play_count=120_000, likes=10_000, comments=300, shares=420,
                  audio_id="AUD1"),
            _reel(pk="r2", days_ago=15, caption="#funny moment",
                  play_count=80_000, likes=5_000, comments=200, shares=None),
            _reel(pk="r_recent", days_ago=2, play_count=999_999),  # filtré
            _reel(pk="r_pinned", days_ago=30, is_pinned=True),  # filtré
        ]
        comments = {
            "r1": [
                _comment("MDR ce dépassement", 250),
                _comment("Génial 👏", 80),
                _comment("nope", 0),
                _comment("", 999),
                _comment("On veut la suite !", 30),
            ],
            "r2": [
                _comment("Hilarious", 12),
                _comment("J'en peux plus", 5),
            ],
        }
        client = self._client(medias, comments)
        sleep_fn = MagicMock()

        out = db_mod.collect_training_data(
            "alice",
            _profile(followers=50_000),
            client=client,
            training_path=self.training,
            sleep_fn=sleep_fn,
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(sleep_fn.call_count, 2)  # un sleep par Reel collecté
        client.user_medias.assert_called_once_with("1234567", amount=5)

        # Persistance — schéma à deux datasets (générateur + classifier).
        store = json.loads(self.training.read_text(encoding="utf-8"))
        self.assertEqual(len(store["entries"]), 2)
        e1 = next(e for e in store["entries"] if e["media_id"] == "r1")
        self.assertEqual(e1["username"], "alice")
        self.assertTrue(e1["reel_url"].endswith("/reel/Cr1/"))
        self.assertIn("collected_at", e1)

        # generator_input : features dispo en prod (pas de fuite de réception).
        self.assertEqual(
            e1["generator_input"],
            {
                "t_type": "T2",
                "niche": "humour",
                "caption": "Top moment #F1 #monaco",
                "hashtags": ["f1", "monaco"],
                "audio_id": "AUD1",
            },
        )

        # classifier_context : compteurs bruts + ratios.
        self.assertEqual(
            e1["classifier_context"],
            {
                "views": 120_000,
                "likes": 10_000,
                "comment_count": 300,
                "shares": 420,
                "comment_to_like_ratio": 0.03,
                "share_to_like_ratio": 0.042,
            },
        )

        # top_comments : self-contained (text + likes + t_type + niche).
        self.assertEqual(
            [c["text"] for c in e1["top_comments"]],
            ["MDR ce dépassement", "Génial 👏", "On veut la suite !"],
        )
        self.assertTrue(all(c["t_type"] == "T2" for c in e1["top_comments"]))
        self.assertTrue(all(c["niche"] == "humour" for c in e1["top_comments"]))
        self.assertTrue(all(c["likes"] > 0 for c in e1["top_comments"]))

        # Anciennes clés (rétro-compat hostile) doivent **ne plus** apparaître.
        self.assertNotIn("t_type", e1)
        self.assertNotIn("niche", e1)
        self.assertNotIn("caption", e1)
        self.assertNotIn("hashtags", e1)
        self.assertNotIn("audio_id", e1)
        self.assertNotIn("reel_ratio", e1)
        self.assertNotIn("reel_metrics", e1)

        # r2 : shares=None → share_to_like_ratio doit rester None.
        e2 = next(e for e in store["entries"] if e["media_id"] == "r2")
        self.assertIsNone(e2["classifier_context"]["shares"])
        self.assertIsNone(e2["classifier_context"]["share_to_like_ratio"])
        self.assertAlmostEqual(
            e2["classifier_context"]["comment_to_like_ratio"],
            200 / 5_000,
            places=5,
        )

    def test_dedup_by_media_id_across_calls(self) -> None:
        medias = [_reel(pk="r1", days_ago=10, play_count=5_000)]
        client = self._client(medias, {"r1": [_comment("hi", 10)]})

        out1 = db_mod.collect_training_data(
            "alice", _profile(), client=client,
            training_path=self.training, sleep_fn=lambda: None,
        )
        out2 = db_mod.collect_training_data(
            "alice", _profile(), client=client,
            training_path=self.training, sleep_fn=lambda: None,
        )
        self.assertEqual(len(out1), 1)
        self.assertEqual(out2, [])  # déjà collecté
        store = json.loads(self.training.read_text(encoding="utf-8"))
        self.assertEqual(len(store["entries"]), 1)

    def test_user_medias_failure_returns_empty_no_crash(self) -> None:
        client = MagicMock()
        client.user_id_from_username.return_value = "x"
        client.user_medias.side_effect = RuntimeError("network")
        out = db_mod.collect_training_data(
            "alice", _profile(), client=client, training_path=self.training,
        )
        self.assertEqual(out, [])

    def test_media_comments_failure_skips_reel_continues(self) -> None:
        medias = [
            _reel(pk="ok", days_ago=10, play_count=10_000),
            _reel(pk="boom", days_ago=12, play_count=20_000),
        ]
        client = MagicMock()
        client.user_id_from_username.return_value = "x"
        client.user_medias.return_value = medias

        def _comments(media_id, amount=50):
            if str(media_id) == "boom":
                raise RuntimeError("rate limit")
            return [_comment("yo", 5)]

        client.media_comments.side_effect = _comments
        out = db_mod.collect_training_data(
            "alice", _profile(), client=client,
            training_path=self.training, sleep_fn=lambda: None,
        )
        self.assertEqual([e["media_id"] for e in out], ["ok"])

    def test_collection_works_without_followers(self) -> None:
        """``followers`` n'est plus utilisé par le builder (pas de ``reel_ratio``).

        La collecte doit donc fonctionner même sur un profil avec
        ``followers=0`` (cas dégénéré ou profil fraîchement scrapé).
        """
        medias = [_reel(pk="r1", days_ago=10, play_count=10_000,
                        likes=200, comments=10)]
        client = self._client(medias, {"r1": [_comment("yo", 5)]})
        out = db_mod.collect_training_data(
            "ghost",
            _profile(followers=0),
            client=client,
            training_path=self.training,
            sleep_fn=lambda: None,
        )
        self.assertEqual(len(out), 1)
        # Le builder n'expose plus de reel_ratio — feature retirée du schéma.
        self.assertNotIn("reel_ratio", out[0])
        self.assertIn("classifier_context", out[0])
        self.assertEqual(out[0]["classifier_context"]["views"], 10_000)


# ---------------------------------------------------------------------------
# schedule_pending_collection
# ---------------------------------------------------------------------------


class SchedulePendingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.pending = self.tmp / "pending_collection.json"
        self.addCleanup(self.tmpdir.cleanup)

    def test_schedules_with_seven_days_default(self) -> None:
        anchor = _now_utc().replace(microsecond=0)  # _iso tronque à la seconde
        entry = db_mod.schedule_pending_collection(
            "alice", path=self.pending, now=anchor
        )
        self.assertEqual(entry["username"], "alice")
        self.assertEqual(
            db_mod._parse_iso(entry["collect_after"]),
            anchor + timedelta(days=7),
        )
        store = json.loads(self.pending.read_text(encoding="utf-8"))
        self.assertEqual(len(store["entries"]), 1)

    def test_dedup_refreshes_existing_entry(self) -> None:
        first = db_mod.schedule_pending_collection(
            "@Alice", path=self.pending, now=_now_utc(0)
        )
        second = db_mod.schedule_pending_collection(
            "alice", path=self.pending, now=_now_utc(3)
        )
        self.assertEqual(first["username"], "alice")
        self.assertEqual(second["username"], "alice")
        store = json.loads(self.pending.read_text(encoding="utf-8"))
        # Une seule entrée pour alice (dédup par username normalisé).
        self.assertEqual(len(store["entries"]), 1)
        self.assertEqual(store["entries"][0]["collect_after"], second["collect_after"])

    def test_empty_username_raises(self) -> None:
        with self.assertRaises(db_mod.DatasetIOError):
            db_mod.schedule_pending_collection("", path=self.pending)


# ---------------------------------------------------------------------------
# check_pending_collections
# ---------------------------------------------------------------------------


class CheckPendingCollectionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.pending = self.tmp / "pending_collection.json"
        self.training = self.tmp / "training_comments.json"
        self.dbfile = self.tmp / "database.json"
        self.addCleanup(self.tmpdir.cleanup)

    def _write_db(self, profiles: dict) -> None:
        self.dbfile.write_text(
            json.dumps({"profiles": profiles}, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_pending(self, entries: list[dict]) -> None:
        self.pending.write_text(
            json.dumps({"entries": entries}, ensure_ascii=False),
            encoding="utf-8",
        )

    def test_empty_pending_returns_zero_stats(self) -> None:
        stats = db_mod.check_pending_collections(
            mock=True,
            db_path=self.dbfile,
            pending_path=self.pending,
            training_path=self.training,
        )
        self.assertEqual(stats, {"checked": 0, "collected": 0, "skipped": 0, "removed": 0})

    def test_future_entry_is_kept_and_skipped(self) -> None:
        anchor = _now_utc()
        future = db_mod._iso(anchor + timedelta(days=3))
        self._write_pending([{"username": "alice", "collect_after": future}])
        self._write_db({"alice": _profile()})
        collect_mock = MagicMock(return_value=[])

        stats = db_mod.check_pending_collections(
            db_path=self.dbfile,
            pending_path=self.pending,
            training_path=self.training,
            collect_fn=collect_mock,
            now=anchor,
        )
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(stats["checked"], 0)
        collect_mock.assert_not_called()
        store = json.loads(self.pending.read_text(encoding="utf-8"))
        self.assertEqual(len(store["entries"]), 1)

    def test_due_entry_with_collected_data_is_removed(self) -> None:
        anchor = _now_utc()
        past = db_mod._iso(anchor - timedelta(days=1))
        self._write_pending([{"username": "alice", "collect_after": past}])
        self._write_db({"alice": _profile()})
        collect_mock = MagicMock(return_value=[{"media_id": "r1"}])

        stats = db_mod.check_pending_collections(
            db_path=self.dbfile,
            pending_path=self.pending,
            training_path=self.training,
            collect_fn=collect_mock,
            now=anchor,
        )
        self.assertEqual(stats["checked"], 1)
        self.assertEqual(stats["collected"], 1)
        self.assertEqual(stats["removed"], 1)
        collect_mock.assert_called_once()
        store = json.loads(self.pending.read_text(encoding="utf-8"))
        self.assertEqual(store["entries"], [])

    def test_due_entry_with_no_data_is_rescheduled(self) -> None:
        anchor = _now_utc()
        past = db_mod._iso(anchor - timedelta(days=1))
        self._write_pending([{"username": "alice", "collect_after": past}])
        self._write_db({"alice": _profile()})
        collect_mock = MagicMock(return_value=[])  # toujours pas de Reels

        stats = db_mod.check_pending_collections(
            db_path=self.dbfile,
            pending_path=self.pending,
            training_path=self.training,
            collect_fn=collect_mock,
            now=anchor,
        )
        self.assertEqual(stats["checked"], 1)
        self.assertEqual(stats["collected"], 0)
        store = json.loads(self.pending.read_text(encoding="utf-8"))
        self.assertEqual(len(store["entries"]), 1)
        # collect_after a bien été repoussé de 7 jours par rapport à l'ancien.
        new_after = db_mod._parse_iso(store["entries"][0]["collect_after"])
        self.assertGreater(new_after, anchor)

    def test_entry_without_db_profile_is_dropped(self) -> None:
        anchor = _now_utc()
        past = db_mod._iso(anchor - timedelta(days=1))
        self._write_pending([{"username": "ghost", "collect_after": past}])
        self._write_db({})  # ghost n'existe pas
        collect_mock = MagicMock()

        stats = db_mod.check_pending_collections(
            db_path=self.dbfile,
            pending_path=self.pending,
            training_path=self.training,
            collect_fn=collect_mock,
            now=anchor,
        )
        self.assertEqual(stats["removed"], 1)
        collect_mock.assert_not_called()
        store = json.loads(self.pending.read_text(encoding="utf-8"))
        self.assertEqual(store["entries"], [])

    def test_collect_fn_exception_does_not_crash_cycle(self) -> None:
        anchor = _now_utc()
        past = db_mod._iso(anchor - timedelta(days=1))
        self._write_pending([
            {"username": "boom", "collect_after": past},
            {"username": "alice", "collect_after": past},
        ])
        self._write_db({"boom": _profile(), "alice": _profile()})
        calls = {"n": 0}

        def _collect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("network down")
            return [{"media_id": "ok1"}]

        stats = db_mod.check_pending_collections(
            db_path=self.dbfile,
            pending_path=self.pending,
            training_path=self.training,
            collect_fn=_collect,
            now=anchor,
        )
        # On a quand même traité les deux profils.
        self.assertEqual(calls["n"], 2)
        self.assertEqual(stats["collected"], 1)
        # boom est rescheduled (pas data) ; alice est removed.
        store = json.loads(self.pending.read_text(encoding="utf-8"))
        self.assertEqual(len(store["entries"]), 1)
        self.assertEqual(store["entries"][0]["username"], "boom")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTest(unittest.TestCase):
    def test_cli_mock_only_prints_paths_no_action(self) -> None:
        buf = io.StringIO()
        with patch.object(sys, "argv", ["dataset_builder.py", "--mock"]), \
                patch.object(db_mod, "check_pending_collections") as fake_check:
            with redirect_stdout(buf):
                db_mod._main_cli()
        fake_check.assert_not_called()
        self.assertIn("Mode mock", buf.getvalue())

    def test_cli_pending_calls_check(self) -> None:
        buf = io.StringIO()
        fake_stats = {"checked": 2, "collected": 1, "skipped": 0, "removed": 1}
        with patch.object(sys, "argv", ["dataset_builder.py", "--pending"]), \
                patch.object(
                    db_mod, "check_pending_collections", return_value=fake_stats
                ) as fake_check:
            with redirect_stdout(buf):
                db_mod._main_cli()
        fake_check.assert_called_once()
        out = buf.getvalue()
        self.assertIn("checked=2", out)
        self.assertIn("collected=1", out)


# ---------------------------------------------------------------------------
# Intégration : telegram bot _trigger_dataset_collection
# ---------------------------------------------------------------------------


class TelegramBotTriggerTest(unittest.TestCase):
    """Vérifie que _handle_validate / _handle_set_ttype déclenchent la collecte."""

    def setUp(self) -> None:
        # Import tardif : telegram_discovery_bot importe config qui lit les .env.
        from telegram_discovery_bot import (  # noqa: E402
            _trigger_dataset_collection,
        )
        self._trigger = _trigger_dataset_collection

        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.dbfile = self.tmp / "database.json"
        self.dbfile.write_text(
            json.dumps({"profiles": {"alice": _profile()}}, ensure_ascii=False),
            encoding="utf-8",
        )
        self.addCleanup(self.tmpdir.cleanup)

    def test_calls_collect_when_db_has_profile(self) -> None:
        with patch.object(db_mod, "collect_training_data", return_value=[{"media_id": "r1"}]) as fake_coll, \
             patch.object(db_mod, "schedule_pending_collection") as fake_sched:
            self._trigger("alice", db_path=self.dbfile)
        fake_coll.assert_called_once()
        fake_sched.assert_not_called()

    def test_schedules_pending_when_no_data_collected(self) -> None:
        with patch.object(db_mod, "collect_training_data", return_value=[]) as fake_coll, \
             patch.object(db_mod, "schedule_pending_collection") as fake_sched:
            self._trigger("alice", db_path=self.dbfile)
        fake_coll.assert_called_once()
        fake_sched.assert_called_once_with("alice")

    def test_skip_when_profile_absent(self) -> None:
        with patch.object(db_mod, "collect_training_data") as fake_coll, \
             patch.object(db_mod, "schedule_pending_collection") as fake_sched:
            self._trigger("ghost", db_path=self.dbfile)
        fake_coll.assert_not_called()
        fake_sched.assert_not_called()

    def test_collect_exception_falls_back_to_schedule(self) -> None:
        with patch.object(db_mod, "collect_training_data", side_effect=RuntimeError("boom")) as fake_coll, \
             patch.object(db_mod, "schedule_pending_collection") as fake_sched:
            self._trigger("alice", db_path=self.dbfile)
        fake_coll.assert_called_once()
        fake_sched.assert_called_once_with("alice")


# ---------------------------------------------------------------------------
# Intégration : rescore_scheduler appelle check_pending_collections
# ---------------------------------------------------------------------------


class RescoreSchedulerCallsPendingTest(unittest.TestCase):
    def setUp(self) -> None:
        import rescore_scheduler  # noqa: E402
        self.scheduler = rescore_scheduler
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.dbfile = self.tmp / "database.json"
        self.dbfile.write_text(
            json.dumps({"profiles": {}}, ensure_ascii=False), encoding="utf-8"
        )
        self.addCleanup(self.tmpdir.cleanup)

    def test_pending_called_even_when_no_due_profiles(self) -> None:
        fake_check = MagicMock(return_value={"checked": 1, "collected": 1, "skipped": 0, "removed": 1})
        stats = self.scheduler.run_rescore_cycle(
            mock=True,
            db_path=self.dbfile,
            score_fn=MagicMock(),
            notify_fn=MagicMock(),
            sleep_fn=MagicMock(),
            check_pending_fn=fake_check,
        )
        fake_check.assert_called_once()
        self.assertEqual(stats["pending_collected"], 1)

    def test_pending_called_after_due_processing(self) -> None:
        # 1 profil dû + check_pending appelé une fois en plus.
        from datetime import timedelta as _td

        past = (datetime.now(timezone.utc).replace(tzinfo=None) - _td(days=1)).isoformat(timespec="seconds")
        self.dbfile.write_text(
            json.dumps({
                "profiles": {
                    "alice": {
                        "platform": "instagram",
                        "tier": "B",
                        "archived": False,
                        "next_rescore_at": past,
                        "scores_history": [{"date": past, "score": 500}],
                    }
                }
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        score_mock = MagicMock(return_value={
            "score_result": {"username": "alice", "score": 510},
            "tier": "B",
            "profile": {},
        })
        check_mock = MagicMock(return_value={"checked": 0, "collected": 0, "skipped": 0, "removed": 0})

        stats = self.scheduler.run_rescore_cycle(
            mock=True,
            db_path=self.dbfile,
            score_fn=score_mock,
            notify_fn=MagicMock(),
            sleep_fn=MagicMock(),
            check_pending_fn=check_mock,
        )
        self.assertEqual(stats["due_count"], 1)
        score_mock.assert_called_once()
        check_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
