"""Tests unitaires pour watcher : check_new_post + run_watcher (mocks)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from watcher import (
    DAY_INTERVAL_S,
    MAX_ACCOUNTS_PAUSE_S,
    NEW_POST_VIEW_THRESHOLD,
    NIGHT_INTERVAL_S,
    PRIME_INTERVAL_S,
    check_new_post,
    get_poll_interval,
    run_watcher,
)


def _reel(
    media_id: str,
    view_count: int,
    *,
    is_pinned: bool = False,
    caption: str = "",
    audio_id: str = "",
) -> dict:
    return {
        "media_id": media_id,
        "view_count": view_count,
        "like_count": 0,
        "comment_count": 0,
        "share_count": 0,
        "is_pinned": is_pinned,
        "caption": caption,
        "audio_id": audio_id,
        "thumbnail_url": "",
    }


class CheckNewPostTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = MagicMock()

    @patch("watcher.get_recent_reels")
    def test_new_post_detected(self, mock_reels: MagicMock) -> None:
        mock_reels.return_value = [
            _reel("111", 500, caption="Salut #street #mode", audio_id="music_xyz"),
        ]
        creator = {
            "username": "someone",
            "platform": "instagram",
            "last_post_id": "000",
        }
        out = check_new_post(creator, self.context)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["video_id"], "111")
        self.assertEqual(out["bootstrap"], False)
        self.assertIn("street", out["hashtags"])
        self.assertEqual(out["audio_id"], "music_xyz")
        mock_reels.assert_called_once_with("someone", self.context, max_reels=4)

    @patch("watcher.get_recent_reels")
    def test_same_post_returns_none(self, mock_reels: MagicMock) -> None:
        mock_reels.return_value = [_reel("999", 100)]
        creator = {"username": "u", "platform": "instagram", "last_post_id": "999"}
        self.assertIsNone(check_new_post(creator, self.context))

    @patch("watcher.get_recent_reels")
    def test_bootstrap(self, mock_reels: MagicMock) -> None:
        mock_reels.return_value = [_reel("777", 800)]
        creator = {"username": "u", "platform": "instagram", "last_post_id": None}
        out = check_new_post(creator, self.context)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertTrue(out["bootstrap"])

    @patch("watcher.get_recent_reels")
    def test_no_reels_returns_none(self, mock_reels: MagicMock) -> None:
        mock_reels.return_value = []
        creator = {"username": "ghost", "platform": "instagram", "last_post_id": None}
        self.assertIsNone(check_new_post(creator, self.context))

    @patch("watcher.get_recent_reels")
    def test_tiktok_skipped(self, mock_reels: MagicMock) -> None:
        creator = {"username": "x", "platform": "tiktok", "last_post_id": None}
        self.assertIsNone(check_new_post(creator, self.context))
        mock_reels.assert_not_called()

    @patch("watcher.get_recent_reels")
    def test_pinned_reels_skipped(self, mock_reels: MagicMock) -> None:
        pinned_only = [_reel("pinned", 50, is_pinned=True)]
        mock_reels.side_effect = [pinned_only, pinned_only]
        creator = {"username": "u", "platform": "instagram", "last_post_id": None}
        with self.assertLogs("aitertainment", level="INFO") as logs:
            self.assertIsNone(check_new_post(creator, self.context))
        self.assertEqual(mock_reels.call_count, 2)
        mock_reels.assert_any_call("u", self.context, max_reels=4)
        mock_reels.assert_any_call("u", self.context, max_reels=8)
        self.assertTrue(
            any("aucun reel non épinglé trouvé" in msg for msg in logs.output)
        )

    @patch("watcher.get_recent_reels")
    def test_pinned_retry_finds_non_pinned(self, mock_reels: MagicMock) -> None:
        mock_reels.side_effect = [
            [_reel("p1", 10, is_pinned=True), _reel("p2", 20, is_pinned=True)],
            [_reel("p1", 10, is_pinned=True), _reel("fresh", 400)],
        ]
        creator = {"username": "u", "platform": "instagram", "last_post_id": "old"}
        out = check_new_post(creator, self.context)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["video_id"], "fresh")
        self.assertEqual(mock_reels.call_count, 2)

    @patch("watcher.get_recent_reels")
    def test_high_views_returns_none(self, mock_reels: MagicMock) -> None:
        mock_reels.return_value = [_reel("big", NEW_POST_VIEW_THRESHOLD + 1)]
        creator = {"username": "u", "platform": "instagram", "last_post_id": "old"}
        self.assertIsNone(check_new_post(creator, self.context))

    @patch("watcher.get_recent_reels")
    def test_skips_pinned_uses_first_non_pinned(self, mock_reels: MagicMock) -> None:
        mock_reels.return_value = [
            _reel("pinned", 10, is_pinned=True),
            _reel("fresh", 400, caption="new #drop"),
        ]
        creator = {"username": "u", "platform": "instagram", "last_post_id": "old"}
        out = check_new_post(creator, self.context)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["video_id"], "fresh")


class GetPollIntervalTest(unittest.TestCase):
    def _at(self, hour: int) -> datetime:
        return datetime(2026, 5, 7, hour, 30, 0)

    def test_prime_time_evening(self) -> None:
        self.assertEqual(get_poll_interval(self._at(17)), PRIME_INTERVAL_S)
        self.assertEqual(get_poll_interval(self._at(20)), PRIME_INTERVAL_S)

    def test_day_hours(self) -> None:
        self.assertEqual(get_poll_interval(self._at(9)), DAY_INTERVAL_S)
        self.assertEqual(get_poll_interval(self._at(13)), DAY_INTERVAL_S)
        self.assertEqual(get_poll_interval(self._at(16)), DAY_INTERVAL_S)

    def test_night(self) -> None:
        self.assertEqual(get_poll_interval(self._at(0)), NIGHT_INTERVAL_S)
        self.assertEqual(get_poll_interval(self._at(8)), NIGHT_INTERVAL_S)
        self.assertEqual(get_poll_interval(self._at(21)), NIGHT_INTERVAL_S)
        self.assertEqual(get_poll_interval(self._at(23)), NIGHT_INTERVAL_S)


class RunWatcherMockTest(unittest.TestCase):
    """run_watcher en mode --mock : un cycle, pas d'IO réseau, pas d'écriture."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.wl_path = Path(self.tmp.name) / "watchlist.json"
        self.wl_path.write_text(
            json.dumps(
                {
                    "creators": [
                        {
                            "username": "creator_a",
                            "platform": "instagram",
                            "niche": "streetwear",
                            "t_type": "T2",
                            "engagement_baseline": 0.05,
                            "last_post_id": "old_id_a",
                            "added_at": "2026-04-01T00:00:00",
                        },
                        {
                            "username": "creator_b",
                            "platform": "instagram",
                            "niche": "lifestyle",
                            "t_type": None,  # Discovery pas encore passé
                            "engagement_baseline": None,
                            "last_post_id": None,
                            "added_at": "2026-04-01T00:00:00",
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @patch("watcher.notify_new_post")
    @patch("watcher._generate_for_post")
    @patch("watcher.check_new_post")
    @patch("watcher.time.sleep")
    def test_mock_skips_io_and_does_not_persist(
        self,
        mock_sleep: MagicMock,
        mock_check: MagicMock,
        mock_gen: MagicMock,
        mock_notify: MagicMock,
    ) -> None:
        run_watcher(watchlist_path=self.wl_path, mock=True)

        # Aucun appel réseau réel
        mock_check.assert_not_called()
        mock_gen.assert_not_called()
        mock_notify.assert_not_called()

        # Pas d'écriture sur disque en mode mock
        on_disk = json.loads(self.wl_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["creators"][0]["last_post_id"], "old_id_a")
        self.assertIsNone(on_disk["creators"][1]["last_post_id"])

        # sleep(0) en mock pour ne pas bloquer
        for call in mock_sleep.call_args_list:
            self.assertEqual(call.args[0], 0)


class RunWatcherRealCycleTest(unittest.TestCase):
    """run_watcher hors mock : 1 cycle avec check_new_post mocké."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.wl_path = Path(self.tmp.name) / "watchlist.json"
        self.wl_path.write_text(
            json.dumps(
                {
                    "creators": [
                        {
                            "username": "atelier_xyz",
                            "platform": "instagram",
                            "niche": "streetwear",
                            "t_type": "T2",
                            "engagement_baseline": 0.05,
                            "last_post_id": "old_post_id",
                            "added_at": "2026-04-01T00:00:00",
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @patch("watcher.get_browser_context")
    @patch("watcher.sync_playwright")
    @patch("watcher.notify_new_post")
    @patch("watcher._generate_for_post")
    @patch("watcher.check_new_post")
    @patch("watcher.time.sleep")
    def test_one_cycle_persists_new_post_id(
        self,
        mock_sleep: MagicMock,
        mock_check: MagicMock,
        mock_gen: MagicMock,
        mock_notify: MagicMock,
        mock_pw: MagicMock,
        mock_ctx: MagicMock,
    ) -> None:
        mock_pw.return_value.start.return_value = MagicMock()
        mock_ctx.return_value = MagicMock()
        mock_check.return_value = {
            "video_id": "fresh_post_id",
            "caption": "drop",
            "hashtags": ["fitcheck"],
            "audio_id": "snd_42",
            "url": "https://www.instagram.com/reel/abc/",
            "posted_at": datetime(2026, 5, 7, 17, 0, tzinfo=timezone.utc),
            "bootstrap": False,
        }
        mock_gen.return_value = ["c1", "c2", "c3"]
        mock_notify.return_value = True

        run_watcher(watchlist_path=self.wl_path, mock=False, max_cycles=1)

        mock_check.assert_called_once()
        mock_gen.assert_called_once()
        mock_notify.assert_called_once()

        on_disk = json.loads(self.wl_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["creators"][0]["last_post_id"], "fresh_post_id")

    @patch("watcher.get_browser_context")
    @patch("watcher.sync_playwright")
    @patch("watcher.notify_new_post")
    @patch("watcher._generate_for_post")
    @patch("watcher.check_new_post")
    @patch("watcher.time.sleep")
    def test_bootstrap_persists_without_notify(
        self,
        mock_sleep: MagicMock,
        mock_check: MagicMock,
        mock_gen: MagicMock,
        mock_notify: MagicMock,
        mock_pw: MagicMock,
        mock_ctx: MagicMock,
    ) -> None:
        mock_pw.return_value.start.return_value = MagicMock()
        mock_ctx.return_value = MagicMock()
        # créateur en bootstrap
        self.wl_path.write_text(
            json.dumps(
                {
                    "creators": [
                        {
                            "username": "newbie",
                            "platform": "instagram",
                            "niche": "streetwear",
                            "t_type": "T2",
                            "engagement_baseline": 0.05,
                            "last_post_id": None,
                            "added_at": "2026-04-01T00:00:00",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        mock_check.return_value = {
            "video_id": "first_id",
            "caption": "",
            "hashtags": [],
            "audio_id": None,
            "url": "",
            "posted_at": datetime.now(timezone.utc),
            "bootstrap": True,
        }

        run_watcher(watchlist_path=self.wl_path, mock=False, max_cycles=1)

        mock_gen.assert_not_called()
        mock_notify.assert_not_called()

        on_disk = json.loads(self.wl_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["creators"][0]["last_post_id"], "first_id")


class RunWatcherProtectionsTest(unittest.TestCase):
    """Tests des protections anti-détection : pause anti-flag, recovery session."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.wl_path = Path(self.tmp.name) / "watchlist.json"

    def _write_watchlist(self, n_creators: int) -> None:
        creators = [
            {
                "username": f"creator_{i}",
                "platform": "instagram",
                "niche": "streetwear",
                "t_type": "T2",
                "engagement_baseline": 0.04,
                "last_post_id": f"old_{i}",
                "added_at": "2026-04-01T00:00:00",
            }
            for i in range(n_creators)
        ]
        self.wl_path.write_text(
            json.dumps({"creators": creators}), encoding="utf-8"
        )

    @patch("watcher.get_browser_context")
    @patch("watcher.sync_playwright")
    @patch("watcher.notify_new_post")
    @patch("watcher._generate_for_post")
    @patch("watcher.check_new_post", return_value=None)
    @patch("watcher.time.sleep")
    @patch("watcher.config.MAX_ACCOUNTS_PER_SESSION", 3)
    def test_long_pause_after_max_accounts(
        self,
        mock_sleep: MagicMock,
        mock_check: MagicMock,
        mock_gen: MagicMock,
        mock_notify: MagicMock,
        mock_pw: MagicMock,
        mock_ctx: MagicMock,
    ) -> None:
        mock_pw.return_value.start.return_value = MagicMock()
        mock_ctx.return_value = MagicMock()
        # 4 comptes vérifiés ; seuil = 3 → une pause longue déclenchée
        self._write_watchlist(4)
        run_watcher(watchlist_path=self.wl_path, mock=False, max_cycles=1)
        self.assertEqual(mock_check.call_count, 4)

        long_pauses = [
            c for c in mock_sleep.call_args_list if c.args and c.args[0] == MAX_ACCOUNTS_PAUSE_S
        ]
        self.assertEqual(len(long_pauses), 1)


def _sample_named_axes() -> dict[str, float]:
    return {
        "scripted_vs_raw": 0.11,
        "solo_vs_collab": 0.22,
        "fictional_vs_real": 0.33,
        "energy_level": 0.44,
        "production_quality": 0.55,
        "format_length": 0.66,
        "distance_parasociale": 0.77,
        "interaction_style": 0.88,
        "mainstream_vs_niche": 0.99,
        "safe_vs_edgy": 0.12,
    }


class GenerateWithVectorTest(unittest.TestCase):
    @patch("modules.classifier.generate_comments", return_value=["a", "b", "c"])
    def test_named_axes_from_vector_store_passed_to_generate_comments(
        self, mock_gen: MagicMock
    ) -> None:
        from watcher import _generate_for_post

        ctx = {
            "t_type": "T2",
            "niches": ["humour"],
            "username": "creator1",
        }
        vector_store = {
            "creator1": {
                "username": "creator1",
                "named_axes": _sample_named_axes(),
            }
        }
        with self.assertLogs("aitertainment.watcher", level="INFO") as logs:
            _generate_for_post(ctx, vector_store=vector_store)
        kwargs = mock_gen.call_args.kwargs
        self.assertEqual(kwargs["named_axes"], _sample_named_axes())
        self.assertTrue(
            any("vecteur 32D disponible" in msg for msg in logs.output)
        )

    @patch("modules.classifier.generate_comments", return_value=["a", "b", "c"])
    def test_missing_username_omits_named_axes(self, mock_gen: MagicMock) -> None:
        from watcher import _generate_for_post

        vector_store = {
            "creator1": {
                "username": "creator1",
                "named_axes": _sample_named_axes(),
            }
        }
        with self.assertLogs("aitertainment.watcher", level="INFO") as logs:
            _generate_for_post(
                {"t_type": "T2", "niches": ["humour"], "username": "unknown"},
                vector_store=vector_store,
            )
        kwargs = mock_gen.call_args.kwargs
        self.assertNotIn("named_axes", kwargs)
        self.assertTrue(any("pas de vecteur" in msg for msg in logs.output))

    @patch("modules.classifier.generate_comments", return_value=["a", "b", "c"])
    def test_empty_vector_store_keeps_legacy_call(self, mock_gen: MagicMock) -> None:
        from watcher import _generate_for_post

        _generate_for_post(
            {"t_type": "T2", "niches": ["humour"], "username": "creator1"},
            vector_store={},
        )
        self.assertNotIn("named_axes", mock_gen.call_args.kwargs)

    @patch("modules.classifier.generate_comments", return_value=["a", "b", "c"])
    def test_empty_named_axes_treated_as_missing(self, mock_gen: MagicMock) -> None:
        from watcher import _generate_for_post

        vector_store = {"creator1": {"username": "creator1", "named_axes": {}}}
        with self.assertLogs("aitertainment.watcher", level="INFO") as logs:
            _generate_for_post(
                {"t_type": "T2", "niches": ["humour"], "username": "creator1"},
                vector_store=vector_store,
            )
        self.assertNotIn("named_axes", mock_gen.call_args.kwargs)
        self.assertTrue(any("pas de vecteur" in msg for msg in logs.output))


class GenerateForPostTest(unittest.TestCase):
    """Vérifie le câblage Watcher → ``modules.classifier.generate_comments``."""

    @patch("modules.classifier.generate_comments", return_value=["a", "b", "c"])
    def test_passes_video_context_with_caption_hashtags_audio(
        self, mock_gen: MagicMock
    ) -> None:
        from watcher import _generate_for_post

        # Schéma 2026-05 : ``niches`` (liste) priorité sur ``niche`` (string).
        ctx = {
            "t_type": "T2",
            "niches": ["humour", "sketch"],
            "niche": "humour",  # alias rétro-compat
            "caption": "moment culte F1",
            "hashtags": ["F1", "monaco"],
            "audio_id": "AUD42",
            "username": "raikkonenaf",
        }
        out = _generate_for_post(ctx)
        self.assertEqual(out, ["a", "b", "c"])
        mock_gen.assert_called_once()
        kwargs = mock_gen.call_args.kwargs
        # comments_sample = [] (pas de scrape en phase Watcher).
        self.assertEqual(mock_gen.call_args.args[1], [])
        # ``niches`` (liste) propagée — pas l'alias string.
        self.assertEqual(kwargs["niches"], ["humour", "sketch"])
        # ``t_type_profile`` égal au t_type du créateur (sa persona).
        self.assertEqual(kwargs["t_type_profile"], "T2")
        self.assertEqual(
            kwargs["video_context"],
            {"caption": "moment culte F1", "hashtags": ["F1", "monaco"], "audio_id": "AUD42"},
        )

    @patch("modules.classifier.generate_comments", return_value=["a", "b", "c"])
    def test_falls_back_to_niche_string_for_legacy_context(
        self, mock_gen: MagicMock
    ) -> None:
        from watcher import _generate_for_post

        # Rétro-compat : context ancien schéma (uniquement ``niche`` string,
        # sans ``niches`` liste).
        _generate_for_post(
            {"t_type": "T2", "niche": "humour", "audio": "OLD_KEY"}
        )
        kwargs = mock_gen.call_args.kwargs
        self.assertEqual(kwargs["niches"], "humour")
        self.assertEqual(kwargs["video_context"]["audio_id"], "OLD_KEY")

    @patch("modules.classifier.generate_comments")
    def test_t1_skips_generation(self, mock_gen: MagicMock) -> None:
        from watcher import _generate_for_post

        out = _generate_for_post({"t_type": "T1", "niches": ["x"]})
        self.assertEqual(out, [])
        mock_gen.assert_not_called()

    @patch("modules.classifier.generate_comments")
    def test_t3a_skips_generation(self, mock_gen: MagicMock) -> None:
        from watcher import _generate_for_post

        out = _generate_for_post({"t_type": "T3a", "niches": ["x"]})
        self.assertEqual(out, [])
        mock_gen.assert_not_called()


class WatchlistSchemaMigrationTest(unittest.TestCase):
    """Schéma 2026-05 : ``_normalize_entry`` accepte ``niches`` (liste) et
    migre lazy depuis ``niche`` (string) sans casser les watchlists historiques.
    """

    @staticmethod
    def _entry(**overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "username": "alice",
            "platform": "instagram",
            "t_type": "T2",
            "engagement_baseline": 0.05,
            "last_post_id": None,
            "added_at": "2026-05-09T11:00:00+00:00",
        }
        base.update(overrides)
        return base

    def test_priority_to_niches_list(self) -> None:
        from watcher import _normalize_entry

        out = _normalize_entry(
            self._entry(niches=["humour", "sketch", "imitation"]),
            index=0,
        )
        self.assertEqual(out["niches"], ["humour", "sketch", "imitation"])
        self.assertEqual(out["niche"], "humour")

    def test_lazy_migration_from_legacy_niche_string(self) -> None:
        """Watchlist historique avec uniquement ``niche`` (string) → ``niches=[niche]``."""
        from watcher import _normalize_entry

        out = _normalize_entry(self._entry(niche="streetwear"), index=0)
        self.assertEqual(out["niches"], ["streetwear"])
        self.assertEqual(out["niche"], "streetwear")

    def test_legacy_empty_niche_remains_valid(self) -> None:
        """Rétro-compat : ``niche=""`` historiquement autorisé reste valide."""
        from watcher import _normalize_entry

        out = _normalize_entry(self._entry(niche=""), index=0)
        self.assertEqual(out["niches"], [])
        self.assertEqual(out["niche"], "")

    def test_t_type_placeholder_becomes_none(self) -> None:
        from watcher import _normalize_entry

        for raw in ("T?", " T? ", ""):
            out = _normalize_entry(self._entry(t_type=raw), index=0)
            self.assertIsNone(out["t_type"])

    def test_niches_not_a_list_raises(self) -> None:
        from watcher import WatchlistError, _normalize_entry

        with self.assertRaises(WatchlistError) as ctx:
            _normalize_entry(self._entry(niches="humour"), index=0)
        self.assertIn("niches doit être une liste", str(ctx.exception))

    def test_niches_with_non_string_item_raises(self) -> None:
        from watcher import WatchlistError, _normalize_entry

        with self.assertRaises(WatchlistError) as ctx:
            _normalize_entry(self._entry(niches=["humour", 42]), index=0)
        self.assertIn("niches[1]", str(ctx.exception))

    def test_niches_filters_empty_strings(self) -> None:
        from watcher import _normalize_entry

        out = _normalize_entry(
            self._entry(niches=["", "humour", "   ", "sketch"]),
            index=0,
        )
        self.assertEqual(out["niches"], ["humour", "sketch"])

    def test_niches_priority_overrides_legacy_niche(self) -> None:
        """Si les deux champs sont présents, ``niches`` (liste) prime."""
        from watcher import _normalize_entry

        out = _normalize_entry(
            self._entry(niches=["humour", "réaction"], niche="ignored_legacy"),
            index=0,
        )
        self.assertEqual(out["niches"], ["humour", "réaction"])
        self.assertEqual(out["niche"], "humour")  # alias = niches[0], pas l'ancien

    def test_round_trip_save_load_preserves_niches(self) -> None:
        from watcher import load_watchlist, save_watchlist

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "wl.json"
            save_watchlist(
                [self._entry(niches=["humour", "sketch"])],
                path=p,
            )
            loaded = load_watchlist(path=p)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["niches"], ["humour", "sketch"])
            self.assertEqual(loaded[0]["niche"], "humour")


if __name__ == "__main__":
    unittest.main()
