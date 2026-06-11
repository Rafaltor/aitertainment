"""Tests unitaires pour watcher : check_new_post + run_watcher (mocks)."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

from config import ORDERED_T_TYPES
from watcher import (
    DAY_INTERVAL_S,
    MAX_ACCOUNTS_PAUSE_S,
    NEW_POST_VIEW_THRESHOLD,
    NIGHT_INTERVAL_S,
    PRIME_INTERVAL_S,
    _dual_account_enabled,
    _split_creator_batches,
    _watcher_eligible,
    check_new_post,
    get_poll_interval,
    run_watcher,
)


def _mock_comments_by_type() -> dict[str, str]:
    return {t: f"c-{t}" for t in ORDERED_T_TYPES}


def _reel(
    media_id: str,
    view_count: int,
    *,
    is_pinned: bool = False,
    caption: str = "",
    audio_id: str = "",
    product_type: str = "",
    owner_username: str = "",
) -> dict:
    out = {
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
    if product_type:
        out["product_type"] = product_type
    if owner_username:
        out["owner_username"] = owner_username
    return out


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
        mock_reels.assert_called_once_with(
            "someone", self.context, max_reels=4, spa_wait_ms=ANY
        )

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
    def test_bootstrap_high_views(self, mock_reels: MagicMock) -> None:
        """Bootstrap ignore le seuil de vues — mémorise le Reel le plus récent."""
        mock_reels.return_value = [_reel("viral", NEW_POST_VIEW_THRESHOLD + 50_000)]
        creator = {"username": "u", "platform": "instagram", "last_post_id": None}
        out = check_new_post(creator, self.context)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["video_id"], "viral")
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
        mock_reels.assert_any_call("u", self.context, max_reels=4, spa_wait_ms=ANY)
        mock_reels.assert_any_call("u", self.context, max_reels=8, spa_wait_ms=ANY)
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

    @patch("watcher.VIEW_FILTER_ENABLED", True)
    @patch("watcher.get_recent_reels")
    def test_high_views_returns_none(self, mock_reels: MagicMock) -> None:
        mock_reels.return_value = [_reel("big", NEW_POST_VIEW_THRESHOLD + 1)]
        creator = {"username": "u", "platform": "instagram", "last_post_id": "old"}
        self.assertIsNone(check_new_post(creator, self.context))

    @patch("watcher.VIEW_FILTER_ENABLED", False)
    @patch("watcher.get_recent_reels")
    def test_high_views_alerts_when_filter_disabled(self, mock_reels: MagicMock) -> None:
        mock_reels.return_value = [_reel("big", NEW_POST_VIEW_THRESHOLD + 1)]
        creator = {"username": "u", "platform": "instagram", "last_post_id": "old"}
        out = check_new_post(creator, self.context)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["video_id"], "big")

    @patch("watcher.get_recent_reels")
    def test_skips_carousel_zero_views(self, mock_reels: MagicMock) -> None:
        mock_reels.return_value = [
            _reel("carousel", 0),
            _reel("fresh", 400, caption="new #drop"),
        ]
        creator = {"username": "u", "platform": "instagram", "last_post_id": "old"}
        out = check_new_post(creator, self.context)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["video_id"], "fresh")

    @patch("watcher.get_recent_reels")
    def test_skips_reel_owned_by_other_account(self, mock_reels: MagicMock) -> None:
        mock_reels.return_value = [
            _reel("stolen", 400, product_type="clips", owner_username="other_user"),
        ]
        creator = {"username": "u", "platform": "instagram", "last_post_id": "old"}
        self.assertIsNone(check_new_post(creator, self.context))

    @patch("watcher.get_recent_reels")
    def test_uses_next_reel_when_head_owned_by_collab(self, mock_reels: MagicMock) -> None:
        mock_reels.return_value = [
            _reel("collab", 400, product_type="clips", owner_username="le.corbz"),
            _reel("own", 300, product_type="clips", owner_username="bisou.boge"),
        ]
        creator = {
            "username": "bisou.boge",
            "platform": "instagram",
            "last_post_id": "old",
        }
        out = check_new_post(creator, self.context)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["video_id"], "own")

    @patch("watcher.get_recent_reels")
    def test_skips_non_clips_product_type(self, mock_reels: MagicMock) -> None:
        mock_reels.return_value = [
            _reel("photo", 500, product_type="feed"),
            _reel("fresh", 400, product_type="clips"),
        ]
        creator = {"username": "u", "platform": "instagram", "last_post_id": "old"}
        out = check_new_post(creator, self.context)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["video_id"], "fresh")

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
                            "niches": ["streetwear"],
                            "t_type": "T2",
                            "engagement_baseline": 0.05,
                            "last_post_id": "old_id_a",
                            "added_at": "2026-04-01T00:00:00",
                        },
                        {
                            "username": "creator_b",
                            "platform": "instagram",
                            "niches": ["lifestyle"],
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
                            "niches": ["streetwear"],
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

    @patch("watcher._dual_account_enabled", return_value=False)
    @patch("watcher._describe_reel_visually", return_value="")
    @patch("watcher._sync_post_metadata_from_reel_page", return_value=True)
    @patch("watcher._transcribe_reel", return_value=("", None))
    @patch("watcher.get_browser_context")
    @patch("watcher.session_ok", return_value=True)
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
        mock_session: MagicMock,
        mock_transcribe: MagicMock,
        mock_sync: MagicMock,
        mock_describe: MagicMock,
        _mock_dual: MagicMock,
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
        mock_gen.return_value = _mock_comments_by_type()
        mock_notify.return_value = True

        run_watcher(watchlist_path=self.wl_path, mock=False, max_cycles=1)

        mock_check.assert_called_once()
        mock_gen.assert_called_once()
        mock_notify.assert_called_once()

        on_disk = json.loads(self.wl_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["creators"][0]["last_post_id"], "fresh_post_id")

    @patch("watcher._dual_account_enabled", return_value=False)
    @patch("watcher.get_browser_context")
    @patch("watcher.session_ok", return_value=True)
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
        mock_session: MagicMock,
        _mock_dual: MagicMock,
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
                            "niches": ["streetwear"],
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
    """Tests des protections anti-détection : pause anti-flag optionnelle."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.wl_path = Path(self.tmp.name) / "watchlist.json"

    def _write_watchlist(self, n_creators: int) -> None:
        creators = [
            {
                "username": f"creator_{i}",
                "platform": "instagram",
                "niches": ["streetwear"],
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

    @patch("watcher._dual_account_enabled", return_value=False)
    @patch("watcher.get_browser_context")
    @patch("watcher.session_ok", return_value=True)
    @patch("watcher.sync_playwright")
    @patch("watcher.notify_new_post")
    @patch("watcher._generate_for_post")
    @patch("watcher.check_new_post", return_value=None)
    @patch("watcher.time.sleep")
    @patch("watcher.MAX_ACCOUNTS_PAUSE_S", 180)
    @patch("watcher.config.MAX_ACCOUNTS_PER_SESSION", 3)
    def test_long_pause_after_max_accounts(
        self,
        mock_sleep: MagicMock,
        mock_check: MagicMock,
        mock_gen: MagicMock,
        mock_notify: MagicMock,
        mock_pw: MagicMock,
        mock_ctx: MagicMock,
        mock_session: MagicMock,
        _mock_dual: MagicMock,
    ) -> None:
        mock_pw.return_value.start.return_value = MagicMock()
        mock_ctx.return_value = MagicMock()
        # 4 comptes vérifiés ; seuil = 3 → une pause longue si WATCHER_ACCOUNTS_PAUSE_S > 0
        self._write_watchlist(4)
        run_watcher(watchlist_path=self.wl_path, mock=False, max_cycles=1)
        self.assertEqual(mock_check.call_count, 4)

        long_pauses = [
            c for c in mock_sleep.call_args_list if c.args and c.args[0] == 180
        ]
        self.assertEqual(len(long_pauses), 1)

    @patch("watcher._dual_account_enabled", return_value=False)
    @patch("watcher.MAX_ACCOUNTS_PAUSE_S", 0)
    @patch("watcher.config.MAX_ACCOUNTS_PER_SESSION", 3)
    @patch("watcher.get_browser_context")
    @patch("watcher.session_ok", return_value=True)
    @patch("watcher.sync_playwright")
    @patch("watcher.notify_new_post")
    @patch("watcher._generate_for_post")
    @patch("watcher.check_new_post", return_value=None)
    @patch("watcher.time.sleep")
    def test_no_pause_when_accounts_pause_disabled(
        self,
        mock_sleep: MagicMock,
        mock_check: MagicMock,
        mock_gen: MagicMock,
        mock_notify: MagicMock,
        mock_pw: MagicMock,
        mock_ctx: MagicMock,
        mock_session: MagicMock,
        _mock_dual: MagicMock,
    ) -> None:
        mock_pw.return_value.start.return_value = MagicMock()
        mock_ctx.return_value = MagicMock()
        self._write_watchlist(4)
        run_watcher(watchlist_path=self.wl_path, mock=False, max_cycles=1)
        long_pauses = [
            c
            for c in mock_sleep.call_args_list
            if c.args and c.args[0] == MAX_ACCOUNTS_PAUSE_S
        ]
        self.assertEqual(long_pauses, [])


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
    @patch("modules.classifier.generate_comments_per_category", return_value=_mock_comments_by_type())
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

    @patch("modules.classifier.generate_comments_per_category", return_value=_mock_comments_by_type())
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

    @patch("modules.classifier.generate_comments_per_category", return_value=_mock_comments_by_type())
    def test_empty_vector_store_keeps_legacy_call(self, mock_gen: MagicMock) -> None:
        from watcher import _generate_for_post

        _generate_for_post(
            {"t_type": "T2", "niches": ["humour"], "username": "creator1"},
            vector_store={},
        )
        self.assertNotIn("named_axes", mock_gen.call_args.kwargs)

    @patch("modules.classifier.generate_comments_per_category", return_value=_mock_comments_by_type())
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


class NotifyNewPostTest(unittest.TestCase):
    @patch("telegram_notify.send_telegram_markdown")
    def test_uses_post_username_not_stale_creator(self, mock_send: MagicMock) -> None:
        from watcher import notify_new_post

        notify_new_post(
            {"username": "wrong_watchlist", "t_type": "T2", "niches": ["humour"]},
            {
                "username": "real_owner",
                "video_id": "ABC123",
                "caption": "drop",
                "hashtags": ["drop"],
                "url": "https://www.instagram.com/reel/ABC123/",
            },
            ["mdr", "trop vrai", "dead"],
        )
        text = mock_send.call_args.args[0]
        self.assertIn("👤 @real", text)
        self.assertNotIn("wrong_watchlist", text)
        self.assertIn("reel/ABC123", text)
        self.assertNotIn("(instagram)", text)
        self.assertNotIn("Caption", text)
        self.assertNotIn("Hashtags", text)
        self.assertNotIn("Reel :", text)

    @patch("telegram_notify.send_telegram_markdown")
    def test_t1_lists_generated_comments(self, mock_send: MagicMock) -> None:
        from watcher import notify_new_post

        notify_new_post(
            {"username": "brand_x", "platform": "instagram", "t_type": "T1", "niches": ["mode"]},
            {
                "username": "brand_x",
                "video_id": "x1",
                "caption": "nouveau drop",
                "hashtags": ["mode"],
                "url": "https://instagram.com/reel/x1/",
            },
            ["bravo le drop", "trop fort", "incroyable"],
        )
        text = mock_send.call_args.args[0]
        self.assertIn("1. bravo le drop", text)
        self.assertNotIn("pas de commentaire suggéré", text)

    @patch("telegram_notify.send_telegram_markdown")
    def test_t2_lists_generated_comments(self, mock_send: MagicMock) -> None:
        from watcher import notify_new_post

        notify_new_post(
            {"username": "u", "t_type": "T2", "niches": ["humour"]},
            {"username": "u", "video_id": "r1", "caption": "c", "hashtags": [], "url": "—"},
            ["mdr", "trop vrai", "dead"],
        )
        text = mock_send.call_args.args[0]
        self.assertIn("1. mdr", text)

    @patch("telegram_notify.send_telegram_markdown")
    def test_dict_lists_comments_per_category(self, mock_send: MagicMock) -> None:
        from watcher import notify_new_post

        notify_new_post(
            {"username": "u", "t_type": "T2", "niches": ["humour"]},
            {"username": "u", "video_id": "r1", "caption": "c", "hashtags": [], "url": "—"},
            {"T1": "bravo", "T2": "mdr", "T2b": "punch", "T3a": "nul", "T3b": "ironie", "T4": "mème", "T5": "ratio"},
        )
        text = mock_send.call_args.args[0]
        self.assertIn("*T1* : bravo", text)
        self.assertIn("*T2b* : punch", text)
        self.assertIn("Commentaires par catégorie", text)


class GenerateForPostTest(unittest.TestCase):
    """Vérifie le câblage Watcher → ``generate_comments_per_category``."""

    @patch("modules.classifier.generate_comments_per_category", return_value=_mock_comments_by_type())
    def test_passes_video_context_with_caption_hashtags_audio(
        self, mock_gen: MagicMock
    ) -> None:
        from watcher import _generate_for_post

        ctx = {
            "t_type": "T2",
            "niches": ["humour", "sketch"],
            "caption": "moment culte F1",
            "hashtags": ["F1", "monaco"],
            "audio_id": "AUD42",
            "username": "raikkonenaf",
            "video_id": "REEL42",
        }
        out = _generate_for_post(ctx)
        self.assertEqual(out, _mock_comments_by_type())
        mock_gen.assert_called_once()
        kwargs = mock_gen.call_args.kwargs
        self.assertEqual(kwargs["niches"], ["humour", "sketch"])
        self.assertEqual(
            kwargs["video_context"],
            {
                "caption": "moment culte F1",
                "hashtags": ["F1", "monaco"],
                "audio_id": "AUD42",
                "video_context": "",
                "video_id": "REEL42",
                "username": "raikkonenaf",
            },
        )

    @patch("modules.classifier.generate_comments_per_category", return_value=_mock_comments_by_type())
    def test_missing_niches_yields_empty_list(self, mock_gen: MagicMock) -> None:
        from watcher import _generate_for_post

        # Context sans ``niches`` → liste vide propagée (plus de lecture legacy).
        _generate_for_post({"t_type": "T2", "audio": "OLD_KEY"})
        kwargs = mock_gen.call_args.kwargs
        self.assertEqual(kwargs["niches"], [])
        self.assertEqual(kwargs["video_context"]["audio_id"], "OLD_KEY")

    @patch("modules.classifier.generate_comments_per_category", return_value=_mock_comments_by_type())
    def test_t1_generates_comments(self, mock_gen: MagicMock) -> None:
        from watcher import _generate_for_post

        out = _generate_for_post({"t_type": "T1", "niches": ["x"], "username": "u"})
        self.assertEqual(out, _mock_comments_by_type())
        mock_gen.assert_called_once()

    @patch("modules.classifier.generate_comments_per_category", return_value=_mock_comments_by_type())
    def test_t3a_generates_comments(self, mock_gen: MagicMock) -> None:
        from watcher import _generate_for_post

        out = _generate_for_post({"t_type": "T3a", "niches": ["x"], "username": "u"})
        self.assertEqual(out, _mock_comments_by_type())
        mock_gen.assert_called_once()

    @patch("watcher.notify_new_post")
    @patch("watcher._generate_for_post", return_value=_mock_comments_by_type())
    @patch("watcher._transcribe_reel", return_value=("", None))
    @patch("watcher._sync_post_metadata_from_reel_page", return_value=True)
    @patch("watcher.check_new_post")
    def test_new_post_generates_and_notifies(
        self,
        mock_check: MagicMock,
        mock_sync: MagicMock,
        mock_transcribe: MagicMock,
        mock_gen: MagicMock,
        mock_notify: MagicMock,
    ) -> None:
        from watcher import _process_creator

        creator = {
            "username": "creator_t2",
            "platform": "instagram",
            "niches": ["mode"],
            "t_type": "T2",
            "last_post_id": "old",
        }
        mock_check.return_value = {
            "video_id": "new_reel",
            "username": "creator_t2",
            "caption": "drop",
            "hashtags": ["mode"],
            "audio_id": "snd",
            "url": "https://www.instagram.com/reel/new_reel/",
            "bootstrap": False,
        }
        log = MagicMock()
        _process_creator(
            creator,
            mock=False,
            log=log,
            browser_context=MagicMock(),
        )
        mock_gen.assert_called_once()
        mock_notify.assert_called_once()
        self.assertEqual(mock_notify.call_args.args[2], _mock_comments_by_type())

    @patch("modules.classifier.generate_comments_per_category", return_value=_mock_comments_by_type())
    def test_passes_merged_video_context(self, mock_gen: MagicMock) -> None:
        from watcher import _generate_for_post

        _generate_for_post(
            {
                "t_type": "T2",
                "niches": ["humour"],
                "video_context": "GP Monaco, scène de rue de nuit.",
            }
        )
        self.assertEqual(
            mock_gen.call_args.kwargs["video_context"]["video_context"],
            "GP Monaco, scène de rue de nuit.",
        )


class FuseTranscriptVisualTest(unittest.TestCase):
    @patch("watcher.requests.post")
    def test_fuse_returns_llm_content(self, mock_post: MagicMock) -> None:
        from watcher import _fuse_transcript_visual_for_watcher

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Fusion chronologique du sketch."}}]
        }
        mock_post.return_value = mock_resp
        out = _fuse_transcript_visual_for_watcher(
            "dialogue audio",
            "deux personnes en rue",
            "caption test",
        )
        self.assertEqual(out, "Fusion chronologique du sketch.")

    def test_fuse_skips_when_both_empty(self) -> None:
        from watcher import _fuse_transcript_visual_for_watcher

        self.assertEqual(_fuse_transcript_visual_for_watcher("", "", "cap"), "")


class TranscribeReelTest(unittest.TestCase):
    @patch("scripts.embedder.transcribe_audio")
    @patch("scripts.embedder.extract_wav_from_video")
    @patch("scripts.embedder.download_reel_video")
    def test_returns_transcript_and_mp4(
        self,
        mock_dl: MagicMock,
        mock_extract: MagicMock,
        mock_tr: MagicMock,
    ) -> None:
        from watcher import _transcribe_reel

        mp4 = Path("/tmp/fake.mp4")
        wav = Path("/tmp/fake.wav")
        mock_dl.return_value = mp4
        mock_extract.return_value = wav
        mock_tr.return_value = "bonjour le monde"
        browser_ctx = MagicMock()
        transcript, path = _transcribe_reel("reel123", browser_ctx)
        self.assertEqual(transcript, "bonjour le monde")
        self.assertEqual(path, mp4)
        mock_dl.assert_called_once()
        mock_extract.assert_called_once()
        mock_tr.assert_called_once_with(wav)

    @patch("scripts.embedder.download_reel_video", return_value=None)
    def test_ytdlp_failure_returns_empty(self, mock_dl: MagicMock) -> None:
        from watcher import _transcribe_reel

        self.assertEqual(_transcribe_reel("reel123", MagicMock()), ("", None))

    @patch("scripts.embedder.subprocess.run")
    def test_download_no_video_formats_logs_debug_not_warning(
        self, mock_run: MagicMock
    ) -> None:
        from scripts.embedder import download_reel_video

        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="ERROR: No video formats found",
        )
        with self.assertLogs("aitertainment.embedder", level="DEBUG") as logs:
            out = download_reel_video("carousel_id", MagicMock(), Path(tempfile.mkdtemp()))
        self.assertIsNone(out)
        self.assertTrue(
            any("pas de vidéo (carousel/photo)" in msg for msg in logs.output)
        )
        self.assertFalse(any("yt-dlp vidéo échoué" in msg for msg in logs.output))

    @patch("scripts.embedder.transcribe_audio", return_value="ok")
    @patch("scripts.embedder.extract_wav_from_video")
    @patch("scripts.embedder.download_reel_video")
    def test_tmp_dir_not_removed_when_provided(
        self,
        mock_dl: MagicMock,
        mock_extract: MagicMock,
        mock_tr: MagicMock,
    ) -> None:
        from watcher import _transcribe_reel

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        mp4 = tmp / "reel.mp4"
        mp4.touch()
        mock_dl.return_value = mp4
        mock_extract.return_value = tmp / "reel.wav"
        _transcribe_reel("reel123", MagicMock(), tmp_dir=tmp)
        self.assertTrue(tmp.exists())


class DescribeReelVisuallyTest(unittest.TestCase):
    @patch("watcher.requests.post")
    @patch("watcher.subprocess.run")
    def test_returns_description_from_lm_studio(
        self, mock_run: MagicMock, mock_post: MagicMock
    ) -> None:
        from watcher import _describe_reel_visually

        work = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(work, ignore_errors=True))
        mp4 = work / "v.mp4"
        mp4.write_bytes(b"x" * 2000)

        def fake_run(cmd, **kwargs):
            r = MagicMock()
            r.returncode = 0
            if cmd and cmd[0] == "ffprobe":
                r.stdout = json.dumps(
                    {"streams": [{"codec_type": "video", "duration": "12.0"}]}
                )
            elif cmd and cmd[0] == "ffmpeg":
                out_path = Path(cmd[-1])
                out_path.write_bytes(b"\xff" * 2000)
            return r

        mock_run.side_effect = fake_run
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": " Deux potes dans une cuisine."}}]
        }
        mock_post.return_value = resp

        out = _describe_reel_visually(
            mp4, "http://127.0.0.1:1234/v1", "qwen2.5-vl-7b-instruct"
        )
        self.assertEqual(out, "Deux potes dans une cuisine.")
        self.assertIn("/chat/completions", mock_post.call_args.args[0])


class WatchlistSchemaMigrationTest(unittest.TestCase):
    """``_normalize_entry`` : ``niches`` (liste) est la source de vérité."""

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

    def test_missing_niches_yields_empty_list(self) -> None:
        """Entrée sans ``niches`` → ``[]`` (reste valide)."""
        from watcher import _normalize_entry

        out = _normalize_entry(self._entry(), index=0)
        self.assertEqual(out["niches"], [])
        self.assertNotIn("niche", out)

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

    def test_legacy_niche_field_is_dropped(self) -> None:
        """Un champ ``niche`` (string) résiduel est ignoré, pas réexposé."""
        from watcher import _normalize_entry

        out = _normalize_entry(
            self._entry(niches=["humour", "réaction"], niche="ignored_legacy"),
            index=0,
        )
        self.assertEqual(out["niches"], ["humour", "réaction"])
        # ``niche`` passe par le passthrough des clés inconnues mais n'est plus
        # produit comme alias dérivé de ``niches``.
        self.assertEqual(out.get("niche"), "ignored_legacy")

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
            self.assertNotIn("niche", loaded[0])


class SyncLastPostsTest(unittest.TestCase):
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
                            "niches": ["humour"],
                            "t_type": "T2",
                            "engagement_baseline": 0.05,
                            "last_post_id": "old_reel",
                            "added_at": "2026-04-01T00:00:00",
                        },
                        {
                            "username": "creator_b",
                            "platform": "instagram",
                            "niches": ["mode"],
                            "t_type": "T1",
                            "engagement_baseline": 0.05,
                            "last_post_id": "already_latest",
                            "added_at": "2026-04-01T00:00:00",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

    @patch("watcher._dual_account_enabled", return_value=False)
    @patch("watcher.save_watchlist")
    @patch("watcher.sync_creator_last_post_id")
    @patch("watcher.session_ok", return_value=True)
    @patch("watcher.get_browser_context")
    @patch("watcher.sync_playwright")
    @patch("watcher.time.sleep")
    def test_sync_updates_watchlist(
        self,
        mock_sleep: MagicMock,
        mock_pw: MagicMock,
        mock_ctx: MagicMock,
        mock_session: MagicMock,
        mock_sync_one: MagicMock,
        mock_save: MagicMock,
        _mock_dual: MagicMock,
    ) -> None:
        from watcher import sync_watchlist_last_posts

        mock_pw.return_value.start.return_value = MagicMock()
        mock_ctx.return_value = MagicMock()
        mock_sync_one.side_effect = [True, False]

        sync_watchlist_last_posts(watchlist_path=self.wl_path)

        # creator_b est T1 → exclu par WATCHER_SKIP_T_TYPES par défaut
        self.assertEqual(mock_sync_one.call_count, 1)
        mock_save.assert_called_once()

    @patch("watcher.get_recent_reels")
    def test_sync_creator_updates_last_post_id(self, mock_reels: MagicMock) -> None:
        from watcher import sync_creator_last_post_id

        mock_reels.return_value = [_reel("NEW_ID", 1000)]
        creator = {
            "username": "u",
            "platform": "instagram",
            "last_post_id": "OLD_ID",
        }
        changed = sync_creator_last_post_id(creator, MagicMock())
        self.assertTrue(changed)
        self.assertEqual(creator["last_post_id"], "NEW_ID")

    @patch("watcher.get_recent_reels")
    def test_sync_creator_no_change_when_already_latest(
        self, mock_reels: MagicMock
    ) -> None:
        from watcher import sync_creator_last_post_id

        mock_reels.return_value = [_reel("SAME", 1000)]
        creator = {
            "username": "u",
            "platform": "instagram",
            "last_post_id": "SAME",
        }
        self.assertFalse(sync_creator_last_post_id(creator, MagicMock()))
        self.assertEqual(creator["last_post_id"], "SAME")


class TestWatcherOptimizations(unittest.TestCase):
    def test_split_creator_batches_even(self) -> None:
        creators = [
            {"username": f"u{i}", "t_type": "T2"} for i in range(4)
        ]
        left, right = _split_creator_batches(creators)
        self.assertEqual(len(left), 2)
        self.assertEqual(len(right), 2)
        self.assertEqual(left[0][0], 0)
        self.assertEqual(right[0][0], 2)

    @patch("watcher.config.WATCHER_DUAL_ACCOUNT", True)
    @patch("watcher.Path.is_file", return_value=True)
    def test_dual_account_enabled_with_cookies(self, _mock_is_file: MagicMock) -> None:
        self.assertTrue(_dual_account_enabled())

    @patch("watcher.config.WATCHER_DUAL_ACCOUNT", True)
    @patch("watcher.Path.is_file", return_value=False)
    def test_dual_account_requires_cookies_file(self, _mock_is_file: MagicMock) -> None:
        self.assertFalse(_dual_account_enabled())

    @patch("watcher.config.WATCHER_DUAL_ACCOUNT", False)
    def test_dual_account_disabled_by_config(self) -> None:
        self.assertFalse(_dual_account_enabled())

    @patch("watcher.get_recent_reels_on_page")
    def test_check_new_post_uses_reused_page(self, mock_on_page: MagicMock) -> None:
        mock_on_page.return_value = [_reel("fresh", 400)]
        page = MagicMock()
        creator = {"username": "u", "platform": "instagram", "last_post_id": "old"}
        out = check_new_post(creator, MagicMock(), grid_page=page)
        self.assertIsNotNone(out)
        mock_on_page.assert_called_once()
        self.assertIs(mock_on_page.call_args[0][0], page)


class WatcherEligibleTest(unittest.TestCase):
    @patch("watcher.SKIP_T_TYPES", frozenset({"T1"}))
    def test_t1_excluded_by_default(self) -> None:
        self.assertFalse(_watcher_eligible({"username": "b", "t_type": "T1"}))
        self.assertTrue(_watcher_eligible({"username": "c", "t_type": "T2"}))
        self.assertFalse(_watcher_eligible({"username": "d", "t_type": None}))

    @patch("watcher.SKIP_T_TYPES", frozenset({"T1"}))
    @patch("watcher.check_new_post")
    def test_process_creator_skips_t1(self, mock_check: MagicMock) -> None:
        from watcher import _process_creator

        creator = {"username": "brand", "t_type": "T1", "last_post_id": "old"}
        changed, did_check = _process_creator(
            creator, mock=False, log=MagicMock(), browser_context=MagicMock()
        )
        self.assertFalse(changed)
        self.assertFalse(did_check)
        mock_check.assert_not_called()

    @patch("watcher.SKIP_T_TYPES", frozenset())
    def test_all_types_when_skip_empty(self) -> None:
        self.assertTrue(_watcher_eligible({"username": "b", "t_type": "T1"}))


class SyncPostMetadataTest(unittest.TestCase):
    @patch("scripts.instagram_browser.get_reel_page_metadata")
    def test_ignores_wrong_reel_page_owner_when_grid_empty(
        self, mock_meta: MagicMock
    ) -> None:
        from watcher import _sync_post_metadata_from_reel_page

        mock_meta.return_value = {
            "caption": "fresh drop",
            "owner_username": "tyga",
        }
        post = {
            "video_id": "ABC123",
            "caption": "",
            "grid_owner": "",
        }
        ctx = MagicMock()
        self.assertTrue(
            _sync_post_metadata_from_reel_page(post, "raikkonenaf", ctx)
        )
        self.assertEqual(post["username"], "raikkonenaf")
        self.assertEqual(post["caption"], "fresh drop")

    @patch("scripts.instagram_browser.get_reel_page_metadata")
    def test_rejects_when_grid_owner_conflicts(
        self, mock_meta: MagicMock
    ) -> None:
        from watcher import _sync_post_metadata_from_reel_page

        mock_meta.return_value = {"caption": "", "owner_username": ""}
        post = {
            "video_id": "ABC123",
            "caption": "",
            "grid_owner": "collab_partner",
        }
        self.assertFalse(
            _sync_post_metadata_from_reel_page(
                post, "raikkonenaf", MagicMock()
            )
        )
        mock_meta.assert_not_called()


if __name__ == "__main__":
    unittest.main()
