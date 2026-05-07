"""Tests pour le détecteur."""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from modules.detector import CreatorStats, SignalDetector

try:
    import pandas  # noqa: F401

    _PANDAS_OK = True
except Exception:
    _PANDAS_OK = False

if _PANDAS_OK:
    from modules.detector import load_creators_csv, save_creators_csv


def _dt(hours_ago: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours_ago)


class CreatorStatsBaselineTest(unittest.TestCase):
    def test_get_baseline_last_ten_most_recent(self) -> None:
        old = _dt(200)
        recent = _dt(1)
        stats = CreatorStats(
            creator_id="1",
            platform="instagram",
            username="u",
            followers=1,
            recent_videos=[
                {"video_id": "a", "views": 100, "likes": 0, "comments": 0, "shares": 0, "saves": 0, "posted_at": old},
                {"video_id": "b", "views": 900, "likes": 0, "comments": 0, "shares": 0, "saves": 0, "posted_at": recent},
            ],
        )
        self.assertEqual(stats.get_baseline(), 500.0)

    def test_get_baseline_empty(self) -> None:
        stats = CreatorStats("1", "instagram", "u", 0, [])
        self.assertEqual(stats.get_baseline(), 0.0)


class CreatorStatsVelocityTest(unittest.TestCase):
    def test_get_velocity(self) -> None:
        posted = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        now = datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc)
        stats = CreatorStats(
            creator_id="1",
            platform="instagram",
            username="u",
            followers=1,
            recent_videos=[
                {
                    "video_id": "v1",
                    "views": 2000,
                    "likes": 0,
                    "comments": 0,
                    "shares": 0,
                    "saves": 0,
                    "posted_at": posted,
                },
            ],
        )
        # 2000 vues en 2 h => 1000 vues/h
        self.assertAlmostEqual(stats.get_velocity("v1", now=now), 1000.0)

    def test_get_velocity_uses_two_thirds_second_floor(self) -> None:
        """Juste après publish : dénominateur = 2/3 s, pas le delta réel quasi nul."""
        posted = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        now = posted  # 0 s écoulé
        stats = CreatorStats(
            creator_id="1",
            platform="instagram",
            username="u",
            followers=1,
            recent_videos=[
                {
                    "video_id": "v1",
                    "views": 100,
                    "likes": 0,
                    "comments": 0,
                    "shares": 0,
                    "saves": 0,
                    "posted_at": posted,
                },
            ],
        )
        hours_floor = (2.0 / 3.0) / 3600.0
        expected = 100.0 / hours_floor
        self.assertAlmostEqual(stats.get_velocity("v1", now=now), expected)

    def test_get_velocity_unknown(self) -> None:
        stats = CreatorStats("1", "instagram", "u", 0, [])
        with self.assertRaises(ValueError):
            stats.get_velocity("missing")


class SignalDetectorTest(unittest.TestCase):
    def test_alert_critical_and_serie(self) -> None:
        t0 = datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        old_times = [t0 - timedelta(days=30 - i) for i in range(6)]
        stats = CreatorStats(
            creator_id="x",
            platform="instagram",
            username="u",
            followers=100,
            follower_growth_7d_pct=25.0,
            recent_videos=[
                *[
                    {
                        "video_id": f"old{i}",
                        "views": 100,
                        "likes": 5,
                        "comments": 0,
                        "shares": 0,
                        "saves": 0,
                        "posted_at": old_times[i],
                    }
                    for i in range(6)
                ],
                {
                    "video_id": "v2",
                    "views": 5000,
                    "likes": 5000,
                    "comments": 400,
                    "shares": 100,
                    "saves": 50,
                    "posted_at": t0 - timedelta(hours=72),
                },
                {
                    "video_id": "v1",
                    "views": 5000,
                    "likes": 5000,
                    "comments": 400,
                    "shares": 100,
                    "saves": 50,
                    "posted_at": t0 - timedelta(hours=48),
                },
                {
                    "video_id": "v0",
                    "views": 5000,
                    "likes": 50_000,
                    "comments": 5000,
                    "shares": 2000,
                    "saves": 1000,
                    "posted_at": t0,
                    "audio_id": "snd",
                    "audio_reels_count": 100,
                    "audio_is_recent": True,
                    "duration_sec": 90,
                },
            ],
        )
        now = t0 + timedelta(hours=2)
        out = SignalDetector(stats).detect(now=now, likes_at_30m_ago=10_000)
        self.assertGreater(out["score_viral"], 0.85)
        self.assertEqual(out["alert"], "CRITICAL")
        self.assertEqual(out["groups"]["g1"]["components"]["serie_score"], 1.0)

    def test_alert_none_low_signal(self) -> None:
        posted = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
        stats = CreatorStats(
            creator_id="y",
            platform="instagram",
            username="u",
            followers=100_000,
            recent_videos=[
                {
                    "video_id": "v",
                    "views": 10,
                    "likes": 2,
                    "comments": 0,
                    "shares": 0,
                    "saves": 0,
                    "posted_at": posted,
                },
            ],
        )
        out = SignalDetector(stats).detect(now=posted + timedelta(hours=5))
        self.assertLessEqual(out["score_viral"], 0.75)
        self.assertEqual(out["alert"], "NONE")


@unittest.skipUnless(_PANDAS_OK, "pandas (numpy) indisponible dans cet environnement")
class CreatorsCsvRoundtripTest(unittest.TestCase):
    def test_save_load_roundtrip(self) -> None:
        posted = datetime(2026, 5, 1, 10, 0, 0, tzinfo=timezone.utc)
        original = CreatorStats(
            creator_id="c1",
            platform="tiktok",
            username="tok",
            followers=42,
            recent_videos=[
                {
                    "video_id": "p1",
                    "views": 10,
                    "likes": 1,
                    "comments": 2,
                    "shares": 3,
                    "saves": 4,
                    "posted_at": posted,
                    "audio_id": "snd_1",
                },
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "creators.csv"
            save_creators_csv([original], path)
            loaded = load_creators_csv(path)
        self.assertEqual(len(loaded), 1)
        r = loaded[0]
        self.assertEqual(r.creator_id, "c1")
        self.assertEqual(r.platform, "tiktok")
        self.assertEqual(r.username, "tok")
        self.assertEqual(r.followers, 42)
        self.assertEqual(len(r.recent_videos), 1)
        v = r.recent_videos[0]
        self.assertEqual(v["video_id"], "p1")
        self.assertEqual(v["views"], 10)
        self.assertEqual(v["audio_id"], "snd_1")
        self.assertEqual(_ensure_utc(v["posted_at"]), posted)

    def test_load_legacy_csv_without_creator_id_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.csv"
            path.write_text("platform,username,notes\ninstagram,x,\n", encoding="utf-8")
            self.assertEqual(load_creators_csv(path), [])


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


if __name__ == "__main__":
    unittest.main()
