"""Tests I/O discovery.py (pas de logique réseau)."""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from instagrapi.exceptions import LoginRequired, UserNotFound

import discovery


class DiscoveryIOTest(unittest.TestCase):
    def test_load_seeds_default(self) -> None:
        data = discovery.load_seeds()
        self.assertIn("domains", data)
        self.assertIsInstance(data["domains"], list)
        self.assertGreaterEqual(len(data["domains"]), 1)

    def test_blacklist_candidates_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bp = Path(tmp) / "blacklist.json"
            cp = Path(tmp) / "candidates.json"

            discovery.save_blacklist(
                {
                    "profiles": [
                        {
                            "username": "seen_user",
                            "platform": "instagram",
                            "outcome": "rejected",
                            "added_at": "2026-05-07T12:00:00",
                        }
                    ]
                },
                path=bp,
            )
            discovery.save_candidates(
                {
                    "candidates": [
                        {
                            "username": "pending_user",
                            "platform": "instagram",
                            "domain": "humour",
                            "score": 0.72,
                            "discovered_at": "2026-05-07T12:05:00",
                        }
                    ]
                },
                path=cp,
            )

            bl = discovery.load_blacklist(path=bp)
            self.assertEqual(len(bl["profiles"]), 1)
            self.assertEqual(bl["profiles"][0]["username"], "seen_user")

            cd = discovery.load_candidates(path=cp)
            self.assertEqual(len(cd["candidates"]), 1)
            self.assertEqual(cd["candidates"][0]["score"], 0.72)

    def test_load_blacklist_missing_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "missing.json"
            data = discovery.load_blacklist(path=p)
            self.assertEqual(data, {"profiles": []})


def _make_user(
    *,
    follower_count: int = 50_000,
    media_count: int = 30,
    biography: str = "bio",
    is_private: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        follower_count=follower_count,
        media_count=media_count,
        biography=biography,
        is_private=is_private,
    )


def _make_media(
    *,
    pk: str,
    views: int,
    likes: int,
    comments: int,
    days_ago: int,
    shares: int | None = None,
    product_type: str = "clips",
    is_pinned: bool = False,
    view_count: int | None = None,
    play_count: int | None = None,
) -> SimpleNamespace:
    vc = views if view_count is None else view_count
    pc = views if play_count is None else play_count
    return SimpleNamespace(
        pk=pk,
        id=pk,
        play_count=pc,
        view_count=vc,
        like_count=likes,
        comment_count=comments,
        share_count=shares,
        taken_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
        caption_text=f"caption {pk}",
        product_type=product_type,
        is_pinned=is_pinned,
    )


class ScoreProfileTest(unittest.TestCase):
    """Tests de la logique de scoring (mocks complets, pas de réseau)."""

    DOMAIN = {
        "name": "humour",
        "niche": "humour",
        "t_types_target": ["T2", "T3b"],
    }

    def setUp(self) -> None:
        # Disable les sleeps anti-détection pendant les tests
        self._sleep_patch = patch.object(discovery, "polite_sleep", lambda *a, **k: None)
        self._sleep_patch.start()
        self.addCleanup(self._sleep_patch.stop)

    def _make_client(
        self,
        *,
        user: SimpleNamespace | None = None,
        medias: list[SimpleNamespace] | None = None,
        comments_per_media: dict[str, list[str]] | None = None,
    ) -> MagicMock:
        client = MagicMock()
        client.user_id_from_username.return_value = "111"
        client.user_info.return_value = user or _make_user()
        client.user_medias.return_value = medias or []

        comments_per_media = comments_per_media or {}

        def _media_comments(media_id: str, amount: int = 15):
            texts = comments_per_media.get(str(media_id), [])
            return [SimpleNamespace(text=t) for t in texts]

        client.media_comments.side_effect = _media_comments
        return client

    def test_user_not_found_returns_none(self) -> None:
        client = MagicMock()
        client.user_id_from_username.side_effect = UserNotFound("nope")
        result = discovery.score_profile(
            "ghost",
            self.DOMAIN,
            blacklist={"profiles": []},
            client=client,
        )
        self.assertIsNone(result)

    def test_blacklisted_skipped_before_network(self) -> None:
        client = MagicMock()
        result = discovery.score_profile(
            "@known",
            self.DOMAIN,
            blacklist={"profiles": [{"username": "known"}]},
            client=client,
        )
        self.assertIsNone(result)
        client.user_id_from_username.assert_not_called()

    def test_followers_below_min_returns_none(self) -> None:
        client = self._make_client(user=_make_user(follower_count=500))
        self.assertIsNone(
            discovery.score_profile(
                "tiny", self.DOMAIN, blacklist={"profiles": []}, client=client
            )
        )

    def test_followers_above_max_returns_none(self) -> None:
        client = self._make_client(user=_make_user(follower_count=2_000_000))
        self.assertIsNone(
            discovery.score_profile(
                "mega", self.DOMAIN, blacklist={"profiles": []}, client=client
            )
        )

    def test_private_account_returns_none(self) -> None:
        client = self._make_client(user=_make_user(is_private=True))
        self.assertIsNone(
            discovery.score_profile(
                "secret", self.DOMAIN, blacklist={"profiles": []}, client=client
            )
        )

    def test_media_count_too_low(self) -> None:
        client = self._make_client(user=_make_user(media_count=1))
        self.assertIsNone(
            discovery.score_profile(
                "thin", self.DOMAIN, blacklist={"profiles": []}, client=client
            )
        )

    def test_session_lost_propagates(self) -> None:
        client = MagicMock()
        client.user_id_from_username.side_effect = LoginRequired("session expired")
        with self.assertRaises(discovery.DiscoverySessionLost):
            discovery.score_profile(
                "anyone",
                self.DOMAIN,
                blacklist={"profiles": []},
                client=client,
            )

    def test_full_scoring_rising_trend_with_t2(self) -> None:
        # 12 medias chronologiques, vues qui montent à la fin → trend = rising
        followers = 10_000
        # 8 anciens (ratio ~0.5x) + 4 récents (ratio ~3.0x) → rising
        medias = []
        for i in range(8):
            medias.append(
                _make_media(
                    pk=f"old{i}",
                    views=5_000,
                    likes=400,
                    comments=30,
                    days_ago=30 - i,
                    shares=20,
                )
            )
        for i in range(4):
            medias.append(
                _make_media(
                    pk=f"new{i}",
                    views=30_000,
                    likes=2_500,
                    comments=200,
                    days_ago=4 - i,
                    shares=120,
                )
            )

        comments_map = {
            "new3": ["lol mdr la blague", "gg le comique"],
            "new2": ["t'es serieux", "haha incroyable"],
            "new1": ["meme énergie", "drôle"],
        }
        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
            comments_per_media=comments_map,
        )

        with patch("modules.classifier.CommentClassifier") as cls:
            cls.return_value.classify.return_value = {
                "type": "T2",
                "confidence": 0.9,
                "patterns": [],
                "tone": "humour tribal",
                "brand_risk": "low",
            }
            result = discovery.score_profile(
                "rising_creator",
                self.DOMAIN,
                blacklist={"profiles": []},
                client=client,
            )

        self.assertIsNotNone(result)
        assert result is not None  # type-narrow
        self.assertEqual(result["username"], "rising_creator")
        self.assertEqual(result["domain"], "humour")
        self.assertEqual(result["followers"], followers)
        # 12 reels, 0 posts → reel_weight = 1.0, post_weight = 0.0
        self.assertEqual(result["reels_count"], 12)
        self.assertEqual(result["posts_count"], 0)
        self.assertAlmostEqual(result["reel_weight"], 1.0, places=5)
        self.assertAlmostEqual(result["post_weight"], 0.0, places=5)
        self.assertEqual(result["reel_trend"], "rising")
        self.assertIsNone(result["post_ratio_median"])
        self.assertEqual(result["t_type_dominant"], "T2")
        self.assertAlmostEqual(sum(result["t_type_distribution"].values()), 1.0, places=5)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1000.0)
        # T-type T2 est dans les targets → bonus t_type_match plein régime
        # rising → +150, ratio cap 30 mais ici 3x → 35, etc.
        self.assertGreater(result["score"], 400.0)

    def test_returns_none_when_no_medias(self) -> None:
        client = self._make_client(user=_make_user(media_count=10), medias=[])
        self.assertIsNone(
            discovery.score_profile(
                "empty", self.DOMAIN, blacklist={"profiles": []}, client=client
            )
        )

    def test_below_min_total_medias_returns_none(self) -> None:
        # 2 médias seulement → < MIN_TOTAL_MEDIAS_REQUIRED (3) → None
        followers = 20_000
        medias = [
            _make_media(
                pk=f"m{i}",
                views=10_000,
                likes=500,
                comments=40,
                days_ago=5 - i,
                product_type="clips" if i == 0 else "feed",
            )
            for i in range(2)
        ]
        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=2),
            medias=medias,
        )
        with patch("modules.classifier.CommentClassifier"):
            self.assertIsNone(
                discovery.score_profile(
                    "thin_history",
                    self.DOMAIN,
                    blacklist={"profiles": []},
                    client=client,
                )
            )

    def test_reels_only_creator_post_weight_zero(self) -> None:
        followers = 10_000
        medias = [
            _make_media(
                pk=f"r{i}",
                views=20_000,
                likes=1_500,
                comments=120,
                days_ago=10 - i,
                product_type="clips",
            )
            for i in range(10)
        ]
        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
        )
        with patch("modules.classifier.CommentClassifier"):
            result = discovery.score_profile(
                "reels_only",
                self.DOMAIN,
                blacklist={"profiles": []},
                client=client,
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["reels_count"], 10)
        self.assertEqual(result["posts_count"], 0)
        self.assertAlmostEqual(result["reel_weight"], 1.0, places=5)
        self.assertAlmostEqual(result["post_weight"], 0.0, places=5)
        self.assertIsNone(result["post_ratio_median"])
        self.assertIsNone(result["post_engagement_median"])
        self.assertEqual(result["score_posts"], 0.0)
        # score = score_reels × 1.0
        self.assertAlmostEqual(result["score"], result["score_reels"], places=3)

    def test_posts_only_creator_reel_weight_zero(self) -> None:
        followers = 10_000
        medias = [
            _make_media(
                pk=f"p{i}",
                views=0,
                likes=800,
                comments=60,
                days_ago=10 - i,
                product_type="feed",
            )
            for i in range(8)
        ]
        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
        )
        with patch("modules.classifier.CommentClassifier"):
            result = discovery.score_profile(
                "posts_only",
                self.DOMAIN,
                blacklist={"profiles": []},
                client=client,
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["reels_count"], 0)
        # 8 posts en entrée → cap à MAX_POSTS_FOR_SCORING (4) pour le scoring.
        self.assertEqual(result["posts_count"], 4)
        self.assertAlmostEqual(result["reel_weight"], 0.0, places=5)
        self.assertAlmostEqual(result["post_weight"], 1.0, places=5)
        self.assertIsNone(result["reel_ratio_median"])
        self.assertIsNone(result["reel_trend"])
        self.assertEqual(result["score_reels"], 0.0)
        self.assertGreaterEqual(result["score"], 0.0)
        # score = score_posts × 1.0
        self.assertAlmostEqual(result["score"], result["score_posts"], places=3)
        # post_ratio_median = 800/10000 = 0.08 (4 plus récents, tous identiques)
        self.assertAlmostEqual(result["post_ratio_median"], 0.08, places=4)

    def test_mixed_50_50_balances_weights(self) -> None:
        # 4 reels qualité + 4 posts moyens → weights 50/50 (sous le cap posts).
        followers = 10_000
        medias = []
        for i in range(4):
            medias.append(
                _make_media(
                    pk=f"r{i}",
                    views=25_000,
                    likes=2_000,
                    comments=150,
                    days_ago=15 - i,
                    shares=80,
                    product_type="clips",
                )
            )
        for i in range(4):
            medias.append(
                _make_media(
                    pk=f"p{i}",
                    views=0,
                    likes=600,
                    comments=80,
                    days_ago=5 - i,
                    product_type="feed",
                )
            )
        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
        )
        with patch("modules.classifier.CommentClassifier") as cls:
            cls.return_value.classify.return_value = {
                "type": "T2",
                "confidence": 0.8,
                "patterns": [],
                "tone": "humour",
                "brand_risk": "low",
            }
            result = discovery.score_profile(
                "mixed_50",
                self.DOMAIN,
                blacklist={"profiles": []},
                client=client,
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["reels_count"], 4)
        self.assertEqual(result["posts_count"], 4)
        self.assertAlmostEqual(result["reel_weight"], 0.5, places=5)
        self.assertAlmostEqual(result["post_weight"], 0.5, places=5)
        # SCORE_FINAL = 0.5 × score_reels + 0.5 × score_posts
        expected = 0.5 * result["score_reels"] + 0.5 * result["score_posts"]
        self.assertAlmostEqual(result["score"], expected, places=3)
        # Les Reels qualité ne sont pas dilués par les photos pour leur métrique propre
        self.assertGreater(result["reel_ratio_median"], 2.0)
        self.assertAlmostEqual(result["post_ratio_median"], 0.06, places=4)

    def test_mixed_80_reels_20_posts_weighted_correctly(self) -> None:
        followers = 10_000
        medias = []
        for i in range(8):
            medias.append(
                _make_media(
                    pk=f"r{i}",
                    views=20_000,
                    likes=1_500,
                    comments=120,
                    days_ago=15 - i,
                    product_type="clips",
                )
            )
        for i in range(2):
            medias.append(
                _make_media(
                    pk=f"p{i}",
                    views=0,
                    likes=400,
                    comments=30,
                    days_ago=2 - i,
                    product_type="feed",
                )
            )
        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
        )
        with patch("modules.classifier.CommentClassifier"):
            result = discovery.score_profile(
                "mixed_80",
                self.DOMAIN,
                blacklist={"profiles": []},
                client=client,
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["reels_count"], 8)
        self.assertEqual(result["posts_count"], 2)
        self.assertAlmostEqual(result["reel_weight"], 0.8, places=5)
        self.assertAlmostEqual(result["post_weight"], 0.2, places=5)
        expected = 0.8 * result["score_reels"] + 0.2 * result["score_posts"]
        self.assertAlmostEqual(result["score"], expected, places=3)

    def test_reel_views_top_up_from_media_info(self) -> None:
        """user_medias sans compteurs ; media_info renvoie play_count si view_count=0."""
        followers = 10_000
        medias = [
            _make_media(
                pk=f"r{i}",
                views=0,
                view_count=0,
                play_count=0,
                likes=500,
                comments=40,
                days_ago=5 - i,
                product_type="clips",
            )
            for i in range(3)
        ]
        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
        )
        client.media_info.return_value = SimpleNamespace(
            view_count=0, play_count=50_000
        )
        with patch("modules.classifier.CommentClassifier"):
            result = discovery.score_profile(
                "zero_list_views",
                self.DOMAIN,
                blacklist={"profiles": []},
                client=client,
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(client.media_info.call_count, 3)
        self.assertAlmostEqual(result["reel_ratio_median"], 5.0, places=4)

    def test_top_up_sets_reel_views_from_play_count_when_view_count_zero(self) -> None:
        """media_info avec view_count=0 et play_count=50000 → reel["views"] == 50000."""
        reels = [
            {
                "media_id": "999",
                "views": 0,
                "taken_at": datetime.now(timezone.utc),
            }
        ]
        client = MagicMock()
        client.media_info.return_value = SimpleNamespace(
            view_count=0, play_count=50_000
        )
        discovery._top_up_reel_views_via_media_info(
            client,
            reels,
            log=logging.getLogger("test_top_up"),
            username="testuser",
        )
        self.assertEqual(reels[0]["views"], 50_000)

    def test_reel_view_top_up_no_cap_processes_all_zero_reels(self) -> None:
        """Pas de cap : 16 Reels à 0 vue → 16 appels media_info."""
        followers = 10_000
        medias = [
            _make_media(
                pk=f"r{i}",
                views=0,
                view_count=0,
                play_count=0,
                likes=400,
                comments=30,
                days_ago=30 - i,
                product_type="clips",
            )
            for i in range(16)
        ]
        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
        )
        client.media_info.return_value = SimpleNamespace(
            view_count=0, play_count=50_000
        )
        with patch("modules.classifier.CommentClassifier"):
            result = discovery.score_profile(
                "all_zero_reels",
                self.DOMAIN,
                blacklist={"profiles": []},
                client=client,
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(client.media_info.call_count, 16)
        self.assertAlmostEqual(result["reel_ratio_median"], 5.0, places=4)

    def test_reel_views_skip_media_info_when_list_has_views(self) -> None:
        followers = 10_000
        medias = [
            _make_media(
                pk=f"r{i}",
                views=20_000,
                likes=1_000,
                comments=80,
                days_ago=3 - i,
                product_type="clips",
            )
            for i in range(5)
        ]
        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
        )
        with patch("modules.classifier.CommentClassifier"):
            discovery.score_profile(
                "has_views",
                self.DOMAIN,
                blacklist={"profiles": []},
                client=client,
            )
        client.media_info.assert_not_called()

    def test_viral_outlier_pushes_score_above_median_only(self) -> None:
        """Distribution skewée @raikkonenaf : P90 capte le potentiel viral.

        10 Reels [19k, 28k, 45k, 49k, 64k, 77k, 115k, 500k, 1300k, 1500k] /
        48k followers → reel_ratio_p90 ≈ 27.08x (1300k / 48k, nearest-rank).

        Vérifie aussi que le score est plus élevé qu'une version "médiane
        seule" (P90 forcé à 0.0), via deux appels directs à
        ``_compute_reel_score`` avec exactement les mêmes autres signaux.
        """
        followers = 48_000
        views = [19_000, 28_000, 45_000, 49_000, 64_000,
                 77_000, 115_000, 500_000, 1_300_000, 1_500_000]
        medias = [
            _make_media(
                pk=f"r{i}",
                views=v,
                likes=int(v * 0.05),
                comments=int(v * 0.005),
                days_ago=15 - i,
                product_type="clips",
            )
            for i, v in enumerate(views)
        ]
        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
        )
        with patch("modules.classifier.CommentClassifier"):
            result = discovery.score_profile(
                "viral_skew",
                self.DOMAIN,
                blacklist={"profiles": []},
                client=client,
            )
        self.assertIsNotNone(result)
        assert result is not None

        self.assertAlmostEqual(
            result["reel_ratio_p90"], 1_300_000 / 48_000.0, places=4
        )
        self.assertAlmostEqual(
            result["reel_ratio_median"],
            (64_000 / 48_000.0 + 77_000 / 48_000.0) / 2.0,
            places=4,
        )

        common_kwargs = {
            "reel_ratio_median": result["reel_ratio_median"],
            "reel_engagement_median": result["reel_engagement_median"],
            "reel_trend": result["reel_trend"],
            "t_type_distribution": result["t_type_distribution"],
            "publish_frequency": result["posting_rhythm"],
            "domain": self.DOMAIN,
        }
        score_with_p90 = discovery._compute_reel_score(
            reel_ratio_p90=result["reel_ratio_p90"], **common_kwargs
        )
        score_median_only = discovery._compute_reel_score(
            reel_ratio_p90=0.0, **common_kwargs
        )
        self.assertGreater(score_with_p90, score_median_only)
        # Contribution attendue du P90 : 27.08/50 * 100 ≈ 54.17
        self.assertAlmostEqual(
            score_with_p90 - score_median_only, 27.083333 / 50.0 * 100, places=3
        )

    def test_pinned_medias_excluded_from_scoring(self) -> None:
        """2 épinglés + 10 normaux : épinglés exclus ; media_sampled = brut API."""
        followers = 10_000
        medias = []
        for i in range(2):
            medias.append(
                _make_media(
                    pk=f"pin{i}",
                    views=999_999_999,
                    likes=900_000,
                    comments=80_000,
                    days_ago=200 + i,
                    is_pinned=True,
                )
            )
        for i in range(10):
            medias.append(
                _make_media(
                    pk=f"ok{i}",
                    views=20_000,
                    likes=1_500,
                    comments=120,
                    days_ago=10 - i,
                    is_pinned=False,
                )
            )
        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
        )
        with patch("modules.classifier.CommentClassifier"):
            result = discovery.score_profile(
                "with_pins",
                self.DOMAIN,
                blacklist={"profiles": []},
                client=client,
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["media_sampled"], 12)
        self.assertEqual(result["reels_count"], 10)
        self.assertEqual(result["posts_count"], 0)
        # Médiane des ratios sur les 10 non épinglés uniquement (20k / 10k = 2.0)
        self.assertAlmostEqual(result["reel_ratio_median"], 2.0, places=4)

    def test_posts_capped_at_max_for_scoring_keeps_most_recent(self) -> None:
        """6 reels + 8 posts → posts capés à 4, reel_weight = 6/10 = 0.6."""
        followers = 10_000
        medias = []
        for i in range(6):
            medias.append(
                _make_media(
                    pk=f"r{i}",
                    views=20_000,
                    likes=1_500,
                    comments=120,
                    days_ago=30 - i,
                    product_type="clips",
                )
            )
        # 8 posts : "old0..old3" très anciens (likes faibles, signaux dégradés),
        # "p0..p3" récents (likes élevés). Le cap doit garder les 4 récents.
        for i in range(4):
            medias.append(
                _make_media(
                    pk=f"old_p{i}",
                    views=0,
                    likes=10,
                    comments=1,
                    days_ago=180 - i,
                    product_type="feed",
                )
            )
        for i in range(4):
            medias.append(
                _make_media(
                    pk=f"p{i}",
                    views=0,
                    likes=900,
                    comments=80,
                    days_ago=4 - i,
                    product_type="feed",
                )
            )
        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
        )
        with patch("modules.classifier.CommentClassifier"):
            result = discovery.score_profile(
                "capped_posts",
                self.DOMAIN,
                blacklist={"profiles": []},
                client=client,
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["reels_count"], 6)
        self.assertEqual(result["posts_count"], 4)
        self.assertAlmostEqual(result["reel_weight"], 0.6, places=5)
        self.assertAlmostEqual(result["post_weight"], 0.4, places=5)
        # post_ratio_median = médiane(900/10000) sur les 4 récents uniquement
        # (les anciens "old_p*" à 10 likes seraient ~0.001 et tireraient la
        # médiane vers 0 si pris en compte).
        self.assertAlmostEqual(result["post_ratio_median"], 0.09, places=4)

    def test_classification_uses_only_recent_reels_when_three_or_more(self) -> None:
        """≥ 3 Reels : on classifie sur les 3 plus récents, pas sur les Posts."""
        followers = 10_000
        medias = []
        for i in range(3):
            medias.append(
                _make_media(
                    pk=f"old_r{i}",
                    views=10_000,
                    likes=500,
                    comments=40,
                    days_ago=40 - i,
                    product_type="clips",
                )
            )
        for i in range(3):
            medias.append(
                _make_media(
                    pk=f"recent_r{i}",
                    views=20_000,
                    likes=1_000,
                    comments=80,
                    days_ago=3 - i,
                    product_type="clips",
                )
            )
        for i in range(2):
            medias.append(
                _make_media(
                    pk=f"p{i}",
                    views=0,
                    likes=300,
                    comments=20,
                    days_ago=1 - i,
                    product_type="feed",
                )
            )
        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
            comments_per_media={f"recent_r{i}": ["lol"] for i in range(3)},
        )
        with patch("modules.classifier.CommentClassifier"):
            discovery.score_profile(
                "reels_priority",
                self.DOMAIN,
                blacklist={"profiles": []},
                client=client,
            )
        scraped_pks = {
            str(call.args[0]) for call in client.media_comments.call_args_list
        }
        self.assertEqual(scraped_pks, {"recent_r0", "recent_r1", "recent_r2"})

    def test_classification_completes_with_posts_when_reels_below_three(self) -> None:
        """1 Reel + ≥ 2 Posts : on complète avec les 2 Posts les plus récents."""
        followers = 10_000
        medias = [
            _make_media(
                pk="r_only",
                views=20_000,
                likes=1_500,
                comments=120,
                days_ago=10,
                product_type="clips",
            ),
            _make_media(
                pk="old_p",
                views=0,
                likes=400,
                comments=30,
                days_ago=60,
                product_type="feed",
            ),
            _make_media(
                pk="p_recent_0",
                views=0,
                likes=600,
                comments=50,
                days_ago=4,
                product_type="feed",
            ),
            _make_media(
                pk="p_recent_1",
                views=0,
                likes=800,
                comments=70,
                days_ago=2,
                product_type="feed",
            ),
        ]
        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
            comments_per_media={
                "r_only": ["lol"], "p_recent_0": ["wow"], "p_recent_1": ["nice"],
            },
        )
        with patch("modules.classifier.CommentClassifier"):
            discovery.score_profile(
                "complete_with_posts",
                self.DOMAIN,
                blacklist={"profiles": []},
                client=client,
            )
        scraped_pks = {
            str(call.args[0]) for call in client.media_comments.call_args_list
        }
        # 1 reel + 2 posts les plus récents (jamais "old_p"). Cap classifieur = 3.
        self.assertEqual(scraped_pks, {"r_only", "p_recent_0", "p_recent_1"})

    def test_posting_rhythm_in_result_dict(self) -> None:
        """``publish_frequency`` retiré ; clé renommée ``posting_rhythm``."""
        followers = 10_000
        medias = [
            _make_media(
                pk=f"r{i}",
                views=20_000,
                likes=1_500,
                comments=120,
                days_ago=10 - i,
                product_type="clips",
            )
            for i in range(4)
        ]
        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
        )
        with patch("modules.classifier.CommentClassifier"):
            result = discovery.score_profile(
                "rhythm_check",
                self.DOMAIN,
                blacklist={"profiles": []},
                client=client,
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("posting_rhythm", result)
        self.assertNotIn("publish_frequency", result)
        # 4 médias étalés sur 3 jours (j-10 → j-7) → 4/3 médias/jour.
        self.assertAlmostEqual(result["posting_rhythm"], 4 / 3.0, places=3)

    def test_posting_rhythm_ignores_posts_window(self) -> None:
        """Les anciens Posts ne dilatent plus la fenêtre — calcul sur Reels seuls."""
        followers = 10_000
        medias = []
        # 4 Reels récents : j-10 → j-7 (span 3 jours).
        for i in range(4):
            medias.append(
                _make_media(
                    pk=f"r{i}",
                    views=20_000,
                    likes=1_500,
                    comments=120,
                    days_ago=10 - i,
                    product_type="clips",
                )
            )
        # 4 Posts photos sur l'année passée — sans top-up de vues, leur
        # taken_at était auparavant inclus dans la fenêtre de posting_rhythm.
        for i, ago in enumerate((350, 270, 180, 90)):
            medias.append(
                _make_media(
                    pk=f"p{i}",
                    views=0,
                    likes=600,
                    comments=40,
                    days_ago=ago,
                    product_type="feed",
                )
            )
        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
        )
        with patch("modules.classifier.CommentClassifier"):
            result = discovery.score_profile(
                "reel_rhythm_only",
                self.DOMAIN,
                blacklist={"profiles": []},
                client=client,
            )
        self.assertIsNotNone(result)
        assert result is not None
        # Reels seuls : 4 / 3 jours ≈ 1.33 (au lieu de 8/350 ≈ 0.023 si Posts
        # anciens étaient inclus).
        self.assertAlmostEqual(result["posting_rhythm"], 4 / 3.0, places=3)

    def test_remove_pinned_reels_excludes_two_stale_leaders(self) -> None:
        """2 Reels anciens en tête (j-180, j-200) + 3 récents (j-1..j-3) → 3 Reels."""
        base = datetime.now(timezone.utc)
        reels = [
            {"taken_at": base - timedelta(days=180)},
            {"taken_at": base - timedelta(days=200)},
            {"taken_at": base - timedelta(days=1)},
            {"taken_at": base - timedelta(days=2)},
            {"taken_at": base - timedelta(days=3)},
        ]
        out = discovery._remove_pinned_reels(reels)
        self.assertEqual(len(out), 3)
        self.assertEqual(out, reels[2:])


def _row(views: int) -> dict[str, Any]:
    return {"views": views}


class ReelMetricsZeroViewsTest(unittest.TestCase):
    """Vues à 0 = donnée manquante : ne doivent jamais peser dans les médianes."""

    def test_ratio_median_ignores_zero_view_reels(self) -> None:
        reels = [_row(0), _row(0), _row(20_000), _row(40_000)]
        # Médiane sur [20_000, 40_000] uniquement → 30_000 / 10_000 = 3.0
        self.assertAlmostEqual(
            discovery._reel_ratio_median(reels, 10_000), 3.0, places=6
        )

    def test_ratio_median_all_zero_returns_zero(self) -> None:
        reels = [_row(0), _row(0), _row(0)]
        self.assertEqual(discovery._reel_ratio_median(reels, 10_000), 0.0)

    def test_ratio_p90_picks_high_outlier_with_nearest_rank(self) -> None:
        # Reels [19k, 28k, 45k, 49k, 64k, 77k, 115k, 500k, 1300k, 1500k] / 48k
        # ratios sortés ≈ [0.40, 0.58, 0.94, 1.02, 1.33, 1.60, 2.40, 10.42,
        # 27.08, 31.25]. Nearest-rank P90 = ceil(0.9*10)-1 = 8 → 1300/48 ≈ 27.08.
        views = [19_000, 28_000, 45_000, 49_000, 64_000,
                 77_000, 115_000, 500_000, 1_300_000, 1_500_000]
        reels = [_row(v) for v in views]
        self.assertAlmostEqual(
            discovery._reel_ratio_p90(reels, 48_000),
            1_300_000 / 48_000.0,
            places=4,
        )

    def test_ratio_p90_ignores_zero_view_reels(self) -> None:
        # 3 zéros + 5 valeurs → on travaille sur les 5 valeurs uniquement.
        # P90 nearest-rank sur 5 = ceil(4.5)-1 = 4 → max = 50k/10k = 5.0.
        reels = [_row(0)] * 3 + [_row(v) for v in (10_000, 20_000, 30_000, 40_000, 50_000)]
        self.assertAlmostEqual(
            discovery._reel_ratio_p90(reels, 10_000), 5.0, places=6
        )

    def test_ratio_p90_under_three_reels_with_views_returns_zero(self) -> None:
        reels = [_row(0)] * 8 + [_row(50_000), _row(80_000)]
        self.assertEqual(discovery._reel_ratio_p90(reels, 10_000), 0.0)

    def test_view_trend_requires_eight_reels_with_views(self) -> None:
        # 7 reels avec vues + 1 reel à 0 → < 8 exploitables → "stable"
        reels = [_row(5_000) for _ in range(7)] + [_row(0)]
        self.assertEqual(discovery._reel_view_trend(reels, 10_000), "stable")

    def test_view_trend_rising_on_views_only(self) -> None:
        # Bruit : 4 zéros au milieu (data manquante). On les filtre puis on
        # compare prev4 vs last4.
        prev = [_row(5_000) for _ in range(4)]
        zeros = [_row(0) for _ in range(4)]
        last = [_row(30_000) for _ in range(4)]
        self.assertEqual(
            discovery._reel_view_trend(prev + zeros + last, 10_000), "rising"
        )

    def test_view_trend_declining_on_views_only(self) -> None:
        prev = [_row(30_000) for _ in range(4)]
        last = [_row(5_000) for _ in range(4)]
        self.assertEqual(
            discovery._reel_view_trend(prev + last, 10_000), "declining"
        )


class HumanScheduleHelpersTest(unittest.TestCase):
    def test_is_night_evening_and_morning(self) -> None:
        self.assertTrue(discovery._is_night(datetime(2026, 5, 7, 23, 30)))
        self.assertTrue(discovery._is_night(datetime(2026, 5, 7, 7, 0)))
        self.assertFalse(discovery._is_night(datetime(2026, 5, 7, 10, 0)))

    def test_is_lunch_window(self) -> None:
        self.assertTrue(discovery._is_lunch(datetime(2026, 5, 7, 12, 30)))
        self.assertTrue(discovery._is_lunch(datetime(2026, 5, 7, 13, 59)))
        self.assertFalse(discovery._is_lunch(datetime(2026, 5, 7, 14, 0)))
        self.assertFalse(discovery._is_lunch(datetime(2026, 5, 7, 11, 59)))

    def test_seconds_until_hour_target_today(self) -> None:
        now = datetime(2026, 5, 7, 9, 0)
        secs = discovery._seconds_until_hour(now, 14)
        self.assertEqual(int(secs), 5 * 3600)

    def test_seconds_until_hour_target_tomorrow(self) -> None:
        now = datetime(2026, 5, 7, 23, 30)
        secs = discovery._seconds_until_hour(now, 8)
        self.assertEqual(int(secs), 8 * 3600 + 30 * 60)

    def test_maybe_reset_day(self) -> None:
        s = discovery.DiscoverySession(profiles_today=12, day_key="2026-05-06")
        discovery._maybe_reset_day(s, datetime(2026, 5, 7, 10, 0))
        self.assertEqual(s.profiles_today, 0)
        self.assertEqual(s.day_key, "2026-05-07")

    def test_burst_break_triggers_after_two_hours(self) -> None:
        s = discovery.DiscoverySession(mock=False)
        log = logging.getLogger("test_burst")
        sleeps: list[float] = []
        with patch.object(discovery.config, "DISABLE_HUMAN_SCHEDULE", False):
            # Premier appel : initialise activity_started_at
            discovery._maybe_take_burst_break(
                s, log,
                now_fn=lambda: datetime(2026, 5, 7, 10, 0),
                sleep_fn=sleeps.append,
            )
            self.assertEqual(sleeps, [])
            # 2h05 plus tard : doit déclencher la pause obligatoire
            discovery._maybe_take_burst_break(
                s, log,
                now_fn=lambda: datetime(2026, 5, 7, 12, 5),
                sleep_fn=sleeps.append,
            )
        self.assertEqual(sleeps, [discovery.ACTIVITY_PAUSE_S])

    def test_wait_for_active_window_mock_returns_immediately(self) -> None:
        s = discovery.DiscoverySession(mock=True)
        log = logging.getLogger("test_active_window")
        # En pleine nuit, en mock → return sans sleep
        sleeps: list[float] = []
        with patch.object(discovery.config, "DISABLE_HUMAN_SCHEDULE", False):
            discovery._wait_for_active_window(
                s, log,
                now_fn=lambda: datetime(2026, 5, 7, 2, 0),
                sleep_fn=sleeps.append,
            )
        self.assertEqual(sleeps, [])

    def test_disable_human_schedule_bypasses_night_window(self) -> None:
        s = discovery.DiscoverySession(mock=False)
        log = logging.getLogger("test_disable_schedule")
        sleeps: list[float] = []
        with patch.object(discovery.config, "DISABLE_HUMAN_SCHEDULE", True):
            discovery._wait_for_active_window(
                s, log,
                now_fn=lambda: datetime(2026, 5, 7, 2, 0),
                sleep_fn=sleeps.append,
            )
        self.assertEqual(sleeps, [])

    def test_disable_human_schedule_bypasses_burst_break(self) -> None:
        s = discovery.DiscoverySession(
            mock=False,
            activity_started_at=datetime(2026, 5, 7, 8, 0),
        )
        log = logging.getLogger("test_disable_burst")
        sleeps: list[float] = []
        with patch.object(discovery.config, "DISABLE_HUMAN_SCHEDULE", True):
            # 4h après le début → normalement pause 30min, ici no-op.
            discovery._maybe_take_burst_break(
                s, log,
                now_fn=lambda: datetime(2026, 5, 7, 12, 0),
                sleep_fn=sleeps.append,
            )
        self.assertEqual(sleeps, [])


class ExploreNetworkTest(unittest.TestCase):
    DOMAIN = {
        "name": "humour",
        "niche": "humour",
        "seeds": ["seed_one"],
        "t_types_target": ["T2", "T3b"],
    }

    def setUp(self) -> None:
        # Désactive complètement les sleeps anti-détection.
        self._sleep_patch = patch.object(discovery, "polite_sleep", lambda *a, **k: None)
        self._sleep_patch.start()
        self.addCleanup(self._sleep_patch.stop)
        # Neutralise la fenêtre horaire (nuit / déjeuner) : sans ça, lancer la
        # suite à 23h ferait dormir le test jusqu'à 8h. Les fenêtres elles-mêmes
        # sont testées dans HumanScheduleHelpersTest.
        self._window_patch = patch.object(
            discovery, "_wait_for_active_window", lambda *a, **k: None
        )
        self._window_patch.start()
        self.addCleanup(self._window_patch.stop)
        # Idem : la pause de burst (2h → 30min) est testée à part.
        self._burst_patch = patch.object(
            discovery, "_maybe_take_burst_break", lambda *a, **k: None
        )
        self._burst_patch.start()
        self.addCleanup(self._burst_patch.stop)

    @staticmethod
    def _make_following_client(
        followings: list[str],
    ) -> MagicMock:
        client = MagicMock()
        client.user_id_from_username.return_value = "111"
        # user_following renvoie un Dict[pk, UserShort]
        client.user_following.return_value = {
            f"pk_{i}": SimpleNamespace(username=u, pk=f"pk_{i}")
            for i, u in enumerate(followings)
        }
        return client

    def test_filters_already_in_blacklist_and_watchlist(self) -> None:
        client = self._make_following_client(["alpha", "beta", "gamma"])
        blacklist = {"profiles": [{"username": "beta"}]}
        watchlist = [{"username": "gamma"}]
        candidates: dict[str, Any] = {"candidates": []}
        session = discovery.DiscoverySession(mock=False)

        scored: list[str] = []

        def fake_score(username, domain, **kw):
            scored.append(username)
            return None  # ineligible — toujours blacklist mais pas candidat

        with patch.object(discovery, "score_profile", side_effect=fake_score):
            discovery.explore_network(
                self.DOMAIN,
                blacklist=blacklist,
                watchlist=watchlist,
                candidates=candidates,
                client=client,
                session=session,
                blacklist_path=Path("/tmp/skip_persist_bl.json"),
                candidates_path=Path("/tmp/skip_persist_cand.json"),
            )

        self.assertEqual(scored, ["alpha"])  # beta blacklist, gamma watchlist
        self.assertEqual(candidates["candidates"], [])
        # alpha a été ajouté à la blacklist (outcome=ineligible)
        bl_users = [p["username"] for p in blacklist["profiles"]]
        self.assertIn("alpha", bl_users)

    def test_high_score_added_to_candidates_and_notified(self) -> None:
        client = self._make_following_client(["promising"])
        blacklist = {"profiles": []}
        candidates: dict[str, Any] = {"candidates": []}
        session = discovery.DiscoverySession(mock=False)

        good_result = {
            "username": "promising",
            "domain": "humour",
            "score": 720.0,
            "ratio_median": 1.2,
            "ratio_trend": "rising",
            "t_type_dominant": "T2",
            "t_type_distribution": {"T2": 1.0},
            "biography": "bio",
            "followers": 12_000,
        }

        with patch.object(discovery, "score_profile", return_value=good_result), \
             patch.object(discovery, "_notify_candidate") as notif, \
             patch.object(discovery, "save_blacklist"), \
             patch.object(discovery, "save_candidates"):
            discovery.explore_network(
                self.DOMAIN,
                blacklist=blacklist,
                watchlist=[],
                candidates=candidates,
                client=client,
                session=session,
            )

        self.assertEqual(len(candidates["candidates"]), 1)
        self.assertEqual(candidates["candidates"][0]["username"], "promising")
        self.assertTrue(candidates["candidates"][0]["validated"] is False)
        self.assertEqual(session.candidates_found, 1)
        notif.assert_called_once()
        # Toujours blacklisté, même si retenu (jamais reproposé)
        self.assertIn("promising", [p["username"] for p in blacklist["profiles"]])

    def test_low_score_blacklisted_but_not_candidate(self) -> None:
        client = self._make_following_client(["meh"])
        blacklist = {"profiles": []}
        candidates: dict[str, Any] = {"candidates": []}
        session = discovery.DiscoverySession(mock=False)

        low_result = {
            "username": "meh",
            "domain": "humour",
            "score": 250.0,
            "ratio_median": 0.3,
            "ratio_trend": "declining",
            "t_type_dominant": "T1",
            "t_type_distribution": {"T1": 1.0},
            "biography": "",
            "followers": 9_000,
        }

        with patch.object(discovery, "score_profile", return_value=low_result), \
             patch.object(discovery, "_notify_candidate") as notif, \
             patch.object(discovery, "save_blacklist"), \
             patch.object(discovery, "save_candidates"):
            discovery.explore_network(
                self.DOMAIN,
                blacklist=blacklist,
                watchlist=[],
                candidates=candidates,
                client=client,
                session=session,
            )

        self.assertEqual(candidates["candidates"], [])
        notif.assert_not_called()
        self.assertEqual(
            blacklist["profiles"][0]["outcome"], "rejected"
        )

    def test_daily_quota_stops_exploration(self) -> None:
        client = self._make_following_client([f"u{i}" for i in range(10)])
        blacklist = {"profiles": []}
        candidates: dict[str, Any] = {"candidates": []}
        session = discovery.DiscoverySession(
            mock=False,
            profiles_today=discovery.MAX_PROFILES_PER_DAY - 2,
            day_key=datetime.now().strftime("%Y-%m-%d"),
        )

        scored: list[str] = []

        def fake_score(username, domain, **kw):
            scored.append(username)
            return None

        with patch.object(discovery, "score_profile", side_effect=fake_score), \
             patch.object(discovery, "save_blacklist"), \
             patch.object(discovery, "save_candidates"):
            discovery.explore_network(
                self.DOMAIN,
                blacklist=blacklist,
                watchlist=[],
                candidates=candidates,
                client=client,
                session=session,
            )

        # On ne consomme que les 2 derniers slots du quota.
        self.assertEqual(len(scored), 2)
        self.assertEqual(session.profiles_today, discovery.MAX_PROFILES_PER_DAY)

    def test_session_lost_propagates_from_explore(self) -> None:
        client = self._make_following_client(["x"])
        client.user_following.side_effect = LoginRequired("kicked")
        with self.assertRaises(discovery.DiscoverySessionLost):
            discovery.explore_network(
                self.DOMAIN,
                blacklist={"profiles": []},
                watchlist=[],
                candidates={"candidates": []},
                client=client,
                session=discovery.DiscoverySession(mock=False),
            )

    def test_mock_mode_runs_without_network(self) -> None:
        session = discovery.DiscoverySession(mock=True)
        candidates: dict[str, Any] = {"candidates": []}
        blacklist = {"profiles": []}

        # Ni client ni I/O ne doivent être touchés en mock.
        with patch.object(discovery, "save_blacklist") as bl_save, \
             patch.object(discovery, "save_candidates") as cd_save:
            discovery.explore_network(
                self.DOMAIN,
                blacklist=blacklist,
                watchlist=[],
                candidates=candidates,
                client=None,
                session=session,
            )
            bl_save.assert_not_called()
            cd_save.assert_not_called()

        # On a bien parcouru les followings simulés (3 par seed).
        self.assertGreater(session.profiles_today, 0)


class RunDiscoveryCliTest(unittest.TestCase):
    def test_run_discovery_mock_iterates_all_domains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seeds_p = Path(tmp) / "seeds.json"
            bl_p = Path(tmp) / "blacklist.json"
            cd_p = Path(tmp) / "candidates.json"
            seeds_p.write_text(
                json.dumps(
                    {
                        "domains": [
                            {
                                "name": "humour",
                                "niche": "humour",
                                "seeds": ["alpha"],
                                "t_types_target": ["T2"],
                            },
                            {
                                "name": "streetwear",
                                "niche": "streetwear",
                                "seeds": ["beta"],
                                "t_types_target": ["T2", "T3b"],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            bl_p.write_text(json.dumps({"profiles": []}), encoding="utf-8")
            cd_p.write_text(json.dumps({"candidates": []}), encoding="utf-8")

            sleeps: list[float] = []
            session = discovery.run_discovery(
                mock=True,
                seeds_path=seeds_p,
                blacklist_path=bl_p,
                candidates_path=cd_p,
                sleep_fn=sleeps.append,
            )
            self.assertEqual(session.mock, True)
            # Aucun sleep réel en mock.
            self.assertEqual(sleeps, [])
            # Les deux domaines ont été parcourus en mock (≥ 1 profil chacun).
            self.assertGreater(session.profiles_today, 0)

    def test_run_discovery_only_seed_picks_first_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seeds_p = Path(tmp) / "seeds.json"
            seeds_p.write_text(
                json.dumps(
                    {
                        "domains": [
                            {
                                "name": "humour",
                                "niche": "humour",
                                "seeds": ["already_in"],
                                "t_types_target": ["T2"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            bl_p = Path(tmp) / "blacklist.json"
            cd_p = Path(tmp) / "candidates.json"
            bl_p.write_text(json.dumps({"profiles": []}), encoding="utf-8")
            cd_p.write_text(json.dumps({"candidates": []}), encoding="utf-8")

            session = discovery.run_discovery(
                only_seed="@brand_new",
                mock=True,
                seeds_path=seeds_p,
                blacklist_path=bl_p,
                candidates_path=cd_p,
            )
            self.assertGreater(session.profiles_today, 0)


if __name__ == "__main__":
    unittest.main()
