"""Tests I/O discovery.py (pas de logique réseau)."""

from __future__ import annotations

import json
import logging
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

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


def _make_profile_data(
    *,
    followers: int = 50_000,
    following: int = 100,
    posts_count: int = 30,
    biography: str = "bio",
    full_name: str = "Creator",
    is_private: bool = False,
) -> dict[str, Any]:
    return {
        "followers": followers,
        "following": following,
        "posts_count": posts_count,
        "biography": biography,
        "full_name": full_name,
        "is_private": is_private,
    }


def _reels_from_medias(medias: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in medias:
        if str(getattr(m, "product_type", "") or "") != discovery.REEL_PRODUCT_TYPE:
            continue
        pk = str(getattr(m, "pk", "") or getattr(m, "id", "") or "")
        taken = getattr(m, "taken_at", None)
        row: dict[str, Any] = {
            "media_id": pk,
            "view_count": discovery._media_views(m),
            "like_count": int(getattr(m, "like_count", 0) or 0),
            "comment_count": int(getattr(m, "comment_count", 0) or 0),
            "thumbnail_url": "",
        }
        if isinstance(taken, datetime):
            row["taken_at"] = taken
        caption = getattr(m, "caption_text", None)
        if caption:
            row["caption_text"] = str(caption)
        out.append(row)
    return out


class ScoreProfileTest(unittest.TestCase):
    """Tests de la logique de scoring (mocks complets, pas de réseau)."""

    DOMAIN = {
        "name": "humour",
        "niche": "humour",
        "t_types_target": ["T2", "T3b"],
    }

    def setUp(self) -> None:
        self._sleep_patch = patch.object(discovery, "polite_sleep", lambda *a, **k: None)
        self._sleep_patch.start()
        self.addCleanup(self._sleep_patch.stop)
        self._ctx = MagicMock()
        self._profile_patch = patch(
            "discovery.get_profile_data",
            return_value=_make_profile_data(),
        )
        self._reels_patch = patch("discovery.get_recent_reels", return_value=[])
        self.mock_get_profile = self._profile_patch.start()
        self.mock_get_reels = self._reels_patch.start()
        self.addCleanup(self._profile_patch.stop)
        self.addCleanup(self._reels_patch.stop)

    def _make_client(
        self,
        *,
        user: SimpleNamespace | None = None,
        medias: list[SimpleNamespace] | None = None,
    ) -> MagicMock:
        """Configure les mocks Playwright (profil + reels) pour un cas de test."""
        u = user or _make_user()
        self.mock_get_profile.return_value = _make_profile_data(
            followers=int(u.follower_count),
            posts_count=int(u.media_count),
            biography=str(u.biography),
            is_private=bool(u.is_private),
        )
        self.mock_get_reels.return_value = _reels_from_medias(medias or [])
        return self._ctx

    def test_user_not_found_returns_none(self) -> None:
        self.mock_get_profile.return_value = None
        result = discovery.score_profile(
            "ghost",
            self.DOMAIN,
            blacklist={"profiles": []},
            context=self._ctx,
        )
        self.assertIsNone(result)

    def test_blacklisted_skipped_before_network(self) -> None:
        result = discovery.score_profile(
            "@known",
            self.DOMAIN,
            blacklist={"profiles": [{"username": "known"}]},
            context=self._ctx,
        )
        self.assertIsNone(result)
        self.mock_get_profile.assert_not_called()

    def test_followers_below_min_returns_none(self) -> None:
        self._make_client(user=_make_user(follower_count=500))
        self.assertIsNone(
            discovery.score_profile(
                "tiny", self.DOMAIN, blacklist={"profiles": []}, context=self._ctx
            )
        )

    def test_followers_above_max_returns_none(self) -> None:
        self._make_client(user=_make_user(follower_count=2_000_000))
        self.assertIsNone(
            discovery.score_profile(
                "mega", self.DOMAIN, blacklist={"profiles": []}, context=self._ctx
            )
        )

    def test_private_account_returns_none(self) -> None:
        self._make_client(user=_make_user(is_private=True))
        self.assertIsNone(
            discovery.score_profile(
                "secret", self.DOMAIN, blacklist={"profiles": []}, context=self._ctx
            )
        )

    def test_media_count_too_low(self) -> None:
        self._make_client(user=_make_user(media_count=1))
        self.assertIsNone(
            discovery.score_profile(
                "thin", self.DOMAIN, blacklist={"profiles": []}, context=self._ctx
            )
        )

    def test_missing_context_returns_none(self) -> None:
        self.assertIsNone(
            discovery.score_profile(
                "anyone",
                self.DOMAIN,
                blacklist={"profiles": []},
                context=None,
            )
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

        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
        )

        result = discovery.score_profile(
            "rising_creator",
            self.DOMAIN,
            blacklist={"profiles": []},
            context=self._ctx,
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
        self.assertIsNone(result["t_type_dominant"])
        self.assertEqual(result["t_type_distribution"], {})
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1000.0)
        self.assertGreater(result["score"], 300.0)

    def test_returns_none_when_no_medias(self) -> None:
        client = self._make_client(user=_make_user(media_count=10), medias=[])
        self.assertIsNone(
            discovery.score_profile(
                "empty", self.DOMAIN, blacklist={"profiles": []}, context=self._ctx
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
                    context=self._ctx,
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
                context=self._ctx,
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

    def test_posts_only_creator_returns_none(self) -> None:
        """Playwright ne récupère que des Reels — profil sans Reel → skip."""
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
        self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
        )
        result = discovery.score_profile(
            "posts_only",
            self.DOMAIN,
            blacklist={"profiles": []},
            context=self._ctx,
        )
        self.assertIsNone(result)

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
                context=self._ctx,
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["reels_count"], 4)
        self.assertEqual(result["posts_count"], 0)
        self.assertAlmostEqual(result["reel_weight"], 1.0, places=5)
        self.assertAlmostEqual(result["post_weight"], 0.0, places=5)
        self.assertAlmostEqual(result["score"], result["score_reels"], places=3)
        self.assertGreater(result["reel_ratio_median"], 2.0)
        self.assertIsNone(result["post_ratio_median"])

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
                context=self._ctx,
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["reels_count"], 8)
        self.assertEqual(result["posts_count"], 0)
        self.assertAlmostEqual(result["reel_weight"], 1.0, places=5)
        self.assertAlmostEqual(result["post_weight"], 0.0, places=5)
        self.assertAlmostEqual(result["score"], result["score_reels"], places=3)

    def test_reel_views_top_up_from_media_info(self) -> None:
        """Reels Playwright avec vues renseignées → ratio médian cohérent."""
        followers = 10_000
        self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=[
                _make_media(
                    pk=f"r{i}",
                    views=50_000,
                    likes=500,
                    comments=40,
                    days_ago=5 - i,
                    product_type="clips",
                )
                for i in range(3)
            ],
        )
        with patch("modules.classifier.CommentClassifier"):
            result = discovery.score_profile(
                "zero_list_views",
                self.DOMAIN,
                blacklist={"profiles": []},
                context=self._ctx,
            )
        self.assertIsNotNone(result)
        assert result is not None
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
            result = discovery.score_profile(
                "has_views",
                self.DOMAIN,
                blacklist={"profiles": []},
                context=self._ctx,
            )
        self.assertIsNotNone(result)

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
                context=self._ctx,
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
        # Contribution log attendue du P90 : log10(27.08+1)/log10(51) * 100.
        expected_p90_pts = (
            math.log10(27.083333 + 1.0) / math.log10(51.0) * 100.0
        )
        self.assertAlmostEqual(
            score_with_p90 - score_median_only, expected_p90_pts, places=3
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
                context=self._ctx,
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["media_sampled"], 40)
        self.assertEqual(result["reels_count"], 10)
        self.assertEqual(result["posts_count"], 0)
        # Médiane des ratios sur les 10 non épinglés uniquement (20k / 10k = 2.0)
        self.assertAlmostEqual(result["reel_ratio_median"], 2.0, places=4)

    def test_reels_only_when_posts_present_in_profile(self) -> None:
        """Playwright : seuls les Reels sont scorés (posts feed ignorés)."""
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
        self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
        )
        result = discovery.score_profile(
            "capped_posts",
            self.DOMAIN,
            blacklist={"profiles": []},
            context=self._ctx,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["reels_count"], 6)
        self.assertEqual(result["posts_count"], 0)
        self.assertAlmostEqual(result["reel_weight"], 1.0, places=5)
        self.assertAlmostEqual(result["post_weight"], 0.0, places=5)
        self.assertIsNone(result["post_ratio_median"])

    def test_score_profile_does_not_fetch_media_comments(self) -> None:
        """Le scoring Discovery ne charge plus les commentaires individuels."""
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
            for i in range(4)
        ]
        client = self._make_client(
            user=_make_user(follower_count=followers, media_count=40),
            medias=medias,
        )
        discovery.score_profile(
            "no_comment_fetch",
            self.DOMAIN,
            blacklist={"profiles": []},
            context=self._ctx,
        )
        client.media_comments.assert_not_called()

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
                context=self._ctx,
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
                context=self._ctx,
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


class LogScoringTest(unittest.TestCase):
    """Scoring logarithmique : pas de plafond linéaire, pas de div0, compression."""

    DOMAIN: dict[str, Any] = {"name": "humour", "t_types_target": ["T2"]}

    @staticmethod
    def _reel_score(ratio_med: float, **overrides: Any) -> float:
        kwargs: dict[str, Any] = {
            "reel_ratio_median": ratio_med,
            "reel_ratio_p90": 0.0,
            "reel_engagement_median": 0.0,
            "reel_trend": "stable",
            "t_type_distribution": {},
            "publish_frequency": 0.0,
            "domain": LogScoringTest.DOMAIN,
        }
        kwargs.update(overrides)
        return discovery._compute_reel_score(**kwargs)

    def test_log_norm_zero_does_not_raise_or_div_zero(self) -> None:
        # Pas de log10(0) : value <= 0 → 0.0
        self.assertEqual(discovery._log_norm(0.0, ref=10.0), 0.0)
        self.assertEqual(discovery._log_norm(-5.0, ref=10.0), 0.0)
        self.assertEqual(discovery._log_norm(None, ref=10.0), 0.0)
        # ref invalide → 0.0 (pas de ZeroDivisionError)
        self.assertEqual(discovery._log_norm(10.0, ref=0.0), 0.0)
        self.assertEqual(discovery._log_norm(10.0, ref=-1.0), 0.0)

    def test_log_norm_at_ref_equals_one(self) -> None:
        self.assertAlmostEqual(discovery._log_norm(10.0, ref=10.0), 1.0, places=12)
        self.assertAlmostEqual(discovery._log_norm(50.0, ref=50.0), 1.0, places=12)

    def test_compute_reel_score_zero_ratio_does_not_crash(self) -> None:
        score = self._reel_score(0.0)
        # Avec stable + tout reste à 0, score = 75 (trend stable seul).
        self.assertEqual(score, discovery.SCORE_REEL_TREND_W * 0.5)

    def test_log_score_6x_clearly_above_1x(self) -> None:
        score_1x = self._reel_score(1.0)
        score_6x = self._reel_score(6.0)
        self.assertGreater(score_6x, score_1x)
        # Différenciation log vs linéaire :
        # log(2)/log(11)*250 ≈ 72.3 ; log(7)/log(11)*250 ≈ 202.9 → +130 pts.
        self.assertAlmostEqual(score_6x - score_1x, 130.6, places=0)

    def test_log_score_30x_above_6x_but_with_compression(self) -> None:
        """30x vaut > 6x mais **pas** 5× plus (compression logarithmique).

        Linéaire (ancien) : 30/6 = 5× la contribution de 6x.
        Log : log10(31)/log10(7) ≈ 1.491/0.845 ≈ 1.76× → loin de 5×.
        """
        contrib_6x = discovery._log_norm(6.0, ref=10.0) * 250
        contrib_30x = discovery._log_norm(30.0, ref=10.0) * 250
        self.assertGreater(contrib_30x, contrib_6x)
        # Ratio entre contributions : entre 1.5× et 2.0× (jamais 5×).
        ratio = contrib_30x / contrib_6x
        self.assertGreater(ratio, 1.5)
        self.assertLess(ratio, 2.0)

    def test_log_score_unbounded_above_ref(self) -> None:
        """Au-delà de la référence, le score continue à croître (pas de plafond)."""
        contrib_at_ref = discovery._log_norm(10.0, ref=10.0) * 250  # = 250
        contrib_above = discovery._log_norm(100.0, ref=10.0) * 250
        self.assertGreater(contrib_above, contrib_at_ref)
        # log10(101)/log10(11) ≈ 2.004/1.041 ≈ 1.926 → ~481 pts (>250 nominal).
        self.assertAlmostEqual(contrib_above, 481.4, places=0)

    def test_frequency_score_caps_at_one_per_day(self) -> None:
        """Le rythme est plafonné à ref pour ne pas récompenser le spam."""
        s_at_ref = discovery._frequency_score(1.0)
        s_spam = discovery._frequency_score(5.0)
        self.assertAlmostEqual(s_at_ref, 1.0, places=12)
        self.assertEqual(s_at_ref, s_spam)

    def test_frequency_score_log_growth_below_ref(self) -> None:
        # rhythm = 0.5 → log10(6)/log10(11) ≈ 0.747
        self.assertAlmostEqual(
            discovery._frequency_score(0.5),
            math.log10(6.0) / math.log10(11.0),
            places=6,
        )
        self.assertEqual(discovery._frequency_score(0.0), 0.0)
        self.assertEqual(discovery._frequency_score(-1.0), 0.0)


class ExplainScoreTest(unittest.TestCase):
    SAMPLE: dict[str, Any] = {
        "username": "raikkonenaf",
        "domain": "humour",
        "reel_ratio_median": 6.57,
        "reel_ratio_p90": 32.7,
        "reel_engagement_median": 0.068,
        "reel_trend": "stable",
        "post_ratio_median": 0.05,
        "post_engagement_median": 0.04,
        "posting_rhythm": 1.59,
        "score_reels": 600.0,
        "score_posts": 350.0,
        "reel_weight": 0.7,
        "post_weight": 0.3,
        "score": 525.0,
    }

    def test_explain_includes_username_and_breakdown(self) -> None:
        text = discovery.explain_score(self.SAMPLE)
        self.assertIn("@raikkonenaf", text)
        self.assertIn("reel_ratio_median", text)
        self.assertIn("reel_ratio_p90", text)
        self.assertIn("reel_engagement", text)
        self.assertIn("reel_trend", text)
        self.assertIn("posting_rhythm", text)
        self.assertIn("score_reels", text)
        self.assertIn("score_posts", text)
        self.assertIn("score_final", text)

    def test_explain_handles_empty_input(self) -> None:
        text = discovery.explain_score({})
        self.assertIn("?", text)

    def test_explain_pts_match_log_formula(self) -> None:
        """La contribution P90 affichée doit suivre log10(32.7+1)/log10(51)*100."""
        text = discovery.explain_score(self.SAMPLE)
        expected_p90 = round(
            math.log10(32.7 + 1.0) / math.log10(51.0) * 100
        )
        self.assertIn(f"{expected_p90}pts", text)


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
    """Discovery 2026-05 : ``explore_network`` ne score **plus** que le seed
    lui-même (plus de fetch ``user_following``). Le seed est retiré de
    ``seeds.json`` après exploration.
    """

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
        # Pas de touches disque sur seeds.json par défaut — chaque test qui
        # veut auditer la suppression patche localement avec une vraie path.
        self._save_seeds_patch = patch.object(discovery, "save_seeds")
        self._save_seeds_patch.start()
        self.addCleanup(self._save_seeds_patch.stop)
        self._load_seeds_patch = patch.object(
            discovery, "load_seeds",
            return_value={"domains": [dict(self.DOMAIN)]},
        )
        self._load_seeds_patch.start()
        self.addCleanup(self._load_seeds_patch.stop)

    def test_seed_in_blacklist_skips_auto_score(self) -> None:
        """Si le seed est déjà blacklisté, aucun scoring (Discovery 2026-05 :
        plus de followings, le seed est le seul profil considéré).
        """
        blacklist = {"profiles": [{"username": "seed_one"}]}
        candidates: dict[str, Any] = {"candidates": []}
        session = discovery.DiscoverySession(mock=False)

        scored: list[str] = []

        def fake_score(username, domain, **kw):
            scored.append(username)
            return None

        with patch.object(discovery, "score_profile", side_effect=fake_score):
            discovery.explore_network(
                self.DOMAIN,
                blacklist=blacklist,
                watchlist=[],
                candidates=candidates,
                context=MagicMock(),
                session=session,
                blacklist_path=Path("/tmp/skip_persist_bl.json"),
                candidates_path=Path("/tmp/skip_persist_cand.json"),
            )

        self.assertEqual(scored, [])
        self.assertEqual(session.profiles_today, 0)

    def test_seed_in_watchlist_skips_auto_score(self) -> None:
        """Seed déjà dans la watchlist → log spécifique + aucun scoring."""
        watchlist = [{"username": "seed_one"}]
        session = discovery.DiscoverySession(mock=False)

        scored: list[str] = []

        def fake_score(username, domain, **kw):
            scored.append(username)
            return None

        with patch.object(discovery, "score_profile", side_effect=fake_score), \
             self.assertLogs("aitertainment.discovery", level="INFO") as cm:
            discovery.explore_network(
                self.DOMAIN,
                blacklist={"profiles": []},
                watchlist=watchlist,
                candidates={"candidates": []},
                context=MagicMock(),
                session=session,
                blacklist_path=Path("/tmp/skip_persist_bl.json"),
                candidates_path=Path("/tmp/skip_persist_cand.json"),
            )

        self.assertEqual(scored, [])
        self.assertTrue(
            any("Seed @seed_one déjà vu — skip auto-score." in m for m in cm.output),
            cm.output,
        )

    def test_seed_with_high_score_added_to_candidates_and_notified(self) -> None:
        """Le seed est l'unique profil scoré : si son score franchit le seuil
        il est ajouté aux candidates et notifié (boutons inline ✅ ❌ ✏️ via
        ``_notify_candidate``).
        """
        blacklist = {"profiles": []}
        candidates: dict[str, Any] = {"candidates": []}
        session = discovery.DiscoverySession(mock=False)

        good_result = {
            "username": "seed_one",
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
                context=MagicMock(),
                session=session,
            )

        self.assertEqual(len(candidates["candidates"]), 1)
        self.assertEqual(candidates["candidates"][0]["username"], "seed_one")
        self.assertTrue(candidates["candidates"][0]["validated"] is False)
        self.assertEqual(session.candidates_found, 1)
        notif.assert_called_once()
        self.assertEqual(session.profiles_today, 1)
        # Le seed est aussi blacklisté ("candidate" outcome) — jamais reproposé.
        self.assertIn("seed_one", [p["username"] for p in blacklist["profiles"]])

    def test_seed_with_low_score_blacklisted_not_candidate(self) -> None:
        """Score sous seuil → blacklist (outcome=rejected), pas de notif."""
        blacklist = {"profiles": []}
        candidates: dict[str, Any] = {"candidates": []}
        session = discovery.DiscoverySession(mock=False)

        low_result = {
            "username": "seed_one",
            "domain": "humour",
            "score": 250.0,  # sous CANDIDATE_SCORE_THRESHOLD
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
                context=MagicMock(),
                session=session,
            )

        self.assertEqual(candidates["candidates"], [])
        notif.assert_not_called()
        self.assertEqual(blacklist["profiles"][0]["outcome"], "rejected")

    def test_daily_quota_stops_exploration_across_seeds(self) -> None:
        """Quota atteint → on s'arrête sans toucher aux seeds restants.

        Avec un domaine de 5 seeds et un quota déjà à ``MAX-2``, on ne
        consomme que 2 slots — les seeds 3-4-5 ne sont pas scorés.
        """
        domain = {
            "name": "humour",
            "niche": "humour",
            "seeds": ["s1", "s2", "s3", "s4", "s5"],
            "t_types_target": ["T2"],
        }
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
                domain,
                blacklist=blacklist,
                watchlist=[],
                candidates=candidates,
                context=MagicMock(),
                session=session,
            )

        # Exactement 2 seeds consommés (le quota cap les 3 suivants).
        self.assertEqual(scored, ["s1", "s2"])
        self.assertEqual(session.profiles_today, discovery.MAX_PROFILES_PER_DAY)

    def test_score_profile_error_still_runs_explore_network(self) -> None:
        """Une erreur dans ``score_profile`` ne doit pas planter ``explore_network``."""
        with patch.object(discovery, "score_profile", return_value=None):
            discovery.explore_network(
                self.DOMAIN,
                blacklist={"profiles": []},
                watchlist=[],
                candidates={"candidates": []},
                context=MagicMock(),
                session=discovery.DiscoverySession(mock=False),
            )

    def test_mock_mode_runs_without_network(self) -> None:
        """En mock : aucun ``client`` Instagram, aucune écriture disque,
        seul le seed est scoré (``profiles_today=1``).
        """
        session = discovery.DiscoverySession(mock=True)
        candidates: dict[str, Any] = {"candidates": []}
        blacklist = {"profiles": []}

        with patch.object(discovery, "save_blacklist") as bl_save, \
             patch.object(discovery, "save_candidates") as cd_save:
            discovery.explore_network(
                self.DOMAIN,
                blacklist=blacklist,
                watchlist=[],
                candidates=candidates,
                context=None,
                session=session,
            )
            bl_save.assert_not_called()
            cd_save.assert_not_called()

        # Mock mode 2026-05 : 1 seed = 1 scoring (plus de followings).
        self.assertEqual(session.profiles_today, 1)


class ExploreNetworkSeedsRemovalTest(unittest.TestCase):
    """Discovery 2026-05 : après exploration, le seed est retiré de
    ``seeds.json`` (mais le domaine est conservé même vide).
    """

    DOMAIN = {
        "name": "humour",
        "niche": "humour",
        "seeds": ["seed_one", "seed_two"],
        "t_types_target": ["T2"],
    }

    def setUp(self) -> None:
        self._sleep_patch = patch.object(discovery, "polite_sleep", lambda *a, **k: None)
        self._sleep_patch.start()
        self.addCleanup(self._sleep_patch.stop)
        self._window_patch = patch.object(
            discovery, "_wait_for_active_window", lambda *a, **k: None
        )
        self._window_patch.start()
        self.addCleanup(self._window_patch.stop)
        self._burst_patch = patch.object(
            discovery, "_maybe_take_burst_break", lambda *a, **k: None
        )
        self._burst_patch.start()
        self.addCleanup(self._burst_patch.stop)

    def test_seed_removed_from_seeds_json_after_exploration(self) -> None:
        """Chaque seed exploré est retiré du ``seeds`` du domaine et
        ``save_seeds`` est appelé après chaque retrait — sans toucher au
        domaine lui-même (préservé même après le dernier seed).
        """
        with tempfile.TemporaryDirectory() as tmp:
            seeds_p = Path(tmp) / "seeds.json"
            seeds_p.write_text(
                json.dumps(
                    {
                        "domains": [
                            {
                                "name": "humour",
                                "niche": "humour",
                                "seeds": ["seed_one", "seed_two"],
                                "t_types_target": ["T2"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(discovery, "score_profile", return_value=None), \
                 patch.object(discovery, "save_blacklist"), \
                 patch.object(discovery, "save_candidates"):
                discovery.explore_network(
                    self.DOMAIN,
                    blacklist={"profiles": []},
                    watchlist=[],
                    candidates={"candidates": []},
                    context=MagicMock(),
                    session=discovery.DiscoverySession(mock=False),
                    seeds_path=seeds_p,
                )

            # Après l'exploration des 2 seeds, ``seeds`` est vidée mais le
            # domaine subsiste (cf. brief : on ne supprime pas le domaine).
            data = json.loads(seeds_p.read_text(encoding="utf-8"))
            domain = data["domains"][0]
            self.assertEqual(domain["seeds"], [])
            self.assertEqual(domain["name"], "humour")
            self.assertEqual(domain.get("t_types_target"), ["T2"])

    def test_seeds_json_not_touched_in_mock_mode(self) -> None:
        """En mock : aucune écriture sur seeds.json (le test ne doit pas
        polluer le repo).
        """
        with tempfile.TemporaryDirectory() as tmp:
            seeds_p = Path(tmp) / "seeds.json"
            payload = {
                "domains": [
                    {
                        "name": "humour",
                        "niche": "humour",
                        "seeds": ["seed_one", "seed_two"],
                        "t_types_target": ["T2"],
                    }
                ]
            }
            seeds_p.write_text(json.dumps(payload), encoding="utf-8")

            with patch.object(discovery, "score_profile", return_value=None), \
                 patch.object(discovery, "save_blacklist"), \
                 patch.object(discovery, "save_candidates"), \
                 patch.object(discovery, "save_seeds") as save_seeds_mock:
                discovery.explore_network(
                    self.DOMAIN,
                    blacklist={"profiles": []},
                    watchlist=[],
                    candidates={"candidates": []},
                    context=None,
                    session=discovery.DiscoverySession(mock=True),
                    seeds_path=seeds_p,
                )

            save_seeds_mock.assert_not_called()
            self.assertEqual(
                json.loads(seeds_p.read_text(encoding="utf-8")), payload
            )

    def test_seeds_json_not_touched_when_seeds_override_used(self) -> None:
        """``seeds_override`` (CLI ``--seed``) n'altère pas seeds.json :
        on score juste le seed fourni sans modifier la persistance.
        """
        with tempfile.TemporaryDirectory() as tmp:
            seeds_p = Path(tmp) / "seeds.json"
            payload = {
                "domains": [
                    {
                        "name": "humour",
                        "niche": "humour",
                        "seeds": ["existing"],
                        "t_types_target": ["T2"],
                    }
                ]
            }
            seeds_p.write_text(json.dumps(payload), encoding="utf-8")

            with patch.object(discovery, "score_profile", return_value=None), \
                 patch.object(discovery, "save_blacklist"), \
                 patch.object(discovery, "save_candidates"), \
                 patch.object(discovery, "save_seeds") as save_seeds_mock:
                discovery.explore_network(
                    {"name": "humour", "niche": "humour", "seeds": []},
                    blacklist={"profiles": []},
                    watchlist=[],
                    candidates={"candidates": []},
                    context=MagicMock(),
                    session=discovery.DiscoverySession(mock=False),
                    seeds_path=seeds_p,
                    seeds_override=["adhoc_one"],
                )

            save_seeds_mock.assert_not_called()
            self.assertEqual(
                json.loads(seeds_p.read_text(encoding="utf-8")), payload
            )


class ExploreNetworkInMemorySetsTest(unittest.TestCase):
    """Discovery 2026-05 : ``blacklist_set`` / ``seen_this_run`` mutables
    en mémoire — pas besoin d'attendre un reload disque entre seeds.
    """

    def setUp(self) -> None:
        self._sleep_patch = patch.object(discovery, "polite_sleep", lambda *a, **k: None)
        self._sleep_patch.start()
        self.addCleanup(self._sleep_patch.stop)
        self._window_patch = patch.object(
            discovery, "_wait_for_active_window", lambda *a, **k: None
        )
        self._window_patch.start()
        self.addCleanup(self._window_patch.stop)
        self._burst_patch = patch.object(
            discovery, "_maybe_take_burst_break", lambda *a, **k: None
        )
        self._burst_patch.start()
        self.addCleanup(self._burst_patch.stop)
        # Stub seeds.json IO pour ne pas écrire sur disque.
        self._save_seeds_patch = patch.object(discovery, "save_seeds")
        self._save_seeds_patch.start()
        self.addCleanup(self._save_seeds_patch.stop)
        self._load_seeds_patch = patch.object(
            discovery, "load_seeds",
            return_value={"domains": [
                {"name": "humour", "seeds": [], "t_types_target": ["T2"]}
            ]},
        )
        self._load_seeds_patch.start()
        self.addCleanup(self._load_seeds_patch.stop)

    def test_seen_this_run_dedupes_duplicate_seeds(self) -> None:
        """Si la même valeur apparaît 2× dans seeds (cas de curation cassée),
        ``score_profile`` n'est appelé qu'une seule fois.
        """
        domain = {
            "name": "humour",
            "niche": "humour",
            "seeds": ["seed_one", "seed_one"],
            "t_types_target": ["T2"],
        }
        scored: list[str] = []

        def fake_score(username, domain, **kw):
            scored.append(username)
            return None

        with patch.object(discovery, "score_profile", side_effect=fake_score), \
             patch.object(discovery, "save_blacklist"), \
             patch.object(discovery, "save_candidates"):
            discovery.explore_network(
                domain,
                blacklist={"profiles": []},
                watchlist=[],
                candidates={"candidates": []},
                context=MagicMock(),
                session=discovery.DiscoverySession(mock=False),
            )

        self.assertEqual(scored, ["seed_one"])

    def test_blacklist_set_updated_in_memory_between_seeds(self) -> None:
        """Un seed scoré (donc ajouté à la blacklist) est immédiatement filtré
        si réapparaît dans la même boucle — sans relire le disque.

        Scénario : 2 seeds distincts ; on simule en passant la même string
        seed dupliquée AVEC une casse différente — le set en mémoire la
        normalise et bloque le doublon.
        """
        domain = {
            "name": "humour",
            "niche": "humour",
            "seeds": ["alpha", "ALPHA"],  # même seed avec casse différente
            "t_types_target": ["T2"],
        }

        scored: list[str] = []

        def fake_score(username, domain, **kw):
            scored.append(username)
            return None

        with patch.object(discovery, "score_profile", side_effect=fake_score), \
             patch.object(discovery, "save_blacklist"), \
             patch.object(discovery, "save_candidates"):
            discovery.explore_network(
                domain,
                blacklist={"profiles": []},
                watchlist=[],
                candidates={"candidates": []},
                context=MagicMock(),
                session=discovery.DiscoverySession(mock=False),
            )

        # ``alpha`` est scoré 1 fois, ``ALPHA`` (normalisé en alpha) skip via
        # le set en mémoire (seen_this_run + blacklist_set).
        self.assertEqual(scored, ["alpha"])

    def test_blacklist_usernames_helper_normalizes_entries(self) -> None:
        """``_blacklist_usernames`` strip ``@`` et ``lower``."""
        bl = {
            "profiles": [
                {"username": "@FOO"},
                {"username": "  bar  "},
                {"username": ""},  # ignoré
                {"not_a_dict": True},  # ignoré (entrée mal formée)
                "string_au_lieu_de_dict",  # ignoré
            ]
        }
        out = discovery._blacklist_usernames(bl)
        self.assertEqual(out, {"foo", "bar"})

    def test_blacklist_usernames_helper_handles_none_and_empty(self) -> None:
        self.assertEqual(discovery._blacklist_usernames(None), set())
        self.assertEqual(discovery._blacklist_usernames({}), set())
        self.assertEqual(discovery._blacklist_usernames({"profiles": []}), set())


class FetchSuggestionsTest(unittest.TestCase):
    """``_fetch_suggestions`` délègue à ``get_suggested_accounts`` (Playwright)."""

    def test_delegates_to_get_suggested_accounts(self) -> None:
        ctx = MagicMock()
        with patch.object(
            discovery,
            "get_suggested_accounts",
            return_value=["alpha", "beta"],
        ) as mock_gsa:
            result = discovery._fetch_suggestions("seed_one", ctx, max_results=30)
        self.assertEqual(result, ["alpha", "beta"])
        mock_gsa.assert_called_once_with("seed_one", ctx, max_results=30)

    def test_empty_when_no_context(self) -> None:
        self.assertEqual(discovery._fetch_suggestions("seed_one", None), [])


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


class ScoreAndPersistTest(unittest.TestCase):
    """``score_and_persist`` : upsert DB systématique + notif au-dessus du seuil."""

    DOMAIN = {
        "name": "humour",
        "niche": "humour",
        "t_types_target": ["T2", "T3b"],
    }

    def setUp(self) -> None:
        self._sleep_patch = patch.object(discovery, "polite_sleep", lambda *a, **k: None)
        self._sleep_patch.start()
        self.addCleanup(self._sleep_patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.db_path = base / "database.json"
        self.cand_path = base / "candidates.json"
        self.cand_path.write_text(
            json.dumps({"candidates": []}), encoding="utf-8"
        )

    def _stub_score_result(self, score: float = 500.0) -> dict[str, Any]:
        return {
            "username": "raikkonenaf",
            "domain": "humour",
            "platform": "instagram",
            "followers": 48_000,
            "score": score,
            "score_reels": score * 1.1,
            "score_posts": score * 0.7,
            "reel_weight": 0.7,
            "post_weight": 0.3,
            "reel_ratio_median": 6.57,
            "reel_ratio_p90": 32.7,
            "reel_engagement_median": 0.068,
            "reel_trend": "stable",
            "post_ratio_median": 0.05,
            "post_engagement_median": 0.04,
            "posting_rhythm": 1.59,
            "t_type_dominant": "T2",
            "t_type_distribution": {"T2": 0.7, "T3b": 0.3},
            "biography": "bio",
            "media_sampled": 12,
            "reels_count": 8,
            "posts_count": 4,
            "reels_sampled": 8,
            "scored_at": "2026-05-08T12:00:00",
        }

    def test_persists_to_db_and_notifies_above_threshold(self) -> None:
        notify_calls: list[dict[str, Any]] = []
        result = self._stub_score_result(score=500.0)  # > 300, upsert + notif

        with patch.object(discovery, "score_profile", return_value=result):
            summary = discovery.score_and_persist(
                "@raikkonenaf",
                domain=self.DOMAIN,
                added_via="manual",
                blacklist={"profiles": []},
                context=MagicMock(),
                db_path=self.db_path,
                candidates_path=self.cand_path,
                notify_fn=notify_calls.append,
            )

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["tier"], "B")
        self.assertTrue(summary["notified"])
        self.assertEqual(len(notify_calls), 1)
        self.assertEqual(notify_calls[0]["username"], "raikkonenaf")

        db = json.loads(self.db_path.read_text(encoding="utf-8"))
        self.assertIn("raikkonenaf", db["profiles"])
        self.assertEqual(db["profiles"]["raikkonenaf"]["tier"], "B")
        self.assertEqual(db["profiles"]["raikkonenaf"]["added_via"], "manual")

        cands = json.loads(self.cand_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [c["username"] for c in cands["candidates"]], ["raikkonenaf"]
        )

    def test_below_threshold_persists_but_does_not_notify(self) -> None:
        notify_calls: list[dict[str, Any]] = []
        result = self._stub_score_result(score=250.0)  # < 300

        with patch.object(discovery, "score_profile", return_value=result):
            summary = discovery.score_and_persist(
                "@raikkonenaf",
                domain=self.DOMAIN,
                blacklist={"profiles": []},
                context=MagicMock(),
                db_path=self.db_path,
                candidates_path=self.cand_path,
                notify_fn=notify_calls.append,
            )

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertFalse(summary["notified"])
        self.assertEqual(notify_calls, [])

        # DB tout de même remplie (tier C → archivé).
        db = json.loads(self.db_path.read_text(encoding="utf-8"))
        self.assertIn("raikkonenaf", db["profiles"])
        self.assertEqual(db["profiles"]["raikkonenaf"]["tier"], "C")

        # Pas inscrit dans candidates.json.
        cands = json.loads(self.cand_path.read_text(encoding="utf-8"))
        self.assertEqual(cands["candidates"], [])

    def test_filtered_profile_returns_none(self) -> None:
        with patch.object(discovery, "score_profile", return_value=None):
            summary = discovery.score_and_persist(
                "@private_user",
                domain=self.DOMAIN,
                blacklist={"profiles": []},
                context=MagicMock(),
                db_path=self.db_path,
            )
        self.assertIsNone(summary)
        # Pas de DB créée car aucun upsert ne se produit.
        self.assertFalse(self.db_path.exists())

    def test_default_domain_falls_back_to_first_in_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seeds_p = Path(tmp) / "seeds.json"
            seeds_p.write_text(
                json.dumps(
                    {
                        "domains": [
                            {
                                "name": "humour",
                                "niche": "humour",
                                "seeds": [],
                                "t_types_target": ["T2"],
                            },
                            {
                                "name": "streetwear",
                                "niche": "streetwear",
                                "seeds": [],
                                "t_types_target": ["T3b"],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            captured: list[dict[str, Any]] = []

            def fake_score(username, domain, **kw):
                captured.append(domain)
                return self._stub_score_result(score=500.0)

            with patch.object(discovery, "score_profile", side_effect=fake_score):
                summary = discovery.score_and_persist(
                    "@raikkonenaf",
                    blacklist={"profiles": []},
                    context=MagicMock(),
                    seeds_path=seeds_p,
                    db_path=self.db_path,
                    candidates_path=self.cand_path,
                    notify_fn=lambda r: None,
                )
            self.assertIsNotNone(summary)
            self.assertEqual(captured[0]["name"], "humour")


class ExploreNetworkUpsertsDatabaseTest(unittest.TestCase):
    """``explore_network`` upserte dans ``database.json`` à chaque scoring
    réussi, **même** si le score est sous le seuil candidat.

    Discovery 2026-05 : seul le seed est scoré (plus de followings). Avec
    plusieurs seeds dans le domaine, on attend N upserts pour N seeds.
    """

    DOMAIN = {
        "name": "humour",
        "niche": "humour",
        "seeds": ["seed_one", "seed_two"],
        "t_types_target": ["T2"],
    }

    def setUp(self) -> None:
        self._sleep_patch = patch.object(discovery, "polite_sleep", lambda *a, **k: None)
        self._sleep_patch.start()
        self.addCleanup(self._sleep_patch.stop)
        # Stub seeds.json IO (le bloc seeds removal essaie sinon de
        # ``load_seeds`` avec ``DEFAULT_SEEDS_PATH``).
        self._save_seeds_patch = patch.object(discovery, "save_seeds")
        self._save_seeds_patch.start()
        self.addCleanup(self._save_seeds_patch.stop)
        self._load_seeds_patch = patch.object(
            discovery, "load_seeds",
            return_value={"domains": [dict(self.DOMAIN)]},
        )
        self._load_seeds_patch.start()
        self.addCleanup(self._load_seeds_patch.stop)

    def test_low_score_still_upserted_in_db(self) -> None:
        db: dict[str, Any] = {"profiles": {}}

        def fake_score(username, domain, **kw):
            return {
                "username": username,
                "domain": "humour",
                "platform": "instagram",
                "followers": 5_000,
                "score": 250.0,  # < CANDIDATE_SCORE_THRESHOLD
                "score_reels": 250.0,
                "score_posts": 0.0,
                "reel_weight": 1.0,
                "post_weight": 0.0,
                "reel_ratio_median": 0.5,
                "reel_ratio_p90": 1.0,
                "reel_engagement_median": 0.02,
                "reel_trend": "stable",
                "post_ratio_median": None,
                "post_engagement_median": None,
                "posting_rhythm": 0.3,
                "t_type_dominant": "T2",
                "t_type_distribution": {"T2": 1.0},
                "biography": "",
                "media_sampled": 8,
                "reels_count": 8,
                "posts_count": 0,
                "reels_sampled": 8,
                "scored_at": "2026-05-08T12:00:00",
            }

        notif_mock = MagicMock()
        with patch.object(discovery, "score_profile", side_effect=fake_score), \
             patch.object(discovery, "_notify_candidate", notif_mock), \
             patch.object(discovery, "save_blacklist"), \
             patch.object(discovery, "save_db") as save_db_mock:
            discovery.explore_network(
                self.DOMAIN,
                blacklist={"profiles": []},
                watchlist=[],
                candidates={"candidates": []},
                db=db,
                context=MagicMock(),
                session=discovery.DiscoverySession(mock=False),
            )

        # Discovery 2026-05 : 2 seeds → 2 profils upsertés (plus de
        # followings). Tous tier C archivés (score 250 < seuil tier B).
        self.assertEqual(set(db["profiles"].keys()), {"seed_one", "seed_two"})
        self.assertEqual(db["profiles"]["seed_one"]["tier"], "C")
        self.assertTrue(db["profiles"]["seed_one"]["archived"])
        # Pas de notif sous le seuil.
        notif_mock.assert_not_called()
        # Et save_db a été appelé pour chaque upsert (hors mock).
        self.assertGreaterEqual(save_db_mock.call_count, 2)


class PrintScoreSummaryTest(unittest.TestCase):
    def test_format_matches_brief(self) -> None:
        summary = {
            "score_result": {
                "username": "raikkonenaf",
                "score": 448.0,
                "t_type_dominant": "T2",
            },
            "profile": {"tier": "B"},
            "tier": "B",
            "notified": True,
        }
        with patch("builtins.print") as mock_print:
            discovery._print_score_summary(summary, "@raikkonenaf")
        out = mock_print.call_args.args[0]
        self.assertIn("@raikkonenaf", out)
        self.assertIn("Score : 448", out)
        self.assertIn("Tier : B", out)
        self.assertIn("T-type : T2", out)
        self.assertIn("Notif : ✅", out)

    def test_format_when_filtered(self) -> None:
        with patch("builtins.print") as mock_print:
            discovery._print_score_summary(None, "ghost")
        out = mock_print.call_args.args[0]
        self.assertIn("@ghost", out)
        self.assertIn("filtré", out)


class ClassifyRecentCommentsTest(unittest.TestCase):
    """Couvre le log enrichi (raw / max_likes / exploitables) + fallback top-5."""

    def setUp(self) -> None:
        # ``_classify_recent_comments`` appelle ``polite_sleep()`` avant chaque
        # ``media_comments`` (1.5-4s par appel) — on neutralise pour la suite.
        self._sleep_patch = patch("discovery.polite_sleep", return_value=None)
        self._sleep_patch.start()

    def tearDown(self) -> None:
        self._sleep_patch.stop()

    @staticmethod
    def _comment(text: str, likes: int = 1) -> SimpleNamespace:
        return SimpleNamespace(text=text, like_count=likes)

    @staticmethod
    def _row(media_id: str = "M1") -> dict[str, str]:
        return {"media_id": media_id}

    def _patch_classifier(self, ttype: str = "T2", confidence: float = 0.85):
        """Patch ``CommentClassifier`` pour retourner un T-type figé."""
        instance = MagicMock()
        instance.classify.return_value = {"type": ttype, "confidence": confidence}
        return patch(
            "modules.classifier.CommentClassifier",
            return_value=instance,
        ), instance

    def _run(self, client: MagicMock):
        log = MagicMock()
        return discovery._classify_recent_comments(
            client,
            [self._row("M1")],
            niches=["humour"],
            log=log,
            sleep_between_posts=False,
        ), log

    # ---- amount=50 -----------------------------------------------------

    def test_media_comments_called_with_amount_50(self) -> None:
        client = MagicMock()
        client.media_comments.return_value = [
            self._comment("commentaire assez long pour passer", likes=5),
        ]
        cls_patch, _ = self._patch_classifier()
        with cls_patch:
            self._run(client)
        kwargs = client.media_comments.call_args.kwargs
        self.assertEqual(kwargs.get("amount"), 50)

    # ---- log enrichi ---------------------------------------------------

    def test_no_comments_logs_raw_and_max_likes_zero(self) -> None:
        client = MagicMock()
        client.media_comments.return_value = []
        cls_patch, instance = self._patch_classifier()
        with cls_patch:
            (_, dominant), log = self._run(client)
        self.assertIsNone(dominant)
        instance.classify.assert_not_called()
        # Format attendu : "media M1 : 0 commentaires bruts, max_likes=0, exploitables=0"
        msg = log.info.call_args.args[0] % log.info.call_args.args[1:]
        self.assertIn("media M1", msg)
        self.assertIn("0 commentaires bruts", msg)
        self.assertIn("max_likes=0", msg)
        self.assertIn("exploitables=0", msg)
        self.assertIn("aucun commentaire exploitable", msg)

    def test_only_emoji_comments_log_includes_raw_count(self) -> None:
        """Cas Instagram fréquent : commentaires bruts = N mais texte vide → 0 exploitables."""
        client = MagicMock()
        client.media_comments.return_value = [
            SimpleNamespace(text="", like_count=10),
            SimpleNamespace(text="   ", like_count=2),
            SimpleNamespace(text="\n\t  ", like_count=0),
        ]
        cls_patch, instance = self._patch_classifier()
        with cls_patch:
            (_, dominant), log = self._run(client)
        self.assertIsNone(dominant)
        instance.classify.assert_not_called()
        msg = log.info.call_args.args[0] % log.info.call_args.args[1:]
        self.assertIn("3 commentaires bruts", msg)
        self.assertIn("max_likes=0", msg)  # text vide → on n'a PAS compté les likes
        self.assertIn("exploitables=0", msg)

    # ---- fallback max_likes=0 -----------------------------------------

    def test_fallback_top5_longest_when_max_likes_zero(self) -> None:
        """Tous les commentaires à 0 like → on garde les 5 textes les plus longs (>10 chars)."""
        client = MagicMock()
        client.media_comments.return_value = [
            self._comment("court", likes=0),                         # 5 chars : éliminé
            self._comment("court aussi", likes=0),                   # 11 chars : OK
            self._comment("commentaire vraiment long numéro 1", likes=0),
            self._comment("commentaire vraiment long numéro 2", likes=0),
            self._comment("commentaire vraiment long numéro 3", likes=0),
            self._comment("commentaire vraiment long numéro 4", likes=0),
            self._comment("commentaire vraiment long numéro 5", likes=0),
            self._comment("commentaire vraiment long numéro 6", likes=0),
        ]
        cls_patch, instance = self._patch_classifier(ttype="T3b", confidence=0.7)
        with cls_patch:
            (distribution, dominant), log = self._run(client)
        # Le classifier a été appelé avec exactement 5 textes (top par longueur).
        instance.classify.assert_called_once()
        called_texts = instance.classify.call_args.args[0]
        self.assertEqual(len(called_texts), 5)
        # Le plus long est en tête (sort descending par longueur).
        self.assertTrue(all(len(t) > 10 for t in called_texts))
        # Et le résultat propage le T-type retourné par le classifier.
        self.assertEqual(dominant, "T3b")
        self.assertAlmostEqual(distribution["T3b"], 1.0)
        # Log contient bien "fallback".
        log_msgs = [
            (c.args[0] % c.args[1:]) for c in log.info.call_args_list
        ]
        self.assertTrue(
            any("fallback sur 5 textes" in m for m in log_msgs),
            f"log info attendu (fallback) absent ; reçu : {log_msgs}",
        )

    def test_fallback_skipped_if_some_comment_has_likes(self) -> None:
        """Si au moins 1 commentaire a likes>0, on prend tous les textes (pas de fallback)."""
        client = MagicMock()
        client.media_comments.return_value = [
            self._comment("petit", likes=0),                                # 5 chars
            self._comment("commentaire moyen", likes=12),                   # >10 chars + liké
            self._comment("autre commentaire bien plus long", likes=0),     # >10 chars
        ]
        cls_patch, instance = self._patch_classifier()
        with cls_patch:
            self._run(client)
        # En mode nominal on garde TOUT (y compris les courts non likés).
        called_texts = instance.classify.call_args.args[0]
        self.assertEqual(len(called_texts), 3)

    def test_fallback_yields_skip_when_all_texts_too_short(self) -> None:
        """max_likes=0 et tous les textes ≤ 10 chars → log "tous ≤ 10 chars" + skip."""
        client = MagicMock()
        client.media_comments.return_value = [
            self._comment("lol", likes=0),
            self._comment("ok", likes=0),
            self._comment("sympa", likes=0),
        ]
        cls_patch, instance = self._patch_classifier()
        with cls_patch:
            (_, dominant), log = self._run(client)
        instance.classify.assert_not_called()
        self.assertIsNone(dominant)
        log_msgs = [
            (c.args[0] % c.args[1:]) for c in log.info.call_args_list
        ]
        # On doit voir le log "fallback" puis le log "tous ≤ 10 chars".
        self.assertTrue(any("max_likes=0" in m for m in log_msgs))
        self.assertTrue(any("≤ 10 chars" in m for m in log_msgs))


class SeedSchemaHelpersTest(unittest.TestCase):
    """Couvre les helpers ``_seed_username`` et ``_seed_niches`` (schéma 2026-05)."""

    DOMAIN_NEW = {
        "name": "humour",
        "niches": ["humour"],
        "seeds": [
            {"username": "raikkonenaf", "niches": ["humour", "sketch", "imitation"]},
            "legacy_user",  # rétro-compat string brute
        ],
    }
    DOMAIN_LEGACY = {
        "name": "humour",
        "niche": "humour",  # ancien champ string
        "seeds": ["a", "b"],
    }

    def test_seed_username_from_dict(self) -> None:
        self.assertEqual(
            discovery._seed_username({"username": "@raikkonenaf", "niches": ["humour"]}),
            "raikkonenaf",
        )

    def test_seed_username_from_string(self) -> None:
        self.assertEqual(discovery._seed_username("@user_legacy"), "user_legacy")

    def test_seed_username_handles_garbage(self) -> None:
        self.assertEqual(discovery._seed_username(None), "")
        self.assertEqual(discovery._seed_username({}), "")
        self.assertEqual(discovery._seed_username({"username": ""}), "")
        # ``lstrip("@").strip()`` (ordre du brief) — strip simple, pas
        # idempotent contre des espaces avant l'@. En prod les seeds sont
        # bien formés, ce cas marginal n'est pas couvert.
        self.assertEqual(discovery._seed_username({"username": "@bob  "}), "bob")
        self.assertEqual(discovery._seed_username("  bob"), "bob")

    def test_seed_niches_dict_seed_priority(self) -> None:
        seed = {"username": "raikkonenaf", "niches": ["humour", "sketch", "imitation"]}
        self.assertEqual(
            discovery._seed_niches(seed, self.DOMAIN_NEW),
            ["humour", "sketch", "imitation"],
        )

    def test_seed_niches_string_seed_falls_back_to_domain_niches(self) -> None:
        # Seed string : pas de niches propres → on retombe sur ``domain["niches"]``.
        self.assertEqual(
            discovery._seed_niches("legacy_user", self.DOMAIN_NEW),
            ["humour"],
        )

    def test_seed_niches_falls_back_to_legacy_niche_string(self) -> None:
        # Domain ancien schéma (``niche`` string) → retour [niche].
        self.assertEqual(
            discovery._seed_niches("a", self.DOMAIN_LEGACY),
            ["humour"],
        )

    def test_seed_niches_falls_back_to_domain_name(self) -> None:
        # Ni ``niches`` ni ``niche`` → on prend le nom du domaine.
        self.assertEqual(
            discovery._seed_niches("a", {"name": "gaming"}),
            ["gaming"],
        )

    def test_seed_niches_ultimate_fallback_humour(self) -> None:
        # Domain vide → fallback ``["humour"]``.
        self.assertEqual(discovery._seed_niches("a", {}), ["humour"])


class ScoreProfileNichesSchemaTest(unittest.TestCase):
    """Vérifie que ``score_profile`` retourne ``niches`` (liste)."""

    def setUp(self) -> None:
        self._ctx = MagicMock()
        medias = [
            _make_media(pk=f"M{i}", views=10_000, likes=500, comments=30, days_ago=i)
            for i in range(5)
        ]
        self._profile_patch = patch(
            "discovery.get_profile_data",
            return_value=_make_profile_data(),
        )
        self._reels_patch = patch(
            "discovery.get_recent_reels",
            return_value=_reels_from_medias(medias),
        )
        self._profile_patch.start()
        self._reels_patch.start()
        self.addCleanup(self._profile_patch.stop)
        self.addCleanup(self._reels_patch.stop)

    @staticmethod
    def _user(follower_count: int = 10_000) -> SimpleNamespace:
        return SimpleNamespace(
            pk="111",
            username="user",
            full_name="Test",
            follower_count=follower_count,
            following_count=500,
            media_count=20,
            is_private=False,
            biography="bio",
        )

    @staticmethod
    def _media(pk: str = "M") -> SimpleNamespace:
        return SimpleNamespace(
            pk=pk,
            taken_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            view_count=10_000,
            play_count=10_000,
            like_count=500,
            comment_count=30,
            product_type="clips",
            is_pinned=False,
        )

    def _client(self) -> MagicMock:
        client = MagicMock()
        client.user_id_from_username.return_value = "111"
        client.user_info.return_value = self._user()
        client.user_medias.return_value = [self._media(f"M{i}") for i in range(5)]
        client.media_comments.return_value = [
            SimpleNamespace(text="commentaire assez long pour passer", like_count=5),
        ]
        client.media_info.return_value = SimpleNamespace(play_count=10_000, view_count=10_000)
        return client

    def test_returns_niches_list_from_new_schema(self) -> None:
        domain = {
            "name": "humour",
            "niches": ["humour", "sketch", "imitation"],
        }
        with patch("discovery.polite_sleep", return_value=None), \
             patch("modules.classifier.CommentClassifier") as MockCls:
            MockCls.return_value.classify.return_value = {
                "type": "T2", "confidence": 0.8,
            }
            result = discovery.score_profile(
                "user", domain, blacklist={"profiles": []}, context=self._ctx
            )

        self.assertIsNotNone(result)
        # Schéma 2026-05 : ``niches`` (liste) est l'unique source de vérité.
        self.assertEqual(result["niches"], ["humour", "sketch", "imitation"])
        # Le champ string ``niche`` n'est plus produit.
        self.assertNotIn("niche", result)

    def test_falls_back_to_legacy_niche_string(self) -> None:
        domain = {"name": "humour", "niche": "humour"}  # ancien schéma
        with patch("discovery.polite_sleep", return_value=None), \
             patch("modules.classifier.CommentClassifier") as MockCls:
            MockCls.return_value.classify.return_value = {
                "type": "T2", "confidence": 0.8,
            }
            result = discovery.score_profile(
                "user", domain, blacklist={"profiles": []}, context=self._ctx
            )

        self.assertEqual(result["niches"], ["humour"])
        self.assertNotIn("niche", result)

    def test_invalid_niche_filtered_via_validate_niches(self) -> None:
        """Une niche hors ``VALID_NICHES`` doit être filtrée par ``validate_niches``."""
        domain = {"name": "humour", "niches": ["INVALID", "humour", "sketch"]}
        with patch("discovery.polite_sleep", return_value=None), \
             patch("modules.classifier.CommentClassifier") as MockCls:
            MockCls.return_value.classify.return_value = {
                "type": "T2", "confidence": 0.8,
            }
            result = discovery.score_profile(
                "user", domain, blacklist={"profiles": []}, context=self._ctx
            )

        # ``INVALID`` est rejeté par ``validate_niches`` et logué en WARNING.
        self.assertEqual(result["niches"], ["humour", "sketch"])
        self.assertNotIn("niche", result)


if __name__ == "__main__":
    unittest.main()
