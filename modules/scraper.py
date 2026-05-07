"""Récupération des données publiques (Apify Instagram / TikTok)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from modules.detector import CreatorStats


def fetch_creator_recent_posts(platform: str, username: str) -> list[dict]:
    """Retourne les posts récents et métriques brutes pour un créateur."""
    raise NotImplementedError


def fetch_creator_stats_mock(creator: CreatorStats) -> CreatorStats:
    """Jeu de données de démo pour le MVP (remplacé plus tard par Apify / scraper)."""
    t0 = datetime.now(timezone.utc)
    old_times = [t0 - timedelta(days=30 - i) for i in range(6)]
    olds: list[dict[str, Any]] = [
        {
            "video_id": f"mock_old_{i}",
            "views": 100,
            "likes": 5,
            "comments": 0,
            "shares": 0,
            "saves": 0,
            "posted_at": old_times[i],
        }
        for i in range(6)
    ]
    v2 = {
        "video_id": "mock_v2",
        "views": 5000,
        "likes": 5000,
        "comments": 400,
        "shares": 100,
        "saves": 50,
        "posted_at": t0 - timedelta(hours=72),
    }
    v1 = {
        "video_id": "mock_v1",
        "views": 5000,
        "likes": 5000,
        "comments": 400,
        "shares": 100,
        "saves": 50,
        "posted_at": t0 - timedelta(hours=48),
    }
    if creator.platform.lower() == "instagram":
        link = f"https://www.instagram.com/reel/{creator.username}_mock/"
    else:
        link = f"https://www.tiktok.com/@{creator.username}/video/mock123"
    v0: dict[str, Any] = {
        "video_id": "mock_headline",
        "views": 5000,
        "likes": 50_000,
        "comments": 5000,
        "shares": 2000,
        "saves": 1000,
        "posted_at": t0 - timedelta(hours=2),
        "audio_id": "mock_snd",
        "audio_reels_count": 100,
        "audio_is_recent": True,
        "duration_sec": 90,
        "url": link,
    }
    return CreatorStats(
        creator_id=creator.creator_id,
        platform=creator.platform,
        username=creator.username,
        followers=max(int(creator.followers), 100),
        recent_videos=olds + [v2, v1, v0],
        follower_growth_7d_pct=creator.follower_growth_7d_pct
        if creator.follower_growth_7d_pct is not None
        else 25.0,
    )


def mock_recent_comments(username: str, n: int = 30) -> list[str]:
    """Commentaires fictifs pour brancher CommentClassifier (Apify plus tard)."""
    u = username.lstrip("@")
    return [f"[mock] com #{i+1} — réaction sur @{u}" for i in range(n)]
