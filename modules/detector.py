"""Score composite et détection (groupes de signaux MVP)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_DATA = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CREATORS_CSV = _PROJECT_DATA / "creators.csv"

# Délai minimal avant de compter la vélocité (évite division ~0 juste après publish)
_VELOCITY_MIN_ELAPSED_S = 2.0 / 3.0


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _video_to_serializable(v: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "video_id": v["video_id"],
        "views": int(v["views"]),
        "likes": int(v["likes"]),
        "comments": int(v["comments"]),
        "shares": int(v["shares"]),
        "saves": int(v["saves"]),
        "posted_at": _ensure_utc(v["posted_at"]).isoformat(),
    }
    aid = v.get("audio_id")
    if aid is not None:
        out["audio_id"] = aid
    if "duration_sec" in v and v["duration_sec"] is not None:
        out["duration_sec"] = int(v["duration_sec"])
    if "audio_reels_count" in v and v["audio_reels_count"] is not None:
        out["audio_reels_count"] = int(v["audio_reels_count"])
    if "audio_is_recent" in v and v["audio_is_recent"] is not None:
        out["audio_is_recent"] = bool(v["audio_is_recent"])
    return out


def _video_from_dict(d: dict[str, Any]) -> dict[str, Any]:
    posted = d["posted_at"]
    if isinstance(posted, str):
        posted = datetime.fromisoformat(posted.replace("Z", "+00:00"))
    v: dict[str, Any] = {
        "video_id": str(d["video_id"]),
        "views": int(d["views"]),
        "likes": int(d["likes"]),
        "comments": int(d["comments"]),
        "shares": int(d["shares"]),
        "saves": int(d["saves"]),
        "posted_at": posted,
    }
    if "audio_id" in d and d["audio_id"] is not None and d["audio_id"] != "":
        v["audio_id"] = d["audio_id"]
    if "duration_sec" in d and d["duration_sec"] is not None and d["duration_sec"] != "":
        v["duration_sec"] = int(float(d["duration_sec"]))
    if "audio_reels_count" in d and d["audio_reels_count"] is not None and d["audio_reels_count"] != "":
        v["audio_reels_count"] = int(float(d["audio_reels_count"]))
    if "audio_is_recent" in d and d["audio_is_recent"] is not None and d["audio_is_recent"] != "":
        v["audio_is_recent"] = bool(d["audio_is_recent"])
    return v


def _sorted_videos_recent_first(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(videos, key=lambda x: _ensure_utc(x["posted_at"]), reverse=True)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class CreatorStats:
    """Métriques agrégées d'un créateur et de ses vidéos récentes."""

    creator_id: str
    platform: str  # "instagram" ou "tiktok"
    username: str
    followers: int
    recent_videos: list[dict[str, Any]] = field(default_factory=list)
    follower_growth_7d_pct: float | None = None  # ex. 5.0 pour +5 % sur 7 jours

    def get_baseline(self) -> float:
        """Moyenne des vues sur les 10 vidéos les plus récentes (ou moins s'il y en a moins)."""
        if not self.recent_videos:
            return 0.0
        sorted_v = _sorted_videos_recent_first(self.recent_videos)
        last_n = sorted_v[:10]
        total_views = sum(int(v["views"]) for v in last_n)
        return total_views / len(last_n)

    def get_velocity(self, video_id: str, *, now: datetime | None = None) -> float:
        """Ratio vues / heures écoulées depuis la publication (vues par heure).

        Si moins de 2/3 s se sont écoulées depuis ``posted_at``, le calcul part
        comme si 2/3 s s'étaient écoulées (plancher anti pic artificiel).
        """
        now_utc = _ensure_utc(now) if now else datetime.now(timezone.utc)
        for v in self.recent_videos:
            if str(v["video_id"]) != str(video_id):
                continue
            posted = _ensure_utc(v["posted_at"])
            elapsed_s = max((now_utc - posted).total_seconds(), _VELOCITY_MIN_ELAPSED_S)
            hours = elapsed_s / 3600.0
            return float(v["views"]) / hours
        raise ValueError(f"video_id introuvable: {video_id}")

    def to_csv_row(self) -> dict[str, Any]:
        serializable = [_video_to_serializable(v) for v in self.recent_videos]
        row: dict[str, Any] = {
            "creator_id": self.creator_id,
            "platform": self.platform,
            "username": self.username,
            "followers": int(self.followers),
            "recent_videos": json.dumps(serializable, ensure_ascii=False),
        }
        if self.follower_growth_7d_pct is not None:
            row["follower_growth_7d_pct"] = float(self.follower_growth_7d_pct)
        else:
            row["follower_growth_7d_pct"] = ""
        return row

    @classmethod
    def from_csv_row(cls, row: dict[str, Any]) -> CreatorStats:
        import math

        raw = row.get("recent_videos", "[]")
        if isinstance(raw, float) and (raw != raw or math.isnan(raw)):  # NaN
            raw = "[]"
        if not isinstance(raw, str):
            raw = str(raw)
        parsed = json.loads(raw) if raw.strip() else []
        videos = [_video_from_dict(x) for x in parsed]
        g = row.get("follower_growth_7d_pct")
        fg: float | None
        if g is None or (isinstance(g, float) and (g != g or math.isnan(g))):
            fg = None
        elif isinstance(g, str) and not str(g).strip():
            fg = None
        else:
            try:
                fg = float(g)
            except (TypeError, ValueError):
                fg = None
        return cls(
            creator_id=str(row["creator_id"]),
            platform=str(row["platform"]),
            username=str(row["username"]),
            followers=int(row["followers"]),
            recent_videos=videos,
            follower_growth_7d_pct=fg,
        )


def save_creators_csv(
    creators: list[CreatorStats],
    path: str | Path | None = None,
) -> None:
    """Écrit une liste de CreatorStats dans un CSV via pandas."""
    import pandas as pd

    p = Path(path) if path else DEFAULT_CREATORS_CSV
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [c.to_csv_row() for c in creators]
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)


def _load_creators_csv_stdlib(p: Path) -> list[CreatorStats]:
    import csv

    with p.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "creator_id" not in reader.fieldnames:
            return []
        rows = list(reader)
    if not rows:
        return []
    return [CreatorStats.from_csv_row(dict(r)) for r in rows]


def load_creators_csv(path: str | Path | None = None) -> list[CreatorStats]:
    """Charge les CreatorStats depuis un CSV (pandas si dispo, sinon csv stdlib)."""
    p = Path(path) if path else DEFAULT_CREATORS_CSV
    if not p.exists():
        return []
    try:
        import pandas as pd

        df = pd.read_csv(p)
        if df.empty:
            return []
        if "creator_id" not in df.columns:
            return []
        return [CreatorStats.from_csv_row(row.to_dict()) for _, row in df.iterrows()]
    except Exception:
        return _load_creators_csv_stdlib(p)


def _minutes_since_post(posted_at: datetime, now: datetime) -> float:
    elapsed_s = max(
        (_ensure_utc(now) - _ensure_utc(posted_at)).total_seconds(),
        _VELOCITY_MIN_ELAPSED_S,
    )
    return elapsed_s / 60.0


class SignalDetector:
    """Score viral composite à partir de ``CreatorStats`` (4 groupes de signaux)."""

    WEIGHT_G1 = 0.35
    WEIGHT_G2 = 0.35
    WEIGHT_G3 = 0.15
    WEIGHT_G4 = 0.15

    ALERT_THRESHOLD = 0.75
    CRITICAL_THRESHOLD = 0.85

    def __init__(self, stats: CreatorStats) -> None:
        self.stats = stats

    def detect(
        self,
        *,
        now: datetime | None = None,
        likes_at_30m_ago: int | None = None,
    ) -> dict[str, Any]:
        """Calcule le score viral et le détail par groupe.

        ``likes_at_30m_ago`` : nombre cumulé de likes sur la vidéo la plus récente
        tel qu'il était il y a 30 minutes (pour l'accélération G2). Si absent ou
        âge de la vidéo < 30 min, le sous-score accélération vaut 0.5 (neutre).
        """
        now_utc = _ensure_utc(now) if now else datetime.now(timezone.utc)
        sorted_v = _sorted_videos_recent_first(self.stats.recent_videos)
        headline = sorted_v[0] if sorted_v else None

        g1, d1 = self._group1_creator(sorted_v)
        g2, d2 = self._group2_temporal(headline, now_utc, likes_at_30m_ago=likes_at_30m_ago)
        g3, d3 = self._group3_content(headline)
        g4, d4 = self._group4_community(headline)

        score_viral = (
            self.WEIGHT_G1 * g1
            + self.WEIGHT_G2 * g2
            + self.WEIGHT_G3 * g3
            + self.WEIGHT_G4 * g4
        )

        if score_viral > self.CRITICAL_THRESHOLD:
            alert = "CRITICAL"
        elif score_viral > self.ALERT_THRESHOLD:
            alert = "ALERT"
        else:
            alert = "NONE"

        return {
            "score_viral": float(score_viral),
            "alert": alert,
            "groups": {
                "g1": {"score": g1, "weight": self.WEIGHT_G1, "components": d1},
                "g2": {"score": g2, "weight": self.WEIGHT_G2, "components": d2},
                "g3": {"score": g3, "weight": self.WEIGHT_G3, "components": d3},
                "g4": {"score": g4, "weight": self.WEIGHT_G4, "components": d4},
            },
        }

    def _group1_creator(
        self, sorted_v: list[dict[str, Any]]
    ) -> tuple[float, dict[str, Any]]:
        baseline = self.stats.get_baseline()
        k = min(3, len(sorted_v))
        if len(sorted_v) < 2 or baseline <= 0:
            serie_score = 0.0
        else:
            window = sorted_v[:k]
            serie_score = 1.0 if all(int(v["views"]) > baseline * 1.5 for v in window) else 0.0

        if sorted_v and self.stats.followers > 0:
            last_views = int(sorted_v[0]["views"])
            ratio_raw = last_views / float(self.stats.followers)
            ratio_vues_followers = _clip01(ratio_raw / 10.0)
        else:
            ratio_raw = 0.0
            ratio_vues_followers = 0.0

        if self.stats.follower_growth_7d_pct is None:
            croissance_norm = 0.0
            growth_raw: float | None = None
        else:
            growth_raw = max(float(self.stats.follower_growth_7d_pct), 0.0)
            croissance_norm = _clip01(growth_raw / 20.0)

        g1 = (serie_score + ratio_vues_followers + croissance_norm) / 3.0
        details = {
            "serie_score": serie_score,
            "ratio_vues_followers_raw": float(ratio_raw),
            "ratio_vues_followers_norm": float(ratio_vues_followers),
            "croissance_followers_7j_pct": growth_raw,
            "croissance_followers_7j_norm": float(croissance_norm),
        }
        return g1, details

    def _group2_temporal(
        self,
        headline: dict[str, Any] | None,
        now_utc: datetime,
        *,
        likes_at_30m_ago: int | None,
    ) -> tuple[float, dict[str, Any]]:
        if not headline:
            empty = {
                "velocite_likes_per_min": 0.0,
                "velocite_likes_norm": 0.0,
                "velocite_commentaires_per_min": 0.0,
                "velocite_commentaires_norm": 0.0,
                "velocite_partages_per_min": 0.0,
                "velocite_partages_norm": 0.0,
                "acceleration": 0.0,
            }
            return 0.0, empty

        posted = _ensure_utc(headline["posted_at"])
        minutes = _minutes_since_post(posted, now_utc)
        likes = int(headline["likes"])
        comments = int(headline["comments"])
        shares_saves = int(headline["shares"]) + int(headline["saves"])

        lpm = likes / minutes
        cpm = comments / minutes
        spm = shares_saves / minutes

        velocite_likes = _clip01(lpm / 100.0)
        velocite_commentaires = _clip01(cpm / 20.0)
        velocite_partages = _clip01(spm / 10.0)

        acceleration = self._acceleration_score(
            likes_total=likes,
            minutes_since_post=minutes,
            likes_at_30m_ago=likes_at_30m_ago,
        )

        g2 = (
            velocite_likes
            + velocite_commentaires
            + velocite_partages
            + acceleration
        ) / 4.0
        details = {
            "velocite_likes_per_min": float(lpm),
            "velocite_likes_norm": float(velocite_likes),
            "velocite_commentaires_per_min": float(cpm),
            "velocite_commentaires_norm": float(velocite_commentaires),
            "velocite_partages_per_min": float(spm),
            "velocite_partages_norm": float(velocite_partages),
            "acceleration": float(acceleration),
        }
        return g2, details

    @staticmethod
    def _acceleration_score(
        *,
        likes_total: int,
        minutes_since_post: float,
        likes_at_30m_ago: int | None,
    ) -> float:
        if likes_at_30m_ago is None or minutes_since_post <= 30.0:
            return 0.5
        minutes_then = max(minutes_since_post - 30.0, 1.0 / 60.0)
        vel_then = likes_at_30m_ago / minutes_then
        vel_now = likes_total / max(minutes_since_post, 1.0 / 60.0)
        if vel_then <= 1e-9:
            return 0.5
        ratio = vel_now / vel_then
        return _clip01((ratio - 0.8) / 0.7)

    def _group3_content(
        self, headline: dict[str, Any] | None
    ) -> tuple[float, dict[str, Any]]:
        if not headline:
            return 0.0, {"audio_trending": False, "duree_optimale": False}

        audio_id = headline.get("audio_id")
        reels = headline.get("audio_reels_count")
        is_recent = headline.get("audio_is_recent", True)
        if not isinstance(is_recent, bool):
            is_recent = bool(is_recent)

        audio_trending = bool(
            audio_id
            and reels is not None
            and int(reels) < 5000
            and is_recent
        )

        dur = headline.get("duration_sec")
        if dur is not None:
            d = int(dur)
            duree_optimale = 60 <= d <= 120
        else:
            duree_optimale = False

        audio_score = 1.0 if audio_trending else 0.0
        duree_score = 1.0 if duree_optimale else 0.0
        g3 = (audio_score + duree_score) / 2.0
        details = {
            "audio_trending": audio_trending,
            "audio_reels_count": int(reels) if reels is not None else None,
            "duree_optimale": duree_optimale,
            "duration_sec": int(dur) if dur is not None else None,
        }
        return g3, details

    def _group4_community(
        self, headline: dict[str, Any] | None
    ) -> tuple[float, dict[str, Any]]:
        pattern_recurrence = 0.5
        if not headline:
            ratio_norm = 0.0
            ratio_raw = 0.0
        else:
            likes = int(headline["likes"])
            comments = int(headline["comments"])
            if likes > 0:
                ratio_raw = comments / float(likes)
                ratio_norm = _clip01(ratio_raw / 0.05)
            else:
                ratio_raw = 0.0
                ratio_norm = 0.0

        g4 = (ratio_norm + pattern_recurrence) / 2.0
        details = {
            "ratio_comments_likes_raw": float(ratio_raw),
            "ratio_comments_likes_norm": float(ratio_norm),
            "pattern_recurrence": pattern_recurrence,
        }
        return g4, details


def score_post(metrics: dict, comments_sample: list[str]) -> float:
    """Calcule un score 0→1 pour un post selon les signaux configurés."""
    raise NotImplementedError
