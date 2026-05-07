"""Récupération des données publiques Instagram via Apify (avec repli mock)."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, NoReturn

import config
from modules.detector import CreatorStats

_HASHTAG_RE = re.compile(r"#(\w+)")

_LOGGER = logging.getLogger("aitertainment")

_REEL_SCRAPER_ACTOR = "apify/instagram-reel-scraper"
_COMMENT_SCRAPER_ACTOR = "apify/instagram-comment-scraper"

# Plafond du run côté Apify (s)
_REELS_TIMEOUT_S = 60
_COMMENTS_TIMEOUT_S = 30


# ---------- Exceptions ----------

class ApifyError(RuntimeError):
    """Erreur générique côté Apify (réseau, schéma, run KO…)."""


class ApifyAccountPrivateError(ApifyError):
    """Compte ou vidéo privé, supprimé ou inaccessible."""


class ApifyRateLimitError(ApifyError):
    """Limite de requêtes atteinte (429)."""


class ApifyQuotaExceededError(ApifyError):
    """Quota / crédits Apify dépassé (402/403)."""


class ApifyTimeoutError(ApifyError):
    """L'acteur n'a pas terminé dans le délai imparti."""


# ---------- Détection mode mock ----------

def _is_mock_mode() -> bool:
    """Mock automatique si ``APIFY_TOKEN`` est absent ou vide."""
    return not (config.APIFY_TOKEN and config.APIFY_TOKEN.strip())


def _get_apify_client() -> Any:
    """Instancie un ``ApifyClient`` (lève ``ApifyError`` si la lib manque)."""
    try:
        from apify_client import ApifyClient  # type: ignore
    except ImportError as e:
        raise ApifyError(f"apify-client non installé : {e}") from e
    return ApifyClient(token=config.APIFY_TOKEN.strip())


def _raise_apify_error(e: BaseException, ctx: str) -> NoReturn:
    """Mappe les erreurs apify-client / réseau vers les exceptions du module."""
    msg = str(e)
    low = msg.lower()
    sc = getattr(e, "status_code", None)

    has_api_shape = (
        sc is not None
        or e.__class__.__name__ in ("ApifyApiError", "ApifyClientError")
    )
    if has_api_shape:
        if sc == 429 or "rate limit" in low or "too many requests" in low:
            raise ApifyRateLimitError(f"{ctx}: rate limit ({msg})") from e
        if sc in (402, 403) or "quota" in low or "credit" in low or "monthly usage" in low:
            raise ApifyQuotaExceededError(f"{ctx}: quota Apify dépassé ({msg})") from e
        if sc == 404 or any(k in low for k in ("private", "not found", "deleted", "unavailable")):
            raise ApifyAccountPrivateError(f"{ctx}: privé/introuvable ({msg})") from e
        raise ApifyError(f"{ctx}: erreur Apify (status={sc}) {msg}") from e

    raise ApifyError(f"{ctx}: erreur réseau / inattendue : {msg}") from e


def _check_run_succeeded(run: dict[str, Any] | None, ctx: str) -> dict[str, Any]:
    """Valide le statut du run Apify (lève en cas de timeout / échec)."""
    if not run:
        raise ApifyTimeoutError(f"{ctx}: pas de run retourné (timeout/abort)")
    status = str(run.get("status") or "").upper().replace("_", "-")
    if status in ("TIMED-OUT", "TIMEOUT"):
        raise ApifyTimeoutError(f"{ctx}: actor timeout ({status})")
    if status and status != "SUCCEEDED":
        raise ApifyError(f"{ctx}: actor status={status}")
    return run


def _iterate_dataset_items(client: Any, dataset_id: str) -> list[dict[str, Any]]:
    if not dataset_id:
        return []
    try:
        return list(client.dataset(dataset_id).iterate_items())
    except Exception as e:
        raise ApifyError(f"lecture dataset {dataset_id} : {e}") from e


# ---------- Normalisation des items Apify ----------

def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return datetime.now(timezone.utc)
    if isinstance(value, str):
        s = value.strip()
        if s:
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            except ValueError:
                pass
    return datetime.now(timezone.utc)


def _is_private_marker(item: dict[str, Any]) -> bool:
    err = item.get("error") or item.get("errorMessage") or item.get("errorDescription")
    if isinstance(err, str):
        low = err.lower()
        if any(k in low for k in (
            "private", "not found", "blocked", "unavailable", "deleted",
            "no longer", "restricted",
        )):
            return True
    if item.get("isPrivate") is True:
        return True
    if item.get("private") is True:
        return True
    return False


def _coerce_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _extract_audio_id(item: dict[str, Any]) -> str | None:
    for key in ("audioId", "audio_id", "musicId", "music_id"):
        v = item.get(key)
        if v:
            return str(v)
    music = item.get("musicInfo") or item.get("music") or item.get("audio")
    if isinstance(music, dict):
        for key in ("audio_id", "audioId", "id", "musicCanonicalId"):
            v = music.get(key)
            if v:
                return str(v)
    return None


def _extract_caption(item: dict[str, Any]) -> str:
    """Caption brute du reel, en repli sur les variantes Apify connues."""
    for key in ("caption", "captionText", "text", "edge_media_to_caption"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _extract_hashtags(item: dict[str, Any], caption: str) -> list[str]:
    """Hashtags Apify si dispo, sinon fallback sur regex ``#tag`` dans caption."""
    raw = item.get("hashtags")
    if isinstance(raw, list):
        out = [str(h).lstrip("#").strip() for h in raw if h]
        out = [h for h in out if h]
        if out:
            return out
    if caption:
        return _HASHTAG_RE.findall(caption)
    return []


def _normalize_reel(item: dict[str, Any]) -> dict[str, Any]:
    video_id = (
        item.get("id")
        or item.get("shortCode")
        or item.get("postId")
        or item.get("code")
        or ""
    )
    views = (
        item.get("videoViewCount")
        or item.get("videoPlayCount")
        or item.get("playCount")
        or item.get("views")
        or 0
    )
    likes = item.get("likesCount") or item.get("likeCount") or item.get("likes") or 0
    comments = (
        item.get("commentsCount")
        or item.get("commentCount")
        or item.get("comments")
        or 0
    )
    posted = item.get("timestamp") or item.get("takenAt") or item.get("postedAt")
    url = item.get("url") or item.get("permalink") or item.get("videoUrl") or ""
    caption = _extract_caption(item)
    return {
        "video_id": str(video_id),
        "views": _coerce_int(views),
        "likes": _coerce_int(likes),
        "comments_count": _coerce_int(comments),
        "posted_at": _parse_dt(posted),
        "url": str(url),
        "audio_id": _extract_audio_id(item),
        "caption": caption,
        "hashtags": _extract_hashtags(item, caption),
    }


def _normalize_comment(item: dict[str, Any]) -> str | None:
    txt = item.get("text") or item.get("comment") or item.get("content")
    if not isinstance(txt, str):
        return None
    s = txt.strip()
    return s or None


# ---------- API publique : Apify ----------

def get_creator_reels(username: str, max_reels: int = 10) -> list[dict[str, Any]]:
    """Derniers reels Instagram d'un compte via ``apify/instagram-reel-scraper``.

    Repli automatique sur des données mock si ``APIFY_TOKEN`` est vide.
    Lève ``ApifyAccountPrivateError`` / ``ApifyRateLimitError`` /
    ``ApifyQuotaExceededError`` / ``ApifyTimeoutError`` selon le cas.
    """
    u = (username or "").lstrip("@").strip()
    if not u:
        raise ValueError("username vide")
    n = max(1, int(max_reels))

    if _is_mock_mode():
        _LOGGER.info("Apify[reels] MOCK @%s (n=%d) — APIFY_TOKEN vide", u, n)
        return _mock_creator_reels(u, n)

    ctx = f"reels @{u}"
    _LOGGER.info("Apify[reels] CALL @%s (n=%d)", u, n)
    run_input: dict[str, Any] = {
        "directUrls": [f"https://www.instagram.com/{u}/"],
        "maxReelsPerProfile": n,
    }

    try:
        client = _get_apify_client()
        run = client.actor(_REEL_SCRAPER_ACTOR).call(
            run_input=run_input,
            timeout_secs=_REELS_TIMEOUT_S,
        )
        run = _check_run_succeeded(run, ctx)
        items = _iterate_dataset_items(client, str(run.get("defaultDatasetId") or ""))
    except ApifyError:
        _LOGGER.exception("Apify[reels] ECHEC %s", ctx)
        raise
    except Exception as e:
        _LOGGER.exception("Apify[reels] ECHEC %s", ctx)
        _raise_apify_error(e, ctx)

    out: list[dict[str, Any]] = []
    for item in items[:n]:
        if not isinstance(item, dict):
            continue
        if _is_private_marker(item):
            _LOGGER.warning("Apify[reels] %s : compte privé/indisponible", ctx)
            raise ApifyAccountPrivateError(f"{ctx}: compte privé / indisponible")
        out.append(_normalize_reel(item))

    _LOGGER.info("Apify[reels] OK %s -> %d reel(s)", ctx, len(out))
    return out


def get_video_comments(video_url: str, limit: int = 15) -> list[str]:
    """Commentaires (texte uniquement) d'une vidéo via ``apify/instagram-comment-scraper``.

    Repli automatique sur des données mock si ``APIFY_TOKEN`` est vide.
    Lève ``ApifyAccountPrivateError`` si la vidéo est privée/supprimée,
    ``ApifyTimeoutError`` si l'acteur dépasse le délai imparti, etc.
    """
    url = (video_url or "").strip()
    if not url:
        raise ValueError("video_url vide")
    n = max(1, int(limit))

    if _is_mock_mode():
        _LOGGER.info("Apify[comments] MOCK %s (n=%d) — APIFY_TOKEN vide", url, n)
        return _mock_video_comments(url, n)

    ctx = f"comments {url}"
    _LOGGER.info("Apify[comments] CALL %s (n=%d)", url, n)
    run_input: dict[str, Any] = {
        "directUrls": [url],
        "maxComments": n,
        "includeReplies": False,
    }

    try:
        client = _get_apify_client()
        run = client.actor(_COMMENT_SCRAPER_ACTOR).call(
            run_input=run_input,
            timeout_secs=_COMMENTS_TIMEOUT_S,
        )
        run = _check_run_succeeded(run, ctx)
        items = _iterate_dataset_items(client, str(run.get("defaultDatasetId") or ""))
    except ApifyError:
        _LOGGER.exception("Apify[comments] ECHEC %s", ctx)
        raise
    except Exception as e:
        _LOGGER.exception("Apify[comments] ECHEC %s", ctx)
        _raise_apify_error(e, ctx)

    if items and isinstance(items[0], dict) and _is_private_marker(items[0]):
        _LOGGER.warning("Apify[comments] %s : vidéo privée/supprimée", ctx)
        raise ApifyAccountPrivateError(f"{ctx}: vidéo privée / supprimée")

    out: list[str] = []
    for item in items[:n]:
        if not isinstance(item, dict):
            continue
        s = _normalize_comment(item)
        if s:
            out.append(s)

    _LOGGER.info("Apify[comments] OK %s -> %d commentaire(s)", ctx, len(out))
    return out


# ---------- Mocks ----------

def _mock_creator_reels(username: str, n: int) -> list[dict[str, Any]]:
    t0 = datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    for i in range(n):
        caption = (
            f"[mock] caption #{i + 1} pour @{username} "
            f"#streetwear #fitcheck #archive"
        )
        out.append(
            {
                "video_id": f"mock_{username}_{i}",
                "views": 5_000 + 1_000 * i,
                "likes": 800 + 100 * i,
                "comments_count": 50 + 10 * i,
                "posted_at": t0 - timedelta(hours=2 + 24 * i),
                "url": f"https://www.instagram.com/reel/mock_{username}_{i}/",
                "audio_id": f"mock_snd_{i % 3}",
                "caption": caption,
                "hashtags": ["streetwear", "fitcheck", "archive"],
            }
        )
    return out


def _mock_video_comments(video_url: str, n: int) -> list[str]:
    return [f"[mock] commentaire #{i + 1} sur {video_url}" for i in range(n)]


# ---------- Compat héritée (utilisée par main.py / tests) ----------

def fetch_creator_recent_posts(platform: str, username: str) -> list[dict[str, Any]]:
    """Adapter rétro-compatible : reels Instagram via Apify (ou mock)."""
    if platform.lower() != "instagram":
        raise NotImplementedError(f"plateforme non supportée: {platform}")
    return get_creator_reels(username)


def fetch_creator_stats_mock(creator: CreatorStats) -> CreatorStats:
    """Jeu de données de démo pour le MVP (utilisé tant qu'Apify n'est pas branché)."""
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
    return [f"[mock] com #{i + 1} — réaction sur @{u}" for i in range(n)]
