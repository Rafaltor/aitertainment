"""Watcher AItertainment — détection temps réel sur des créateurs déjà profilés.

============================================================================
ARCHITECTURE — séparation stricte Discovery vs Watcher
============================================================================

Deux phases distinctes :

PHASE DISCOVERY (``discovery.py`` + bot Telegram)
    - Score les profils (Reels), écrit ``database.json``, propose des
      candidats ; validation humaine → ``watchlist.json``.

PHASE WATCHER (ce fichier)
    - **Ne scrape JAMAIS les commentaires** d'un post frais.
      Raison : les premiers commentaires d'un post sont majoritairement des
      bots / contributeurs très précoces — bruit pur pour la classification.
    - Le ``t_type`` du créateur est **déjà connu** via ``watchlist.json``.
    - Objectif : détecter un nouveau post et émettre un commentaire calibré
      en moins de 3 minutes après publication.
    - Le contexte du commentaire suggéré provient de deux sources :
        1. Profil créateur (``t_type``, ``niches``) → déjà dans
           ``watchlist.json`` (ne change pas à chaque tick).
        2. Contexte vidéo (caption, hashtags, audio_id, transcript Whisper,
           description visuelle Qwen2.5-VL) → Playwright + GraphQL ; une seule
           passe yt-dlp (MP4) puis Whisper + LM Studio vision avant génération.

Corpus commentaires d'entraînement : ``scripts/scrape_viral_comments.py`` →
``viral_comments.json`` (hors watchlist temps réel).

``generate_comments`` utilise le ``t_type`` du créateur + métadonnées du reel.

Ce fichier expose :
    - ``load_watchlist`` / ``save_watchlist`` : persistance de la watchlist.
    - ``check_new_post`` : détection via Playwright + GraphQL (vues faibles).
    - ``get_poll_interval`` : intervalle de polling adaptatif (heure locale).
    - ``run_watcher`` : boucle principale (CLI : ``python watcher.py`` /
      ``python watcher.py --mock``).

Lancement :
    python watcher.py            # boucle réelle (Instagram + Ollama + Telegram)
    python watcher.py --mock     # un cycle, post & génération synthétiques,
                                 # pas d'appel Instagram/Telegram
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

import config
import requests
from config import VALID_T_TYPES
from modules.atomic_json import atomic_write_json
from telegram_notify import setup_watcher_logger
from playwright.sync_api import BrowserContext, sync_playwright

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.instagram_browser import get_browser_context, get_recent_reels

_HASHTAG_RE = re.compile(r"#(\w+)")

DEFAULT_WATCHLIST_PATH = _PROJECT_ROOT / "data" / "watchlist.json"
VECTOR_STORE_PATH = Path("data/vector_store.json")

VALID_PLATFORMS = frozenset({"instagram", "tiktok"})
NEW_POST_VIEW_THRESHOLD = 2000  # vues < seuil = post récent
REEL_PRODUCT_TYPE = "clips"  # product_type Instagram pour les Reels vidéo

VISION_MODEL = os.environ.get(
    "LM_STUDIO_VISION_MODEL", "qwen2.5-vl-7b-instruct"
).strip()

_LOGGER = logging.getLogger("aitertainment")


def _hashtags_from_caption(caption: str) -> list[str]:
    return _HASHTAG_RE.findall(caption or "")


def _creator_primary_niche(creator: dict[str, Any]) -> str:
    niches = creator.get("niches")
    if isinstance(niches, list) and niches:
        return str(niches[0] or "?")
    return "?"


def _creator_niches(creator: dict[str, Any]) -> list[str]:
    niches = creator.get("niches")
    if isinstance(niches, list) and niches:
        return list(niches)
    return []


def _is_video_reel(reel: dict[str, Any]) -> bool:
    """True si le média est un Reel vidéo (exclut carrousels / posts photo)."""
    pt = reel.get("product_type")
    if pt is not None and str(pt).strip():
        return str(pt).strip().lower() == REEL_PRODUCT_TYPE
    return int(reel.get("view_count") or 0) > 0


def _sync_post_metadata_from_reel_page(
    post: dict[str, Any],
    expected_username: str,
    browser_context: BrowserContext,
) -> bool:
    """Aligne caption + @ Instagram depuis ``/reel/{id}/`` (évite le mélange grille).

    Retourne ``False`` si le propriétaire détecté ne correspond pas au créateur
    surveillé (pas de notif / génération sur ce cycle).
    """
    from scripts.instagram_browser import get_reel_page_metadata

    media_id = str(post.get("video_id") or "").strip()
    expected = str(expected_username or "").lstrip("@").strip().lower()
    if not media_id or not expected:
        return False

    grid_caption = str(post.get("caption") or "").strip()
    try:
        meta = get_reel_page_metadata(media_id, browser_context)
    except Exception as e:
        _LOGGER.warning(
            "@%s reel %s : métadonnées reel échouées (%s) — repli grille.",
            expected,
            media_id,
            e,
        )
        post["username"] = expected
        post["url"] = f"https://www.instagram.com/reel/{media_id}/"
        return True

    owner = str(meta.get("owner_username") or "").strip().lower()
    if owner and owner != expected:
        _LOGGER.warning(
            "@%s reel %s : propriétaire reel=@%s — skip (mauvais compte).",
            expected,
            media_id,
            owner,
        )
        return False

    caption = str(meta.get("caption") or "").strip() or grid_caption
    if grid_caption and caption and caption != grid_caption:
        _LOGGER.debug(
            "@%s reel %s : caption grille remplacée (grille=%r → reel=%r)",
            expected,
            media_id,
            grid_caption[:80],
            caption[:80],
        )

    post["username"] = owner or expected
    post["caption"] = caption
    post["hashtags"] = _hashtags_from_caption(caption)
    post["url"] = f"https://www.instagram.com/reel/{media_id}/"
    return True


def _filter_video_reels(reels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ne garde que les Reels vidéo (``product_type=clips`` ou vues > 0)."""
    filtered = [r for r in reels if _is_video_reel(r)]
    dropped = len(reels) - len(filtered)
    if dropped:
        _LOGGER.debug(
            "check_new_post : %d média(s) non-vidéo ignoré(s) (carousel/photo)",
            dropped,
        )
    return filtered


def check_new_post(
    creator: dict[str, Any], context: BrowserContext
) -> dict[str, Any] | None:
    """Détecte un nouveau Reel via Playwright (hors épinglés).

    - **Bootstrap** (``last_post_id`` absent) : mémorise le Reel le plus récent
      sans condition de vues — point de départ pour la surveillance.
    - **Surveillance** (``last_post_id`` connu) : alerte seulement si le Reel
      en tête change *et* a des vues sous le seuil dynamique (post frais).
    """
    if not isinstance(creator, dict):
        _LOGGER.warning("check_new_post : creator doit être un dict, reçu %s", type(creator).__name__)
        return None

    username = str(creator.get("username") or "").lstrip("@").strip()
    if not username:
        _LOGGER.warning("check_new_post : 'username' manquant dans creator")
        return None

    platform = str(creator.get("platform", "") or "").strip().lower()
    if platform and platform != "instagram":
        _LOGGER.warning(
            "check_new_post @%s : plateforme %r non supportée (skip)",
            username,
            platform,
        )
        return None

    try:
        reels = get_recent_reels(username, context, max_reels=4)
    except Exception as e:
        _LOGGER.warning("check_new_post @%s : erreur (%s)", username, e)
        return None

    if not reels:
        _LOGGER.warning("check_new_post @%s : aucun reel récupéré (session IG ?)", username)
        return None

    reels = _filter_video_reels(reels)
    if not reels:
        _LOGGER.debug(
            "check_new_post @%s : aucun reel vidéo après filtre carousel/photo",
            username,
        )
        return None

    non_pinned = [r for r in reels if not r.get("is_pinned", False)]
    if not non_pinned:
        try:
            reels = get_recent_reels(username, context, max_reels=8)
        except Exception as e:
            _LOGGER.warning("check_new_post @%s : erreur retry (%s)", username, e)
            return None
        reels = _filter_video_reels(reels)
        non_pinned = [r for r in reels if not r.get("is_pinned", False)]

    if not non_pinned:
        _LOGGER.info("check_new_post @%s : aucun reel non épinglé trouvé", username)
        return None

    first = non_pinned[0]
    first_views = int(first.get("view_count") or 0)

    grid_owner = str(first.get("owner_username") or "").lstrip("@").strip().lower()
    if grid_owner and grid_owner != username.lower():
        _LOGGER.warning(
            "check_new_post @%s : reel en tête appartient à @%s — ignoré.",
            username,
            grid_owner,
        )
        return None

    media_id = str(first.get("media_id") or "")
    if not media_id:
        return None

    last_post_id = str(creator.get("last_post_id") or "")
    if last_post_id and media_id == last_post_id:
        return None

    caption = str(first.get("caption") or "")
    hashtags = _hashtags_from_caption(caption)

    if not last_post_id:
        _LOGGER.info(
            "check_new_post @%s : bootstrap (last_post_id null) -> media_id=%s (%d vues)",
            username,
            media_id,
            first_views,
        )
        return {
            "video_id": media_id,
            "username": username,
            "caption": caption,
            "hashtags": hashtags,
            "audio_id": str(first.get("audio_id") or ""),
            "url": f"https://www.instagram.com/reel/{media_id}/",
            "posted_at": datetime.now(timezone.utc),
            "bootstrap": True,
        }

    views_list = [int(r["view_count"]) for r in non_pinned if int(r.get("view_count") or 0) > 0]
    if len(views_list) >= 2:
        avg_views = sum(views_list[1:]) / len(views_list[1:])
        threshold = max(NEW_POST_VIEW_THRESHOLD, avg_views * 0.05)
    else:
        threshold = float(NEW_POST_VIEW_THRESHOLD)

    if first_views >= threshold:
        _LOGGER.debug(
            "check_new_post @%s : pas de nouveau post (%d vues >= seuil %.0f)",
            username,
            first_views,
            threshold,
        )
        return None

    _LOGGER.info(
        "check_new_post @%s : nouveau post détecté %s (%d vues < seuil %.0f)",
        username,
        media_id,
        first_views,
        threshold,
    )

    return {
        "video_id": media_id,
        "username": username,
        "caption": caption,
        "hashtags": hashtags,
        "audio_id": str(first.get("audio_id") or ""),
        "url": f"https://www.instagram.com/reel/{media_id}/",
        "posted_at": datetime.now(timezone.utc),
        "bootstrap": False,
    }


class WatchlistError(ValueError):
    """Schéma watchlist invalide ou fichier illisible."""


def _empty_watchlist() -> dict[str, Any]:
    return {"creators": []}


def _normalize_entry(entry: Any, *, index: int) -> dict[str, Any]:
    """Valide / normalise une entrée créateur (lève ``WatchlistError`` si KO)."""
    if not isinstance(entry, dict):
        raise WatchlistError(f"creators[{index}] doit être un objet, reçu {type(entry).__name__}")

    username = entry.get("username")
    if not isinstance(username, str) or not username.strip():
        raise WatchlistError(f"creators[{index}].username manquant ou vide")
    username = username.lstrip("@").strip()

    platform = entry.get("platform")
    if not isinstance(platform, str) or platform.strip().lower() not in VALID_PLATFORMS:
        raise WatchlistError(
            f"creators[{index}].platform invalide ({platform!r}), "
            f"attendu un parmi {sorted(VALID_PLATFORMS)}"
        )
    platform = platform.strip().lower()

    # ``niches`` (liste) est la source de vérité. Une entrée sans ``niches``
    # (ou avec une liste vide) reste valide → ``[]``.
    niches_raw = entry.get("niches")
    niches: list[str] = []
    if niches_raw is not None:
        if not isinstance(niches_raw, list):
            raise WatchlistError(
                f"creators[{index}].niches doit être une liste, "
                f"reçu {type(niches_raw).__name__}"
            )
        for j, n in enumerate(niches_raw):
            if not isinstance(n, str):
                raise WatchlistError(
                    f"creators[{index}].niches[{j}] doit être une chaîne, "
                    f"reçu {type(n).__name__}"
                )
            cleaned = n.strip()
            if cleaned:
                niches.append(cleaned)

    t_type_raw = entry.get("t_type")
    if isinstance(t_type_raw, str):
        t_type_raw = t_type_raw.strip()
        if t_type_raw in ("", "T?"):
            t_type_raw = None
    t_type: str | None
    if t_type_raw is None or t_type_raw == "":
        t_type = None
    elif isinstance(t_type_raw, str) and t_type_raw in VALID_T_TYPES:
        t_type = t_type_raw
    else:
        raise WatchlistError(
            f"creators[{index}].t_type invalide ({t_type_raw!r}), "
            f"attendu un parmi {sorted(VALID_T_TYPES)} ou null"
        )

    eb_raw = entry.get("engagement_baseline")
    if eb_raw is None or eb_raw == "":
        engagement_baseline: float | None = None
    else:
        try:
            engagement_baseline = float(eb_raw)
        except (TypeError, ValueError) as e:
            raise WatchlistError(
                f"creators[{index}].engagement_baseline doit être un nombre ou null"
            ) from e

    last_post_id_raw = entry.get("last_post_id")
    if last_post_id_raw in (None, ""):
        last_post_id: str | None = None
    else:
        last_post_id = str(last_post_id_raw)

    added_at_raw = entry.get("added_at")
    added_at: str
    if added_at_raw in (None, ""):
        added_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    elif isinstance(added_at_raw, str):
        added_at = added_at_raw.strip()
    else:
        raise WatchlistError(
            f"creators[{index}].added_at doit être une chaîne ISO 8601"
        )

    out: dict[str, Any] = {
        "username": username,
        "platform": platform,
        "niches": list(niches),
        "t_type": t_type,
        "engagement_baseline": engagement_baseline,
        "last_post_id": last_post_id,
        "added_at": added_at,
    }
    for k, v in entry.items():
        if k not in out:
            out[k] = v
    return out


def load_watchlist(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Charge ``watchlist.json`` et renvoie la liste normalisée des créateurs.

    Schéma attendu : ``{"creators": [ {username, platform, niches, t_type,
    engagement_baseline, last_post_id, added_at}, ... ]}``.

    - Fichier absent → liste vide (et log warning).
    - Fichier vide / JSON invalide → ``WatchlistError``.
    - Schéma invalide (entrée mal formée) → ``WatchlistError``.
    """
    p = Path(path) if path else DEFAULT_WATCHLIST_PATH
    if not p.exists():
        _LOGGER.warning("Watchlist absente : %s — démarrage avec liste vide.", p)
        return []

    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        raise WatchlistError(f"Lecture impossible de {p} : {e}") from e

    text = raw.strip()
    if not text:
        _LOGGER.warning("Watchlist vide : %s — démarrage avec liste vide.", p)
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise WatchlistError(f"JSON invalide dans {p} : {e}") from e

    if not isinstance(data, dict):
        raise WatchlistError(
            f"Racine de {p} doit être un objet JSON, reçu {type(data).__name__}"
        )
    creators = data.get("creators", [])
    if not isinstance(creators, list):
        raise WatchlistError(f'"creators" dans {p} doit être une liste')

    normalized = [_normalize_entry(c, index=i) for i, c in enumerate(creators)]
    _LOGGER.info("Watchlist chargée : %d créateur(s) depuis %s", len(normalized), p)
    return normalized


def save_watchlist(
    watchlist: list[dict[str, Any]],
    path: str | Path | None = None,
) -> None:
    """Sérialise la watchlist vers ``watchlist.json`` (UTF-8, indent=2).

    Écriture atomique via fichier temporaire ``.tmp`` + ``replace`` pour éviter
    un fichier corrompu si l'écriture est interrompue.
    """
    if not isinstance(watchlist, list):
        raise WatchlistError(
            f"watchlist doit être une liste, reçu {type(watchlist).__name__}"
        )

    p = Path(path) if path else DEFAULT_WATCHLIST_PATH
    normalized = [_normalize_entry(c, index=i) for i, c in enumerate(watchlist)]
    atomic_write_json(p, {"creators": normalized})
    _LOGGER.info("Watchlist sauvegardée : %d créateur(s) -> %s", len(normalized), p)


def load_watchlist_synced(
    path: str | Path | None = None,
    *,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Charge la watchlist puis re-dérive ses métadonnées depuis ``database.json``.

    ``database.json`` est la source de vérité (``niches`` / ``t_type`` /
    archivage) ; la watchlist ne conserve en propre que son état runtime
    (curseur ``last_post_id`` …). Voir ``database.rebuild_watchlist``.

    Best-effort : si la DB est illisible / absente, on renvoie la watchlist
    brute. Le Watcher ne doit **jamais** s'arrêter pour un souci de dérivation.
    """
    creators = load_watchlist(path)
    try:
        from database import load_db, rebuild_watchlist

        db = load_db(path=db_path)
    except Exception as e:  # noqa: BLE001 — best-effort, on log et on continue
        _LOGGER.warning(
            "Re-dérivation watchlist depuis database.json impossible (%s) — "
            "watchlist brute utilisée.",
            e,
        )
        return creators
    return rebuild_watchlist(db, creators)


# =============================================================================
# Boucle Watcher
# =============================================================================

PRIME_INTERVAL_S = config.WATCHER_PRIME_INTERVAL_S
DAY_INTERVAL_S = config.WATCHER_DAY_INTERVAL_S
NIGHT_INTERVAL_S = config.WATCHER_NIGHT_INTERVAL_S
SLEEP_BETWEEN_CREATORS_S = config.WATCHER_SLEEP_BETWEEN_CREATORS_S
MAX_ACCOUNTS_PAUSE_S = config.WATCHER_ACCOUNTS_PAUSE_S


def get_poll_interval(now: datetime | None = None) -> int:
    """Intervalle de polling (secondes) selon l'heure **locale**.

    - 17h ≤ h < 21h : prime time → 300 s
    - 09h ≤ h < 17h : journée    → 600 s
    - sinon (21h–09h, nuit)      → 1800 s
    """
    h = (now or datetime.now()).hour
    if 17 <= h < 21:
        return PRIME_INTERVAL_S
    if 9 <= h < 17:
        return DAY_INTERVAL_S
    return NIGHT_INTERVAL_S


def _truncate(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _telegram_md_escape(text: str) -> str:
    """Échappe les caractères spéciaux du Markdown classique Telegram."""
    out: list[str] = []
    for ch in text or "":
        if ch in ("\\", "_", "*", "[", "`"):
            out.append("\\")
        out.append(ch)
    return "".join(out)


def load_vector_store(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Charge ``vector_store.json`` et indexe les entrées par username."""
    p = Path(path) if path is not None else VECTOR_STORE_PATH
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    if not p.exists():
        return {}

    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        entries = [entry for entry in data if isinstance(entry, dict)]
    elif isinstance(data, dict):
        raw_entries = data.get("entries") or data.get("profiles") or []
        entries = [entry for entry in raw_entries if isinstance(entry, dict)]
    else:
        entries = []

    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        username = str(entry.get("username") or "").lstrip("@").strip().lower()
        if username:
            out[username] = entry
    return out


def _describe_reel_visually(
    mp4_path: Path, lm_studio_url: str, vision_model: str
) -> str:
    """Décrit le contenu visible d'un Reel (frames ffmpeg + Qwen2.5-VL)."""
    mp4_path = Path(mp4_path)
    if not mp4_path.exists():
        return ""

    duration = 10.0
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                str(mp4_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if probe.returncode == 0 and probe.stdout:
            data = json.loads(probe.stdout)
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    duration = float(stream.get("duration", 10))
                    break
    except Exception:
        pass

    frames: list[Path] = []
    for pct in (0.25, 0.50, 0.75):
        t = duration * pct
        frame_path = mp4_path.parent / f"frame_{pct:.0%}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(t),
                "-i",
                str(mp4_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(frame_path),
            ],
            capture_output=True,
            check=False,
        )
        if frame_path.exists() and frame_path.stat().st_size > 1000:
            frames.append(frame_path)

    if not frames:
        _LOGGER.warning("aucune frame extraite pour %s", mp4_path)
        return ""

    frame = frames[0]
    img_b64 = base64.b64encode(frame.read_bytes()).decode()
    base_url = lm_studio_url.rstrip("/")
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Décris ce que tu vois en 2-3 phrases courtes. "
                                    "Contexte : reel Instagram humour français. "
                                    "Décris les personnes, actions, décor visible. "
                                    "Ne mentionne pas de noms de personnes publiques."
                                ),
                            },
                        ],
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.3,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return (
            resp.json()["choices"][0]["message"]["content"].strip()
        )
    except Exception as e:
        _LOGGER.warning("description visuelle échouée pour %s : %s", mp4_path, e)
        return ""


def _transcribe_reel(
    media_id: str,
    browser_context: BrowserContext,
    tmp_dir: Path | None = None,
) -> tuple[str, Path | None]:
    """Télécharge le MP4, transcrit l'audio (Whisper), renvoie aussi le chemin vidéo.

    Si ``tmp_dir`` est fourni, le répertoire n'est pas supprimé (le caller gère).
    """
    from scripts.embedder import (
        download_reel_video,
        extract_wav_from_video,
        transcribe_audio,
    )

    media_id = str(media_id or "").strip()
    if not media_id:
        return "", None

    own_tmp = tmp_dir is None
    work_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="watcher_reel_"))
    try:
        mp4_path = download_reel_video(media_id, browser_context, work_dir)
        if mp4_path is None:
            return "", None
        wav_path = extract_wav_from_video(mp4_path, work_dir)
        if wav_path is None:
            _LOGGER.warning(
                "transcription ignorée pour reel %s (ffmpeg audio)", media_id
            )
            return "", mp4_path
        text = transcribe_audio(wav_path)
        if not text:
            _LOGGER.warning(
                "transcription vide pour reel %s (Whisper)", media_id
            )
        return text, mp4_path
    except Exception as e:
        _LOGGER.warning("transcription reel %s échouée : %s", media_id, e)
        return "", None
    finally:
        if own_tmp:
            shutil.rmtree(work_dir, ignore_errors=True)


def _build_classification_from_t_type(t_type: str) -> dict[str, Any]:
    """Classification synthétique depuis ``t_type`` (Discovery a déjà tranché).

    En phase Watcher, on ne reclassifie pas : on injecte directement le
    ``t_type`` figé en Discovery dans ``generate_comments`` sous forme de dict
    minimal compatible.
    """
    return {
        "type": t_type,
        "confidence": 0.95,
        "patterns": [],
        "tone": f"Registre figé en Discovery ({t_type}).",
        "brand_risk": "low" if t_type in ("T1", "T2", "T4") else "medium",
    }


def _generate_for_post(
    context: dict[str, Any],
    vector_store: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Produit 3 commentaires depuis le ``context`` (pas de ``comments_sample``).

    Le contexte vidéo (caption, hashtags, audio_id) est passé directement à
    ``generate_comments`` via le paramètre ``video_context``. Le ``t_type``
    du créateur (sa **personnalité de commentateur**, validée humainement
    en Discovery) sert à la fois à choisir le template T-type et à
    renseigner ``t_type_profile`` dans le prompt — pas de classification
    online en phase Watcher (post frais, distribution non stabilisée).

    Renvoie ``[]`` si ``t_type`` est invalide (hors ``VALID_T_TYPES``).
    """
    from modules.classifier import generate_comments  # import local : Ollama

    log = logging.getLogger("aitertainment.watcher")
    vector_store = vector_store or {}
    t_type = str(context.get("t_type") or "")
    # ``niches`` (liste) : le caller (``run_watcher``) passe ``creator["niches"]``
    # à la construction du context.
    niches_raw = context.get("niches")
    niches: list[str] = list(niches_raw) if isinstance(niches_raw, list) else []

    if t_type not in VALID_T_TYPES:
        return []

    classification = _build_classification_from_t_type(t_type)
    username = str(context.get("username") or "").lstrip("@").strip().lower()
    vs_entry = vector_store.get(username, {})
    named_axes = vs_entry.get("named_axes") or {}
    if not isinstance(named_axes, dict):
        named_axes = {}

    gen_kwargs: dict[str, Any] = {
        "niches": niches,
        "t_type_profile": t_type,
        "video_context": {
            "caption": context.get("caption"),
            "hashtags": context.get("hashtags"),
            "audio_id": context.get("audio") or context.get("audio_id"),
            "transcript": context.get("transcript") or "",
            "visual_description": context.get("visual_description") or "",
            "video_id": context.get("video_id") or "",
            "username": context.get("username") or "",
        },
    }
    reel_id = str(context.get("video_id") or "")
    log.info(
        "generate @%s reel=%s — caption=%d chars, transcript=%d, visuel=%d",
        username or "?",
        reel_id or "?",
        len(str(context.get("caption") or "")),
        len(str(context.get("transcript") or "")),
        len(str(context.get("visual_description") or "")),
    )
    if named_axes:
        log.info(
            "generate @%s : vecteur 32D disponible (%d axes)",
            username or "?",
            len(named_axes),
        )
        gen_kwargs["named_axes"] = named_axes
    else:
        log.info(
            "generate @%s : pas de vecteur — génération sans profil créateur",
            username or "?",
        )

    return generate_comments(
        classification,
        [],  # pas de comments_sample en phase Watcher (post frais)
        **gen_kwargs,
    )


def notify_new_post(
    creator: dict[str, Any],
    post: dict[str, Any],
    comments: list[str],
) -> bool:
    """Envoie une notif Telegram pour un nouveau post détecté.

    Retourne ``True`` si l'envoi a réussi, ``False`` sinon (config absente,
    erreur réseau…). N'interrompt jamais la boucle en cas d'échec.
    """
    log = logging.getLogger("aitertainment.watcher")
    from telegram_notify import send_telegram_markdown

    display_user = str(
        post.get("username") or creator.get("username") or "?"
    ).lstrip("@").strip()
    t_type = str(creator.get("t_type") or "?")
    niche = _creator_primary_niche(creator)
    reel_id = str(post.get("video_id") or "")

    url = str(post.get("url") or "").strip()
    if not url and reel_id:
        url = f"https://www.instagram.com/reel/{reel_id}/"

    padded = (list(comments) + ["—", "—", "—"])[:3]
    c1, c2, c3 = padded
    comments_block = (
        f"📝 *Commentaires suggérés :*\n"
        f"1. {_telegram_md_escape(c1)}\n"
        f"2. {_telegram_md_escape(c2)}\n"
        f"3. {_telegram_md_escape(c3)}"
    )

    text = (
        f"📢 *Nouveau post détecté*\n\n"
        f"👤 @{_telegram_md_escape(display_user)}\n"
        f"🎭 Type figé : {_telegram_md_escape(t_type)}  · "
        f"Niche : {_telegram_md_escape(niche)}\n\n"
        f"{comments_block}\n\n"
        f"🔗 {_telegram_md_escape(url)}"
    )

    try:
        send_telegram_markdown(text, parse_mode="Markdown")
    except ValueError as e:
        log.warning(
            "Telegram non envoyée @%s (config manquante) : %s", display_user, e
        )
        return False
    except Exception as e:
        log.warning("Telegram échec @%s : %s", display_user, e)
        return False
    log.info("Telegram envoyée pour @%s reel=%s", display_user, reel_id or "?")
    return True


def _mock_post(creator: dict[str, Any]) -> dict[str, Any]:
    """Post synthétique pour ``--mock`` (jamais appelle Instagram)."""
    u = str(creator.get("username") or "x")
    return {
        "video_id": f"mock_post_{u}",
        "caption": f"[mock] caption pour @{u} — drop limité ce vendredi #fitcheck",
        "hashtags": ["fitcheck", "archive", "streetwear"],
        "audio_id": "mock_snd_42",
        "url": f"https://www.instagram.com/reel/mock_{u}/",
        "posted_at": datetime.now(timezone.utc),
        "bootstrap": False,
    }


def _mock_generate(context: dict[str, Any]) -> list[str]:
    return [
        f"[mock] commentaire 1 — {context.get('t_type')} / "
        f"{(context.get('niches') or ['?'])[0]}",
        "[mock] commentaire 2 — registre figé en Discovery",
        "[mock] commentaire 3 — placeholder Watcher",
    ]


def _process_creator(
    creator: dict[str, Any],
    *,
    mock: bool,
    log: logging.Logger,
    vector_store: dict[str, dict[str, Any]] | None = None,
    browser_context: BrowserContext | None = None,
) -> tuple[bool, bool]:
    """Traite un créateur.

    Returns
    -------
    tuple[bool, bool]
        ``(changed, did_check)`` :
            - ``changed`` : la watchlist a été modifiée.
            - ``did_check`` : un appel API Instagram a été tenté
              (sert à incrémenter le compteur ``MAX_ACCOUNTS_PER_SESSION``).
    """
    username = str(creator.get("username") or "?")
    t_type = creator.get("t_type")

    if t_type is None:
        log.info("@%s : t_type absent (Discovery requis), skip.", username)
        return False, False

    did_check = not mock  # le mock ne consomme rien côté Instagram
    if mock:
        post: dict[str, Any] | None = _mock_post(creator)
    elif browser_context is None:
        log.warning("@%s : context Playwright manquant — skip.", username)
        return False, False
    else:
        post = check_new_post(creator, browser_context)

    if post is None:
        return False, did_check

    if post.get("bootstrap"):
        log.info(
            "@%s : bootstrap, mémorise last_post_id=%s sans notification.",
            username,
            post.get("video_id"),
        )
        creator["last_post_id"] = str(post["video_id"])
        return True, did_check

    media_id = str(post.get("video_id") or "")
    if not mock and browser_context is not None:
        if not _sync_post_metadata_from_reel_page(post, username, browser_context):
            log.warning(
                "@%s : métadonnées reel incohérentes — pas de notif ni génération.",
                username,
            )
            return False, did_check

    transcript = ""
    visual_description = ""
    if not mock and browser_context is not None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="ait_watch_"))
        try:
            transcript, mp4_path = _transcribe_reel(
                media_id, browser_context, tmp_dir=tmp_dir
            )
            if transcript:
                log.info("@%s transcript (%d chars)", username, len(transcript))
            if mp4_path and config.LM_STUDIO_URL and VISION_MODEL:
                visual_description = _describe_reel_visually(
                    mp4_path,
                    config.LM_STUDIO_URL,
                    VISION_MODEL,
                )
                if visual_description:
                    log.info(
                        "@%s visuel (%d chars)", username, len(visual_description)
                    )
        except Exception as e:
            log.warning("@%s transcript/visuel échoué : %s", username, e)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    gen_context = {
        "t_type": t_type,
        "niches": _creator_niches(creator),
        "caption": post.get("caption"),
        "hashtags": post.get("hashtags"),
        "audio": post.get("audio_id"),
        "url": post.get("url"),
        "username": username,
        "video_id": media_id,
        "transcript": transcript,
        "visual_description": visual_description,
    }

    comments: list[str] = []
    try:
        comments = (
            _mock_generate(gen_context)
            if mock
            else _generate_for_post(gen_context, vector_store=vector_store or {})
        )
    except Exception as e:
        log.exception("@%s : génération de commentaires échouée (%s)", username, e)
        comments = []

    if not comments:
        log.info("@%s : pas de commentaires générés (erreur ou t_type invalide).", username)
    else:
        log.info("@%s : %d commentaire(s) généré(s).", username, len(comments))

    if mock:
        log.info("[mock] notification Telegram skip (post=%s)", post.get("video_id"))
    else:
        notify_new_post(creator, post, comments)

    creator["last_post_id"] = str(post["video_id"])
    log.info(
        "@%s : last_post_id mis à jour -> %s",
        username,
        creator["last_post_id"],
    )
    return True, did_check


def run_watcher(
    *,
    watchlist_path: str | Path | None = None,
    mock: bool = False,
    max_cycles: int | None = None,
) -> None:
    """Boucle principale du Watcher.

    - Charge ``watchlist.json``.
    - Pour chaque créateur : ``check_new_post`` → si nouveau post, génère 3
      commentaires + notifie Telegram + met à jour ``last_post_id``.
    - 3 s entre chaque créateur (anti-détection).
    - À la fin du cycle, sleep ``get_poll_interval()`` puis boucle.
    - ``mock=True`` : un post synthétique par créateur, pas de Telegram, pas
      d'écriture sur ``watchlist.json``, ``max_cycles=1`` par défaut.
    - ``KeyboardInterrupt`` → arrêt propre.
    """
    setup_watcher_logger()
    log = logging.getLogger("aitertainment.watcher")
    log.info(
        "=== Watcher démarré (mock=%s, max_accounts/session=%d) ===",
        mock,
        config.MAX_ACCOUNTS_PER_SESSION,
    )

    if mock and max_cycles is None:
        max_cycles = 1

    cycle = 0
    accounts_since_pause = 0  # compteur global, reset après pause longue
    vector_store = load_vector_store(VECTOR_STORE_PATH)
    log.info("vector_store chargé : %d comptes", len(vector_store))
    last_vector_store_reload = datetime.utcnow()

    playwright_instance = None
    context: BrowserContext | None = None
    if not mock:
        playwright_instance = sync_playwright().start()
        context = get_browser_context(playwright_instance)

    try:
        while True:
            cycle += 1
            log.info("--- Cycle %d ---", cycle)

            if (datetime.utcnow() - last_vector_store_reload).total_seconds() > 21600:
                vector_store = load_vector_store(VECTOR_STORE_PATH)
                last_vector_store_reload = datetime.utcnow()
                log.info("vector_store rechargé : %d comptes", len(vector_store))

            try:
                creators = load_watchlist_synced(watchlist_path)
            except WatchlistError as e:
                log.error("Watchlist illisible : %s — abandon.", e)
                return

            if not creators:
                log.warning("Watchlist vide — rien à surveiller ce cycle.")

            for i, creator in enumerate(creators):
                try:
                    changed, did_check = _process_creator(
                        creator,
                        mock=mock,
                        log=log,
                        vector_store=vector_store,
                        browser_context=context,
                    )
                except Exception as e:
                    log.exception(
                        "@%s : erreur non gérée pendant le traitement (%s)",
                        creator.get("username", "?"),
                        e,
                    )
                    changed, did_check = False, False

                if changed and not mock:
                    try:
                        save_watchlist(creators, watchlist_path)
                    except Exception as e:
                        log.exception("Sauvegarde watchlist échouée : %s", e)

                if did_check:
                    accounts_since_pause += 1
                    if accounts_since_pause >= config.MAX_ACCOUNTS_PER_SESSION:
                        log.info(
                            "Seuil de %d comptes vérifiés atteint — pause %ds anti-flag.",
                            config.MAX_ACCOUNTS_PER_SESSION,
                            MAX_ACCOUNTS_PAUSE_S,
                        )
                        time.sleep(0 if mock else MAX_ACCOUNTS_PAUSE_S)
                        accounts_since_pause = 0

                if i < len(creators) - 1:
                    time.sleep(0 if mock else SLEEP_BETWEEN_CREATORS_S)

            if max_cycles is not None and cycle >= max_cycles:
                log.info("max_cycles=%d atteint, arrêt.", max_cycles)
                return

            interval = get_poll_interval()
            log.info("Cycle %d terminé. Pause %ds.", cycle, interval)
            time.sleep(0 if mock else interval)
    except KeyboardInterrupt:
        log.info("Watcher arrêté (Ctrl+C).")
    finally:
        if context is not None:
            context.close()
            br = context.browser
            if br:
                br.close()
        if playwright_instance is not None:
            playwright_instance.stop()


def _main_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="AItertainment Watcher — détection temps réel.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mode test : post synthétique, pas d'Instagram, pas de Telegram, "
             "pas d'écriture sur watchlist.json (un seul cycle).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Effectue un seul cycle puis quitte (utile pour cron / debug).",
    )
    args = parser.parse_args()

    run_watcher(
        mock=args.mock,
        max_cycles=1 if (args.mock or args.once) else None,
    )


__all__ = [
    "DEFAULT_WATCHLIST_PATH",
    "VALID_T_TYPES",
    "VALID_PLATFORMS",
    "NEW_POST_VIEW_THRESHOLD",
    "PRIME_INTERVAL_S",
    "DAY_INTERVAL_S",
    "NIGHT_INTERVAL_S",
    "SLEEP_BETWEEN_CREATORS_S",
    "WatchlistError",
    "load_watchlist",
    "load_watchlist_synced",
    "save_watchlist",
    "check_new_post",
    "get_poll_interval",
    "load_vector_store",
    "notify_new_post",
    "run_watcher",
    "VECTOR_STORE_PATH",
]


if __name__ == "__main__":
    _main_cli()
