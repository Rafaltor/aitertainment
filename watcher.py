"""Watcher AItertainment — détection temps réel sur des créateurs déjà profilés.

============================================================================
ARCHITECTURE — séparation stricte Discovery vs Watcher
============================================================================

Le pipeline historique (``main.py``) scrapait à la fois le contenu ET les
commentaires à chaque cycle. Cette architecture est dépréciée au profit de
deux phases distinctes :

PHASE DISCOVERY (``discovery.py``, futur module)
    - Scrape l'historique des reels d'un créateur (10–30 derniers posts).
    - Scrape les commentaires passés sur ces posts (réactions humaines réelles
      sur du contenu installé, donc exploitables par le classifieur T1→T5).
    - Classifie le créateur (``CommentClassifier``) sur ces données
      historiques pour fixer son ``t_type``.
    - Calcule ``engagement_baseline`` (likes+comments)/(views * followers).
    - Persiste le profil enrichi dans ``watchlist.json``.
    - Tourne **périodiquement** (1x/semaine typiquement), **pas en temps
      réel**. Coût Apify amorti, qualité de la classification maximale.

PHASE WATCHER (ce fichier)
    - **Ne scrape JAMAIS les commentaires** d'un post frais.
      Raison : les premiers commentaires d'un post sont majoritairement des
      bots / contributeurs très précoces — bruit pur pour la classification.
    - Le ``t_type`` du créateur est **déjà connu** via ``watchlist.json``.
    - Objectif : détecter un nouveau post et émettre un commentaire calibré
      en moins de 3 minutes après publication.
    - Le contexte du commentaire suggéré provient de deux sources :
        1. Profil créateur (``t_type``, ``niche``) → déjà dans
           ``watchlist.json`` (ne change pas à chaque tick).
        2. Contexte vidéo (caption, hashtags, audio_id) → extrait du post
           lui-même via ``instagrapi`` (session du compte dédié, sans commentaires).

Conséquence : ``generate_comments(...)`` est appelé sans ``comments_sample``,
le ton est piloté par le ``t_type`` figé en Discovery + les métadonnées du
nouveau reel.

Ce fichier expose :
    - ``load_watchlist`` / ``save_watchlist`` : persistance de la watchlist.
    - ``check_new_post`` : détection 1-shot via instagrapi.
    - ``get_poll_interval`` : intervalle de polling adaptatif (heure locale).
    - ``run_watcher`` : boucle principale (CLI : ``python watcher.py`` /
      ``python watcher.py --mock``).

Lancement :
    python watcher.py            # boucle réelle (Instagram + Ollama + Telegram)
    python watcher.py --mock     # un cycle, post & génération synthétiques,
                                 # pas d'appel Instagram/Telegram
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from instagrapi.exceptions import (
    ClientThrottledError,
    LoginRequired,
    PleaseWaitFewMinutes,
    PrivateAccount,
    PrivateError,
    RateLimitError,
    UserNotFound,
)

import config
from instagram_client import (
    InstagramAuthError,
    WatcherStopRequested,
    get_client,
    polite_sleep,
    recover_from_session_loss,
    setup_watcher_logger,
)

_HASHTAG_RE = re.compile(r"#(\w+)")

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WATCHLIST_PATH = _PROJECT_ROOT / "watchlist.json"
VECTOR_STORE_PATH = Path("data/vector_store.json")

VALID_T_TYPES = frozenset({"T1", "T2", "T2b", "T3a", "T3b", "T4", "T5"})
VALID_PLATFORMS = frozenset({"instagram", "tiktok"})

_LOGGER = logging.getLogger("aitertainment")


def _hashtags_from_caption(caption: str) -> list[str]:
    return _HASHTAG_RE.findall(caption or "")


def _media_permalink(media: Any) -> str:
    code = str(getattr(media, "code", "") or "").strip()
    if not code:
        return ""
    product = str(getattr(media, "product_type", "") or "").strip().lower()
    if product == "clips":
        return f"https://www.instagram.com/reel/{code}/"
    return f"https://www.instagram.com/p/{code}/"


def _taken_at_utc(media: Any) -> datetime:
    t = getattr(media, "taken_at", None)
    if isinstance(t, datetime):
        if t.tzinfo is None:
            return t.replace(tzinfo=timezone.utc)
        return t.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _extract_reel_music_id(media: Any) -> str | None:
    """Audio associé au reel (``product_type == clips``), sinon ``None``."""
    product = str(getattr(media, "product_type", "") or "").strip().lower()
    if product != "clips":
        return None

    cm = getattr(media, "clips_metadata", None)
    if cm is None:
        return None

    def _from_dict(d: dict[str, Any]) -> str | None:
        mid = d.get("music_canonical_id")
        if mid:
            return str(mid)
        mi = d.get("music_info")
        if isinstance(mi, dict):
            for k in ("id", "pk", "audio_cluster_id", "music_canonical_id"):
                v = mi.get(k)
                if v:
                    return str(v)
        oi = d.get("original_sound_info")
        if isinstance(oi, dict):
            for k in ("music_canonical_id", "audio_asset_id", "original_media_id"):
                v = oi.get(k)
                if v is not None:
                    return str(v)
        return None

    if isinstance(cm, dict):
        return _from_dict(cm)

    mid = getattr(cm, "music_canonical_id", None)
    if mid:
        return str(mid)
    mi = getattr(cm, "music_info", None)
    if isinstance(mi, dict):
        for k in ("id", "pk", "audio_cluster_id", "music_canonical_id"):
            v = mi.get(k)
            if v:
                return str(v)
    oi = getattr(cm, "original_sound_info", None)
    if oi is not None:
        for attr in ("music_canonical_id", "audio_asset_id", "original_media_id"):
            v = getattr(oi, attr, None)
            if v is not None:
                return str(v)
    return None


class _SessionLost(Exception):
    """Signal interne : la session est perdue, ``run_watcher`` doit déclencher
    ``recover_from_session_loss`` (pause 15 min + retry login)."""


def _fetch_latest_media_instagram(client: Any, username: str) -> Any | None:
    """``user_medias(..., 1)`` avec :

    - délai aléatoire ``IG_SLEEP_MIN..IG_SLEEP_MAX`` avant chaque tentative,
    - retry unique sur rate limit court (``RateLimitError`` /
      ``ClientThrottledError``) → sleep 30 s,
    - propagation ``_SessionLost`` sur ``LoginRequired`` /
      ``PleaseWaitFewMinutes`` → la boucle décidera de la pause longue +
      retry login (cf. ``recover_from_session_loss``).
    """
    wlog = logging.getLogger("aitertainment.watcher")

    for attempt in range(2):
        try:
            user_id = client.user_id_from_username(username)
            polite_sleep()
            medias = client.user_medias(str(user_id), amount=1)
            if not medias:
                wlog.info("check_new_post @%s : aucun média sur le profil", username)
                return None
            return medias[0]
        except (RateLimitError, ClientThrottledError) as e:
            if attempt == 0:
                wlog.warning(
                    "check_new_post @%s : rate limit court Instagram (%s), pause 30s puis retry",
                    username,
                    e,
                )
                time.sleep(30)
                continue
            wlog.warning(
                "check_new_post @%s : rate limit après retry — abandon (%s)",
                username,
                e,
            )
            return None
        except (LoginRequired, PleaseWaitFewMinutes) as e:
            wlog.warning(
                "check_new_post @%s : session Instagram perdue (%s) — recovery.",
                username,
                type(e).__name__,
            )
            raise _SessionLost(str(e)) from e
        except UserNotFound as e:
            wlog.warning("check_new_post @%s : compte inexistant (%s)", username, e)
            return None
        except PrivateAccount as e:
            wlog.warning("check_new_post @%s : compte privé (%s)", username, e)
            return None
        except PrivateError as e:
            wlog.warning(
                "check_new_post @%s : médias inaccessibles (API privée) (%s)",
                username,
                e,
            )
            return None
        except Exception as e:
            wlog.warning(
                "check_new_post @%s : erreur instagrapi inattendue (%s)",
                username,
                e,
            )
            return None

    return None


def check_new_post(creator: dict[str, Any]) -> dict[str, Any] | None:
    """Détecte si ``creator`` a publié un média plus récent que ``last_post_id``.

    Utilise **instagrapi** (session du compte dédié, cf. ``instagram_client``) :
    pas d'Apify sur cette étape — coût réduit pour le polling fréquent.

    Étapes :
        1. ``get_client()`` — session réutilisée via ``session.json``.
        2. ``user_id_from_username`` puis ``user_medias(..., amount=1)``.
        3. Compare ``str(media.pk)`` à ``creator['last_post_id']``.

    Rate limit Instagram : pause **30s** puis **une seule** nouvelle tentative ;
    compte privé / inexistant / erreur API → ``None`` (log ``logs/watcher.log``).

    Comportement identique au contrat précédent :
        - identique → ``None`` ;
        - différent ou ``last_post_id`` null (bootstrap) → dict avec ``video_id``
          (alias stable du media pk), ``caption``, ``hashtags``, ``audio_id``,
          ``url``, ``posted_at``, ``bootstrap``.

    Cette fonction **ne modifie pas** ``watchlist.json``.
    """
    if not isinstance(creator, dict):
        _LOGGER.warning("check_new_post : creator doit être un dict, reçu %s", type(creator).__name__)
        return None

    username_raw = creator.get("username", "")
    username = str(username_raw).lstrip("@").strip() if username_raw else ""
    if not username:
        _LOGGER.warning("check_new_post : 'username' manquant dans creator")
        return None

    platform = str(creator.get("platform", "") or "").strip().lower()
    if platform and platform != "instagram":
        _LOGGER.warning(
            "check_new_post @%s : plateforme %r non supportée par instagrapi (skip)",
            username,
            platform,
        )
        return None

    last_post_id_raw = creator.get("last_post_id")
    last_post_id = str(last_post_id_raw).strip() if last_post_id_raw else None

    setup_watcher_logger()
    wlog = logging.getLogger("aitertainment.watcher")

    try:
        client = get_client()
    except InstagramAuthError as e:
        wlog.warning("check_new_post @%s : connexion Instagram impossible (%s)", username, e)
        return None

    media = _fetch_latest_media_instagram(client, username)
    if media is None:
        return None

    media_id = str(getattr(media, "pk", "") or "").strip()
    if not media_id:
        wlog.warning("check_new_post @%s : média sans pk", username)
        return None

    if last_post_id is not None and media_id == last_post_id:
        _LOGGER.debug(
            "check_new_post @%s : pas de nouveau post (last_post_id=%s)",
            username,
            last_post_id,
        )
        return None

    bootstrap = last_post_id is None
    caption = str(getattr(media, "caption_text", "") or "")
    hashtags = _hashtags_from_caption(caption)
    audio_id = _extract_reel_music_id(media)
    url = _media_permalink(media)
    posted_at = _taken_at_utc(media)

    if bootstrap:
        wlog.info(
            "check_new_post @%s : bootstrap (last_post_id null) -> media_id=%s",
            username,
            media_id,
        )
    else:
        wlog.info(
            "check_new_post @%s : nouveau post media_id=%s (précédent %s)",
            username,
            media_id,
            last_post_id,
        )

    return {
        "video_id": media_id,
        "caption": caption,
        "hashtags": hashtags,
        "audio_id": audio_id,
        "url": url,
        "posted_at": posted_at,
        "bootstrap": bootstrap,
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

    # Schéma 2026-05 : ``niches`` (liste) est la source de vérité, ``niche``
    # (string) reste exposé en alias pour les callers legacy. Migration lazy :
    # une entrée historique avec uniquement ``"niche": "humour"`` reste valide
    # et est promue à ``"niches": ["humour"]`` à la lecture.
    niches_raw = entry.get("niches")
    if niches_raw is not None:
        if not isinstance(niches_raw, list):
            raise WatchlistError(
                f"creators[{index}].niches doit être une liste, "
                f"reçu {type(niches_raw).__name__}"
            )
        niches: list[str] = []
        for j, n in enumerate(niches_raw):
            if not isinstance(n, str):
                raise WatchlistError(
                    f"creators[{index}].niches[{j}] doit être une chaîne, "
                    f"reçu {type(n).__name__}"
                )
            cleaned = n.strip()
            if cleaned:
                niches.append(cleaned)
    else:
        niche_legacy = entry.get("niche", "")
        if not isinstance(niche_legacy, str):
            raise WatchlistError(f"creators[{index}].niche doit être une chaîne")
        niche_legacy = niche_legacy.strip()
        niches = [niche_legacy] if niche_legacy else []
    # Alias string : ``niche = niches[0]`` (ou ``""`` si la liste est vide,
    # autorisé pour rétro-compat avec les watchlists historiques).
    niche = niches[0] if niches else ""

    t_type_raw = entry.get("t_type")
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
        "niche": niche,
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

    Schéma attendu : ``{"creators": [ {username, platform, niche, t_type,
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
    payload = {"creators": normalized}

    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)
    _LOGGER.info("Watchlist sauvegardée : %d créateur(s) -> %s", len(normalized), p)


# =============================================================================
# Boucle Watcher
# =============================================================================

PRIME_INTERVAL_S = 300    # 17h–21h : prime time, polling toutes les 5 min
DAY_INTERVAL_S = 600      # 09h–17h : journée, polling toutes les 10 min
NIGHT_INTERVAL_S = 1800   # 21h–09h : nuit, polling toutes les 30 min
SLEEP_BETWEEN_CREATORS_S = 3  # anti-détection (rate limit human-like)
# Pause longue déclenchée tous les ``MAX_ACCOUNTS_PER_SESSION`` comptes vérifiés
# (voir ``config.py``) — protège contre le profilage anti-bot.
MAX_ACCOUNTS_PAUSE_S = 600  # 10 min


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

    Renvoie ``[]`` si ``t_type`` est ``T1`` ou ``T3a`` (le brief les exclut
    de la génération).
    """
    from modules.classifier import generate_comments  # import local : Ollama

    log = logging.getLogger("aitertainment.watcher")
    vector_store = vector_store or {}
    t_type = str(context.get("t_type") or "")
    # Schéma 2026-05 : ``niches`` (liste) prioritaire avec rétro-compat sur
    # l'ancien champ ``niche`` (string). Le caller (``run_watcher``) passe
    # désormais ``creator["niches"]`` à la construction du context.
    niches_raw = context.get("niches")
    if isinstance(niches_raw, list) and niches_raw:
        niches: list[str] | str = list(niches_raw)
    else:
        niches = str(context.get("niche") or "")

    if t_type in ("T1", "T3a") or t_type not in VALID_T_TYPES:
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
        },
    }
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
    from instagram_client import send_telegram_markdown

    username = str(creator.get("username") or "?")
    t_type = str(creator.get("t_type") or "?")
    niche = str(creator.get("niche") or "?")

    caption_short = _truncate(str(post.get("caption") or ""), 240)
    hashtags = post.get("hashtags") or []
    hashtags_str = " ".join(f"#{str(h)}" for h in hashtags) or "—"
    url = str(post.get("url") or "—")

    padded = (list(comments) + ["—", "—", "—"])[:3]
    c1, c2, c3 = padded

    text = (
        f"📢 *Nouveau post détecté*\n\n"
        f"👤 @{_telegram_md_escape(username)} "
        f"({_telegram_md_escape(str(creator.get('platform') or 'instagram'))})\n"
        f"🎭 Type figé : {_telegram_md_escape(t_type)}  · "
        f"Niche : {_telegram_md_escape(niche)}\n\n"
        f"💬 *Caption :* {_telegram_md_escape(caption_short) or '—'}\n"
        f"🏷  *Hashtags :* {_telegram_md_escape(hashtags_str)}\n\n"
        f"📝 *Commentaires suggérés :*\n"
        f"1. {_telegram_md_escape(c1)}\n"
        f"2. {_telegram_md_escape(c2)}\n"
        f"3. {_telegram_md_escape(c3)}\n\n"
        f"🔗 {_telegram_md_escape(url)}"
    )

    try:
        send_telegram_markdown(text, parse_mode="Markdown")
    except ValueError as e:
        log.warning("Telegram non envoyée @%s (config manquante) : %s", username, e)
        return False
    except Exception as e:
        log.warning("Telegram échec @%s : %s", username, e)
        return False
    log.info("Telegram envoyée pour @%s", username)
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
        f"[mock] commentaire 1 — {context.get('t_type')} / {context.get('niche')}",
        "[mock] commentaire 2 — registre figé en Discovery",
        "[mock] commentaire 3 — placeholder Watcher",
    ]


def _process_creator(
    creator: dict[str, Any],
    *,
    mock: bool,
    log: logging.Logger,
    vector_store: dict[str, dict[str, Any]] | None = None,
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
    else:
        post = check_new_post(creator)

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

    context = {
        "t_type": t_type,
        # Schéma 2026-05 : on propage la liste ``niches`` ; ``niche`` est
        # conservé en alias rétro-compat (lu par d'éventuels callers de
        # mock / debug encore sur l'ancien schéma).
        "niches": creator.get("niches") or (
            [creator["niche"]] if isinstance(creator.get("niche"), str) and creator["niche"] else []
        ),
        "niche": creator.get("niche"),
        "caption": post.get("caption"),
        "hashtags": post.get("hashtags"),
        "audio": post.get("audio_id"),
        "url": post.get("url"),
        "username": username,
    }

    try:
        comments = (
            _mock_generate(context)
            if mock
            else _generate_for_post(context, vector_store=vector_store or {})
        )
    except Exception as e:
        log.exception("@%s : génération de commentaires échouée (%s)", username, e)
        comments = []

    if not comments:
        log.info("@%s : pas de commentaires générés (T1/T3a ou erreur).", username)
    else:
        log.info("@%s : %d commentaire(s) généré(s).", username, len(comments))

    if mock:
        log.info("[mock] notification Telegram skip (post=%s)", post.get("video_id"))
    elif comments:
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

    try:
        while True:
            cycle += 1
            log.info("--- Cycle %d ---", cycle)

            if (datetime.utcnow() - last_vector_store_reload).total_seconds() > 21600:
                vector_store = load_vector_store(VECTOR_STORE_PATH)
                last_vector_store_reload = datetime.utcnow()
                log.info("vector_store rechargé : %d comptes", len(vector_store))

            try:
                creators = load_watchlist(watchlist_path)
            except WatchlistError as e:
                log.error("Watchlist illisible : %s — abandon.", e)
                return

            if not creators:
                log.warning("Watchlist vide — rien à surveiller ce cycle.")

            for i, creator in enumerate(creators):
                # Recovery propagée : si la session est perdue pendant ce cycle,
                # on tente recover_from_session_loss qui peut lever
                # WatcherStopRequested (capture en dehors du for).
                try:
                    changed, did_check = _process_creator(
                        creator,
                        mock=mock,
                        log=log,
                        vector_store=vector_store,
                    )
                except _SessionLost as e:
                    log.warning(
                        "Session perdue pendant @%s : %s → recovery.",
                        creator.get("username", "?"),
                        e,
                    )
                    recover_from_session_loss()  # peut lever WatcherStopRequested
                    log.info("Reconnexion réussie, on continue le cycle.")
                    changed, did_check = False, False
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
    except WatcherStopRequested as e:
        log.error("Watcher stoppé proprement (recovery KO) : %s", e)
        return
    except KeyboardInterrupt:
        log.info("Watcher arrêté (Ctrl+C).")


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
    "PRIME_INTERVAL_S",
    "DAY_INTERVAL_S",
    "NIGHT_INTERVAL_S",
    "SLEEP_BETWEEN_CREATORS_S",
    "WatchlistError",
    "load_watchlist",
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
