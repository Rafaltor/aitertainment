"""Discovery — Layer 0 du pipeline AItertainment (exploration Instagram).

============================================================================
Architecture (phase lente, comportement humain simulé)
============================================================================

``discovery.py`` tourne **volontairement lent** pour explorer le réseau
Instagram sans déclencher de flags anti-bot. Il est **domain-agnostic** :
la niche / le vertical est toujours un **paramètre de données**
(``seeds.json``), jamais codé en dur dans la logique métier.

Trois responsabilités (implémentation future) :

1. **Explorer** depuis des comptes seed (graph related, hashtags, etc.)
   pour remonter des profils candidats.

2. **Scorer** chaque candidat sur son **historique** (≥10 posts), via une
   médiane sur les métriques agrégées — **score profil**, pas score vidéo
   isolée.

3. **Notifier** le Bot Telegram #2 pour **validation humaine** avant toute
   insertion dans ``watchlist.json``.

Principes :

- Sleeps longs et aléatoires (anti-détection).
- **Blacklist** : ne jamais reproposer un profil déjà vu (validé, rejeté ou
  simplement exploré) — ``blacklist.json``.
- **Feedback loop** : apprendre des décisions humaines (à brancher plus tard).

Ce module expose uniquement la **persistance** et les chemins par défaut :
chargement / sauvegarde atomique de ``seeds``, ``blacklist``, ``candidates``.
La boucle d'exploration et les appels instagrapi viendront ensuite.

============================================================================
Fichiers de données (répertoire ``data/``)
============================================================================

- ``data/seeds.json`` — comptes de départ et métadonnées par domaine.
- ``data/blacklist.json`` — profils déjà traités (plus de proposition).
- ``data/candidates.json`` — file d'attente avant validation Telegram.

"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
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
from database import (
    DatabaseIOError,
    load_db,
    save_db,
    upsert_profile,
)
from instagram_client import (
    InstagramAuthError,
    WatcherStopRequested,
    get_client,
    polite_sleep,
    recover_from_session_loss,
    setup_watcher_logger,
)

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_SEEDS_PATH = DEFAULT_DATA_DIR / "seeds.json"
DEFAULT_BLACKLIST_PATH = DEFAULT_DATA_DIR / "blacklist.json"
DEFAULT_CANDIDATES_PATH = DEFAULT_DATA_DIR / "candidates.json"

_LOGGER = logging.getLogger("aitertainment.discovery")

# Bornes pour la phase Discovery — on sleep beaucoup plus longtemps que le
# Watcher : exploration humaine simulée, pas de course à la milliseconde.
# Les valeurs vivent dans config.py (overridable via .env), on les ré-exporte
# ici pour rétro-compat (les autres modules / tests référencent ces noms).
DISCOVERY_BETWEEN_POSTS_MIN_S = float(config.DISCOVERY_BETWEEN_POSTS_MIN_S)
DISCOVERY_BETWEEN_POSTS_MAX_S = float(config.DISCOVERY_BETWEEN_POSTS_MAX_S)

MIN_FOLLOWERS = 1_000
MAX_FOLLOWERS = 1_000_000
MIN_MEDIA_COUNT = 2
HISTORY_MEDIAS_TO_FETCH = 15          # historique (+ marge vs épinglés exclus)
MIN_TOTAL_MEDIAS_REQUIRED = 3          # nb min de médias TOTAL (reels + posts) pour scorer
MAX_POSTS_FOR_SCORING = 4              # cap sur les Posts scorés (les 4 plus récents non-épinglés)
COMMENTS_MEDIA_SAMPLE = 3              # nombre de médias dont on scrape les commentaires
COMMENTS_PER_MEDIA = 50                # ↑ depuis 15 : améliore le signal classifier sur les
                                       # posts à faible engagement (la plupart des Reels
                                       # n'ont que 5-15 commentaires likés sur les 50 premiers)
REEL_PRODUCT_TYPE = "clips"

# Pondérations & valeurs de **référence** SCORE_PROFIL
# ----------------------------------------------------------------------------
#
# La normalisation de chaque signal continu est **logarithmique** :
#
#     contribution = log10(value*scale + 1) / log10(ref*scale + 1) * weight
#
# où ``ref`` est la valeur cible où la métrique doit délivrer 100 % du poids.
# **Pas de plafond** : un signal au-delà de ``ref`` continue à scorer (avec
# rendement décroissant), un signal proche de zéro scoré faiblement, un
# signal exactement à ``ref`` scoré exactement le poids.
#
# Exception : ``posting_rhythm`` est plafonné à ``ref`` (1 média/jour) — on
# ne veut pas récompenser le spam.

# Valeurs de référence (cible de qualité = 100 % du poids)
SCORE_REEL_RATIO_REF = 10.0       # views/followers — créateur viral établi
SCORE_REEL_RATIO_P90_REF = 50.0   # views/followers — viralité épisodique forte
SCORE_REEL_ENGAGEMENT_REF = 0.10  # 10 % (likes+comments)/followers — excellent
SCORE_POST_RATIO_REF = 0.15       # 15 % likes/followers — excellent post
SCORE_POST_ENGAGEMENT_REF = 0.15  # 15 % (likes+comments)/followers — excellent
SCORE_RHYTHM_REF = 1.0            # 1 média/jour — optimum (plafonné)

# Pondérations Reels (Σ = 925)
SCORE_REEL_RATIO_W = 250          # médiane (stabilité)
SCORE_REEL_RATIO_P90_W = 100      # P90 (potentiel viral)
SCORE_REEL_ENGAGEMENT_W = 200
SCORE_REEL_TREND_W = 150
SCORE_REEL_T_TYPE_W = 200
SCORE_REEL_FREQ_W = 25            # ne pas sur-pondérer le rythme

# Pondérations Posts (Σ = 725)
SCORE_POST_RATIO_W = 300
SCORE_POST_ENGAGEMENT_W = 250
SCORE_POST_T_TYPE_W = 150
SCORE_POST_FREQ_W = 25

# --- Aliases rétro-compat (les anciens ``*_CAP`` désignent désormais des
# **références** log, pas des plafonds linéaires — la sémantique a changé) ---
SCORE_REEL_RATIO_CAP = SCORE_REEL_RATIO_REF
SCORE_REEL_RATIO_P90_CAP = SCORE_REEL_RATIO_P90_REF
SCORE_ENGAGEMENT_CAP = SCORE_REEL_ENGAGEMENT_REF
SCORE_POST_RATIO_CAP = SCORE_POST_RATIO_REF
SCORE_RATIO_CAP = SCORE_REEL_RATIO_REF
SCORE_RATIO_W = SCORE_REEL_RATIO_W
SCORE_ENGAGEMENT_W = SCORE_REEL_ENGAGEMENT_W
SCORE_TREND_W = SCORE_REEL_TREND_W
SCORE_T_TYPE_W = SCORE_REEL_T_TYPE_W
SCORE_FREQ_W = SCORE_REEL_FREQ_W

CANDIDATE_SCORE_THRESHOLD = 500.0

# Seuil distinct (plus bas) au-dessus duquel ``score_and_persist`` envoie une
# notif Telegram. Tout score est de toute façon upserté dans ``database.json``.
# Source de vérité : ``config.DISCOVERY_NOTIFY_THRESHOLD`` (overridable .env).
DISCOVERY_NOTIFY_THRESHOLD = float(config.DISCOVERY_NOTIFY_THRESHOLD)

# Profils explorés par seed (suffisant pour découvrir 50 candidats sans
# attaquer le rate limit instagrapi sur user_following).
FOLLOWING_FETCH_AMOUNT = 50

# ----------------------------------------------------------------------------
# Limites humaines simulées (Layer 0)
# ----------------------------------------------------------------------------
# Quota humain quotidien (config.py — overridable via .env).
MAX_PROFILES_PER_DAY = int(config.MAX_PROFILES_PER_DAY)

# Nuit : inactif 23h00 → 08h00 (heure locale).
NIGHT_START_HOUR = 23
NIGHT_END_HOUR = 8
# Pause déjeuner obligatoire : 12h00 → 14h00 (heure locale).
LUNCH_START_HOUR = 12
LUNCH_END_HOUR = 14

# Burst max : 2h d'activité, puis 30 min de pause obligatoire.
ACTIVITY_BURST_S = 2 * 3600
ACTIVITY_PAUSE_S = 30 * 60

# Sleep aléatoire entre deux profils scorés (config.py — overridable via .env).
DISCOVERY_BETWEEN_PROFILES_MIN_S = float(config.DISCOVERY_BETWEEN_PROFILES_MIN_S)
DISCOVERY_BETWEEN_PROFILES_MAX_S = float(config.DISCOVERY_BETWEEN_PROFILES_MAX_S)
# Alias rétro-compat (anciens noms locaux).
INTER_PROFILE_MIN_S = DISCOVERY_BETWEEN_PROFILES_MIN_S
INTER_PROFILE_MAX_S = DISCOVERY_BETWEEN_PROFILES_MAX_S

# Sleep aléatoire entre deux domaines (boucle ``run_discovery``).
INTER_DOMAIN_MIN_S = 1 * 3600
INTER_DOMAIN_MAX_S = 3 * 3600

# Logs Discovery dédiés (séparés du Watcher).
DEFAULT_DISCOVERY_LOG_PATH = _PROJECT_ROOT / "logs" / "discovery.log"


class DiscoveryIOError(ValueError):
    """Erreur de lecture / écriture ou schéma JSON discovery invalide."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DiscoveryIOError(f"fichier absent : {path}")
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise DiscoveryIOError(f"lecture impossible : {path} ({e})") from e
    if not raw:
        raise DiscoveryIOError(f"fichier vide : {path}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise DiscoveryIOError(f"JSON invalide dans {path} : {e}") from e
    if not isinstance(data, dict):
        raise DiscoveryIOError(
            f"racine JSON doit être un objet dans {path}, reçu {type(data).__name__}"
        )
    return data


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


# --- seeds ---


def load_seeds(path: str | Path | None = None) -> dict[str, Any]:
    """Charge ``seeds.json``.

    Schéma attendu::

        {
          "domains": [
            {
              "name": str,
              "seeds": [username, ...],
              "keywords": [...],
              "t_types_target": ["T2", ...]
            },
            ...
          ]
        }

    Returns
    -------
    dict
        Objet racine complet (au minimum une clé ``domains`` liste).
    """
    p = Path(path) if path else DEFAULT_SEEDS_PATH
    data = _read_json(p)
    domains = data.get("domains")
    if domains is None:
        raise DiscoveryIOError(f'"domains" manquant dans {p}')
    if not isinstance(domains, list):
        raise DiscoveryIOError(f'"domains" doit être une liste dans {p}')
    _LOGGER.debug("Seeds chargés : %d domaine(s) depuis %s", len(domains), p.name)
    return data


# --- blacklist ---


def load_blacklist(path: str | Path | None = None) -> dict[str, Any]:
    """Charge ``blacklist.json``.

    Schéma minimal::

        { "profiles": [ { "username", "platform", "outcome", "added_at", ... }, ... ] }

    Si le fichier est absent, retourne une blacklist vide (premier run).
    """
    p = Path(path) if path else DEFAULT_BLACKLIST_PATH
    if not p.exists():
        _LOGGER.info("Blacklist absente — démarrage vide (%s)", p.name)
        return {"profiles": []}
    data = _read_json(p)
    profiles = data.get("profiles")
    if profiles is None:
        raise DiscoveryIOError(f'"profiles" manquant dans {p}')
    if not isinstance(profiles, list):
        raise DiscoveryIOError(f'"profiles" doit être une liste dans {p}')
    _LOGGER.debug(
        "Blacklist chargée : %d entrée(s) depuis %s", len(profiles), p.name
    )
    return data


def save_blacklist(
    blacklist: dict[str, Any],
    path: str | Path | None = None,
) -> None:
    """Écrit ``blacklist.json`` de façon atomique."""
    if not isinstance(blacklist, dict):
        raise DiscoveryIOError(
            f"blacklist doit être un dict, reçu {type(blacklist).__name__}"
        )
    profiles = blacklist.get("profiles")
    if profiles is None or not isinstance(profiles, list):
        raise DiscoveryIOError('"profiles" doit être une liste non absente')
    p = Path(path) if path else DEFAULT_BLACKLIST_PATH
    _atomic_write_json(p, {"profiles": profiles})
    _LOGGER.info("Blacklist sauvegardée : %d entrée(s) -> %s", len(profiles), p)


# --- candidates ---


def load_candidates(path: str | Path | None = None) -> dict[str, Any]:
    """Charge ``candidates.json``.

    Schéma minimal::

        { "candidates": [ { ... }, ... ] }

    Fichier absent → file vide.
    """
    p = Path(path) if path else DEFAULT_CANDIDATES_PATH
    if not p.exists():
        _LOGGER.info("Candidates absent — démarrage vide (%s)", p.name)
        return {"candidates": []}
    data = _read_json(p)
    cands = data.get("candidates")
    if cands is None:
        raise DiscoveryIOError(f'"candidates" manquant dans {p}')
    if not isinstance(cands, list):
        raise DiscoveryIOError(f'"candidates" doit être une liste dans {p}')
    _LOGGER.debug(
        "Candidates chargés : %d entrée(s) depuis %s", len(cands), p.name
    )
    return data


def save_candidates(
    candidates_state: dict[str, Any],
    path: str | Path | None = None,
) -> None:
    """Écrit ``candidates.json`` de façon atomique."""
    if not isinstance(candidates_state, dict):
        raise DiscoveryIOError(
            f"candidates_state doit être un dict, reçu {type(candidates_state).__name__}"
        )
    cands = candidates_state.get("candidates")
    if cands is None or not isinstance(cands, list):
        raise DiscoveryIOError('"candidates" doit être une liste non absente')
    p = Path(path) if path else DEFAULT_CANDIDATES_PATH
    _atomic_write_json(p, {"candidates": cands})
    _LOGGER.info(
        "Candidates sauvegardés : %d entrée(s) -> %s", len(cands), p.name
    )


# =============================================================================
# Scoring de profil (Layer 0)
# =============================================================================


class DiscoverySessionLost(RuntimeError):
    """Session Instagram perdue pendant Discovery (LoginRequired / wait)."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_blacklisted(username: str, blacklist: dict[str, Any]) -> bool:
    target = username.lower().lstrip("@").strip()
    for p in blacklist.get("profiles", []):
        if not isinstance(p, dict):
            continue
        u = str(p.get("username") or "").lower().lstrip("@").strip()
        if u == target:
            return True
    return False


def _is_media_pinned(media: Any) -> bool:
    """True si le média est épinglé en haut du profil (hors scope scoring).

    Instagrapi expose ``is_pinned`` sur les objets Media ; on accepte aussi un
    dict pour les tests / mocks.
    """
    if isinstance(media, dict):
        return bool(media.get("is_pinned", False))
    return getattr(media, "is_pinned", None) is True


def _media_views(media: Any) -> int:
    """Vues d'un media (``play_count`` pour reels, ``view_count`` sinon)."""
    pc = getattr(media, "play_count", None)
    if pc:
        return int(pc)
    vc = getattr(media, "view_count", None)
    if vc:
        return int(vc)
    return 0


def _media_metric_row(media: Any) -> dict[str, Any]:
    return {
        "media_id": str(getattr(media, "pk", "") or getattr(media, "id", "") or ""),
        "views": _media_views(media),
        "likes": int(getattr(media, "like_count", 0) or 0),
        "comments": int(getattr(media, "comment_count", 0) or 0),
        "shares": getattr(media, "share_count", None),
        "taken_at": getattr(media, "taken_at", None),
        "caption_text": str(getattr(media, "caption_text", "") or ""),
        "product_type": str(getattr(media, "product_type", "") or ""),
    }


def debug_reel_views(username: str) -> None:
    """Debug brut : compare les compteurs de vues Reels ``user_medias`` vs ``media_info``.

    Utile pour diagnostiquer pourquoi ``view_count`` / ``play_count`` sont à 0
    sur les lignes ``user_medias`` alors que le détail média les expose.

    Pas de scoring, pas de ``polite_sleep`` — uniquement des ``print`` ligne par ligne.
    """
    u = (username or "").lstrip("@").strip()
    if not u:
        print("debug_reel_views: username vide")
        return

    client = get_client()
    user_id = client.user_id_from_username(u)
    medias = client.user_medias(str(user_id), amount=20)
    reels = [
        m
        for m in medias
        if str(getattr(m, "product_type", "") or "") == REEL_PRODUCT_TYPE
    ]

    print(f"=== debug_reel_views @{u} ===")
    print(f"user_id={user_id}  total_medias={len(medias)}  reels(clips)={len(reels)}")
    print()

    for idx, m in enumerate(reels, start=1):
        pk = getattr(m, "pk", None) or getattr(m, "id", None)
        pk_str = str(pk) if pk is not None else ""

        vc_um = getattr(m, "view_count", None)
        pc_um = getattr(m, "play_count", None)

        print(f"--- Reel #{idx} (user_medias) pk={pk_str!r} ---")
        print(f"  media_id (pk): {pk_str!r}")
        print(f"  taken_at: {getattr(m, 'taken_at', None)!r}")
        print(f"  is_pinned: {getattr(m, 'is_pinned', None)!r}")
        print(f"  product_type: {getattr(m, 'product_type', None)!r}")
        print(f"  view_count (user_medias): {vc_um!r}")
        print(f"  play_count (user_medias): {pc_um!r}")
        print(f"  views (_media_views): {_media_views(m)}")
        print()

        if not pk_str:
            print("  media_info: skip (pk vide)")
            print()
            continue

        try:
            full = client.media_info(pk_str)
        except Exception as e:
            print(f"  media_info({pk_str}) ERROR: {type(e).__name__}: {e}")
            print()
            continue

        vc_mi = getattr(full, "view_count", None)
        pc_mi = getattr(full, "play_count", None)
        vvc = getattr(full, "video_view_count", None)
        mt = getattr(full, "media_type", None)
        pt_mi = getattr(full, "product_type", None)
        meta = getattr(full, "clips_metadata", None)
        if meta is None:
            clips_vc_repr = "<clips_metadata absent>"
        elif isinstance(meta, dict):
            clips_vc_repr = repr(meta.get("view_count", "<pas de clé view_count>"))
        else:
            clips_vc_repr = repr(getattr(meta, "view_count", "<pas d'attribut view_count>"))

        print(f"  --- media_info({pk_str}) ---")
        print(f"  view_count (media_info): {vc_mi!r}")
        print(f"  play_count (media_info): {pc_mi!r}")
        print(f"  video_view_count (media_info): {vvc!r}")
        print(f"  media_type: {mt!r}")
        print(f"  product_type (media_info): {pt_mi!r}")
        print(f"  clips_metadata.get('view_count') si dict / équivalent: {clips_vc_repr}")
        print()


def _top_up_reel_views_via_media_info(
    client: Any,
    reels: list[dict[str, Any]],
    *,
    log: logging.Logger,
    username: str,
) -> None:
    """Complète ``views`` pour les Reels où ``user_medias`` n'a pas les compteurs.

    Le debug @raikkonenaf a confirmé que ``user_medias`` renvoie **toujours**
    ``view_count = play_count = 0`` sur les Reels, alors que ``media_info(pk)``
    expose ``play_count`` non-nul. ``media_info`` est donc la **seule source
    fiable** : on top-up tous les Reels à ``views == 0`` (pas de cap).
    ``polite_sleep()`` est appelé après chaque appel.
    """
    if not reels:
        return
    zero_views = [r for r in reels if int(r.get("views") or 0) == 0]
    if not zero_views:
        return
    zero_views.sort(key=lambda r: r["taken_at"], reverse=True)
    for reel in zero_views:
        pk = str(reel.get("media_id") or "").strip()
        if not pk:
            continue
        try:
            media_full = client.media_info(pk)
        except (LoginRequired, PleaseWaitFewMinutes) as e:
            raise DiscoverySessionLost(
                f"media_info({pk}) @{username}: {e}"
            ) from e
        except (RateLimitError, ClientThrottledError) as e:
            log.warning(
                "score_profile @%s : rate limit media_info(%s) (%s) — skip top-up.",
                username,
                pk,
                e,
            )
            continue
        except PrivateError as e:
            log.warning(
                "score_profile @%s : media_info(%s) privé (%s) — skip.",
                username,
                pk,
                e,
            )
            continue
        except Exception as e:
            log.warning(
                "score_profile @%s : media_info(%s) erreur (%s) — skip.",
                username,
                pk,
                e,
            )
            continue
        vc = int(getattr(media_full, "view_count", None) or 0)
        pc = int(getattr(media_full, "play_count", None) or 0)
        reel["views"] = pc or vc or 0
        polite_sleep()


# Métriques Reels (basées sur ``views``)
# ---------------------------------------------------------------------------


def _reel_ratio_median(reels: list[dict[str, Any]], followers: int) -> float:
    """Médiane(views / followers) — ratio "viral" propre aux Reels.

    On ne considère que les Reels avec ``views > 0`` (post top-up via
    ``media_info``). Les vues à 0 sont des "données manquantes", pas un signal
    de portée nulle : les inclure tirerait artificiellement la médiane vers le
    bas et pénaliserait à tort le profil.
    """
    if followers <= 0 or not reels:
        return 0.0
    reels_with_views = [r for r in reels if int(r.get("views", 0) or 0) > 0]
    if not reels_with_views:
        return 0.0
    return float(
        statistics.median(
            int(r["views"]) / float(followers) for r in reels_with_views
        )
    )


def _reel_ratio_p90(reels: list[dict[str, Any]], followers: int) -> float:
    """90e percentile de ``views / followers`` — capte le **potentiel viral**.

    Méthode "nearest-rank" : sur ``N`` Reels triés croissants, on retourne le
    ratio à l'indice ``ceil(0.9 * N) - 1`` (un Reel réellement publié, pas
    une interpolation entre deux). Ainsi, sur 10 Reels, c'est le 9ᵉ le plus
    haut qui sort.

    Comme ``_reel_ratio_median``, on ne considère que les Reels avec
    ``views > 0``. Avec **moins de 3** Reels exploitables, on retourne ``0.0``
    (signal pas significatif).
    """
    if followers <= 0 or not reels:
        return 0.0
    ratios = sorted(
        int(r["views"]) / float(followers)
        for r in reels
        if int(r.get("views", 0) or 0) > 0
    )
    if len(ratios) < 3:
        return 0.0
    idx = max(0, math.ceil(0.9 * len(ratios)) - 1)
    return float(ratios[idx])


def _reel_view_trend(reels_chronological: list[dict[str, Any]], followers: int) -> str:
    """``rising`` / ``stable`` / ``declining`` sur ``view_count`` — Reels uniquement.

    Médiane des 4 plus récents vs 4 précédents, **uniquement sur les Reels
    avec ``views > 0``** (les 0 sont des données manquantes, pas une portée
    nulle). Avec < 8 reels exploitables, fallback ``stable``.
    """
    if followers <= 0:
        return "stable"
    reels_with_views = [
        r for r in reels_chronological if int(r.get("views", 0) or 0) > 0
    ]
    if len(reels_with_views) < 8:
        return "stable"
    last4 = reels_with_views[-4:]
    prev4 = reels_with_views[-8:-4]
    median_last = statistics.median(
        int(r["views"]) / float(followers) for r in last4
    )
    median_prev = statistics.median(
        int(r["views"]) / float(followers) for r in prev4
    )
    if median_prev <= 1e-9:
        return "rising" if median_last > 0 else "stable"
    ratio = median_last / median_prev
    if ratio >= 1.2:
        return "rising"
    if ratio <= 0.8:
        return "declining"
    return "stable"


# Alias rétro-compat (anciens noms — même implémentation)
_ratio_median = _reel_ratio_median
_ratio_trend = _reel_view_trend


# ---------------------------------------------------------------------------
# Métriques Posts (basées sur ``likes`` / ``comments``)
# ---------------------------------------------------------------------------


def _post_ratio_median(posts: list[dict[str, Any]], followers: int) -> float:
    """Médiane(likes / followers) — proxy de portée pour les Posts (pas de views fiable)."""
    if followers <= 0 or not posts:
        return 0.0
    return float(statistics.median(int(p["likes"]) / float(followers) for p in posts))


# ---------------------------------------------------------------------------
# Engagement (commun Reels & Posts)
# ---------------------------------------------------------------------------


def _engagement_median(rows: list[dict[str, Any]], followers: int) -> float:
    """Médiane((likes + comments) / followers). Identique pour Reels et Posts."""
    if followers <= 0 or not rows:
        return 0.0
    eng = [
        (int(r["likes"]) + int(r["comments"])) / float(followers) for r in rows
    ]
    return float(statistics.median(eng))


def _median_datetime(values: list[datetime]) -> datetime:
    """Médiane temporelle (milieu exact si ``n`` pair — ``statistics.median``
    ne sait pas additionner deux ``datetime`` en 3.14+).
    """
    if not values:
        raise ValueError("values must be non-empty")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return s[mid - 1] + (s[mid] - s[mid - 1]) / 2


def _remove_pinned_reels(reels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Heuristique : Reels « épinglés » sur la page Reels (``is_pinned`` souvent ``None``).

    Ils apparaissent en **tête** du feed avec un ``taken_at`` très antérieur aux
    Reels suivants. On ne peut pas s'appuyer sur ``is_pinned`` seul.

    **Entrée** : ordre de récupération ``user_medias`` (indices 0, 1, 2…).

    **Logique** :

    1. Médiane des ``taken_at`` de **tous** les Reels.
    2. Pour les indices 0, 1 et 2 uniquement : si ``taken_at`` est strictement
       antérieur à ``médiane - 30 jours`` → exclure (traité comme épinglé).

    Si un ``taken_at`` manque ou n'est pas un ``datetime``, la liste est
    retournée **inchangée** (fail-open).
    """
    if not reels:
        return []
    if any(not isinstance(r.get("taken_at"), datetime) for r in reels):
        return list(reels)
    dates = [r["taken_at"] for r in reels]
    median_dt = _median_datetime(dates)
    threshold = median_dt - timedelta(days=30)
    drop = {
        i
        for i in range(min(3, len(reels)))
        if reels[i]["taken_at"] < threshold
    }
    return [r for i, r in enumerate(reels) if i not in drop]


def _publish_frequency(rows_chronological: list[dict[str, Any]]) -> float:
    """Médias par jour sur la période couverte.

    ``score_profile`` ne passe que les **Reels** retenus après
    ``_remove_pinned_reels`` (épingles page Reels sans ``is_pinned`` fiable).
    """
    dates = [
        r["taken_at"] for r in rows_chronological
        if isinstance(r.get("taken_at"), datetime)
    ]
    if len(dates) < 2:
        return 0.0
    delta = max((dates[-1] - dates[0]).total_seconds() / 86400.0, 1.0)
    return len(dates) / delta


def _classify_recent_comments(
    client: Any,
    chronological_rows: list[dict[str, Any]],
    *,
    niche: str,
    log: logging.Logger,
    sleep_between_posts: bool = True,
) -> tuple[dict[str, float], str | None]:
    """Classifie les commentaires des derniers médias passés (au plus
    ``COMMENTS_MEDIA_SAMPLE``, ordre chronologique ancien → récent).

    Le caller (``score_profile``) priorise les Reels et complète au besoin
    avec des Posts non-épinglés ; cette fonction se contente de scraper et
    classifier ce qu'on lui passe.

    Retourne ``(distribution, dominant_type)``. La distribution est pondérée
    par ``confidence`` retourné par le classifieur, normalisée pour sommer à 1.
    Si aucun post n'est classifiable, ``({}, None)`` est retourné.
    """
    from modules.classifier import (  # import local pour ne pas alourdir start
        ClassificationError,
        CommentClassifier,
    )

    sampled = list(reversed(chronological_rows[-COMMENTS_MEDIA_SAMPLE:]))
    classifier = CommentClassifier()
    weights: dict[str, float] = {}

    for i, row in enumerate(sampled):
        media_id = row.get("media_id")
        if not media_id:
            continue

        polite_sleep()
        try:
            comments = client.media_comments(str(media_id), amount=COMMENTS_PER_MEDIA)
        except (LoginRequired, PleaseWaitFewMinutes) as e:
            raise DiscoverySessionLost(
                f"session perdue pendant media_comments({media_id}) : {e}"
            ) from e
        except (RateLimitError, ClientThrottledError) as e:
            log.warning(
                "media_comments(%s) : rate limit court (%s) — skip ce post.",
                media_id,
                e,
            )
            continue
        except PrivateError as e:
            log.warning(
                "media_comments(%s) : commentaires inaccessibles (%s) — skip.",
                media_id,
                e,
            )
            continue
        except Exception as e:
            log.warning("media_comments(%s) : erreur (%s) — skip.", media_id, e)
            continue

        # On parse texte + likes en un seul pass pour pouvoir loguer un
        # diagnostic riche (utile quand un profil scoré ressort sans T-type
        # dominant — la 1re question est toujours "y avait-il du signal ?").
        # Matérialisation explicite : ``media_comments`` peut renvoyer un
        # générateur paresseux, on ne veut pas l'épuiser à la 1re passe.
        raw_list = list(comments or [])
        raw_total = len(raw_list)
        parsed: list[tuple[str, int]] = []
        for c in raw_list:
            text = str(getattr(c, "text", "")).strip()
            if not text:
                continue
            try:
                likes = int(getattr(c, "like_count", 0) or 0)
            except (TypeError, ValueError):
                likes = 0
            parsed.append((text, likes))

        max_likes = max((lk for _, lk in parsed), default=0)
        exploitable = len(parsed)

        if exploitable == 0:
            log.info(
                "media %s : %d commentaires bruts, max_likes=%d, exploitables=0 — "
                "aucun commentaire exploitable.",
                media_id, raw_total, max_likes,
            )
            texts: list[str] = []
        elif max_likes == 0:
            # Fallback : aucun commentaire liké → on garde les 5 textes les
            # plus longs (>10 chars) pour donner quand même un signal au
            # classifier. Sur les Reels jeunes ou faibles engagements, c'est
            # fréquent que les premiers commentaires soient à 0 like ; sans
            # ce fallback on perdrait toute info de T-type sur ces profils.
            long_texts = sorted(
                (t for t, _ in parsed if len(t) > 10),
                key=len,
                reverse=True,
            )[:5]
            texts = long_texts
            log.info(
                "media %s : %d commentaires bruts, max_likes=0, exploitables=%d — "
                "fallback sur %d textes les plus longs (len>10).",
                media_id, raw_total, exploitable, len(texts),
            )
        else:
            texts = [t for t, _ in parsed]

        if not texts:
            # Cas dégénéré : exploitable > 0 mais tous textes ≤ 10 chars
            # (uniquement des emojis, "lol", "ok", ...). On loggue et on skip.
            if exploitable > 0:
                log.info(
                    "media %s : %d commentaires bruts, max_likes=%d, "
                    "exploitables=%d (tous ≤ 10 chars après fallback) — skip.",
                    media_id, raw_total, max_likes, exploitable,
                )
        else:
            try:
                result = classifier.classify(texts, niche=niche)
                ttype = str(result.get("type") or "")
                conf = float(result.get("confidence") or 0.0)
                if ttype:
                    weights[ttype] = weights.get(ttype, 0.0) + max(conf, 1e-3)
            except (ValueError, ClassificationError) as e:
                log.warning(
                    "Classification échouée pour media %s (%s) — skip.",
                    media_id,
                    e,
                )

        if sleep_between_posts and i < len(sampled) - 1:
            polite_sleep(
                min_s=DISCOVERY_BETWEEN_POSTS_MIN_S,
                max_s=DISCOVERY_BETWEEN_POSTS_MAX_S,
            )

    if not weights:
        return ({}, None)
    total = sum(weights.values()) or 1.0
    distribution = {k: v / total for k, v in weights.items()}
    dominant = max(distribution.items(), key=lambda kv: kv[1])[0]
    return distribution, dominant


def _t_type_match_score(
    distribution: dict[str, float], targets: list[str] | tuple[str, ...]
) -> float:
    if not distribution or not targets:
        return 0.0
    target_set = {str(t).strip() for t in targets if str(t).strip()}
    return float(sum(p for t, p in distribution.items() if t in target_set))


def _log_norm(value: float | None, *, ref: float, scale: float = 1.0) -> float:
    """Normalisation **logarithmique** : ``log10(value*scale + 1) / log10(ref*scale + 1)``.

    Propriétés voulues :

    - ``value <= 0`` ou ``None`` → ``0.0`` (pas de ``ZeroDivisionError`` ni
      de ``ValueError`` sur ``log10(0)``).
    - ``value == ref`` → exactement ``1.0`` (le poids plein du signal).
    - **Pas de plafond** : ``value > ref`` produit ``> 1.0`` mais avec
      rendement décroissant (compression log).
    - ``ref <= 0`` → ``0.0`` (sécurité défensive — ne devrait pas arriver).

    ``scale`` permet de travailler sur des entiers lisibles pour les ratios
    en pourcentage (ex : engagement 0.068 × 100 = 6.8) — la formule reste
    invariante par changement d'échelle.
    """
    if value is None or value <= 0:
        return 0.0
    if ref is None or ref <= 0:
        return 0.0
    den = math.log10(ref * scale + 1.0)
    if den <= 0:
        return 0.0
    return math.log10(value * scale + 1.0) / den


def _frequency_score(freq_per_day: float) -> float:
    """Normalisation log du rythme de publication, **plafonnée** à
    ``SCORE_RHYTHM_REF`` (= 1 média/jour).

    Au-delà de la référence, on **n'augmente plus** : on ne veut pas
    récompenser le spam de contenu (un compte qui poste 5 Reels/jour n'est
    pas « 5 fois meilleur » qu'un compte qui en poste 1/jour — c'est juste
    un signe de spam ou de gestion par agence).
    """
    if freq_per_day is None or freq_per_day <= 0:
        return 0.0
    capped = min(float(freq_per_day), SCORE_RHYTHM_REF)
    return _log_norm(capped, ref=SCORE_RHYTHM_REF, scale=10.0)


def _compute_reel_score(
    *,
    reel_ratio_median: float,
    reel_ratio_p90: float,
    reel_engagement_median: float,
    reel_trend: str,
    t_type_distribution: dict[str, float],
    publish_frequency: float,
    domain: dict[str, Any],
) -> float:
    """SCORE_REELS — somme pondérée logarithmique. Voir docstring ``score_profile``.

    Σ poids = 925, mais le score peut **dépasser** ce plafond pour les profils
    exceptionnels (ratios très au-dessus de la référence) car les signaux
    log ne sont plus écrêtés.
    """
    ratio_median_norm = _log_norm(reel_ratio_median, ref=SCORE_REEL_RATIO_REF)
    ratio_p90_norm = _log_norm(reel_ratio_p90, ref=SCORE_REEL_RATIO_P90_REF)
    eng_norm = _log_norm(
        reel_engagement_median, ref=SCORE_REEL_ENGAGEMENT_REF, scale=100.0
    )
    trend_norm = {"rising": 1.0, "stable": 0.5, "declining": 0.0}.get(
        reel_trend, 0.5
    )
    t_match = _t_type_match_score(
        t_type_distribution, domain.get("t_types_target") or []
    )
    freq_norm = _frequency_score(publish_frequency)
    return float(
        ratio_median_norm * SCORE_REEL_RATIO_W
        + ratio_p90_norm * SCORE_REEL_RATIO_P90_W
        + eng_norm * SCORE_REEL_ENGAGEMENT_W
        + trend_norm * SCORE_REEL_TREND_W
        + t_match * SCORE_REEL_T_TYPE_W
        + freq_norm * SCORE_REEL_FREQ_W
    )


def _compute_post_score(
    *,
    post_ratio_median: float,
    post_engagement_median: float,
    t_type_distribution: dict[str, float],
    publish_frequency: float,
    domain: dict[str, Any],
) -> float:
    """SCORE_POSTS — somme pondérée logarithmique. Voir docstring ``score_profile``.

    Σ poids = 725, idem que SCORE_REELS le score peut excéder ce plafond pour
    les ratios > référence.
    """
    ratio_norm = _log_norm(
        post_ratio_median, ref=SCORE_POST_RATIO_REF, scale=100.0
    )
    eng_norm = _log_norm(
        post_engagement_median, ref=SCORE_POST_ENGAGEMENT_REF, scale=100.0
    )
    t_match = _t_type_match_score(
        t_type_distribution, domain.get("t_types_target") or []
    )
    freq_norm = _frequency_score(publish_frequency)
    return float(
        ratio_norm * SCORE_POST_RATIO_W
        + eng_norm * SCORE_POST_ENGAGEMENT_W
        + t_match * SCORE_POST_T_TYPE_W
        + freq_norm * SCORE_POST_FREQ_W
    )


# Alias rétro-compat
_compute_score = _compute_reel_score


def explain_score(score_result: dict[str, Any]) -> str:
    """Décompose un ``score_result`` en contributions log10 lisibles.

    Format multi-lignes utilisable dans logs / CLI / debug. Pour ``t_match``
    on **déduit** la contribution observée (``score_reels`` − somme des
    autres contributions) parce que la distribution seule ne suffit pas à
    recalculer le score sans connaître les ``t_types_target`` du domaine.
    """
    sr = score_result or {}

    def _f(key: str, default: float = 0.0) -> float:
        try:
            v = sr.get(key)
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    rrm = _f("reel_ratio_median")
    rp90 = _f("reel_ratio_p90")
    rem = _f("reel_engagement_median")
    rt = str(sr.get("reel_trend") or "")
    prm = _f("post_ratio_median")
    pem = _f("post_engagement_median")
    rhy = _f("posting_rhythm")
    score_reels = _f("score_reels")
    score_posts = _f("score_posts")
    reel_w = _f("reel_weight")
    post_w = _f("post_weight")
    score_final = _f("score")

    rrm_pts = _log_norm(rrm, ref=SCORE_REEL_RATIO_REF) * SCORE_REEL_RATIO_W
    rp90_pts = _log_norm(rp90, ref=SCORE_REEL_RATIO_P90_REF) * SCORE_REEL_RATIO_P90_W
    rem_pts = (
        _log_norm(rem, ref=SCORE_REEL_ENGAGEMENT_REF, scale=100.0)
        * SCORE_REEL_ENGAGEMENT_W
    )
    rt_pts = (
        {"rising": 1.0, "stable": 0.5, "declining": 0.0}.get(rt, 0.0)
        * SCORE_REEL_TREND_W
    )
    rhy_reel_pts = _frequency_score(rhy) * SCORE_REEL_FREQ_W
    rhy_post_pts = _frequency_score(rhy) * SCORE_POST_FREQ_W
    prm_pts = (
        _log_norm(prm, ref=SCORE_POST_RATIO_REF, scale=100.0) * SCORE_POST_RATIO_W
    )
    pem_pts = (
        _log_norm(pem, ref=SCORE_POST_ENGAGEMENT_REF, scale=100.0)
        * SCORE_POST_ENGAGEMENT_W
    )

    reel_t_match_pts = max(
        0.0,
        score_reels - (rrm_pts + rp90_pts + rem_pts + rt_pts + rhy_reel_pts),
    )
    post_t_match_pts = max(
        0.0, score_posts - (prm_pts + pem_pts + rhy_post_pts)
    )

    reel_total_w = (
        SCORE_REEL_RATIO_W
        + SCORE_REEL_RATIO_P90_W
        + SCORE_REEL_ENGAGEMENT_W
        + SCORE_REEL_TREND_W
        + SCORE_REEL_T_TYPE_W
        + SCORE_REEL_FREQ_W
    )
    post_total_w = (
        SCORE_POST_RATIO_W
        + SCORE_POST_ENGAGEMENT_W
        + SCORE_POST_T_TYPE_W
        + SCORE_POST_FREQ_W
    )
    bar = "─" * 46

    lines: list[str] = []
    lines.append(
        f"📊 Score @{sr.get('username') or '?'} (domaine={sr.get('domain') or '?'})"
    )
    lines.append("")
    lines.append("─── Reels " + bar[:36])
    lines.append(
        f"reel_ratio_median : {rrm:>6.2f}x  → {rrm_pts:>6.0f}pts / {SCORE_REEL_RATIO_W}"
    )
    lines.append(
        f"reel_ratio_p90    : {rp90:>6.2f}x  → {rp90_pts:>6.0f}pts / {SCORE_REEL_RATIO_P90_W}"
    )
    lines.append(
        f"reel_engagement   : {rem * 100:>5.1f}%   → {rem_pts:>6.0f}pts / {SCORE_REEL_ENGAGEMENT_W}"
    )
    lines.append(
        f"reel_trend        : {(rt or 'n/a'):<9}→ {rt_pts:>6.0f}pts / {SCORE_REEL_TREND_W}"
    )
    lines.append(
        f"reel_t_type       : (déduit) → {reel_t_match_pts:>6.0f}pts / {SCORE_REEL_T_TYPE_W}"
    )
    lines.append(
        f"posting_rhythm    : {rhy:>6.2f}   → {rhy_reel_pts:>6.0f}pts / {SCORE_REEL_FREQ_W}"
    )
    lines.append(bar)
    lines.append(
        f"score_reels                      → {score_reels:>6.0f}pts / {reel_total_w}  (w={reel_w:.2f})"
    )
    lines.append("")
    lines.append("─── Posts " + bar[:36])
    lines.append(
        f"post_ratio_median : {prm * 100:>5.1f}%   → {prm_pts:>6.0f}pts / {SCORE_POST_RATIO_W}"
    )
    lines.append(
        f"post_engagement   : {pem * 100:>5.1f}%   → {pem_pts:>6.0f}pts / {SCORE_POST_ENGAGEMENT_W}"
    )
    lines.append(
        f"post_t_type       : (déduit) → {post_t_match_pts:>6.0f}pts / {SCORE_POST_T_TYPE_W}"
    )
    lines.append(
        f"posting_rhythm    : {rhy:>6.2f}   → {rhy_post_pts:>6.0f}pts / {SCORE_POST_FREQ_W}"
    )
    lines.append(bar)
    lines.append(
        f"score_posts                      → {score_posts:>6.0f}pts / {post_total_w}  (w={post_w:.2f})"
    )
    lines.append("")
    lines.append(
        f"score_final = {reel_w:.2f}×{score_reels:.0f} + {post_w:.2f}×{score_posts:.0f} = {score_final:.0f}"
    )
    return "\n".join(lines)


def score_profile(
    username: str,
    domain: dict[str, Any],
    *,
    blacklist: dict[str, Any] | None = None,
    client: Any | None = None,
) -> dict[str, Any] | None:
    """Score un profil candidat sur son **historique** (Layer 0).

    Pipeline :

    1. ``client = get_client()`` (sauf si fourni — utile pour les tests).
    2. ``user_info`` : lit ``follower_count``, ``media_count``, ``biography``,
       ``is_private``.
    3. **Filtres d'éligibilité** (retourne ``None`` immédiatement) :
        - followers < 1 000 ou > 1 000 000
        - media_count < 2
        - compte privé
        - profil déjà dans ``blacklist.json``
    4. ``user_medias(amount=15)`` puis exclusion des médias **épinglés**
       (``is_pinned is True``). Le champ ``media_sampled`` du résultat compte
       les médias **retournés par l'API avant** ce filtre.
       **Segmentation** par ``product_type`` (ordre ``user_medias`` conservé) :
        - ``reels`` = ``product_type == "clips"``
        - ``posts`` = tout le reste (carousel, photo, IGTV…)
       Si ``len(reels) + len(posts) < MIN_TOTAL_MEDIAS_REQUIRED`` (3), on
       retourne ``None`` (historique trop maigre).
       **Reels page Reels** : ``_remove_pinned_reels`` retire en tête de liste
       les clips trop anciens vs la médiane des dates (épingles sans
       ``is_pinned`` fiable). Puis tri chronologique **ancien → récent** pour
       le reste du pipeline.
       **Cap Posts** : tri chronologique puis les ``MAX_POSTS_FOR_SCORING`` (4)
       posts non-épinglés les plus récents (les anciens posts vivent dans
       des conditions d'audience / d'algo obsolètes).
    5. **Complément des vues Reels** : ``user_medias`` retourne ``view_count =
       play_count = 0`` sur les Reels (confirmé en debug). Pour **chaque** Reel
       à ``views == 0``, on appelle ``media_info(pk)`` (du plus récent au plus
       ancien) puis ``polite_sleep()`` après chaque appel. Les vues sont
       ``play_count or view_count or 0`` sur la réponse détaillée. Cette étape
       précède toute métrique Reels (médiane, P90, trend) qui dépend de
       ``views``.
    6. Métriques par type :

       **Reels** (si ≥ 1) :
        - ``reel_ratio_median`` = médiane(``views / followers``) — stabilité
        - ``reel_ratio_p90`` = 90e percentile (``views / followers``) — capte
          le **potentiel viral** (sur ≥ 3 Reels avec vues, sinon 0)
        - ``reel_engagement_median`` = médiane((likes+comments)/followers)
        - ``reel_trend`` (``rising``/``stable``/``declining``) sur ``views``

       **Posts** (si ≥ 1, max 4) :
        - ``post_ratio_median`` = médiane(``likes / followers``)
        - ``post_engagement_median`` = médiane((likes+comments)/followers)

       **Commun** :
        - ``posting_rhythm`` = Reels/jour sur les **Reels** retenus après
          ``_remove_pinned_reels`` (les Posts photos peuvent dater de plusieurs
          mois et fausseraient la fenêtre temporelle)
    7. Classification T-type — on **priorise les Reels** (signal le plus pur
       sur la consommation actuelle de la communauté) :
        - si ``reels_count >= 3`` : 3 Reels les plus récents uniquement
        - sinon : on complète avec des Posts non-épinglés jusqu'à 3 médias
       8–20 s de pause aléatoire entre chaque média (anti-détection).
    8. SCORE_PROFIL = ``score_reels × reel_weight + score_posts × post_weight``
       (Option C — voir docstring `_compute_*_score`). Les poids sont calculés
       sur ce qui a *réellement* été scoré : ``reel_weight = reels_count /
       (reels_count + posts_count)`` après cap des posts.

       **Normalisation log10** : chaque signal continu (ratio, engagement,
       rhythm) utilise ``log10(x*scale + 1) / log10(ref*scale + 1)`` — pas de
       plafond linéaire arbitraire ; au-delà de la référence le signal
       continue à scorer mais avec rendement décroissant. Les Σ poids ci-
       dessous sont donc des **valeurs nominales atteintes à la référence** ;
       les profils exceptionnels peuvent les dépasser.

       **SCORE_REELS** (poids nominal 925) :
        - ``reel_ratio_median`` × 250 (ref 10×) — stabilité
        - ``reel_ratio_p90``    × 100 (ref 50×) — potentiel viral
        - ``reel_engagement_median`` × 200 (ref 10 %)
        - ``reel_trend`` × 150 (rising=1.0 / stable=0.5 / declining=0.0)
        - ``t_type_match`` × 200
        - ``posting_rhythm`` × 25 (ref 1/jour, **plafonné** anti-spam)

       **SCORE_POSTS** (poids nominal 725) :
        - ``post_ratio_median`` × 300 (ref 15 %)
        - ``post_engagement_median`` × 250 (ref 15 %)
        - ``t_type_match`` × 150
        - ``posting_rhythm`` × 25 (ref 1/jour, **plafonné** anti-spam)

    Returns
    -------
    dict | None
        ``None`` si le profil est filtré (éligibilité, compte privé,
        blacklist, erreur réseau, compte introuvable…). Sinon un dict avec
        ``username``, ``domain``, ``followers``, ``score``,
        ``t_type_dominant``, ``t_type_distribution``, ``biography``,
        ``scored_at``, plus les détails par type : ``score_reels``,
        ``score_posts``, ``reel_weight``, ``post_weight``,
        ``reel_ratio_median``, ``reel_ratio_p90``, ``reel_engagement_median``,
        ``reel_trend``,
        ``post_ratio_median``, ``post_engagement_median``,
        ``posting_rhythm``, ``media_sampled`` (taille brute renvoyée par
        ``user_medias``, avant exclusion épinglés), ``reels_count``,
        ``posts_count`` (après cap à 4).

    Raises
    ------
    DiscoverySessionLost
        Session Instagram perdue (``LoginRequired`` / ``PleaseWaitFewMinutes``).
        Le caller (boucle Discovery) doit appeler
        ``recover_from_session_loss`` puis re-tenter.
    """
    setup_watcher_logger()
    log = logging.getLogger("aitertainment.discovery")
    if log.level == logging.NOTSET:
        log.setLevel(logging.INFO)

    u = (username or "").lstrip("@").strip()
    if not u:
        log.warning("score_profile : username vide")
        return None
    if not isinstance(domain, dict) or not domain.get("name"):
        log.warning("score_profile @%s : domain invalide (%r)", u, domain)
        return None

    if blacklist is None:
        try:
            blacklist = load_blacklist()
        except DiscoveryIOError as e:
            log.warning("score_profile @%s : blacklist illisible (%s) — vide.", u, e)
            blacklist = {"profiles": []}

    if _is_blacklisted(u, blacklist):
        log.info("score_profile @%s : déjà en blacklist — skip.", u)
        return None

    if client is None:
        try:
            client = get_client()
        except InstagramAuthError as e:
            log.warning("score_profile @%s : connexion Instagram impossible (%s)", u, e)
            return None

    # 2. user_info ---------------------------------------------------------
    polite_sleep()
    try:
        user_id = client.user_id_from_username(u)
    except UserNotFound as e:
        log.info("score_profile @%s : compte introuvable (%s)", u, e)
        return None
    except (LoginRequired, PleaseWaitFewMinutes) as e:
        raise DiscoverySessionLost(f"user_id_from_username @{u}: {e}") from e
    except (RateLimitError, ClientThrottledError) as e:
        log.warning("score_profile @%s : rate limit user_id (%s) — abandon.", u, e)
        return None
    except Exception as e:
        log.warning("score_profile @%s : erreur user_id (%s) — abandon.", u, e)
        return None

    polite_sleep()
    try:
        info = client.user_info(str(user_id))
    except (LoginRequired, PleaseWaitFewMinutes) as e:
        raise DiscoverySessionLost(f"user_info @{u}: {e}") from e
    except (RateLimitError, ClientThrottledError) as e:
        log.warning("score_profile @%s : rate limit user_info (%s) — abandon.", u, e)
        return None
    except (PrivateAccount, PrivateError):
        log.info("score_profile @%s : profil non lisible (privé).", u)
        return None
    except Exception as e:
        log.warning("score_profile @%s : erreur user_info (%s) — abandon.", u, e)
        return None

    follower_count = int(getattr(info, "follower_count", 0) or 0)
    media_count = int(getattr(info, "media_count", 0) or 0)
    biography = str(getattr(info, "biography", "") or "")
    is_private = bool(getattr(info, "is_private", False))

    # 3. Filtres d'éligibilité --------------------------------------------
    if is_private:
        log.info("score_profile @%s : compte privé — skip.", u)
        return None
    if follower_count < MIN_FOLLOWERS or follower_count > MAX_FOLLOWERS:
        log.info(
            "score_profile @%s : followers=%d hors fourchette [%d, %d] — skip.",
            u,
            follower_count,
            MIN_FOLLOWERS,
            MAX_FOLLOWERS,
        )
        return None
    if media_count < MIN_MEDIA_COUNT:
        log.info(
            "score_profile @%s : media_count=%d < %d — skip.",
            u,
            media_count,
            MIN_MEDIA_COUNT,
        )
        return None

    # 4. Historique --------------------------------------------------------
    polite_sleep()
    try:
        medias = client.user_medias(str(user_id), amount=HISTORY_MEDIAS_TO_FETCH)
    except (LoginRequired, PleaseWaitFewMinutes) as e:
        raise DiscoverySessionLost(f"user_medias @{u}: {e}") from e
    except (RateLimitError, ClientThrottledError) as e:
        log.warning("score_profile @%s : rate limit user_medias (%s) — abandon.", u, e)
        return None
    except PrivateError as e:
        log.info("score_profile @%s : médias inaccessibles (%s) — skip.", u, e)
        return None
    except Exception as e:
        log.warning("score_profile @%s : erreur user_medias (%s) — abandon.", u, e)
        return None

    if not medias:
        log.info("score_profile @%s : aucun media retourné — skip.", u)
        return None

    # Total brut API (inchangé par le filtre épinglés) → ``media_sampled`` en sortie.
    media_sampled_raw = len(medias)
    medias = [m for m in medias if not _is_media_pinned(m)]
    if not medias:
        log.info(
            "score_profile @%s : tous les médias sont épinglés (%d) — skip.",
            u,
            media_sampled_raw,
        )
        return None

    rows = [_media_metric_row(m) for m in medias]
    rows = [r for r in rows if isinstance(r.get("taken_at"), datetime)]
    if not rows:
        log.info("score_profile @%s : aucun media exploitable — skip.", u)
        return None

    # Segmentation Reels / Posts — ordre API préservé pour ``_remove_pinned_reels``.
    total_medias = len(rows)
    reels = [r for r in rows if str(r.get("product_type") or "") == REEL_PRODUCT_TYPE]
    posts = [r for r in rows if str(r.get("product_type") or "") != REEL_PRODUCT_TYPE]

    if total_medias < MIN_TOTAL_MEDIAS_REQUIRED:
        log.info(
            "score_profile @%s : %d média(s) au total (< %d) — historique trop maigre, skip.",
            u,
            total_medias,
            MIN_TOTAL_MEDIAS_REQUIRED,
        )
        return None

    reels = _remove_pinned_reels(reels)
    reels.sort(key=lambda r: r["taken_at"])
    posts.sort(key=lambda r: r["taken_at"])

    # Cap les Posts aux 4 plus récents (non-épinglés). Les anciens posts
    # tirent les médianes vers des conditions d'audience / d'algo obsolètes.
    posts = posts[-MAX_POSTS_FOR_SCORING:]
    total_for_scoring = len(reels) + len(posts)

    # ``_top_up_*`` doit s'exécuter AVANT toute métrique Reels (ratio_median,
    # ratio_p90, view_trend) qui dépend du champ ``views``.
    _top_up_reel_views_via_media_info(client, reels, log=log, username=u)

    # 6. Métriques agrégées -----------------------------------------------
    # ``posting_rhythm`` sur **Reels uniquement** (les Posts photos peuvent
    # dater de plusieurs mois et fausseraient la fenêtre temporelle).
    reels_chronological = sorted(reels, key=lambda r: r["taken_at"])
    posting_rhythm = _publish_frequency(reels_chronological)

    # Reels (sinon valeurs neutres : 0 / "stable")
    reel_ratio_med = _reel_ratio_median(reels, follower_count) if reels else 0.0
    reel_ratio_p90 = _reel_ratio_p90(reels, follower_count) if reels else 0.0
    reel_eng_med = _engagement_median(reels, follower_count) if reels else 0.0
    reel_trend = (
        _reel_view_trend(reels_chronological, follower_count) if reels else "stable"
    )

    # Posts (sinon valeurs neutres)
    post_ratio_med = _post_ratio_median(posts, follower_count) if posts else 0.0
    post_eng_med = _engagement_median(posts, follower_count) if posts else 0.0

    # 7. Classification T-type — on priorise les Reels (signal le plus pur sur
    # la consommation actuelle), et on complète avec des Posts non-épinglés
    # uniquement si on a moins de 3 Reels.
    if len(reels) >= COMMENTS_MEDIA_SAMPLE:
        media_for_classification = reels[-COMMENTS_MEDIA_SAMPLE:]
    else:
        needed = COMMENTS_MEDIA_SAMPLE - len(reels)
        media_for_classification = reels + posts[-needed:]
    niche = str(domain.get("niche") or domain.get("name") or "")
    distribution, dominant = _classify_recent_comments(
        client,
        media_for_classification,
        niche=niche,
        log=log,
    )

    # 8. SCORE_PROFIL unifié pondéré --------------------------------------
    score_reels = (
        _compute_reel_score(
            reel_ratio_median=reel_ratio_med,
            reel_ratio_p90=reel_ratio_p90,
            reel_engagement_median=reel_eng_med,
            reel_trend=reel_trend,
            t_type_distribution=distribution,
            publish_frequency=posting_rhythm,
            domain=domain,
        )
        if reels
        else 0.0
    )
    score_posts = (
        _compute_post_score(
            post_ratio_median=post_ratio_med,
            post_engagement_median=post_eng_med,
            t_type_distribution=distribution,
            publish_frequency=posting_rhythm,
            domain=domain,
        )
        if posts
        else 0.0
    )
    # Option C : poids basés sur ce qu'on a *réellement* scoré (Reels + Posts
    # capés à 4), pas sur le brut renvoyé par ``user_medias``.
    reel_weight = len(reels) / float(total_for_scoring)
    post_weight = 1.0 - reel_weight
    score_final = score_reels * reel_weight + score_posts * post_weight

    log.info(
        "score_profile @%s : score=%.1f (reels=%d w=%.2f score=%.1f / posts=%d w=%.2f score=%.1f) "
        "t_type=%s followers=%d",
        u,
        score_final,
        len(reels),
        reel_weight,
        score_reels,
        len(posts),
        post_weight,
        score_posts,
        dominant,
        follower_count,
    )

    return {
        "username": u,
        "domain": str(domain.get("name") or ""),
        "platform": "instagram",
        "followers": follower_count,
        "media_count": media_count,
        # Score unifié + détails par type
        "score": float(score_final),
        "score_reels": float(score_reels),
        "score_posts": float(score_posts),
        "reel_weight": float(reel_weight),
        "post_weight": float(post_weight),
        # Métriques Reels (None si aucun reel)
        "reel_ratio_median": float(reel_ratio_med) if reels else None,
        "reel_ratio_p90": float(reel_ratio_p90) if reels else None,
        "reel_engagement_median": float(reel_eng_med) if reels else None,
        "reel_trend": reel_trend if reels else None,
        # Métriques Posts (None si aucun post)
        "post_ratio_median": float(post_ratio_med) if posts else None,
        "post_engagement_median": float(post_eng_med) if posts else None,
        # Communs
        "posting_rhythm": float(posting_rhythm),
        "t_type_dominant": dominant,
        "t_type_distribution": distribution,
        "biography": biography,
        # Compteurs
        "media_sampled": media_sampled_raw,
        "reels_count": len(reels),
        "posts_count": len(posts),
        "reels_sampled": len(reels),  # rétro-compat
        "scored_at": _now_iso(),
    }


# =============================================================================
# Logger dédié Discovery
# =============================================================================


def setup_discovery_logger(
    *, log_path: Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Configure (idempotent) le logger Discovery : ``logs/discovery.log`` + console."""
    log = logging.getLogger("aitertainment.discovery")
    log.setLevel(level)
    if any(getattr(h, "_aitertainment_discovery", False) for h in log.handlers):
        return log

    target = Path(log_path) if log_path else DEFAULT_DISCOVERY_LOG_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(target, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        fh._aitertainment_discovery = True  # type: ignore[attr-defined]
        log.addHandler(fh)
    except OSError:
        pass

    has_console = any(
        type(h) is logging.StreamHandler for h in log.handlers
    )
    if not has_console:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(sh)
    log.propagate = False
    return log


# =============================================================================
# État de session humaine simulée
# =============================================================================


@dataclass
class DiscoverySession:
    """État partagé pendant une session ``run_discovery``.

    - ``profiles_today`` : profils consommés sur le quota MAX_PROFILES_PER_DAY.
    - ``day_key`` : ``YYYY-MM-DD`` local pour reset automatique à minuit.
    - ``activity_started_at`` : début du burst d'activité courant (pour la
      pause obligatoire de 30 min toutes les 2h).
    - ``mock`` : si True, on log les pauses au lieu de les exécuter.
    """

    profiles_today: int = 0
    day_key: str = ""
    activity_started_at: datetime | None = None
    mock: bool = False
    candidates_found: int = 0
    blacklisted_count: int = 0


def _local_now() -> datetime:
    return datetime.now()


def _today_key(now: datetime) -> str:
    return now.strftime("%Y-%m-%d")


def _maybe_reset_day(session: DiscoverySession, now: datetime) -> None:
    k = _today_key(now)
    if session.day_key != k:
        session.profiles_today = 0
        session.day_key = k


def _is_lunch(now: datetime) -> bool:
    return LUNCH_START_HOUR <= now.hour < LUNCH_END_HOUR


def _is_night(now: datetime) -> bool:
    return now.hour >= NIGHT_START_HOUR or now.hour < NIGHT_END_HOUR


def _seconds_until_hour(now: datetime, target_hour: int) -> float:
    target = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return max(1.0, (target - now).total_seconds())


def _wait_for_active_window(
    session: DiscoverySession,
    log: logging.Logger,
    *,
    now_fn=_local_now,
    sleep_fn=time.sleep,
) -> None:
    """Bloque tant qu'on est en nuit (23h–8h) ou en pause déjeuner (12h–14h).

    En mode mock ou si ``config.DISABLE_HUMAN_SCHEDULE`` est vrai, on renvoie
    immédiatement (utile pour tester en dehors des fenêtres actives).
    """
    if getattr(config, "DISABLE_HUMAN_SCHEDULE", False):
        return
    while True:
        now = now_fn()
        if _is_night(now):
            wait_s = _seconds_until_hour(now, NIGHT_END_HOUR)
            log.info(
                "Nuit (%02dh) — pause %.0f min jusqu'à %02dh.",
                now.hour,
                wait_s / 60,
                NIGHT_END_HOUR,
            )
            if session.mock:
                return
            sleep_fn(wait_s)
            continue
        if _is_lunch(now):
            wait_s = _seconds_until_hour(now, LUNCH_END_HOUR)
            log.info(
                "Pause déjeuner (%02dh) — pause %.0f min jusqu'à %02dh.",
                now.hour,
                wait_s / 60,
                LUNCH_END_HOUR,
            )
            if session.mock:
                return
            sleep_fn(wait_s)
            continue
        break


def _maybe_take_burst_break(
    session: DiscoverySession,
    log: logging.Logger,
    *,
    now_fn=_local_now,
    sleep_fn=time.sleep,
) -> None:
    """Pause 30 min toutes les 2h d'activité continue.

    Si ``config.DISABLE_HUMAN_SCHEDULE`` est vrai, no-op.
    """
    if getattr(config, "DISABLE_HUMAN_SCHEDULE", False):
        return
    now = now_fn()
    if session.activity_started_at is None:
        session.activity_started_at = now
        return
    elapsed = (now - session.activity_started_at).total_seconds()
    if elapsed >= ACTIVITY_BURST_S:
        log.info(
            "2h d'activité atteintes — pause obligatoire %.0f min.",
            ACTIVITY_PAUSE_S / 60,
        )
        if not session.mock:
            sleep_fn(ACTIVITY_PAUSE_S)
        session.activity_started_at = now_fn()


# =============================================================================
# Notification Telegram (Bot #2 dédié à Discovery, fallback sur le principal)
# =============================================================================


def _notify_candidate(
    result: dict[str, Any],
    log: logging.Logger,
    *,
    mock: bool = False,
) -> None:
    """Envoie une alerte au Bot Telegram #2 (best-effort).

    Ne lève jamais — un échec réseau est juste loggé en warning.
    """
    if mock:
        log.info(
            "[mock] notif candidate skip — @%s score=%.0f",
            result.get("username"),
            result.get("score") or 0.0,
        )
        return

    token = (config.TELEGRAM_DISCOVERY_TOKEN or "").strip()
    chat_id = (config.TELEGRAM_DISCOVERY_CHAT_ID or "").strip()
    if not token or not chat_id:
        log.info(
            "Bot Discovery non configuré — candidat @%s loggé seulement.",
            result.get("username"),
        )
        return

    distrib = result.get("t_type_distribution") or {}
    distrib_str = ", ".join(
        f"{k}={v:.2f}" for k, v in sorted(distrib.items(), key=lambda kv: -kv[1])
    ) or "n/a"
    bio = (result.get("biography") or "").strip()
    if len(bio) > 160:
        bio = bio[:157] + "..."

    reels_n = int(result.get("reels_count") or 0)
    posts_n = int(result.get("posts_count") or 0)
    reel_w = float(result.get("reel_weight") or 0.0)
    post_w = float(result.get("post_weight") or 0.0)
    reel_ratio = result.get("reel_ratio_median")
    post_ratio = result.get("post_ratio_median")
    reel_trend = result.get("reel_trend")

    metric_lines: list[str] = []
    if reels_n:
        metric_lines.append(
            f"🎞 Reels {reels_n} (w={reel_w:.2f}) · ratio_med={float(reel_ratio or 0):.2f}x"
            + (f" · trend={reel_trend}" if reel_trend else "")
        )
    if posts_n:
        metric_lines.append(
            f"🖼 Posts {posts_n} (w={post_w:.2f}) · "
            f"likes/follow={float(post_ratio or 0):.3f}"
        )
    metrics_block = ("\n" + "\n".join(metric_lines)) if metric_lines else ""

    text = (
        "🔎 *Nouveau candidat Discovery*\n"
        f"👤 @{result.get('username')} ({result.get('domain')})\n"
        f"👥 {result.get('followers')} followers · "
        f"score *{result.get('score', 0):.0f}/1000*"
        f"{metrics_block}\n"
        f"🎭 dominant={result.get('t_type_dominant')} ({distrib_str})\n"
        f"📝 {bio}"
    )
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        if r.status_code != 200:
            log.warning(
                "Telegram Discovery non-200 (%s) pour @%s",
                r.status_code,
                result.get("username"),
            )
    except requests.RequestException as e:
        log.warning("Telegram Discovery erreur (%s) pour @%s", e, result.get("username"))


# =============================================================================
# Persistance incrémentale (blacklist / candidates)
# =============================================================================


def _record_seen(
    blacklist: dict[str, Any],
    username: str,
    *,
    outcome: str,
    domain_name: str,
    score: float | None,
    mock: bool = False,
    blacklist_path: Path | None = None,
) -> None:
    """Ajoute ``username`` à la blacklist (jamais reproposé) puis persiste."""
    u = username.lstrip("@").strip().lower()
    if not u or _is_blacklisted(u, blacklist):
        return
    blacklist.setdefault("profiles", []).append(
        {
            "username": u,
            "platform": "instagram",
            "domain": domain_name,
            "outcome": outcome,
            "score": float(score) if score is not None else None,
            "added_at": _now_iso(),
        }
    )
    if not mock:
        try:
            save_blacklist(blacklist, path=blacklist_path)
        except DiscoveryIOError as e:
            logging.getLogger("aitertainment.discovery").warning(
                "save_blacklist a échoué (%s) — on continue en mémoire.", e
            )


def _record_candidate(
    candidates: dict[str, Any],
    result: dict[str, Any],
    *,
    mock: bool = False,
    candidates_path: Path | None = None,
) -> None:
    """Ajoute ``result`` à candidates.json (idempotent par username) + persiste."""
    target = (result.get("username") or "").lstrip("@").strip().lower()
    if not target:
        return
    arr = candidates.setdefault("candidates", [])
    for c in arr:
        if isinstance(c, dict) and str(c.get("username") or "").lower() == target:
            return  # déjà candidat
    arr.append({**result, "validated": False, "discovered_at": _now_iso()})
    if not mock:
        try:
            save_candidates(candidates, path=candidates_path)
        except DiscoveryIOError as e:
            logging.getLogger("aitertainment.discovery").warning(
                "save_candidates a échoué (%s) — on continue en mémoire.", e
            )


def _watchlist_usernames(
    watchlist: list[dict[str, Any]] | None,
) -> set[str]:
    if not watchlist:
        return set()
    out: set[str] = set()
    for c in watchlist:
        if not isinstance(c, dict):
            continue
        u = str(c.get("username") or "").lstrip("@").strip().lower()
        if u:
            out.add(u)
    return out


# =============================================================================
# Exploration réseau
# =============================================================================


def _seed_followings(
    client: Any,
    seed: str,
    *,
    log: logging.Logger,
) -> list[str]:
    """Retourne les usernames suivis par ``seed`` (max ``FOLLOWING_FETCH_AMOUNT``).

    Propagation des erreurs :
    - ``DiscoverySessionLost`` pour ``LoginRequired`` / ``PleaseWaitFewMinutes``.
    - Les autres erreurs sont loggées et renvoient ``[]`` (on passe au seed suivant).
    """
    s = (seed or "").lstrip("@").strip()
    if not s:
        return []
    polite_sleep()
    try:
        seed_id = client.user_id_from_username(s)
    except UserNotFound:
        log.warning("Seed @%s introuvable — skip.", s)
        return []
    except (LoginRequired, PleaseWaitFewMinutes) as e:
        raise DiscoverySessionLost(f"user_id_from_username @{s}: {e}") from e
    except (RateLimitError, ClientThrottledError) as e:
        log.warning("Seed @%s : rate limit user_id (%s) — skip.", s, e)
        return []
    except Exception as e:
        log.warning("Seed @%s : erreur user_id (%s) — skip.", s, e)
        return []

    polite_sleep()
    try:
        following = client.user_following(str(seed_id), amount=FOLLOWING_FETCH_AMOUNT)
    except (LoginRequired, PleaseWaitFewMinutes) as e:
        raise DiscoverySessionLost(f"user_following @{s}: {e}") from e
    except (RateLimitError, ClientThrottledError) as e:
        log.warning("Seed @%s : rate limit user_following (%s) — skip.", s, e)
        return []
    except (PrivateAccount, PrivateError):
        log.info("Seed @%s : following non lisible (privé) — skip.", s)
        return []
    except Exception as e:
        log.warning("Seed @%s : erreur user_following (%s) — skip.", s, e)
        return []

    out: list[str] = []
    if isinstance(following, dict):
        for _, ushort in following.items():
            uname = str(getattr(ushort, "username", "") or "").strip()
            if uname:
                out.append(uname)
    else:
        # Compat : certains adaptateurs renvoient une liste
        for ushort in following or []:
            uname = str(getattr(ushort, "username", "") or "").strip()
            if uname:
                out.append(uname)
    log.info("Seed @%s : %d followings collectés.", s, len(out))
    return out


def explore_network(
    domain: dict[str, Any],
    *,
    blacklist: dict[str, Any] | None = None,
    watchlist: list[dict[str, Any]] | None = None,
    candidates: dict[str, Any] | None = None,
    db: dict[str, Any] | None = None,
    client: Any | None = None,
    session: DiscoverySession | None = None,
    blacklist_path: Path | None = None,
    candidates_path: Path | None = None,
    db_path: Path | None = None,
    seeds_override: list[str] | None = None,
) -> None:
    """Explore le réseau depuis les seeds d'un domaine et alimente la file
    de candidats (cf. brief Layer 0).

    Pour chaque seed :
        1. ``client.user_following(user_id, amount=50)``
        2. Filtre déjà-vus (blacklist + watchlist).
        3. Pour chaque candidat survivant :
            - ``score_profile(username, domain)``
            - Si ``score > 500`` → ``candidates.json`` + notif Telegram
            - **Toujours** ``blacklist.json`` (jamais reproposé).
            - ``polite_sleep(45–180s)`` entre chaque profil.

    Limites humaines :
        - ``MAX_PROFILES_PER_DAY = 40`` (compteur dans ``session``).
        - Inactif 23h–8h, pause obligatoire 12h–14h.
        - Pause 30 min toutes les 2h d'activité.

    Erreurs :
        - ``DiscoverySessionLost`` → recovery déléguée au caller (run_discovery).
    """
    log = setup_discovery_logger()
    if not isinstance(domain, dict) or not domain.get("name"):
        log.warning("explore_network : domain invalide (%r) — skip.", domain)
        return

    domain_name = str(domain["name"])
    session = session or DiscoverySession(mock=False)

    if blacklist is None:
        try:
            blacklist = load_blacklist()
        except DiscoveryIOError as e:
            log.warning("blacklist illisible (%s) — repart vide.", e)
            blacklist = {"profiles": []}
    if candidates is None:
        try:
            candidates = load_candidates()
        except DiscoveryIOError as e:
            log.warning("candidates illisibles (%s) — repart vide.", e)
            candidates = {"candidates": []}
    if db is None:
        try:
            db = load_db(path=db_path)
        except DatabaseIOError as e:
            log.warning("database illisible (%s) — repart vide.", e)
            db = {"profiles": {}}

    seeds_input = seeds_override if seeds_override is not None else (
        domain.get("seeds") or []
    )
    seeds: list[str] = [
        str(s).lstrip("@").strip()
        for s in seeds_input
        if str(s or "").strip()
    ]
    if not seeds:
        log.info("Domaine %s : aucun seed — skip.", domain_name)
        return

    if not session.mock and client is None:
        try:
            client = get_client()
        except InstagramAuthError as e:
            log.error("Connexion Instagram impossible (%s) — abandon domaine.", e)
            return

    watchlist_set = _watchlist_usernames(watchlist)
    log.info(
        "Domaine %s : %d seeds, watchlist=%d, blacklist=%d.",
        domain_name,
        len(seeds),
        len(watchlist_set),
        len(blacklist.get("profiles", [])),
    )

    for seed in seeds:
        # Quota global avant chaque seed
        _maybe_reset_day(session, _local_now())
        if session.profiles_today >= MAX_PROFILES_PER_DAY:
            log.info(
                "Quota quotidien atteint (%d) — arrêt domaine %s.",
                MAX_PROFILES_PER_DAY,
                domain_name,
            )
            return

        if session.mock:
            following = _mock_followings_for(seed)
        else:
            following = _seed_followings(client, seed, log=log)

        for username in following:
            uname = username.lstrip("@").strip().lower()
            if not uname or uname == seed.lower():
                continue

            _maybe_reset_day(session, _local_now())
            if session.profiles_today >= MAX_PROFILES_PER_DAY:
                log.info(
                    "Quota quotidien atteint (%d) — arrêt seed %s.",
                    MAX_PROFILES_PER_DAY,
                    seed,
                )
                return

            _wait_for_active_window(session, log)
            _maybe_take_burst_break(session, log)

            if uname in watchlist_set:
                log.info("@%s déjà dans watchlist — skip.", uname)
                continue
            if _is_blacklisted(uname, blacklist):
                continue

            log.info("Scoring @%s (depuis seed @%s, domaine %s)...", uname, seed, domain_name)

            try:
                if session.mock:
                    result = _mock_score_profile(uname, domain)
                else:
                    result = score_profile(
                        uname, domain, blacklist=blacklist, client=client
                    )
            except DiscoverySessionLost:
                # Stop net : le caller (run_discovery) appellera recovery.
                raise

            session.profiles_today += 1
            score_val = (result or {}).get("score")
            score_passes = (
                result is not None
                and float(result.get("score") or 0.0) > CANDIDATE_SCORE_THRESHOLD
            )

            # Persistance database : **tout** scoring réussi est upserté
            # (contrairement à la blacklist / candidates qui sont conditionnels).
            if result is not None:
                try:
                    upsert_profile(db, result, added_via="discovery")
                    if not session.mock:
                        save_db(db, path=db_path)
                except DatabaseIOError as e:
                    log.warning(
                        "upsert_profile @%s a échoué (%s) — on continue.",
                        uname,
                        e,
                    )

            if score_passes:
                _record_candidate(
                    candidates,
                    result,  # type: ignore[arg-type]
                    mock=session.mock,
                    candidates_path=candidates_path,
                )
                _notify_candidate(result, log, mock=session.mock)  # type: ignore[arg-type]
                session.candidates_found += 1
                outcome = "candidate"
            else:
                outcome = "rejected" if result is not None else "ineligible"

            _record_seen(
                blacklist,
                uname,
                outcome=outcome,
                domain_name=domain_name,
                score=score_val,
                mock=session.mock,
                blacklist_path=blacklist_path,
            )
            session.blacklisted_count += 1

            if not session.mock:
                polite_sleep(
                    min_s=DISCOVERY_BETWEEN_PROFILES_MIN_S,
                    max_s=DISCOVERY_BETWEEN_PROFILES_MAX_S,
                )


# =============================================================================
# Helpers mock (mode --mock)
# =============================================================================


def _mock_followings_for(seed: str) -> list[str]:
    base = seed.lower().replace("@", "")
    return [f"{base}_follow_{i}" for i in range(3)]


def _mock_score_profile(username: str, domain: dict[str, Any]) -> dict[str, Any] | None:
    """Score factice déterministe (basé sur le hash) pour les tests CLI."""
    h = abs(hash((username, domain.get("name")))) % 1000
    if h < 200:
        return None  # ineligible simulé
    score = 200.0 + (h % 700)  # entre 200 et 899
    targets = list(domain.get("t_types_target") or ["T2"])
    dom_t = targets[0] if targets else "T2"
    # Distribution Reels/Posts factice (entre 100% reels et 50/50)
    reels_n = 6 + (h % 7)             # 6..12 reels
    posts_n = 12 - reels_n if reels_n < 12 else 0
    total = reels_n + posts_n or 1
    reel_w = reels_n / total
    post_w = 1.0 - reel_w
    return {
        "username": username,
        "domain": domain.get("name"),
        "platform": "instagram",
        "followers": 5_000 + (h * 17) % 50_000,
        "score": float(score),
        "score_reels": float(score) if reels_n else 0.0,
        "score_posts": float(score * 0.7) if posts_n else 0.0,
        "reel_weight": reel_w,
        "post_weight": post_w,
        "reel_ratio_median": (h % 50) / 10.0 if reels_n else None,
        "reel_ratio_p90": (h % 50) / 10.0 * 3.0 if reels_n else None,
        "reel_engagement_median": 0.04 if reels_n else None,
        "reel_trend": ["rising", "stable", "declining"][h % 3] if reels_n else None,
        "post_ratio_median": 0.05 if posts_n else None,
        "post_engagement_median": 0.04 if posts_n else None,
        "posting_rhythm": 0.7,
        "t_type_dominant": dom_t,
        "t_type_distribution": {dom_t: 1.0},
        "biography": f"mock bio {username}",
        "media_sampled": total,
        "reels_count": reels_n,
        "posts_count": posts_n,
        "reels_sampled": reels_n,
        "scored_at": _now_iso(),
    }


# =============================================================================
# run_discovery + CLI
# =============================================================================


def run_discovery(
    *,
    only_domain: str | None = None,
    only_seed: str | None = None,
    mock: bool = False,
    seeds_path: Path | None = None,
    blacklist_path: Path | None = None,
    candidates_path: Path | None = None,
    db_path: Path | None = None,
    sleep_fn=time.sleep,
) -> DiscoverySession:
    """Boucle principale Discovery.

    - Sans option : itère sur tous les domaines de ``seeds.json``,
      avec un sleep aléatoire 1h–3h entre chaque domaine.
    - ``only_domain`` : restreint à ce domaine.
    - ``only_seed`` : explore depuis un seed unique. Si le seed est déjà
      référencé dans un domaine, ce domaine est utilisé ; sinon on prend le
      premier domaine de ``seeds.json`` et on remplace ses ``seeds``.
    - ``mock=True`` : aucun appel Instagram, aucune écriture disque,
      aucune notif Telegram, aucun sleep réel.

    Retourne la ``DiscoverySession`` finale (utile pour les tests / CLI).
    """
    log = setup_discovery_logger()
    log.info(
        "=== Discovery démarrée (mock=%s, only_domain=%s, only_seed=%s) ===",
        mock,
        only_domain,
        only_seed,
    )

    seeds_data = load_seeds(path=seeds_path)
    domains = list(seeds_data.get("domains") or [])
    if not domains:
        log.warning("Aucun domaine dans seeds.json — rien à faire.")
        return DiscoverySession(mock=mock)

    if only_domain:
        domains = [d for d in domains if (d.get("name") or "") == only_domain]
        if not domains:
            log.warning("Domaine '%s' introuvable dans seeds.json.", only_domain)
            return DiscoverySession(mock=mock)

    if only_seed:
        seed_clean = only_seed.lstrip("@").strip().lower()
        chosen: dict[str, Any] | None = None
        for d in domains:
            for s in d.get("seeds") or []:
                if str(s).lstrip("@").strip().lower() == seed_clean:
                    chosen = d
                    break
            if chosen:
                break
        if chosen is None:
            chosen = domains[0]
            log.info(
                "Seed @%s pas dans seeds.json — rattaché au domaine '%s' (premier).",
                seed_clean,
                chosen.get("name"),
            )
        domains = [{**chosen, "_seeds_override": [seed_clean]}]

    blacklist = load_blacklist(path=blacklist_path)
    candidates = load_candidates(path=candidates_path)
    try:
        db = load_db(path=db_path)
    except DatabaseIOError as e:
        log.warning("database illisible (%s) — repart vide.", e)
        db = {"profiles": {}}
    try:
        from watcher import load_watchlist  # import tardif (évite cycle)
        watchlist = load_watchlist()
    except Exception as e:
        log.warning("watchlist illisible (%s) — exploration sans filtrage watchlist.", e)
        watchlist = []

    session = DiscoverySession(mock=mock, day_key=_today_key(_local_now()))
    client = None
    if not mock:
        try:
            client = get_client()
        except InstagramAuthError as e:
            log.error("Connexion Instagram impossible (%s) — abandon Discovery.", e)
            return session

    try:
        for i, domain in enumerate(domains):
            seeds_override = domain.get("_seeds_override")
            try:
                explore_network(
                    domain,
                    blacklist=blacklist,
                    watchlist=watchlist,
                    candidates=candidates,
                    db=db,
                    client=client,
                    session=session,
                    blacklist_path=blacklist_path,
                    candidates_path=candidates_path,
                    db_path=db_path,
                    seeds_override=seeds_override,
                )
            except DiscoverySessionLost as e:
                log.warning(
                    "Session perdue dans domaine %s (%s) — recovery.",
                    domain.get("name"),
                    e,
                )
                if mock:
                    raise
                try:
                    client = recover_from_session_loss()
                except WatcherStopRequested:
                    log.error(
                        "Recovery Instagram échouée — arrêt Discovery proprement."
                    )
                    return session

            if i < len(domains) - 1:
                wait_s = random.uniform(INTER_DOMAIN_MIN_S, INTER_DOMAIN_MAX_S)
                log.info(
                    "Domaine %s terminé — pause %.0f min avant le suivant.",
                    domain.get("name"),
                    wait_s / 60,
                )
                if not mock:
                    sleep_fn(wait_s)
    except KeyboardInterrupt:
        log.info("Interruption clavier — arrêt Discovery.")
    finally:
        log.info(
            "=== Discovery terminée (profiles_today=%d, candidates=%d, blacklisted=%d) ===",
            session.profiles_today,
            session.candidates_found,
            session.blacklisted_count,
        )
    return session


def score_and_persist(
    username: str,
    *,
    domain: dict[str, Any] | None = None,
    added_via: str = "manual",
    notify_threshold: float = DISCOVERY_NOTIFY_THRESHOLD,
    blacklist: dict[str, Any] | None = None,
    client: Any | None = None,
    seeds_path: Path | None = None,
    db_path: Path | None = None,
    candidates_path: Path | None = None,
    mock: bool = False,
    notify_fn: Any | None = None,
) -> dict[str, Any] | None:
    """Score un profil unique, **upserte systématiquement** dans
    ``database.json`` (toutes les exécutions sont historisées) et envoie une
    notif Telegram (Bot #2 Discovery) au-dessus de ``notify_threshold``.

    Pipeline :

    1. Si ``domain`` est ``None`` → premier domaine de ``seeds.json``.
    2. ``score_profile(username, domain)``. Si ``None`` (privé / hors fourchette /
       pas assez d'historique), on retourne ``None`` sans rien écrire.
    3. ``upsert_profile(db, score_result, added_via)`` puis ``save_db``.
    4. Si ``score > notify_threshold`` :
        - ``_record_candidate`` (le bot a besoin du candidat dans
          ``candidates.json`` pour résoudre les boutons inline plus tard) ;
        - ``notify_fn(result)`` si fourni, sinon
          ``telegram_discovery_bot.notify_candidate(result, mock=mock)``.

    Returns
    -------
    dict | None
        ``None`` si le profil est filtré, sinon ::

            {
              "score_result": <résultat score_profile>,
              "profile":      <entrée database.json après upsert>,
              "tier":         "A" | "B" | "C",
              "notified":     bool,
            }
    """
    log = setup_discovery_logger()
    username = (username or "").lstrip("@").strip().lower()
    if not username:
        log.warning("score_and_persist : username vide.")
        return None

    if domain is None:
        seeds = load_seeds(path=seeds_path)
        domains = list(seeds.get("domains") or [])
        if not domains:
            log.warning("score_and_persist : aucun domaine dans seeds.json.")
            return None
        domain = domains[0]
        log.info(
            "score_and_persist @%s : domaine par défaut '%s'.",
            username,
            domain.get("name"),
        )

    if blacklist is None and not mock:
        try:
            blacklist = load_blacklist()
        except DiscoveryIOError:
            blacklist = {"profiles": []}

    if mock:
        result = _mock_score_profile(username, domain)
    else:
        if client is None:
            try:
                client = get_client()
            except InstagramAuthError as e:
                log.error(
                    "score_and_persist @%s : client Instagram KO (%s).", username, e
                )
                return None
        try:
            result = score_profile(
                username, domain, blacklist=blacklist, client=client
            )
        except DiscoverySessionLost as e:
            log.error("score_and_persist @%s : session perdue (%s).", username, e)
            return None

    if result is None:
        log.info("score_and_persist @%s : filtré (None).", username)
        return None

    try:
        db = load_db(path=db_path)
    except DatabaseIOError as e:
        log.warning("database illisible (%s) — repart vide.", e)
        db = {"profiles": {}}

    profile = upsert_profile(db, result, added_via=added_via)
    if not mock:
        try:
            save_db(db, path=db_path)
        except DatabaseIOError as e:
            log.warning("save_db a échoué (%s) — on continue en mémoire.", e)

    notified = False
    score = float(result.get("score") or 0.0)
    if score > float(notify_threshold):
        # Le bot Telegram (callbacks ✅ ❌ ✏️) lit ``candidates.json`` au moment
        # du clic — on y inscrit donc systématiquement le résultat avant la notif.
        try:
            cands = load_candidates(path=candidates_path)
        except DiscoveryIOError:
            cands = {"candidates": []}
        _record_candidate(cands, result, mock=mock, candidates_path=candidates_path)

        if notify_fn is not None:
            try:
                notify_fn(result)
                notified = True
            except Exception as e:
                log.warning("notify_fn a échoué (%s).", e)
        else:
            try:
                from telegram_discovery_bot import (  # tardif : évite cycle
                    notify_candidate as _bot_notify,
                )
                _bot_notify(result, mock=mock)
                notified = not mock
            except Exception as e:
                log.warning("notify_candidate a échoué (%s).", e)

    return {
        "score_result": result,
        "profile": profile,
        "tier": profile.get("tier"),
        "notified": notified,
    }


def _print_score_summary(summary: dict[str, Any] | None, username: str) -> None:
    """Format CLI : ``Score : 448 | Tier : B | T-type : T2 | Notif : ✅``."""
    u = username.lstrip("@").strip().lower()
    if summary is None:
        print(
            f"@{u} : profil filtré (privé, hors fourchette, ou historique insuffisant)."
        )
        return
    res = summary["score_result"]
    score = float(res.get("score") or 0.0)
    notif_icon = "✅" if summary.get("notified") else "—"
    print(
        f"@{res.get('username') or u} | "
        f"Score : {score:.0f} | "
        f"Tier : {summary.get('tier') or '?'} | "
        f"T-type : {res.get('t_type_dominant') or '?'} | "
        f"Notif : {notif_icon}"
    )


def _main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Discovery — exploration réseau Instagram (Layer 0)."
    )
    parser.add_argument(
        "--domain",
        help="Restreindre à un domaine de seeds.json (ex: humour).",
    )
    parser.add_argument(
        "--seed",
        help="Explorer depuis un seed unique (ex: @username).",
    )
    parser.add_argument(
        "--score",
        metavar="USERNAME",
        help=(
            "Score un profil unique (ex: @raikkonenaf). "
            "Upserte database.json et notifie Telegram si score > "
            f"{int(DISCOVERY_NOTIFY_THRESHOLD)}."
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mode test : pas d'Instagram, pas de Telegram, pas d'écritures disque.",
    )
    args = parser.parse_args()

    if args.score:
        summary = score_and_persist(args.score, mock=args.mock)
        _print_score_summary(summary, args.score)
        return

    run_discovery(
        only_domain=args.domain,
        only_seed=args.seed,
        mock=args.mock,
    )


if __name__ == "__main__":
    _main_cli()


__all__ = [
    "ACTIVITY_BURST_S",
    "ACTIVITY_PAUSE_S",
    "CANDIDATE_SCORE_THRESHOLD",
    "DISCOVERY_NOTIFY_THRESHOLD",
    "DEFAULT_BLACKLIST_PATH",
    "DEFAULT_CANDIDATES_PATH",
    "DEFAULT_DATA_DIR",
    "DEFAULT_DISCOVERY_LOG_PATH",
    "DEFAULT_SEEDS_PATH",
    "DISCOVERY_BETWEEN_POSTS_MAX_S",
    "DISCOVERY_BETWEEN_POSTS_MIN_S",
    "DISCOVERY_BETWEEN_PROFILES_MAX_S",
    "DISCOVERY_BETWEEN_PROFILES_MIN_S",
    "DiscoveryIOError",
    "DiscoverySession",
    "DiscoverySessionLost",
    "INTER_DOMAIN_MAX_S",
    "INTER_DOMAIN_MIN_S",
    "INTER_PROFILE_MAX_S",
    "INTER_PROFILE_MIN_S",
    "LUNCH_END_HOUR",
    "LUNCH_START_HOUR",
    "MAX_PROFILES_PER_DAY",
    "NIGHT_END_HOUR",
    "NIGHT_START_HOUR",
    "debug_reel_views",
    "explain_score",
    "explore_network",
    "load_blacklist",
    "load_candidates",
    "load_seeds",
    "run_discovery",
    "save_blacklist",
    "save_candidates",
    "score_and_persist",
    "score_profile",
    "setup_discovery_logger",
]
