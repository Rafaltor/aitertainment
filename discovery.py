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
- **Database** : ne jamais reproposer un profil déjà présent dans
  ``database.json`` (déjà scoré / exploré).
- **Feedback loop** : apprendre des décisions humaines (à brancher plus tard).

Ce module expose uniquement la **persistance** et les chemins par défaut :
chargement / sauvegarde atomique de ``seeds``, ``candidates``, ``database``.
La boucle d'exploration utilise Playwright (``scripts/instagram_browser``).

============================================================================
Fichiers de données (répertoire ``data/``)
============================================================================

- ``data/seeds.json`` — comptes de départ et métadonnées par domaine.
- ``data/database.json`` — profils déjà scorés (plus de re-score automatique).
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

import config
from database import (
    DatabaseIOError,
    load_db,
    save_db,
    upsert_profile,
)
from playwright.sync_api import BrowserContext, sync_playwright
from scripts.instagram_browser import (
    BASE_URL,
    get_browser_context,
    get_profile_data,
    get_recent_reels,
    get_suggested_accounts,
    fetch_reel_comment_likes,
    polite_sleep,
)

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_SEEDS_PATH = DEFAULT_DATA_DIR / "seeds.json"
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
HISTORY_MEDIAS_TO_FETCH = 20          # historique (4 lignes × 5 colonnes sur /reels/)
MIN_TOTAL_MEDIAS_REQUIRED = 3          # nb min de médias TOTAL (reels + posts) pour scorer
MAX_POSTS_FOR_SCORING = 4              # cap sur les Posts scorés (les 4 plus récents non-épinglés)
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

# --- Comment-heat : signal DOMINANT (2026-06) -----------------------------
# Ce qui nous importe vraiment : la section commentaires d'un compte est-elle
# « chaude » ? Si les top commentaires récoltent beaucoup de likes, c'est là
# qu'un commentaire bien placé (notre produit) sera vu et liké. On en fait le
# signal dominant du score, devant les vues/engagement bruts des reels.
#
# Deux lectures, le ratio primant (cf. choix produit) :
#   - ratio  = top_comment_likes / reel_likes  → culture commentaire, taille-agnostique
#   - absolu = top_comment_likes               → garde-fou de visibilité réelle
# Calibration 2026-06 (donnée live @marrant_club : top comment ≈ 0.2 % des likes
# du reel sur un compte 470k). Les ratios top_comment/reel_likes réels sont de
# l'ordre de 0.1-1 %, pas 5 % → ref abaissée à 1 % (= excellente culture
# commentaire). Le ratio mène (poids 750), l'absolu n'est qu'un garde-fou de
# visibilité (poids 175, ref haute pour ne pas dominer les gros comptes).
SCORE_COMMENT_RATIO_REF = 0.01    # 1 % des likes du reel sur le top comment = excellent
SCORE_COMMENT_ABS_REF = 1000.0    # likes absolus sur le top comment (garde-fou doux)
SCORE_COMMENT_RATIO_W = 750       # ratio (primaire, dominant)
SCORE_COMMENT_ABS_W = 175         # absolu (garde-fou)  → Σ comment-heat = 925
# Part de la comment-heat dans le score final (le reste = contenu reels/posts).
COMMENT_HEAT_WEIGHT = 0.65
# Nb de reels visités pour lire les commentaires (coût/risque 429 vs signal).
COMMENT_HEAT_REELS_SAMPLE = 3

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

# Seuil **unifié** entre les deux chemins de scoring (résout l'ancien bug où
# ``explore_network`` enregistrait sans notif les profils 350 < score < 500
# alors que ``--score`` les notifiait correctement) :
#
# - score > seuil → record_candidate + notif Telegram (Bot #2 Discovery).
# - score ≤ seuil → database (jamais reproposé).
#
# Tout score reste upserté dans ``database.json`` indépendamment du seuil —
# seul le ratio candidat/database + la notif sont gouvernés ici.
#
# Source de vérité : ``config.DISCOVERY_NOTIFY_THRESHOLD`` (overridable .env).
DISCOVERY_NOTIFY_THRESHOLD = float(config.DISCOVERY_NOTIFY_THRESHOLD)

# Alias historique : ``_score_one_in_domain`` continuait d'utiliser ce nom
# avec une valeur hardcodée à 500. On l'aligne désormais sur
# ``DISCOVERY_NOTIFY_THRESHOLD`` pour que les deux chemins (CLI ``--score``
# via ``score_and_persist`` et ``explore_network`` via ``_score_one_in_domain``)
# prennent les mêmes décisions notif/database.
CANDIDATE_SCORE_THRESHOLD = DISCOVERY_NOTIFY_THRESHOLD

# Mock score_profile : nombre de reels synthétiques dans le résultat.
TOP_COMMENTS_REELS_MAX = 3

# Suggestions Instagram (``discover/*``) en source principale ; followings du
# seed uniquement en fallback final via ``_fetch_followings``.
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
    from modules.atomic_json import atomic_write_json

    atomic_write_json(path, payload)


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


def save_seeds(seeds_data: dict[str, Any], path: str | Path | None = None) -> None:
    """Écrit ``seeds.json`` de façon atomique.

    Symétrique de ``load_seeds`` : valide la forme racine
    (``{"domains": [...]}``), conserve l'intégralité de l'objet (clés
    additionnelles éventuelles) puis remplace le fichier via tmp+rename.

    Lève ``DiscoveryIOError`` si le payload n'a pas la forme attendue —
    c'est volontaire : on préfère **planter visiblement** plutôt que
    persister un seeds.json corrompu (perte de plusieurs heures de
    curation manuelle).
    """
    if not isinstance(seeds_data, dict):
        raise DiscoveryIOError(
            f"seeds_data doit être un dict, reçu {type(seeds_data).__name__}"
        )
    domains = seeds_data.get("domains")
    if not isinstance(domains, list):
        raise DiscoveryIOError('"domains" doit être une liste non absente')
    p = Path(path) if path else DEFAULT_SEEDS_PATH
    _atomic_write_json(p, seeds_data)
    _LOGGER.info("Seeds sauvegardés : %d domaine(s) -> %s", len(domains), p)


def _remove_seed_from_seeds_data(
    seeds_data: dict[str, Any],
    *,
    domain_name: str,
    seed_username: str,
) -> bool:
    """Retire ``seed_username`` du domaine ``domain_name`` dans ``seeds_data``.

    La comparaison se fait sur le username **normalisé** (``lower()``,
    sans ``@``) via ``_seed_username`` — donc transparent au schéma
    string-vs-dict des seeds. Si plusieurs entrées ciblent le même
    username (cas d'invariant cassé en curation manuelle), toutes sont
    retirées.

    Le domaine est conservé même si sa liste ``seeds`` devient vide
    (cf. brief : ne pas le supprimer).

    Retourne ``True`` si au moins une entrée a été retirée — sinon
    ``False`` (le caller décide s'il faut sauvegarder ou non).
    """
    target = (seed_username or "").lstrip("@").strip().lower()
    if not target:
        return False
    domains = seeds_data.get("domains") or []
    for d in domains:
        if not isinstance(d, dict):
            continue
        if str(d.get("name") or "") != domain_name:
            continue
        original = list(d.get("seeds") or [])
        kept = [s for s in original if _seed_username(s).lower() != target]
        if len(kept) != len(original):
            d["seeds"] = kept
            return True
    return False


# --- database (profils déjà vus) ---


def _database_usernames(db: dict[str, Any] | None) -> set[str]:
    """Usernames déjà présents dans ``database.json`` (clés du dict profiles)."""
    if not db or not isinstance(db.get("profiles"), dict):
        return set()
    return {
        str(k).lstrip("@").strip().lower()
        for k in db["profiles"]
        if str(k).strip()
    }


def _is_in_database(username: str, db: dict[str, Any]) -> bool:
    u = str(username or "").lstrip("@").strip().lower()
    if not u:
        return False
    profiles = db.get("profiles")
    return isinstance(profiles, dict) and u in profiles


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
    """Debug Reels via Playwright (``get_recent_reels``)."""
    u = (username or "").lstrip("@").strip()
    if not u:
        print("debug_reel_views: username vide")
        return

    playwright_instance = sync_playwright().start()
    context = get_browser_context(playwright_instance)
    try:
        reels = get_recent_reels(u, context, max_reels=HISTORY_MEDIAS_TO_FETCH)
        print(f"=== debug_reel_views @{u} (Playwright) ===")
        print(f"reels={len(reels)}")
        print()
        for idx, r in enumerate(reels, start=1):
            print(f"--- Reel #{idx} ---")
            print(f"  media_id: {r.get('media_id')!r}")
            print(f"  view_count: {r.get('view_count')!r}")
            print(f"  like_count: {r.get('like_count')!r}")
            print(f"  thumbnail_url: {(r.get('thumbnail_url') or '')[:80]!r}...")
            print()
    finally:
        context.close()
        playwright_instance.stop()


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


def _top_comment_likes_by_media(
    comments: list[dict[str, Any]],
) -> dict[str, int]:
    """``media_id`` → likes du commentaire **le plus liké** du reel.

    Les commentaires IG affichés par défaut sont déjà classés par engagement,
    donc le max des likes parmi ceux scrapés approxime bien le « top comment ».
    """
    out: dict[str, int] = {}
    for c in comments or []:
        mid = str(c.get("media_id") or "").strip()
        if not mid:
            continue
        likes = int(c.get("comment_likes") or c.get("like_count") or 0)
        if likes > out.get(mid, -1):
            out[mid] = likes
    return out


def _comment_ratio_median(
    top_by_media: dict[str, int], reels: list[dict[str, Any]]
) -> float:
    """Médiane(``top_comment_likes / reel_likes``) sur les reels échantillonnés.

    Ratio taille-agnostique : capte la *culture commentaire* (les gens likent
    les commentaires) indépendamment du nombre d'abonnés. On ignore les reels
    sans likes (données manquantes) pour ne pas fausser la médiane.
    """
    reel_likes = {
        str(r.get("media_id") or ""): int(r.get("likes") or 0) for r in reels
    }
    ratios: list[float] = []
    for mid, top_likes in top_by_media.items():
        rl = reel_likes.get(mid, 0)
        if rl > 0 and top_likes > 0:
            ratios.append(top_likes / float(rl))
    return float(statistics.median(ratios)) if ratios else 0.0


def _comment_likes_median(top_by_media: dict[str, int]) -> float:
    """Médiane des likes absolus du top comment (garde-fou de visibilité)."""
    vals = [v for v in top_by_media.values() if v > 0]
    return float(statistics.median(vals)) if vals else 0.0


def _compute_comment_heat_score(
    *,
    comment_ratio_median: float,
    comment_likes_median: float,
) -> float:
    """SCORE_COMMENT_HEAT — somme pondérée log (Σ nominal = 925).

    Ratio primaire (culture commentaire) + absolu secondaire (visibilité).
    """
    ratio_norm = _log_norm(
        comment_ratio_median, ref=SCORE_COMMENT_RATIO_REF, scale=100.0
    )
    abs_norm = _log_norm(comment_likes_median, ref=SCORE_COMMENT_ABS_REF)
    return float(
        ratio_norm * SCORE_COMMENT_RATIO_W + abs_norm * SCORE_COMMENT_ABS_W
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
    crm = _f("comment_ratio_median")
    clm = _f("comment_likes_median")
    score_comment_heat = _f("score_comment_heat")
    content_score = _f("score_content")
    has_comment_heat = sr.get("score_comment_heat") is not None

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
        f"score_contenu = {reel_w:.2f}×{score_reels:.0f} + {post_w:.2f}×{score_posts:.0f} = {content_score:.0f}"
    )
    lines.append("")
    crm_pts = (
        _log_norm(crm, ref=SCORE_COMMENT_RATIO_REF, scale=100.0)
        * SCORE_COMMENT_RATIO_W
    )
    clm_pts = _log_norm(clm, ref=SCORE_COMMENT_ABS_REF) * SCORE_COMMENT_ABS_W
    lines.append("─── Comment-heat (DOMINANT) " + bar[:18])
    lines.append(
        f"comment_ratio_med : {crm * 100:>5.2f}%  → {crm_pts:>6.0f}pts / {SCORE_COMMENT_RATIO_W}"
    )
    lines.append(
        f"comment_likes_med : {clm:>6.0f}   → {clm_pts:>6.0f}pts / {SCORE_COMMENT_ABS_W}"
    )
    lines.append(bar)
    lines.append(f"score_comment_heat               → {score_comment_heat:>6.0f}pts")
    lines.append("")
    if has_comment_heat:
        lines.append(
            f"score_final = {COMMENT_HEAT_WEIGHT:.2f}×{score_comment_heat:.0f} (heat) + "
            f"{1 - COMMENT_HEAT_WEIGHT:.2f}×{content_score:.0f} (contenu) = {score_final:.0f}"
        )
    else:
        lines.append(
            f"score_final = {content_score:.0f} (contenu seul — pas de donnée commentaire)"
        )
    return "\n".join(lines)


def _playwright_reel_rows(raw_reels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convertit la sortie ``get_recent_reels`` en lignes métriques discovery."""
    if not raw_reels:
        return []
    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    # Ordre feed page Reels (épinglés en tête) — ne pas inverser.
    for i, r in enumerate(raw_reels):
        taken = r.get("taken_at")
        if not isinstance(taken, datetime):
            taken = now - timedelta(days=i)
        rows.append(
            {
                "media_id": str(r.get("media_id") or ""),
                # pk numérique conservé pour l'API commentaires (comment-heat).
                "pk": str(r.get("pk") or ""),
                "views": int(r.get("view_count") or 0),
                "likes": int(r.get("like_count") or 0),
                "comments": int(r.get("comment_count") or 0),
                "shares": r.get("share_count"),
                "taken_at": taken,
                "caption_text": str(r.get("caption_text") or ""),
                "product_type": REEL_PRODUCT_TYPE,
            }
        )
    return rows


def score_profile(
    username: str,
    domain: dict[str, Any],
    *,
    db: dict[str, Any] | None = None,
    context: BrowserContext | None = None,
) -> dict[str, Any] | None:
    """Score un profil candidat sur son **historique** (Layer 0).

    .. note:: Stratégie 2026-06 — **comment-heat dominante**.
       Le score final est pondéré à ``COMMENT_HEAT_WEIGHT`` (~65 %) par la
       « chaleur » de la section commentaires (likes des top commentaires,
       surtout le ratio ``top_comment_likes / reel_likes``), le score contenu
       (reels + posts ci-dessous) ne pesant que le reste. Rationale : un compte
       n'a de valeur pour nous que si commenter y est vu/liké. Repli défensif :
       si aucune donnée commentaire n'a pu être scrapée (rate-limit, etc.), on
       retombe sur le score contenu seul plutôt que de pénaliser le profil.

    Pipeline :

    1. ``get_profile_data(username, context)`` (Playwright + GraphQL).
    2. **Filtres d'éligibilité** sur followers / posts_count / privé.
    3. ``get_recent_reels`` pour vues et likes des Reels récents.
    4. **Filtres d'éligibilité** (retourne ``None`` immédiatement) :
        - followers < 1 000 ou > 1 000 000
        - media_count < 2
        - compte privé
        - profil déjà dans ``database.json``
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
        database, erreur réseau, compte introuvable…). Sinon un dict avec
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

    """
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

    if db is None:
        try:
            db = load_db()
        except DatabaseIOError as e:
            log.warning("score_profile @%s : database illisible (%s) — vide.", u, e)
            db = {"profiles": {}}

    if _is_in_database(u, db):
        log.info("score_profile @%s : déjà en database — skip.", u)
        return None

    if context is None:
        log.warning("score_profile @%s : context Playwright manquant — abandon.", u)
        return None

    profile_data = get_profile_data(u, context)
    if profile_data is None:
        log.info("score_profile @%s : profil introuvable ou privé — skip.", u)
        return None

    follower_count = int(profile_data.get("followers") or 0)
    media_count = int(profile_data.get("posts_count") or 0)
    biography = str(profile_data.get("biography") or "")
    is_private = bool(profile_data.get("is_private", False))

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
            "score_profile @%s : posts_count=%d < %d — skip.",
            u,
            media_count,
            MIN_MEDIA_COUNT,
        )
        return None

    raw_reels = get_recent_reels(u, context, max_reels=HISTORY_MEDIAS_TO_FETCH)
    media_sampled_raw = media_count
    reels = _playwright_reel_rows(raw_reels)
    posts: list[dict[str, Any]] = []

    if not reels:
        log.info("score_profile @%s : aucun Reel récupéré — skip.", u)
        return None

    total_medias = len(reels) + len(posts)
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
    total_for_scoring = len(reels) + len(posts)

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

    # Niches (liste) depuis seeds.json (``domain["niches"]``), fallback
    # ``domain["name"]``. Validée contre ``config.VALID_NICHES`` —
    # ``validate_niches`` garantit ``len(niches) >= 1`` (fallback ``["humour"]``).
    niches = (
        list(domain.get("niches") or [])
        or [str(domain.get("name") or "humour")]
    )
    niches = config.validate_niches(niches)
    log.info(
        "score_profile @%s : pas de scrape commentaires profil "
        "(corpus = fil Reels / viral_comments.json)",
        u,
    )
    distribution: dict[str, float] = {}
    dominant: str | None = None

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
    content_score = score_reels * reel_weight + score_posts * post_weight

    # 8.b COMMENT-HEAT (signal dominant) ----------------------------------
    # On visite quelques reels pour lire les top commentaires et mesurer s'ils
    # récoltent des likes (= section commentaires « chaude », là où un
    # commentaire bien placé est vu/liké). C'est le critère décisif.
    comment_ratio_med = 0.0
    comment_likes_med = 0.0
    comment_heat = 0.0
    comment_reels_sampled = 0
    comment_data_ok = False
    if reels:
        reels_for_comments = sorted(
            reels, key=lambda r: int(r.get("comments") or 0), reverse=True
        )[:COMMENT_HEAT_REELS_SAMPLE]
        try:
            scraped = fetch_reel_comment_likes(
                context,
                u,
                reels_for_comments,
                max_reels=COMMENT_HEAT_REELS_SAMPLE,
                logger=log,
            )
        except Exception as e:  # noqa: BLE001 — best-effort, ne bloque pas le scoring
            log.warning("score_profile @%s : fetch commentaires KO (%s).", u, e)
            scraped = []
        top_by_media = _top_comment_likes_by_media(scraped)
        comment_reels_sampled = len(top_by_media)
        if top_by_media:
            comment_data_ok = True
            comment_ratio_med = _comment_ratio_median(top_by_media, reels)
            comment_likes_med = _comment_likes_median(top_by_media)
            comment_heat = _compute_comment_heat_score(
                comment_ratio_median=comment_ratio_med,
                comment_likes_median=comment_likes_med,
            )

    # 8.c SCORE FINAL : comment-heat dominante, contenu en secondaire.
    # Repli défensif : si AUCUNE donnée commentaire (scrape échoué/vide, p.ex.
    # rate-limit), on ne pénalise pas un bon profil — on retombe sur le score
    # contenu seul plutôt que de l'enterrer pour une raison technique.
    if comment_data_ok:
        score_final = (
            COMMENT_HEAT_WEIGHT * comment_heat
            + (1.0 - COMMENT_HEAT_WEIGHT) * content_score
        )
    else:
        score_final = content_score

    log.info(
        "score_profile @%s : score=%.1f [comment_heat=%.1f (ratio_med=%.3f likes_med=%.0f "
        "reels=%d ok=%s) + contenu=%.1f (reels=%d/posts=%d)] t_type=%s followers=%d",
        u,
        score_final,
        comment_heat,
        comment_ratio_med,
        comment_likes_med,
        comment_reels_sampled,
        comment_data_ok,
        content_score,
        len(reels),
        len(posts),
        dominant,
        follower_count,
    )

    return {
        "username": u,
        "domain": str(domain.get("name") or ""),
        # Schéma 2026-05 : ``niches`` (liste validée contre ``VALID_NICHES``)
        # est désormais l'**unique** source de vérité — le champ string
        # ``niche`` n'est plus produit. Les callers downstream (database,
        # telegram bot) lisent ``niches`` en priorité avec
        # un fallback rétro-compat sur l'ancien champ.
        "niches": list(niches),
        "platform": "instagram",
        "followers": follower_count,
        "media_count": media_count,
        # Score unifié + détails par type
        "score": float(score_final),
        "score_reels": float(score_reels),
        "score_posts": float(score_posts),
        "score_content": float(content_score),
        "reel_weight": float(reel_weight),
        "post_weight": float(post_weight),
        # Comment-heat (signal dominant) : None si pas de donnée commentaire.
        "score_comment_heat": float(comment_heat) if comment_data_ok else None,
        "comment_ratio_median": (
            float(comment_ratio_med) if comment_data_ok else None
        ),
        "comment_likes_median": (
            float(comment_likes_med) if comment_data_ok else None
        ),
        "comment_reels_sampled": comment_reels_sampled,
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
        "reels": [
            {
                "media_id": str(r.get("media_id") or ""),
                "view_count": int(r.get("views") or 0),
                "like_count": int(r.get("likes") or 0),
                "comment_count": int(r.get("comments") or 0),
            }
            for r in reels
        ],
        "scored_at": _now_iso(),
    }


def build_dedup_key(media_id: str, text: str) -> str:
    return f"{media_id}||{text.strip().lower()}"


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
    profiles_recorded_count: int = 0


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
) -> dict[str, Any] | None:
    """Envoie une alerte au Bot Telegram #2 — **avec boutons inline**.

    Délègue au vrai ``telegram_discovery_bot.notify_candidate`` (qui construit
    le ``reply_markup`` ✅ ❌ ✏️ 👁 📈 attendu par le validateur). L'import
    est tardif pour éviter le cycle ``discovery → telegram_discovery_bot →
    discovery``.

    Auparavant, cette fonction faisait directement un ``requests.post`` vers
    ``sendMessage`` sans ``reply_markup`` — d'où les notifications "sans
    boutons" rapportées par le validateur quand la notif venait de
    ``explore_network`` (alors que ``score_and_persist`` passait déjà par
    le bot et avait les boutons).

    Retourne la réponse Telegram (``dict``) ou ``None`` en mock / config
    manquante / erreur. Ne lève jamais.
    """
    username = result.get("username")
    if mock:
        log.info(
            "[mock] notif candidate skip — @%s score=%.0f",
            username,
            result.get("score") or 0.0,
        )
        return None

    try:
        from telegram_discovery_bot import (  # tardif : évite cycle
            notify_candidate as _bot_notify,
        )
    except ImportError as e:
        log.warning(
            "telegram_discovery_bot indisponible (%s) — candidat @%s loggé seulement.",
            e, username,
        )
        return None

    try:
        return _bot_notify(result, mock=mock)
    except Exception as e:  # noqa: BLE001 — best-effort, on ne veut JAMAIS planter explore_network
        log.warning("notify_candidate a échoué (%s) pour @%s.", e, username)
        return None


# =============================================================================
# Persistance incrémentale (candidates)
# =============================================================================


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


def _usernames_from_suggestion_payload(payload: Any) -> list[str]:
    """Extrait les usernames d'une réponse discover/chaining/suggestions."""
    if payload is None:
        return []
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        candidates = (
            payload.get("users")
            or payload.get("user_list")
            or payload.get("items")
            or []
        )
    else:
        return []

    out: list[str] = []
    for item in candidates:
        if item is None:
            continue
        if isinstance(item, dict):
            uname = item.get("username")
        else:
            uname = getattr(item, "username", None)
        if uname is None:
            continue
        text = str(uname).lstrip("@").strip()
        if text:
            out.append(text)
    return out


def _filter_suggestion_usernames(
    usernames: list[str],
    *,
    max_results: int,
) -> list[str]:
    """Filtre les usernames vides et tronque à ``max_results``."""
    out: list[str] = []
    for uname in usernames:
        if uname is None:
            continue
        text = str(uname).lstrip("@").strip()
        if not text:
            continue
        out.append(text)
        if len(out) >= max_results:
            break
    return out


def _fetch_suggestion_details_usernames(client: Any, user_id: str) -> list[str]:
    """Niveau 2 : ``fetch_suggestion_details`` (ou ``chaining`` en secours)."""
    if not hasattr(client, "fetch_suggestion_details"):
        raise AttributeError("fetch_suggestion_details indisponible")

    try:
        payload = client.fetch_suggestion_details(user_id)
    except TypeError:
        if not hasattr(client, "chaining"):
            raise
        chaining = client.chaining(user_id)
        users = chaining.get("users") if isinstance(chaining, dict) else []
        chained_ids = ",".join(
            str(u.get("pk") or u.get("id") or "")
            for u in users
            if isinstance(u, dict) and (u.get("pk") or u.get("id"))
        )
        if not chained_ids:
            return _usernames_from_suggestion_payload(chaining)
        payload = client.fetch_suggestion_details(user_id, chained_ids)

    return _usernames_from_suggestion_payload(payload)


def _fetch_followings(username: str, client: Any) -> list[str]:
    """Fallback final : followings du seed via ``user_following``."""
    s = (username or "").lstrip("@").strip()
    if not s or client is None:
        return []

    try:
        seed_id = client.user_id_from_username(s)
    except Exception as e:
        _LOGGER.debug(
            "_fetch_followings user_id_from_username: %s: %s",
            type(e).__name__,
            e,
        )
        return []

    try:
        following = client.user_following(
            str(seed_id), amount=FOLLOWING_FETCH_AMOUNT
        )
    except Exception as e:
        _LOGGER.debug(
            "_fetch_followings user_following: %s: %s",
            type(e).__name__,
            e,
        )
        return []

    out: list[str] = []
    if isinstance(following, dict):
        for _, ushort in following.items():
            uname = str(getattr(ushort, "username", "") or "").strip()
            if uname:
                out.append(uname)
    else:
        for ushort in following or []:
            uname = str(getattr(ushort, "username", "") or "").strip()
            if uname:
                out.append(uname)
    return out


def _fetch_suggestions(
    username: str,
    context: BrowserContext,
    max_results: int = 30,
) -> list[str]:
    """Suggestions Instagram via Playwright (follow + vision « Voir tout »)."""
    s = (username or "").lstrip("@").strip()
    if not s or context is None:
        return []
    return get_suggested_accounts(s, context, max_results=max_results)


# ---------------------------------------------------------------------------
# Helpers de schéma — seeds.json (rétro-compat string ⇄ dict enrichi)
# ---------------------------------------------------------------------------
#
# Depuis 2026-05, ``seeds.json`` autorise deux formes pour chaque seed :
#
# * Forme historique (string) : ``"raikkonenaf"``
# * Forme enrichie (dict)     : ``{"username": "raikkonenaf", "niches": ["humour", "sketch", "imitation"]}``
#
# Les helpers suivants sont les **seuls points** par lesquels on doit accéder
# au username / niches d'un seed — toute autre lecture directe (``str(s)``,
# ``s["username"]``, ``s["niches"]``) doit passer par eux pour rester
# transparente au schéma.


def _seed_username(seed: Any) -> str:
    """Retourne le username propre depuis un seed string ou dict.

    Tolère tous les cas dégénérés : ``None``, dict sans clé ``username``,
    chaîne avec ``@`` initial ou espaces parasites. Retourne ``""`` si
    rien d'exploitable — le caller doit filtrer.
    """
    if isinstance(seed, dict):
        return str(seed.get("username") or "").lstrip("@").strip()
    return str(seed or "").lstrip("@").strip()


def _seed_niches(seed: Any, domain: dict[str, Any]) -> list[str]:
    """Retourne les niches du seed, avec fallback en cascade :

    1. ``seed["niches"]`` si seed est un dict avec une liste non vide.
    2. ``domain["niches"]`` (clé liste, prioritaire).
    3. ``domain["name"]`` (nom du domaine, ex : ``"humour"``).
    4. ``"humour"`` en dernier recours (ne devrait jamais arriver).

    Le résultat est **toujours** une liste avec au moins un élément. Pas
    de validation contre ``VALID_NICHES`` ici — c'est la responsabilité
    de ``score_profile`` (cf. helper ``validate_niches`` de ``config``).
    """
    if isinstance(seed, dict) and seed.get("niches"):
        return list(seed["niches"])
    domain_niches = domain.get("niches") or []
    if domain_niches:
        return list(domain_niches)
    fallback = str(domain.get("name") or "humour")
    return [fallback]


def explore_network(
    domain: dict[str, Any],
    *,
    watchlist: list[dict[str, Any]] | None = None,
    candidates: dict[str, Any] | None = None,
    db: dict[str, Any] | None = None,
    context: BrowserContext | None = None,
    session: DiscoverySession | None = None,
    candidates_path: Path | None = None,
    db_path: Path | None = None,
    seeds_path: Path | None = None,
    seeds_override: list[str] | None = None,
) -> None:
    """Discovery : score le seed puis ses suggestions Instagram.

    Pour chaque seed (objet brut, string ou dict avec ``niches``) :

    1. ``_score_one_in_domain(seed)`` — auto-scoring du seed.
    2. Si le quota quotidien (``MAX_PROFILES_PER_DAY``) est atteint
       → ``return`` (le seed reste dans ``seeds.json`` pour la prochaine run).
    3. ``_fetch_suggestions(seed)`` puis scoring de chaque suggestion
       (quota, ``seen_this_run``, database/watchlist en mémoire).
    4. Retirer le seed de ``seeds.json`` (sauf ``seeds_override`` ou mock)
       et persister atomiquement.
    5. Passer au seed suivant.

    Sets de filtrage :
        - ``database_set`` : usernames déjà dans ``database.json``, enrichi
          en mémoire après chaque ``upsert_profile``.
        - ``watchlist_set`` : idem, prêt pour de futures additions intra-run.
        - ``seen_this_run`` : dédup intra-session ; un seed déjà traité
          ce run est skip avant tout autre check.

    Limites humaines :
        - ``MAX_PROFILES_PER_DAY`` (compteur dans ``session``).
        - Inactif 23h–8h, pause obligatoire 12h–14h.
        - Pause 30 min toutes les 2h d'activité.

    """
    log = setup_discovery_logger()
    if not isinstance(domain, dict) or not domain.get("name"):
        log.warning("explore_network : domain invalide (%r) — skip.", domain)
        return

    domain_name = str(domain["name"])
    session = session or DiscoverySession(mock=False)

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
    # On conserve l'objet seed brut (string ou dict) pour pouvoir lire ses
    # niches enrichies plus bas via ``_seed_niches``. Le filtre
    # ``_seed_username(s)`` élimine les entrées vides / mal formées.
    seed_objects: list[Any] = [s for s in seeds_input if _seed_username(s)]
    if not seed_objects:
        log.info("Domaine %s : aucun seed — skip.", domain_name)
        return

    if not session.mock and context is None:
        log.error("Context Playwright manquant — abandon domaine %s.", domain_name)
        return

    # Sets initialisés **une seule fois** avant la boucle, puis mis à jour
    # en place par ``_score_one_in_domain`` au fur et à mesure.
    watchlist_set = _watchlist_usernames(watchlist)
    database_set = _database_usernames(db)
    seen_this_run: set[str] = set()

    log.info(
        "Domaine %s : %d seeds, watchlist=%d, database=%d.",
        domain_name,
        len(seed_objects),
        len(watchlist_set),
        len(database_set),
    )

    for seed_obj in seed_objects:
        seed_user = _seed_username(seed_obj)
        seed_niches = _seed_niches(seed_obj, domain)
        # Domain enrichi pour ce seed : ``score_profile`` lit ``domain["niches"]``
        # → on injecte celles du seed (plus précises que celles du domain root).
        seed_domain = {**domain, "niches": list(seed_niches)}
        seed_lower = seed_user.lower()

        seed_state = _score_one_in_domain(
            seed_lower,
            domain=seed_domain,
            domain_name=domain_name,
            parent_seed=None,  # is_seed=True
            candidates=candidates,
            db=db,
            watchlist_set=watchlist_set,
            database_set=database_set,
            seen_this_run=seen_this_run,
            context=context,
            session=session,
            candidates_path=candidates_path,
            db_path=db_path,
            log=log,
        )
        if seed_state == "quota":
            return

        if session.mock:
            suggestions: list[str] = []
        else:
            suggestions = _fetch_suggestions(
                seed_user, context, max_results=30
            )

        for username in suggestions:
            uname = username.lstrip("@").strip().lower()
            if not uname or uname == seed_lower:
                continue

            state = _score_one_in_domain(
                uname,
                domain=seed_domain,
                domain_name=domain_name,
                parent_seed=seed_user,
                candidates=candidates,
                db=db,
                watchlist_set=watchlist_set,
                database_set=database_set,
                seen_this_run=seen_this_run,
                context=context,
                session=session,
                candidates_path=candidates_path,
                db_path=db_path,
                log=log,
            )
            if state == "quota":
                return

        # Retrait du seed de seeds.json après exploration complète.
        # On ne touche pas seeds.json en mode ``seeds_override`` (CLI
        # ``--seed`` qui ré-explore explicitement un seul seed) ni en mock
        # (les tests s'attendent à ce que les disques restent intacts).
        if seeds_override is None and not session.mock:
            try:
                seeds_data = load_seeds(path=seeds_path)
                if _remove_seed_from_seeds_data(
                    seeds_data,
                    domain_name=domain_name,
                    seed_username=seed_lower,
                ):
                    save_seeds(seeds_data, path=seeds_path)
                    log.info(
                        "Seed @%s exploré et retiré de seeds.json.",
                        seed_lower,
                    )
            except DiscoveryIOError as e:
                # Best-effort : si on ne peut pas persister la suppression,
                # on continue le run — le seed sera juste re-scoré au prochain
                # passage et resservi par ``seen_this_run`` ce run-ci.
                log.warning(
                    "Impossible de retirer @%s de seeds.json (%s) — on continue.",
                    seed_lower, e,
                )


def _score_one_in_domain(
    uname: str,
    *,
    domain: dict[str, Any],
    domain_name: str,
    parent_seed: str | None,
    candidates: dict[str, Any],
    db: dict[str, Any],
    watchlist_set: set[str],
    database_set: set[str],
    seen_this_run: set[str],
    context: BrowserContext | None,
    session: DiscoverySession,
    candidates_path: Path | None,
    db_path: Path | None,
    log: logging.Logger,
) -> str:
    """Score & persiste **un** username dans le contexte d'un domaine.

    Pipeline (ordre des checks) :

    0. ``seen_this_run`` : dédup intra-session, premier check absolu.
    1. Quota quotidien (``MAX_PROFILES_PER_DAY``).
    2. Fenêtre d'activité (nuit / déjeuner) + pause de burst.
    3. Filtres ``watchlist_set`` / ``database_set`` (O(1) chacun).
    4. ``score_profile`` (ou ``_mock_score_profile`` en mode mock).
    5. ``upsert_profile`` (toujours si le scoring a abouti).
    6. ``_record_candidate`` + ``_notify_candidate`` si score > seuil.
    7. Ajout à ``database_set`` en mémoire après upsert.
    8. ``polite_sleep`` entre profils (sauf mock).

    Tout ``uname`` traité (même skip ou ineligible) est ajouté à
    ``seen_this_run`` à la fin pour éviter qu'un même username repasse
    dans la boucle pendant le même run.

    ``parent_seed=None`` indique que ``uname`` **est** le seed lui-même
    (auto-scoring). Les messages de log sont adaptés en conséquence.

    Retourne :
        - ``"quota"``   : quota quotidien atteint, le caller doit ``return``.
        - ``"skipped"`` : profil filtré (seen / watchlist / database),
          aucun scoring.
        - ``"scored"``  : scoring tenté (succès ou ``None`` = inéligible).

    """
    is_seed = parent_seed is None

    # 0) Dédup intra-session : premier check, **avant** tout autre.
    if uname in seen_this_run:
        log.debug("@%s déjà vu ce run — skip.", uname)
        return "skipped"

    _maybe_reset_day(session, _local_now())
    if session.profiles_today >= MAX_PROFILES_PER_DAY:
        log.info(
            "Quota quotidien atteint (%d) — arrêt domaine %s.",
            MAX_PROFILES_PER_DAY,
            domain_name,
        )
        return "quota"

    _wait_for_active_window(session, log)
    _maybe_take_burst_break(session, log)

    if uname in watchlist_set:
        if is_seed:
            log.info("Seed @%s déjà vu — skip auto-score.", uname)
        else:
            log.info("@%s déjà dans watchlist — skip.", uname)
        seen_this_run.add(uname)
        return "skipped"
    # Check O(1) sur le set en mémoire (mis à jour par les itérations
    # précédentes de la même boucle ``explore_network``).
    if uname in database_set:
        if is_seed:
            log.info("Seed @%s déjà en database — skip auto-score.", uname)
        else:
            log.info("@%s déjà en database — skip.", uname)
        seen_this_run.add(uname)
        return "skipped"

    if is_seed:
        log.info("Scoring seed @%s lui-même (domaine %s)...", uname, domain_name)
    else:
        log.info(
            "Scoring @%s (depuis seed @%s, domaine %s)...",
            uname,
            parent_seed,
            domain_name,
        )

    if session.mock:
        result = _mock_score_profile(uname, domain)
    else:
        result = score_profile(uname, domain, db=db, context=context)

    session.profiles_today += 1
    score_val = (result or {}).get("score")
    score_passes = (
        result is not None
        and float(result.get("score") or 0.0) > CANDIDATE_SCORE_THRESHOLD
    )

    # Persistance database : **tout** scoring réussi est upserté
    # (contrairement aux candidates qui sont conditionnels).
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
        score_for_log = float((result or {}).get("score") or 0.0)
        _record_candidate(
            candidates,
            result,  # type: ignore[arg-type]
            mock=session.mock,
            candidates_path=candidates_path,
        )
        # Trace explicite avant/après notif : le validateur a signalé des
        # boutons absents → on veut pouvoir corréler dans les logs un
        # candidat scoré avec la réponse exacte de Telegram (ou son
        # absence).
        log.info(
            "_score_one_in_domain : tentative notify @%s score=%.1f",
            uname, score_for_log,
        )
        response = _notify_candidate(result, log, mock=session.mock)  # type: ignore[arg-type]
        log.info("_score_one_in_domain : notify retour=%s", response)
        session.candidates_found += 1
        outcome = "candidate"
    else:
        outcome = "rejected" if result is not None else "ineligible"

    if result is not None:
        database_set.add(uname)
    session.profiles_recorded_count += 1
    seen_this_run.add(uname)

    if not session.mock:
        polite_sleep(
            min_s=DISCOVERY_BETWEEN_PROFILES_MIN_S,
            max_s=DISCOVERY_BETWEEN_PROFILES_MAX_S,
        )
    return "scored"


# =============================================================================
# Helpers mock (mode --mock)
# =============================================================================


def _mock_score_profile(username: str, domain: dict[str, Any]) -> dict[str, Any] | None:
    """Score factice déterministe (basé sur le hash) pour les tests CLI."""
    h = abs(hash((username, domain.get("name")))) % 1000
    if h < 200:
        return None  # ineligible simulé
    score = 200.0 + (h % 700)  # entre 200 et 899
    # Distribution Reels/Posts factice (entre 100% reels et 50/50)
    reels_n = 6 + (h % 7)             # 6..12 reels
    posts_n = 12 - reels_n if reels_n < 12 else 0
    total = reels_n + posts_n or 1
    reel_w = reels_n / total
    post_w = 1.0 - reel_w
    # Niches : aligne le mock sur le schéma de ``score_profile`` — liste validée.
    mock_niches = (
        list(domain.get("niches") or [])
        or [str(domain.get("name") or "humour")]
    )
    mock_niches = config.validate_niches(mock_niches)
    return {
        "username": username,
        "domain": domain.get("name"),
        "niches": list(mock_niches),
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
        "t_type_dominant": None,
        "t_type_distribution": {},
        "biography": f"mock bio {username}",
        "media_sampled": total,
        "reels_count": reels_n,
        "posts_count": posts_n,
        "reels_sampled": reels_n,
        "reels": [
            {
                "media_id": f"mock_reel_{i}",
                "view_count": 10_000 + i * 1000,
                "like_count": 100 + i * 10,
                "comment_count": 50 - i,
            }
            for i in range(min(reels_n, TOP_COMMENTS_REELS_MAX))
        ],
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
                if _seed_username(s).lower() == seed_clean:
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
    playwright_instance = None
    context: BrowserContext | None = None
    if not mock:
        try:
            playwright_instance = sync_playwright().start()
            context = get_browser_context(playwright_instance)
        except Exception as e:
            log.error("Playwright / cookies Instagram KO (%s) — abandon Discovery.", e)
            return session

    try:
        for i, domain in enumerate(domains):
            seeds_override = domain.get("_seeds_override")
            explore_network(
                domain,
                watchlist=watchlist,
                candidates=candidates,
                db=db,
                context=context,
                session=session,
                candidates_path=candidates_path,
                db_path=db_path,
                seeds_path=seeds_path,
                seeds_override=seeds_override,
            )

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
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
        if playwright_instance is not None:
            try:
                playwright_instance.stop()
            except Exception:
                pass
        log.info(
            "=== Discovery terminée (profiles_today=%d, candidates=%d, recorded=%d) ===",
            session.profiles_today,
            session.candidates_found,
            session.profiles_recorded_count,
        )
    return session


def score_and_persist(
    username: str,
    *,
    domain: dict[str, Any] | None = None,
    added_via: str = "manual",
    notify_threshold: float = DISCOVERY_NOTIFY_THRESHOLD,
    context: BrowserContext | None = None,
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

    try:
        db = load_db(path=db_path)
    except DatabaseIOError as e:
        log.warning("database illisible (%s) — repart vide.", e)
        db = {"profiles": {}}

    if mock:
        result = _mock_score_profile(username, domain)
    else:
        if context is None:
            log.error(
                "score_and_persist @%s : context Playwright manquant.", username
            )
            return None
        result = score_profile(username, domain, db=db, context=context)

    if result is None:
        log.info("score_and_persist @%s : filtré (None).", username)
        return None

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
        if args.mock:
            summary = score_and_persist(args.score, mock=True)
        else:
            with sync_playwright() as pw:
                context = get_browser_context(pw)
                try:
                    summary = score_and_persist(
                        args.score,
                        mock=False,
                        context=context,
                    )
                finally:
                    context.close()
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
    "load_candidates",
    "load_seeds",
    "run_discovery",
    "save_candidates",
    "_database_usernames",
    "_is_in_database",
    "save_seeds",
    "score_and_persist",
    "score_profile",
    "setup_discovery_logger",
]
