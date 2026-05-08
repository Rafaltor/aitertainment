"""database.py — Layer 0+ : persistance des profils scorés.

============================================================================
Architecture
============================================================================

Ce module est la **mémoire long-terme** du pipeline AItertainment : tous les
profils scorés (par ``discovery.score_profile``, par seed, ou ajoutés à la
main) y sont rangés avec :

- leur **tier** courant (A / B / C) — décide de la fréquence de rescore,
- leur **historique** de scores (chaque exécution append une ligne),
- leur statut **validation humaine** + ``t_type_final`` éventuellement
  corrigé après visionnage,
- leur statut **archived** (tier C ou retrait manuel).

Le fichier ``data/database.json`` a la structure suivante::

    {
      "profiles": {
        "<username>": {
          "platform": "instagram",
          "followers": 48000,
          "niche": "humour",
          "tier": "B",
          "validated": false,
          "t_type_original": "T2",
          "t_type_final": null,
          "added_via": "manual|discovery|seed",
          "added_at": "2026-05-08T00:00:00",
          "last_scored_at": "2026-05-08T00:00:00",
          "next_rescore_at": "2026-05-15T00:00:00",
          "archived": false,
          "scores_history": [
            {
              "date": "2026-05-08T00:00:00",
              "score": 448,
              "score_reels": 496,
              "score_posts": 351,
              "reel_ratio_median": 6.57,
              "reel_ratio_p90": 32.7,
              "reel_engagement_median": 0.068,
              "reel_trend": "stable",
              "posting_rhythm": 1.59,
              "t_type_dominant": "T2",
              "t_type_distribution": {}
            }
          ]
        }
      }
    }

Pas de réseau, pas d'appel instagrapi : c'est de la pure persistance + un peu
de logique de tier / planning de rescore.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"
DB_PATH = DEFAULT_DATA_DIR / "database.json"

_LOGGER = logging.getLogger("aitertainment.database")

# Seuils de score → tier. Tier A = score > 700, tier B = 400 ≤ score ≤ 700,
# tier C = score < 400 (auto-archivé).
TIER_SCORE_THRESHOLDS: dict[str, int] = {"A": 700, "B": 400}

# Délai avant le prochain rescore, par tier. Tier C → pas de rescore (None).
RESCORE_DELAY_DAYS: dict[str, int] = {"A": 7, "B": 30}


class DatabaseIOError(ValueError):
    """Erreur de lecture / écriture ou de schéma database invalide."""


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DatabaseIOError(f"fichier absent : {path}")
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise DatabaseIOError(f"lecture impossible : {path} ({e})") from e
    if not raw:
        raise DatabaseIOError(f"fichier vide : {path}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise DatabaseIOError(f"JSON invalide dans {path} : {e}") from e
    if not isinstance(data, dict):
        raise DatabaseIOError(
            f"racine JSON doit être un objet dans {path}, "
            f"reçu {type(data).__name__}"
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


def _now() -> datetime:
    """Now naïf-UTC (cohérent avec ``discovery._now_iso`` : pas de suffixe Z)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _parse_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        # ``fromisoformat`` accepte naïf et offset-aware. On normalise vers naïf
        # pour rester homogène avec ``_now``.
        dt = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _normalize_username(username: str) -> str:
    """Strip ``@`` éventuel + lowercase. Cohérent avec ``discovery.score_profile``."""
    return str(username or "").lstrip("@").strip().lower()


# ---------------------------------------------------------------------------
# Tier / planning de rescore
# ---------------------------------------------------------------------------


def compute_tier(score: float | int) -> str:
    """A si ``score > 700``, B si ``400 ≤ score ≤ 700``, C sinon."""
    s = float(score)
    if s > TIER_SCORE_THRESHOLDS["A"]:
        return "A"
    if s >= TIER_SCORE_THRESHOLDS["B"]:
        return "B"
    return "C"


def compute_next_rescore_at(
    tier: str, *, anchor: datetime | None = None
) -> str | None:
    """Tier A → ``anchor + 7j``, B → ``+30j``, C → ``None`` (archivé)."""
    if tier == "C":
        return None
    if tier not in RESCORE_DELAY_DAYS:
        raise DatabaseIOError(f"tier inconnu : {tier!r}")
    base = anchor if anchor is not None else _now()
    return _iso(base + timedelta(days=RESCORE_DELAY_DAYS[tier]))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_db(path: str | Path | None = None) -> dict[str, Any]:
    """Charge ``database.json``.

    Si le fichier est absent, retourne ``{"profiles": {}}`` (premier run).
    """
    p = Path(path) if path else DB_PATH
    if not p.exists():
        _LOGGER.info("Database absente — démarrage vide (%s)", p.name)
        return {"profiles": {}}
    data = _read_json(p)
    profiles = data.get("profiles")
    if profiles is None:
        raise DatabaseIOError(f'"profiles" manquant dans {p}')
    if not isinstance(profiles, dict):
        raise DatabaseIOError(f'"profiles" doit être un dict dans {p}')
    _LOGGER.debug(
        "Database chargée : %d profil(s) depuis %s", len(profiles), p.name
    )
    return data


def save_db(db: dict[str, Any], path: str | Path | None = None) -> None:
    """Écrit ``database.json`` de façon atomique."""
    if not isinstance(db, dict):
        raise DatabaseIOError(
            f"db doit être un dict, reçu {type(db).__name__}"
        )
    profiles = db.get("profiles")
    if profiles is None or not isinstance(profiles, dict):
        raise DatabaseIOError('"profiles" doit être un dict non absent')
    p = Path(path) if path else DB_PATH
    _atomic_write_json(p, {"profiles": profiles})
    _LOGGER.info("Database sauvegardée : %d profil(s) -> %s", len(profiles), p)


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


_HISTORY_FIELDS = (
    "score",
    "score_reels",
    "score_posts",
    "reel_ratio_median",
    "reel_ratio_p90",
    "reel_engagement_median",
    "reel_trend",
    "posting_rhythm",
    "t_type_dominant",
    "t_type_distribution",
)


def _build_history_entry(
    score_result: dict[str, Any], *, date_iso: str
) -> dict[str, Any]:
    entry: dict[str, Any] = {"date": date_iso}
    for k in _HISTORY_FIELDS:
        entry[k] = score_result.get(k)
    return entry


def upsert_profile(
    db: dict[str, Any],
    score_result: dict[str, Any],
    added_via: str,
) -> dict[str, Any]:
    """Crée ou met à jour le profil dans ``db`` à partir d'un ``score_result``.

    Pipeline :

    - Recalcule ``tier`` à partir du score (cf. ``compute_tier``).
    - Append une ligne dans ``scores_history`` (un point par exécution).
    - Recalcule ``next_rescore_at`` (None pour tier C).
    - Tier C → ``archived = True`` ; sinon ``archived = False``.
    - Sur **première insertion** : enregistre ``added_at``, ``added_via``,
      ``t_type_original`` (= ``t_type_dominant`` du scoring), ``validated =
      False``, ``t_type_final = None``.
    - Sur **mise à jour** : conserve ``added_at``, ``added_via``,
      ``t_type_original``, ``validated``, ``t_type_final`` (la validation
      humaine est sacrée, on ne l'écrase pas).

    Returns
    -------
    dict
        Le profil mis à jour (référence vivante dans ``db["profiles"]``).
    """
    if not isinstance(db, dict) or not isinstance(db.get("profiles"), dict):
        raise DatabaseIOError("db invalide : attendu {'profiles': {...}}")
    if not isinstance(score_result, dict):
        raise DatabaseIOError("score_result doit être un dict")

    raw_username = score_result.get("username")
    if not raw_username:
        raise DatabaseIOError("score_result['username'] manquant ou vide")
    username = _normalize_username(str(raw_username))

    score = score_result.get("score")
    if score is None:
        raise DatabaseIOError("score_result['score'] manquant")

    last_scored_at = (
        str(score_result.get("scored_at") or "") or _iso(_now())
    )
    last_scored_dt = _parse_iso(last_scored_at) or _now()
    tier = compute_tier(float(score))
    next_rescore = compute_next_rescore_at(tier, anchor=last_scored_dt)
    archived = tier == "C"

    profiles: dict[str, Any] = db["profiles"]
    existing = profiles.get(username)

    if existing is None:
        profile: dict[str, Any] = {
            "platform": str(score_result.get("platform") or "instagram"),
            "followers": int(score_result.get("followers") or 0),
            "niche": str(score_result.get("domain") or ""),
            "tier": tier,
            "validated": False,
            "t_type_original": score_result.get("t_type_dominant"),
            "t_type_final": None,
            "added_via": str(added_via),
            "added_at": last_scored_at,
            "last_scored_at": last_scored_at,
            "next_rescore_at": next_rescore,
            "archived": archived,
            "scores_history": [],
        }
        profiles[username] = profile
    else:
        profile = existing
        # On rafraîchit ce qui change ; on **conserve** ``added_at``,
        # ``added_via``, ``t_type_original``, ``validated``, ``t_type_final``.
        profile["platform"] = str(
            score_result.get("platform") or profile.get("platform") or "instagram"
        )
        profile["followers"] = int(
            score_result.get("followers")
            if score_result.get("followers") is not None
            else profile.get("followers", 0)
        )
        if score_result.get("domain"):
            profile["niche"] = str(score_result["domain"])
        profile["tier"] = tier
        profile["last_scored_at"] = last_scored_at
        profile["next_rescore_at"] = next_rescore
        profile["archived"] = archived
        profile.setdefault("scores_history", [])

    profile["scores_history"].append(
        _build_history_entry(score_result, date_iso=last_scored_at)
    )

    _LOGGER.info(
        "upsert @%s : score=%.1f tier=%s archived=%s next_rescore=%s",
        username,
        float(score),
        tier,
        archived,
        next_rescore or "—",
    )
    return profile


def get_profiles_due_for_rescore(
    db: dict[str, Any], *, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Retourne les profils non archivés dont ``next_rescore_at <= now``.

    Chaque dict retourné est une **copie** enrichie d'une clé ``username`` (les
    profils sont stockés sous forme de mapping ``{username: {...}}``, le caller
    a besoin de l'identifiant pour relancer un scoring).
    """
    if not isinstance(db, dict) or not isinstance(db.get("profiles"), dict):
        raise DatabaseIOError("db invalide : attendu {'profiles': {...}}")
    cutoff = now if now is not None else _now()
    out: list[dict[str, Any]] = []
    for username, profile in db["profiles"].items():
        if not isinstance(profile, dict):
            continue
        if profile.get("archived"):
            continue
        nxt = _parse_iso(profile.get("next_rescore_at"))
        if nxt is None:
            continue
        if nxt <= cutoff:
            enriched = dict(profile)
            enriched["username"] = username
            out.append(enriched)
    return out


def _require_profile(db: dict[str, Any], username: str) -> dict[str, Any]:
    if not isinstance(db, dict) or not isinstance(db.get("profiles"), dict):
        raise DatabaseIOError("db invalide : attendu {'profiles': {...}}")
    key = _normalize_username(username)
    profile = db["profiles"].get(key)
    if profile is None:
        raise DatabaseIOError(f"profil inconnu : @{key}")
    return profile


def promote_tier(
    db: dict[str, Any], username: str, new_tier: str
) -> dict[str, Any]:
    """Force ``tier`` manuellement.

    - Recalcule ``next_rescore_at`` (depuis ``last_scored_at`` si dispo,
      sinon ``now``).
    - ``archived = False`` si nouveau tier ∈ {A, B}, ``True`` si C.
    """
    if new_tier not in {"A", "B", "C"}:
        raise DatabaseIOError(f"tier invalide : {new_tier!r}")
    profile = _require_profile(db, username)
    anchor = _parse_iso(profile.get("last_scored_at")) or _now()
    profile["tier"] = new_tier
    profile["next_rescore_at"] = compute_next_rescore_at(new_tier, anchor=anchor)
    profile["archived"] = new_tier == "C"
    _LOGGER.info(
        "promote_tier @%s -> %s (archived=%s)",
        _normalize_username(username),
        new_tier,
        profile["archived"],
    )
    return profile


def archive_profile(db: dict[str, Any], username: str) -> dict[str, Any]:
    """Retire un profil de la rotation : ``archived=True``, ``tier='C'``,
    ``next_rescore_at=None``.
    """
    profile = _require_profile(db, username)
    profile["archived"] = True
    profile["tier"] = "C"
    profile["next_rescore_at"] = None
    _LOGGER.info("archive_profile @%s", _normalize_username(username))
    return profile


def validate_profile(
    db: dict[str, Any], username: str, t_type_final: str
) -> dict[str, Any]:
    """Marque le profil comme validé humainement et fixe ``t_type_final``."""
    if not t_type_final:
        raise DatabaseIOError("t_type_final ne peut pas être vide")
    profile = _require_profile(db, username)
    profile["validated"] = True
    profile["t_type_final"] = str(t_type_final)
    _LOGGER.info(
        "validate_profile @%s : t_type_final=%s",
        _normalize_username(username),
        t_type_final,
    )
    return profile


__all__ = [
    "DB_PATH",
    "DEFAULT_DATA_DIR",
    "DatabaseIOError",
    "RESCORE_DELAY_DAYS",
    "TIER_SCORE_THRESHOLDS",
    "archive_profile",
    "compute_next_rescore_at",
    "compute_tier",
    "get_profiles_due_for_rescore",
    "load_db",
    "promote_tier",
    "save_db",
    "upsert_profile",
    "validate_profile",
]
