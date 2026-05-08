"""dataset_builder.py — collecte des commentaires Reels pour fine-tuning T-types.

============================================================================
Objectif
============================================================================

Constituer un dataset annoté ``(commentaires → T-type humain validé)`` à
partir des Reels publics des créateurs **validés** dans la watchlist Discovery
(``database.json`` : ``validated=True`` + ``t_type_final`` fixé par l'humain).

Un échantillon par Reel suffit : on récupère 5 Reels récents (mais pas trop
récents — les Reels < 7 jours ont une distribution de commentaires non
stabilisée, voire 0 commentaire). On garde le **top 3** des commentaires
ayant ``like_count > 0`` (les commentaires non likés sont du bruit).

============================================================================
Fichiers persistés
============================================================================

``data/training_comments.json`` est le fichier **mutualisé** servant deux
modèles distincts au moment du fine-tuning. La structure de chaque entrée
sépare explicitement ce qui appartient à chaque rôle ::

    {"entries": [
       {
         "media_id": "...",
         "username": "raikkonenaf",
         "reel_url": "https://www.instagram.com/reel/Cxyz/",
         "collected_at": "2026-05-08T12:34:56",

         # Features disponibles **en prod** quand on génère un commentaire
         # sur un Reel frais (pas encore d'interactions).
         "generator_input": {
           "t_type": "T2",
           "niche": "humour",
           "caption": "...",
           "hashtags": ["f1", "monaco"],
           "audio_id": "12345"
         },

         # Features de **contexte** disponibles uniquement après publication
         # (compteurs + ratios) — pour entraîner le classifier T-type.
         "classifier_context": {
           "views": 500000,
           "likes": 12000,
           "comment_count": 3400,
           "shares": null,
           "comment_to_like_ratio": 0.283,
           "share_to_like_ratio": null
         },

         # Chaque commentaire est **self-contained** : on duplique ``t_type``
         # et ``niche`` pour que le loader classifier puisse itérer sans
         # ré-aller chercher la feature au niveau parent.
         "top_comments": [
           {"text": "...", "likes": 42, "t_type": "T2", "niche": "humour"},
           ...
         ]
       },
       ...
    ]}

Déduplication par ``media_id`` (replays multiples du même Reel ne créent pas
de doublons — utile pour les rescores où on revient sur les mêmes profils).

``data/pending_collection.json`` ::

    {"entries": [
       {"username": "alice",
        "collect_after": "2026-05-15T...",
        "scheduled_at": "2026-05-08T..."},
       ...
    ]}

Quand un profil validé n'a aucun Reel ≥ 7 jours (souvent le cas pour les
créateurs très actifs), on programme la collecte 7 jours plus tard. Le
scheduler de rescore appelle ``check_pending_collections`` 1×/jour qui
rebalaye ce fichier.

============================================================================
CLI
============================================================================

::

    python dataset_builder.py --mock      # affiche les paths, ne fait rien
    python dataset_builder.py --pending   # force le balayage de pending_collection.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_TRAINING_PATH = DEFAULT_DATA_DIR / "training_comments.json"
DEFAULT_PENDING_PATH = DEFAULT_DATA_DIR / "pending_collection.json"

_LOG = logging.getLogger("aitertainment.dataset")

# Paramètres de collecte
MAX_REELS_TO_COLLECT = 5
COLLECT_AGE_MIN_DAYS = 7
COMMENTS_PER_MEDIA = 50
TOP_COMMENTS_PER_MEDIA = 3
PENDING_RETRY_DAYS = 7

REEL_PRODUCT_TYPE = "clips"

_HASHTAG_RE = re.compile(r"#(\w+)")


class DatasetIOError(ValueError):
    """Erreur de lecture / écriture sur les fichiers du dataset."""


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _parse_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _normalize_username(u: str) -> str:
    return str(u or "").lstrip("@").strip().lower()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise DatasetIOError(f"lecture impossible : {path} ({e})") from e
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise DatasetIOError(f"JSON invalide dans {path} : {e}") from e
    if not isinstance(data, dict):
        raise DatasetIOError(
            f"racine JSON doit être un objet ({type(data).__name__})"
        )
    return data


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        suffix=".tmp",
    ) as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        tmp_path = Path(fh.name)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Schéma : training_comments.json
# ---------------------------------------------------------------------------


def _load_training(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_TRAINING_PATH
    if not p.exists():
        return {"entries": []}
    data = _read_json(p)
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise DatasetIOError(f'"entries" doit être une liste dans {p}')
    return {"entries": entries}


def _save_training(payload: dict[str, Any], *, path: Path | None = None) -> None:
    p = Path(path) if path else DEFAULT_TRAINING_PATH
    _atomic_write_json(p, {"entries": list(payload.get("entries") or [])})


# ---------------------------------------------------------------------------
# Schéma : pending_collection.json
# ---------------------------------------------------------------------------


def _load_pending(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_PENDING_PATH
    if not p.exists():
        return {"entries": []}
    data = _read_json(p)
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise DatasetIOError(f'"entries" doit être une liste dans {p}')
    return {"entries": entries}


def _save_pending(payload: dict[str, Any], *, path: Path | None = None) -> None:
    p = Path(path) if path else DEFAULT_PENDING_PATH
    _atomic_write_json(p, {"entries": list(payload.get("entries") or [])})


def schedule_pending_collection(
    username: str,
    *,
    after_days: int = PENDING_RETRY_DAYS,
    path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Programme une collecte différée pour ``username``.

    Si une entrée pour ce username existe déjà, on **rafraîchit** sa date
    ``collect_after`` (pas de doublons). Retourne l'entrée enregistrée.
    """
    u = _normalize_username(username)
    if not u:
        raise DatasetIOError("username vide pour schedule_pending_collection")
    base = now if now is not None else _now()
    collect_after = _iso(base + timedelta(days=int(after_days)))
    scheduled_at = _iso(base)

    pending = _load_pending(path=path)
    entries = list(pending.get("entries") or [])
    entry: dict[str, Any] = {
        "username": u,
        "collect_after": collect_after,
        "scheduled_at": scheduled_at,
    }
    # Dédup par username
    filtered = [e for e in entries if _normalize_username(e.get("username")) != u]
    filtered.append(entry)
    _save_pending({"entries": filtered}, path=path)
    _LOG.info("schedule_pending_collection @%s : collect_after=%s", u, collect_after)
    return entry


# ---------------------------------------------------------------------------
# Extraction depuis les objets instagrapi (ou dicts mock)
# ---------------------------------------------------------------------------


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Lit ``obj.name`` (instagrapi) ou ``obj[name]`` (dict mock), avec fallback."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _is_pinned(media: Any) -> bool:
    raw = _attr(media, "is_pinned", None)
    return raw is True


def _media_taken_at(media: Any) -> datetime | None:
    raw = _attr(media, "taken_at", None)
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.astimezone(timezone.utc).replace(tzinfo=None) if raw.tzinfo else raw
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc).replace(tzinfo=None)
    if isinstance(raw, str):
        return _parse_iso(raw)
    return None


def _media_views(media: Any) -> int:
    """Vue d'un Reel — ``play_count`` prioritaire (cf. discovery._media_views)."""
    pc = _attr(media, "play_count", None) or 0
    if pc:
        try:
            return int(pc)
        except (TypeError, ValueError):
            return 0
    vc = _attr(media, "view_count", None) or 0
    try:
        return int(vc)
    except (TypeError, ValueError):
        return 0


def _media_caption(media: Any) -> str:
    return str(_attr(media, "caption_text", "") or "")


def _media_pk(media: Any) -> str:
    pk = _attr(media, "pk", None) or _attr(media, "id", None) or ""
    return str(pk)


def _media_code(media: Any) -> str:
    return str(_attr(media, "code", "") or "")


def _media_audio_id(media: Any) -> str | None:
    """Tente plusieurs chemins instagrapi. Retombe sur ``None`` proprement.

    Pistes connues (selon la version instagrapi & le type de Reel) :
    - ``music_metadata.audio_cluster_id`` (audio licence)
    - ``clips_metadata.original_sound_info.audio_asset_id`` (son original)
    - ``audio_id`` direct (mocks).
    """
    direct = _attr(media, "audio_id", None)
    if direct:
        return str(direct)
    music = _attr(media, "music_metadata", None)
    if music is not None:
        cluster = _attr(music, "audio_cluster_id", None) or _attr(
            music, "music_canonical_id", None
        )
        if cluster:
            return str(cluster)
    clips = _attr(media, "clips_metadata", None)
    if clips is not None:
        original = _attr(clips, "original_sound_info", None)
        if original is not None:
            asset = _attr(original, "audio_asset_id", None) or _attr(
                original, "music_canonical_id", None
            )
            if asset:
                return str(asset)
    return None


def _extract_hashtags(caption: str) -> list[str]:
    if not caption:
        return []
    return [h.lower() for h in _HASHTAG_RE.findall(caption)]


def _reel_url(media: Any) -> str | None:
    code = _media_code(media)
    if code:
        return f"https://www.instagram.com/reel/{code}/"
    return None


def _media_like_count(media: Any) -> int:
    raw = _attr(media, "like_count", 0)
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _media_comment_count(media: Any) -> int:
    raw = _attr(media, "comment_count", 0)
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _media_share_count(media: Any) -> int | None:
    """``share_count`` est souvent ``None`` côté instagrapi (l'API ne l'expose
    pas pour tous les médias publics). On préserve ``None`` plutôt que de
    forcer 0 — le classifier doit pouvoir distinguer ``shares inconnus`` de
    ``shares = 0 confirmés``.
    """
    raw = _attr(media, "share_count", None)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _build_classifier_context(media: Any) -> dict[str, Any]:
    """Compteurs bruts + ratios pour le **classifier T-type** (pas le scoring).

    Schéma ::

        {
          "views": int,
          "likes": int,
          "comment_count": int,
          "shares": int | None,
          "comment_to_like_ratio": float | None,  # None si likes == 0
          "share_to_like_ratio":   float | None,  # None si likes == 0 ou shares falsy
        }

    Pourquoi ces ratios spécifiquement (cf. brief) :

    - ``comment_to_like_ratio`` : signal **conversationnel**. Très élevé sur
      les T-types qui suscitent du débat (T2 humour vs T4 take/opinion) —
      faible sur T1 esthétique.
    - ``share_to_like_ratio`` : signal **viralité informative**. Très élevé
      sur T3 contenu utile (« ce truc m'a appris X »), faible sur T1.

    ``shares`` ``None`` n'est pas un zéro — c'est une donnée manquante (l'API
    ne renvoie ``share_count`` que pour certains médias). Le classifier doit
    pouvoir filtrer là-dessus, donc on ne dégrade pas en 0.
    """
    views = _media_views(media)
    likes = _media_like_count(media)
    comments = _media_comment_count(media)
    shares = _media_share_count(media)

    if likes > 0:
        c2l: float | None = round(comments / likes, 4)
    else:
        c2l = None

    if likes > 0 and shares:
        s2l: float | None = round(shares / likes, 4)
    else:
        s2l = None

    return {
        "views": views,
        "likes": likes,
        "comment_count": comments,
        "shares": shares,
        "comment_to_like_ratio": c2l,
        "share_to_like_ratio": s2l,
    }


def _build_generator_input(
    media: Any, *, t_type: str | None, niche: str
) -> dict[str, Any]:
    """Features disponibles **en prod** au moment de générer un commentaire.

    Au moment d'inférence, on a un Reel à peine publié — on ne peut donc
    s'appuyer **que** sur ce qui existe dès la publication : T-type cible
    du créateur, niche, caption, hashtags, identifiant audio. Aucun signal
    de réception (likes, vues, partages) ne doit fuiter ici, sinon le
    fine-tuning apprendrait des features qu'il n'aura jamais en prod.
    """
    caption = _media_caption(media)
    return {
        "t_type": t_type,
        "niche": niche,
        "caption": caption,
        "hashtags": _extract_hashtags(caption),
        "audio_id": _media_audio_id(media),
    }


def _comment_like_count(c: Any) -> int:
    raw = _attr(c, "like_count", None)
    if raw is None:
        raw = _attr(c, "comment_like_count", 0)
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _comment_text(c: Any) -> str:
    return str(_attr(c, "text", "") or "").strip()


def _top_comments(
    comments: list[Any],
    *,
    t_type: str | None,
    niche: str,
    n: int = TOP_COMMENTS_PER_MEDIA,
) -> list[dict[str, Any]]:
    """Top ``n`` commentaires par like_count > 0, **self-contained** pour le classifier.

    On duplique ``t_type`` et ``niche`` dans chaque commentaire de sortie : le
    loader classifier itère ``entry["top_comments"]`` directement sans avoir
    à remonter au parent. C'est de la dénormalisation **assumée** (poids JSON
    ×3 sur la liste de commentaires) parce que ces deux champs sont de toute
    façon trivialement courts.
    """
    candidates: list[tuple[int, str]] = []
    for c in comments or []:
        text = _comment_text(c)
        if not text:
            continue
        likes = _comment_like_count(c)
        if likes <= 0:
            continue
        candidates.append((likes, text))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [
        {"text": t, "likes": l, "t_type": t_type, "niche": niche}
        for l, t in candidates[:n]
    ]


# ---------------------------------------------------------------------------
# API publique : collecte + scheduling
# ---------------------------------------------------------------------------


def _select_eligible_reels(
    medias: list[Any], *, now: datetime | None = None
) -> list[Any]:
    """Garde Reels (``product_type == "clips"``) non épinglés et ≥ 7 jours."""
    cutoff = (now or _now()) - timedelta(days=COLLECT_AGE_MIN_DAYS)
    out: list[Any] = []
    for m in medias or []:
        if str(_attr(m, "product_type", "") or "") != REEL_PRODUCT_TYPE:
            continue
        if _is_pinned(m):
            continue
        ta = _media_taken_at(m)
        if ta is None or ta > cutoff:
            continue
        out.append(m)
    return out


def collect_training_data(
    username: str,
    profile: dict[str, Any],
    *,
    client: Any | None = None,
    mock: bool = False,
    training_path: Path | None = None,
    sleep_fn: Callable[[], None] | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Collecte les commentaires des 5 derniers Reels ≥ 7 jours d'``username``.

    Pipeline :

    1. ``client.user_id_from_username`` → ``client.user_medias(amount=5)``.
    2. Filtre Reels ``product_type='clips'``, non épinglés, ``taken_at`` ≥ 7 j.
       Aucun éligible → retour ``[]`` (le caller doit alors ``schedule_pending_collection``).
    3. Pour chaque Reel : ``client.media_comments(amount=50)`` → top 3 par
       ``like_count > 0``.
    4. Construction de l'entrée selon le schéma à **deux datasets** :

       - ``generator_input`` : features dispo en prod (``t_type``, ``niche``,
         ``caption``, ``hashtags``, ``audio_id``).
       - ``classifier_context`` : compteurs + ratios (``views``, ``likes``,
         ``comment_count``, ``shares``, ``comment_to_like_ratio``,
         ``share_to_like_ratio``).
       - ``top_comments`` : chaque commentaire est self-contained (``text``,
         ``likes``, ``t_type``, ``niche``).

    5. Append + dédup par ``media_id`` dans ``training_comments.json`` (atomique).
    6. ``polite_sleep`` (ou ``sleep_fn`` injecté) entre Reels.

    Retourne la liste des entrées **ajoutées** (peut être vide si tout était
    déjà collecté ou si aucun Reel n'est éligible).

    Mode ``mock=True`` → no-op (pas de réseau, pas d'écriture, retour ``[]``).
    """
    u = _normalize_username(username)
    if not u:
        _LOG.warning("collect_training_data : username vide.")
        return []
    if mock:
        _LOG.info("[mock] collect_training_data @%s — no-op.", u)
        return []

    if client is None:
        try:
            from instagram_client import get_client  # tardif (cycle d'import)
            client = get_client()
        except Exception as e:  # ImportError, InstagramAuthError, etc.
            _LOG.warning("collect_training_data @%s : pas de client (%s).", u, e)
            return []

    if sleep_fn is None:
        try:
            from instagram_client import polite_sleep
            sleep_fn = polite_sleep
        except ImportError:
            sleep_fn = lambda: None  # noqa: E731

    try:
        user_id = client.user_id_from_username(u)
    except Exception as e:
        _LOG.warning("collect_training_data @%s : user_id KO (%s).", u, e)
        return []

    try:
        medias = client.user_medias(str(user_id), amount=MAX_REELS_TO_COLLECT)
    except Exception as e:
        _LOG.warning("collect_training_data @%s : user_medias KO (%s).", u, e)
        return []

    eligible = _select_eligible_reels(medias, now=now)
    if not eligible:
        _LOG.info(
            "collect_training_data @%s : aucun Reel éligible (≥ %d jours).",
            u,
            COLLECT_AGE_MIN_DAYS,
        )
        return []

    niche = str(profile.get("niche") or "")
    t_type = profile.get("t_type_final") or profile.get("t_type_original")

    try:
        store = _load_training(path=training_path)
    except DatasetIOError as e:
        _LOG.warning("training_comments illisible (%s) — repart vide.", e)
        store = {"entries": []}
    existing_ids = {
        str(e.get("media_id"))
        for e in store["entries"]
        if isinstance(e, dict) and e.get("media_id")
    }
    collected_at = _iso(now or _now())

    added: list[dict[str, Any]] = []
    for reel in eligible:
        media_id = _media_pk(reel)
        if not media_id:
            continue
        if media_id in existing_ids:
            _LOG.info(
                "collect_training_data @%s : %s déjà présent — skip.",
                u,
                media_id,
            )
            continue

        try:
            comments = client.media_comments(
                media_id, amount=COMMENTS_PER_MEDIA
            )
        except Exception as e:
            _LOG.warning(
                "collect_training_data @%s media %s : comments KO (%s) — skip.",
                u,
                media_id,
                e,
            )
            try:
                sleep_fn()
            except Exception:
                pass
            continue

        entry: dict[str, Any] = {
            "media_id": media_id,
            "username": u,
            "reel_url": _reel_url(reel),
            "collected_at": collected_at,
            "generator_input": _build_generator_input(
                reel, t_type=t_type, niche=niche
            ),
            "classifier_context": _build_classifier_context(reel),
            "top_comments": _top_comments(
                comments or [], t_type=t_type, niche=niche
            ),
        }
        store["entries"].append(entry)
        existing_ids.add(media_id)
        added.append(entry)

        try:
            sleep_fn()
        except Exception as e:
            _LOG.warning("sleep_fn a levé (%s) — on continue.", e)

    if added:
        try:
            _save_training(store, path=training_path)
        except DatasetIOError as e:
            _LOG.warning("save_training a échoué (%s).", e)
            return []
        _LOG.info(
            "collect_training_data @%s : %d nouvelle(s) entrée(s) écrite(s).",
            u,
            len(added),
        )
    else:
        _LOG.info(
            "collect_training_data @%s : aucune nouvelle entrée (déjà collecté).",
            u,
        )
    return added


# ---------------------------------------------------------------------------
# Pending collection — balayage périodique
# ---------------------------------------------------------------------------


def check_pending_collections(
    *,
    client: Any | None = None,
    mock: bool = False,
    db_path: Path | None = None,
    pending_path: Path | None = None,
    training_path: Path | None = None,
    sleep_fn: Callable[[], None] | None = None,
    now: datetime | None = None,
    collect_fn: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, int]:
    """Rebalaye ``pending_collection.json`` et lance les collectes échues.

    Pour chaque entrée dont ``collect_after <= now`` :

    1. On charge le profil depuis ``database.json`` (le ``t_type_final`` peut
       avoir changé entre-temps — c'est tout l'intérêt du différé).
    2. ``collect_training_data(username, profile, ...)`` ; on garde la même
       sémantique : retour ``[]`` si toujours aucun Reel éligible.
    3. **L'entrée est retirée** du pending dans tous les cas où la collecte
       a tourné (réussie ou pas) — sinon les profils orphelins (supprimés
       de la DB) resteraient à vie. On reprogramme **sauf** si tout est OK.

    Retourne ``{"checked", "collected", "skipped", "removed"}``.
    """
    cutoff = now or _now()
    stats = {"checked": 0, "collected": 0, "skipped": 0, "removed": 0}

    try:
        pending = _load_pending(path=pending_path)
    except DatasetIOError as e:
        _LOG.warning("pending illisible (%s) — abort.", e)
        return stats

    entries: list[dict[str, Any]] = list(pending.get("entries") or [])
    if not entries:
        return stats

    if collect_fn is None:
        collect_fn = collect_training_data

    # Charge la DB une seule fois (peut être absente).
    db_profiles: dict[str, Any] = {}
    try:
        from database import load_db  # tardif
        db = load_db(path=db_path)
        db_profiles = db.get("profiles") or {}
    except Exception as e:
        _LOG.warning("load_db a échoué (%s) — on continue avec profil vide.", e)

    remaining: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        u = _normalize_username(entry.get("username"))
        ca = _parse_iso(entry.get("collect_after"))
        if not u or ca is None:
            _LOG.info("pending entrée invalide : %r — drop.", entry)
            stats["removed"] += 1
            continue

        if ca > cutoff:
            remaining.append(entry)
            stats["skipped"] += 1
            continue

        stats["checked"] += 1
        profile = db_profiles.get(u)
        if not isinstance(profile, dict):
            _LOG.info(
                "pending @%s : profil absent de database.json — drop.", u
            )
            stats["removed"] += 1
            continue

        try:
            added = collect_fn(
                u,
                profile,
                client=client,
                mock=mock,
                training_path=training_path,
                sleep_fn=sleep_fn,
                now=now,
            )
        except Exception as e:
            _LOG.warning(
                "pending @%s : collect_training_data a levé (%s).", u, e
            )
            added = []

        if added:
            stats["collected"] += 1
            stats["removed"] += 1
            # Entrée traitée avec succès → retirée définitivement.
        else:
            # Toujours pas de Reel éligible (ou erreur) → reprogramme + 7 j.
            base = entry.get("collect_after") or _iso(cutoff)
            base_dt = _parse_iso(base) or cutoff
            entry["collect_after"] = _iso(
                base_dt + timedelta(days=PENDING_RETRY_DAYS)
            )
            entry["scheduled_at"] = _iso(cutoff)
            remaining.append(entry)
            _LOG.info(
                "pending @%s : pas d'entrée — reprogrammé à %s.",
                u,
                entry["collect_after"],
            )

    try:
        _save_pending({"entries": remaining}, path=pending_path)
    except DatasetIOError as e:
        _LOG.warning("save_pending a échoué (%s).", e)

    _LOG.info(
        "check_pending_collections : checked=%d collected=%d skipped=%d removed=%d",
        stats["checked"],
        stats["collected"],
        stats["skipped"],
        stats["removed"],
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main_cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dataset builder — collecte des commentaires Reels pour fine-tuning T-types."
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mode test : pas d'Instagram, pas d'écriture (no-op visible).",
    )
    parser.add_argument(
        "--pending",
        action="store_true",
        help="Force le balayage de pending_collection.json (collectes différées).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if args.mock and not args.pending:
        print("ℹ️  Mode mock — aucune action (utilisez --pending pour forcer le check).")
        print(f"   training_path : {DEFAULT_TRAINING_PATH}")
        print(f"   pending_path  : {DEFAULT_PENDING_PATH}")
        return

    if args.pending:
        stats = check_pending_collections(mock=args.mock)
        print(
            f"✅ check_pending_collections — checked={stats['checked']} | "
            f"collected={stats['collected']} | skipped={stats['skipped']} | "
            f"removed={stats['removed']}"
        )
        return

    print("Aucune action — utilisez --pending ou --mock.")


if __name__ == "__main__":
    _main_cli()


__all__ = [
    "COLLECT_AGE_MIN_DAYS",
    "COMMENTS_PER_MEDIA",
    "DEFAULT_PENDING_PATH",
    "DEFAULT_TRAINING_PATH",
    "DatasetIOError",
    "MAX_REELS_TO_COLLECT",
    "PENDING_RETRY_DAYS",
    "TOP_COMMENTS_PER_MEDIA",
    "check_pending_collections",
    "collect_training_data",
    "schedule_pending_collection",
]
