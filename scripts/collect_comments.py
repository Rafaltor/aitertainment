"""collect_comments.py — collecte brute de commentaires Reels (watchlist).

Script autonome : ne dépend ni de ``discovery.py``, ni de ``watcher.py``,
ni de ``embedder.py``. Sortie : ``data/raw_comments.json`` (sans T-type).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REELS_PER_ACCOUNT = 3
COMMENTS_PER_REEL = 50
MIN_WORDS = 4
RAW_COMMENTS_PATH = Path("data/raw_comments.json")
WATCHLIST_PATH = Path("data/watchlist.json")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG = logging.getLogger("aitertainment.collect_comments")
_HASHTAG_RE = re.compile(r"(?:^|\s)#(\w+)")
_ALPHA_WORD_RE = re.compile(r"[a-zA-ZÀ-ÿ]{2,}")


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_watchlist(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Charge la watchlist et ne garde que les comptes validés."""
    p = _resolve_path(Path(path) if path is not None else WATCHLIST_PATH)
    if not p.exists():
        raise FileNotFoundError(f"watchlist absente : {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("creators") or data.get("watchlist") or []
    else:
        raise ValueError(f"racine watchlist invalide dans {p}")

    if not isinstance(entries, list):
        raise ValueError(f"entrées watchlist invalides dans {p}")

    out: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        action = entry.get("action")
        if action is not None and action != "validated":
            continue
        out.append(entry)
    return out


def is_valid_comment(text: str, like_count: Any) -> bool:
    """Filtre strict des commentaires exploitables pour le dataset brut."""
    if like_count is None:
        return False
    try:
        likes = int(like_count)
    except (TypeError, ValueError):
        return False
    if likes < 0:
        return False

    cleaned = (text or "").strip()
    if len(cleaned.split()) <= MIN_WORDS:
        return False

    lower = cleaned.lower()
    if "http" in lower or "www" in lower or ".com" in lower:
        return False

    if len(_ALPHA_WORD_RE.findall(cleaned)) < 2:
        return False
    return True


def extract_hashtags(caption: str) -> list[str]:
    """Extrait les hashtags d'une caption Instagram."""
    if not caption:
        return []
    return _HASHTAG_RE.findall(caption)


def build_dedup_key(media_id: str, text: str) -> str:
    """Clé de déduplication ``media_id + texte normalisé``."""
    return f"{media_id}||{text.strip().lower()}"


def load_raw_comments(
    path: Path | str | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Charge ``raw_comments.json`` et construit le set de dédup."""
    p = _resolve_path(Path(path) if path is not None else RAW_COMMENTS_PATH)
    if not p.exists():
        return [], set()

    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        entries = [entry for entry in data if isinstance(entry, dict)]
    elif isinstance(data, dict):
        raw_entries = data.get("entries") or data.get("comments") or []
        entries = [entry for entry in raw_entries if isinstance(entry, dict)]
    else:
        entries = []

    dedup_keys = {
        build_dedup_key(str(entry.get("media_id") or ""), str(entry.get("text") or ""))
        for entry in entries
        if entry.get("media_id") and entry.get("text")
    }
    return entries, dedup_keys


def save_raw_comments(
    entries: list[dict[str, Any]],
    path: Path | str | None = None,
) -> None:
    """Écrit ``raw_comments.json`` de façon atomique."""
    p = _resolve_path(Path(path) if path is not None else RAW_COMMENTS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, p)


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _is_reel(media: Any) -> bool:
    media_type = str(_attr(media, "media_type", "") or "").lower()
    product_type = str(_attr(media, "product_type", "") or "").lower()
    return media_type in {"clip", "clips", "reel"} or product_type == "clips"


def _comment_like_count(comment: Any) -> Any:
    if isinstance(comment, dict):
        if "like_count" not in comment:
            return None
        return comment.get("like_count")
    return getattr(comment, "like_count", None)


def _comment_text(comment: Any) -> str:
    text = _attr(comment, "text", None)
    if text is None:
        text = _attr(comment, "comment", "")
    return str(text or "")


def _media_comment_to_like_ratio(media: Any) -> float:
    likes = int(_attr(media, "like_count", 0) or 0)
    comments = int(_attr(media, "comment_count", 0) or 0)
    if likes <= 0:
        return 0.0
    return float(comments) / float(likes)


def _audio_id(media: Any) -> str:
    audio = _attr(media, "audio", None)
    if audio is None:
        return ""
    if isinstance(audio, dict):
        return str(audio.get("id") or audio.get("pk") or "")
    return str(getattr(audio, "id", None) or getattr(audio, "pk", None) or audio or "")


def collect_for_account(
    username: str,
    creator: dict[str, Any],
    client: Any,
    reels_per_account: int,
    comments_per_reel: int,
    dedup_keys: set[str],
) -> list[dict[str, Any]]:
    """Collecte les commentaires bruts d'un compte watchlisté."""
    from instagram_client import polite_sleep

    uname = str(username or "").lstrip("@").strip()
    if not uname:
        return []

    user_id = client.user_id_from_username(uname)
    medias = client.user_medias(str(user_id), amount=reels_per_account)
    reels = [media for media in medias or [] if _is_reel(media)]
    if not reels:
        _LOG.info("Collecte @%s : aucun Reel récent — skip.", uname)
        return []

    niches = creator.get("niches") or ["humour"]
    collected: list[dict[str, Any]] = []
    kept = 0
    seen = 0

    for index, media in enumerate(reels):
        media_id = str(_attr(media, "pk", "") or _attr(media, "id", "") or "")
        caption = str(_attr(media, "caption_text", "") or _attr(media, "caption", "") or "")
        views = int(_attr(media, "view_count", 0) or 0)
        ratio = _media_comment_to_like_ratio(media)
        hashtags = extract_hashtags(caption)
        audio_id = _audio_id(media)

        raw_comments = client.media_comments(media_id, amount=comments_per_reel)
        for comment in raw_comments or []:
            seen += 1
            text = _comment_text(comment)
            like_count = _comment_like_count(comment)
            if not is_valid_comment(text, like_count):
                continue

            dedup_key = build_dedup_key(media_id, text)
            if dedup_key in dedup_keys:
                continue

            dedup_keys.add(dedup_key)
            kept += 1
            collected.append(
                {
                    "media_id": media_id,
                    "username": uname,
                    "niches": list(niches),
                    "text": text.strip(),
                    "comment_likes": int(like_count),
                    "views": views,
                    "comment_to_like_ratio": ratio,
                    "caption": caption,
                    "hashtags": hashtags,
                    "audio_id": audio_id,
                    "collected_at": _utc_now_iso(),
                }
            )

        if index < len(reels) - 1:
            polite_sleep(min_s=2, max_s=2)

    _LOG.info(
        "Collecte @%s : %d/%d commentaires retenus sur %d reels",
        uname,
        kept,
        seen,
        len(reels),
    )
    return collected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collecte brute de commentaires Reels.")
    parser.add_argument("--account", help="Traiter un seul compte (@username).")
    parser.add_argument(
        "--reels-per-account",
        type=int,
        default=REELS_PER_ACCOUNT,
        help=f"Nombre de Reels par compte (défaut {REELS_PER_ACCOUNT}).",
    )
    parser.add_argument(
        "--comments-per-reel",
        type=int,
        default=COMMENTS_PER_REEL,
        help=f"Nombre de commentaires par Reel (défaut {COMMENTS_PER_REEL}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche le plan sans écrire sur disque.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        creators = load_watchlist()
    except FileNotFoundError as exc:
        _LOG.error("%s", exc)
        return 1
    except ValueError as exc:
        _LOG.error("%s", exc)
        return 1

    if not creators:
        _LOG.error("Watchlist vide — rien à collecter.")
        return 2

    if args.account:
        target = args.account.lstrip("@").strip().lower()
        creators = [
            creator
            for creator in creators
            if str(creator.get("username") or "").lstrip("@").strip().lower() == target
        ]
        if not creators:
            _LOG.error("Compte @%s introuvable dans la watchlist.", target)
            return 1

    entries, dedup_keys = load_raw_comments()
    added = 0

    if args.dry_run:
        for creator in creators:
            username = str(creator.get("username") or "").lstrip("@").strip()
            _LOG.info("DRY-RUN : collecterait les commentaires de @%s", username)
        _LOG.info("=== Collecte terminée : %d nouveaux commentaires ajoutés ===", added)
        return 0

    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from instagram_client import get_client, polite_sleep

    client = get_client()

    for index, creator in enumerate(creators):
        username = str(creator.get("username") or "").lstrip("@").strip()
        new_entries = collect_for_account(
            username,
            creator,
            client,
            args.reels_per_account,
            args.comments_per_reel,
            dedup_keys,
        )
        entries.extend(new_entries)
        added += len(new_entries)
        if index < len(creators) - 1:
            polite_sleep(min_s=5, max_s=5)

    save_raw_comments(entries)
    _LOG.info("=== Collecte terminée : %d nouveaux commentaires ajoutés ===", added)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
