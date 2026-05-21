"""embedder.py — embeddings de profils watchlistés (LM Studio + Whisper).

Script autonome : ne dépend ni de ``discovery.py`` ni de ``watcher.py``.
Embeddings : LM Studio uniquement (``LM_STUDIO_URL`` + ``LM_STUDIO_EMBED_MODEL``).
Commentaires : tous dans ``raw_comments.json`` ; scrape Playwright (3 reels) si besoin.
Captions / transcripts : Playwright + Whisper. Voir ``main()`` pour le CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from playwright.sync_api import BrowserContext, Page, sync_playwright

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import LM_STUDIO_EMBED_MODEL, LM_STUDIO_URL
from scripts.instagram_browser import (
    RAW_COMMENTS_PATH as _RAW_COMMENTS_PATH,
    _list_reel_page_aria_labels,
    _unescape_json_string_fragment,
    click_reel_comment_button,
    collect_top_comments,
    extract_reel_caption_from_dom,
    extract_reel_comments_panel_text,
    get_browser_context,
    get_profile_data,
    get_recent_reels,
    navigate_to_reel_page,
    open_reels_grid,
    parse_comments_from_dom_text,
    polite_sleep,
    return_to_reels_grid,
)

WHISPER_MODEL_SIZE = "small"
REELS_PER_ACCOUNT = 5
RAW_COMMENTS_PATH = _RAW_COMMENTS_PATH
VECTOR_STORE_PATH = Path("data/vector_store.json")
WATCHLIST_PATH = Path("data/watchlist.json")
DATABASE_PATH = Path("data/database.json")
PCA_MODEL_PATH = Path("data/pca_model.pkl")

NAMED_AXES = [
    "scripted_vs_raw",
    "solo_vs_collab",
    "fictional_vs_real",
    "energy_level",
    "production_quality",
    "format_length",
    "distance_parasociale",
    "interaction_style",
    "mainstream_vs_niche",
    "safe_vs_edgy",
]

_LOG = logging.getLogger("aitertainment.embedder")
_HASHTAG_RE = re.compile(r"#(\w+)")
_WHISPER_LOGGERS_QUIETED = False


def _quiet_whisper_logs() -> None:
    """Réduit le bruit faster-whisper / Hugging Face / ctranslate2 dans le terminal."""
    global _WHISPER_LOGGERS_QUIETED
    if _WHISPER_LOGGERS_QUIETED:
        return
    for name in (
        "ctranslate2",
        "faster_whisper",
        "httpx",
        "httpcore",
        "huggingface_hub",
        "filelock",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("ctranslate2").setLevel(logging.ERROR)
    _WHISPER_LOGGERS_QUIETED = True


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_raw_comments_entries(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Charge ``raw_comments.json`` (liste ou ``{"entries": [...]}``)."""
    p = _resolve_path(Path(path) if path is not None else RAW_COMMENTS_PATH)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    if isinstance(data, dict):
        entries = data.get("entries") or []
        if isinstance(entries, list):
            return [e for e in entries if isinstance(e, dict)]
    return []


def _raw_comment_row(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise une entrée ``raw_comments.json`` pour l'embedding."""
    text = str(entry.get("text") or "").strip()
    if not text:
        return None
    return {
        "text": text,
        "comment_likes": int(entry.get("comment_likes") or 0),
        "media_id": str(entry.get("media_id") or "").strip(),
    }


def comments_for_account_from_raw(
    username: str,
    path: Path | str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Tous les commentaires Discovery pour ce compte (aucun plafond ni filtre).

    Retourne ``(liste complète, index par media_id)``, triés par ``comment_likes`` décroissant.
    """
    uname = str(username or "").lstrip("@").strip().lower()
    if not uname:
        return [], {}

    rows: list[dict[str, Any]] = []
    for entry in load_raw_comments_entries(path):
        if str(entry.get("username") or "").lstrip("@").strip().lower() != uname:
            continue
        row = _raw_comment_row(entry)
        if row:
            rows.append(row)

    rows.sort(key=lambda r: int(r.get("comment_likes") or 0), reverse=True)

    by_media: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        mid = str(row.get("media_id") or "").strip()
        if mid:
            by_media.setdefault(mid, []).append(row)

    return rows, by_media


def ensure_comments_for_account(
    username: str,
    reels: list[dict[str, Any]],
    context: BrowserContext,
    niches: list[str] | str,
    *,
    raw_path: Path | str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], str | None]:
    """Commentaires depuis ``raw_comments.json`` ; scrape complémentaire sur 3 reels si besoin."""
    path = _resolve_path(Path(raw_path) if raw_path is not None else RAW_COMMENTS_PATH)
    uname = str(username or "").lstrip("@").strip()
    comments, by_media = comments_for_account_from_raw(uname, path)
    if comments:
        _LOG.info(
            "@%s : %d commentaire(s) en base — scrape complémentaire (3 reels).",
            uname,
            len(comments),
        )
    else:
        _LOG.info(
            "@%s : aucun commentaire en base — collecte Playwright (3 reels).",
            uname,
        )

    n = collect_top_comments(
        uname,
        context,
        reels,
        niches,
        raw_comments_path=path,
        classify=False,
        logger=_LOG,
    )
    comments, by_media = comments_for_account_from_raw(uname, path)
    if comments:
        source = "raw_comments.json" if n == 0 else "raw_comments.json+playwright"
        if n > 0:
            _LOG.info(
                "@%s : %d commentaire(s) au total (%d nouveau(x)) → %s",
                uname,
                len(comments),
                n,
                path,
            )
        return comments, by_media, source

    _LOG.warning("@%s : aucun commentaire après collecte.", uname)
    return [], {}, None


def format_comment_for_embedding(comment: dict[str, Any] | str) -> str:
    """Ligne commentaire pour l'embedding : texte + nombre de likes."""
    if isinstance(comment, str):
        return comment.strip()
    text = str(comment.get("text") or "").strip()
    if not text:
        return ""
    likes = int(comment.get("comment_likes") or 0)
    return f"[{likes} likes] {text}"


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


def load_creators_from_database(
    path: Path | str | None = None,
    *,
    tier: str | None = None,
) -> list[dict[str, Any]]:
    """Charge les profils non archivés depuis ``database.json``.

    Si ``tier`` est ``A``, ``B`` ou ``C``, ne garde que ce tier.
    """
    p = _resolve_path(Path(path) if path is not None else DATABASE_PATH)
    if not p.exists():
        raise FileNotFoundError(f"database absente : {p}")

    tier_filter = str(tier).strip().upper() if tier else None
    if tier_filter and tier_filter not in ("A", "B", "C"):
        raise ValueError(f"tier invalide : {tier!r} (attendu A, B ou C)")

    data = json.loads(p.read_text(encoding="utf-8"))
    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f'"profiles" invalide dans {p}')

    out: list[dict[str, Any]] = []
    for username, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        if profile.get("archived", False):
            continue
        profile_tier = str(profile.get("tier") or "C").strip().upper()
        if tier_filter and profile_tier != tier_filter:
            continue
        u = str(username).lstrip("@").strip().lower()
        if not u:
            continue
        out.append(
            {
                "username": u,
                "action": "validated",
                "niches": profile.get("niches") or [],
                "t_type": profile.get("t_type_final") or profile.get("t_type_original"),
                "followers": profile.get("followers", 0),
                "tier": profile.get("tier", "C"),
            }
        )
    return out


def load_creators(
    source: str = "watchlist",
    *,
    watchlist_path: Path | str | None = None,
    database_path: Path | str | None = None,
    tier: str | None = None,
) -> list[dict[str, Any]]:
    """Charge la liste des créateurs selon ``source`` (``watchlist`` ou ``database``)."""
    if source == "watchlist":
        return load_watchlist(watchlist_path)
    if source == "database":
        return load_creators_from_database(database_path, tier=tier)
    raise ValueError(f"source inconnue : {source!r} (attendu watchlist ou database)")


def export_playwright_cookies(context: BrowserContext, cookie_file: Path) -> None:
    """Exporte les cookies Playwright au format Netscape pour yt-dlp."""
    cookies = context.cookies()
    with cookie_file.open("w", encoding="utf-8") as fh:
        fh.write("# Netscape HTTP Cookie File\n")
        for c in cookies:
            domain = c["domain"]
            flag = "TRUE" if domain.startswith(".") else "FALSE"
            secure = "TRUE" if c.get("secure") else "FALSE"
            expires = c.get("expires", 0)
            expiry = int(expires) if expires and expires > 0 else 0
            fh.write(
                f"{domain}\t{flag}\t{c['path']}\t{secure}\t{expiry}\t"
                f"{c['name']}\t{c['value']}\n"
            )


def download_audio_from_reel(
    media_id: str, context: BrowserContext, tmp_dir: Path
) -> Path | None:
    """Télécharge l'audio d'un Reel via yt-dlp et les cookies Playwright."""
    cookie_file = tmp_dir / "cookies.txt"
    export_playwright_cookies(context, cookie_file)

    wav_path = tmp_dir / f"{media_id}.wav"
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--cookies",
                str(cookie_file),
                "--extract-audio",
                "--audio-format",
                "wav",
                "--audio-quality",
                "0",
                "-o",
                str(tmp_dir / "%(id)s.%(ext)s"),
                "--quiet",
                f"https://www.instagram.com/reel/{media_id}/",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        _LOG.warning("yt-dlp absent — transcription audio ignorée pour %s.", media_id)
        return None
    except subprocess.TimeoutExpired:
        _LOG.warning("yt-dlp timeout pour %s.", media_id)
        return None

    if result.returncode != 0:
        stderr = (result.stderr or "")[:200]
        _LOG.warning("yt-dlp échoué pour %s : %s", media_id, stderr)
        return None

    if wav_path.exists():
        return wav_path
    wav_files = sorted(tmp_dir.glob("*.wav"))
    return wav_files[0] if wav_files else None


_OG_UI_NOISE = ("Ne pas suggérer", "View this reel", "Watch on Instagram")


def _is_usable_caption(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 2:
        return False
    if any(noise in t for noise in _OG_UI_NOISE):
        return False
    return True


def _extract_caption_from_og_description(og_desc: str) -> str:
    """Extrait la caption depuis ``og:description`` (guillemets optionnels en fin)."""
    text = (og_desc or "").strip()
    if not text:
        return ""

    match = re.search(r':\s*"(.+?)"?\s*$', text)
    if not match:
        match = re.search(r':\s*"(.+)', text)
    if match:
        candidate = match.group(1).strip().rstrip('"')
        if _is_usable_caption(candidate):
            return candidate

    if ": " in text:
        tail = text.rsplit(": ", 1)[-1].strip().strip('"').strip()
        if _is_usable_caption(tail) and "likes" not in tail[:30].lower():
            return tail
    return ""


def _caption_from_graphql_text_window(text: str, media_id: str) -> str:
    """Repli regex sur une fenêtre JSON (comme ``get_reel_caption``)."""
    mid = str(media_id or "").strip()
    if not mid or mid not in text:
        return ""
    idx = text.find(mid)
    window = text[idx : idx + 3000]
    patterns = (
        r'"caption"\s*:\s*\{\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"',
        r'"caption_text"\s*:\s*"((?:[^"\\]|\\.)*)"',
        r'"(?:caption_text|text)"\s*:\s*"((?:[^"\\]|\\.){2,500})"',
    )
    for pattern in patterns:
        for raw in re.findall(pattern, window):
            decoded = _unescape_json_string_fragment(raw)
            if _is_usable_caption(decoded):
                return decoded
    return ""


def _walk_graphql_json_for_caption(node: Any, media_id: str, *, depth: int = 0) -> str:
    """Parcourt un JSON GraphQL et retourne la caption du reel ciblé."""
    if depth > 15:
        return ""
    mid = str(media_id or "").strip()
    if isinstance(node, dict):
        code = node.get("code") or node.get("shortcode")
        code_s = str(code or "").strip()
        cap = node.get("caption")
        if code_s == mid and isinstance(cap, dict):
            t = str(cap.get("text") or "").strip()
            if _is_usable_caption(t):
                return t
        cap_text = node.get("caption_text")
        if code_s == mid and cap_text is not None:
            t = str(cap_text).strip()
            if _is_usable_caption(t):
                return t
        for value in node.values():
            found = _walk_graphql_json_for_caption(value, media_id, depth=depth + 1)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _walk_graphql_json_for_caption(item, media_id, depth=depth + 1)
            if found:
                return found
    return ""


def _extract_comments_from_reel_page(
    page: Page,
    *,
    username: str = "",
    media_id: str = "",
) -> list[str]:
    """Ouvre le panneau commentaires et retourne les textes parsés."""
    u = username.lstrip("@").strip() or "?"
    mid = media_id or "?"
    try:
        clicked = click_reel_comment_button(page)
        if not clicked:
            labels = _list_reel_page_aria_labels(page)
            _LOG.warning(
                "Commentaires @%s reel %s : bouton absent. aria-labels sur page : %s",
                u,
                mid,
                labels[:25] or "(aucun)",
            )
            return []
        page.wait_for_timeout(3500)
        panel_text = extract_reel_comments_panel_text(page)
    except Exception as e:
        _LOG.warning(
            "Commentaires @%s reel %s : extraction DOM échouée (%s).",
            u,
            mid,
            e,
        )
        return []

    parsed = parse_comments_from_dom_text(str(panel_text or ""))
    texts = [str(c.get("text") or "").strip() for c in parsed if c.get("text")]
    if not texts and clicked:
        preview = str(panel_text or "").strip().replace("\n", " ")[:120]
        _LOG.warning(
            "Commentaires @%s reel %s : panneau ouvert (%s) mais 0 commentaire parsé "
            "(aperçu DOM : %r).",
            u,
            mid,
            clicked,
            preview or "(vide)",
        )
    return texts


def _read_caption_from_loaded_reel_page(
    page: Page,
    media_id: str,
    *,
    json_caption: str = "",
) -> str:
    """Lit la caption sur une page reel déjà ouverte."""
    mid = str(media_id or "").strip()
    caption = ""
    try:
        og_desc = (
            page.get_attribute(
                'meta[property="og:description"]',
                "content",
                timeout=5000,
            )
            or ""
        )
    except Exception:
        og_desc = ""
    caption = _extract_caption_from_og_description(og_desc)
    if not caption:
        caption = json_caption
    if not caption:
        caption = extract_reel_caption_from_dom(page)
    if caption:
        caption = caption.strip().rstrip('".').strip()
    return caption


def _make_graphql_caption_listener(media_id: str) -> tuple[Any, list[str]]:
    """Retourne ``(handler, holder)`` — ``holder[0]`` reçoit la caption GraphQL."""
    mid = str(media_id or "").strip()
    holder: list[str] = [""]

    def on_response(response: Any) -> None:
        if holder[0] or "graphql" not in response.url:
            return
        try:
            body = response.text()
        except Exception:
            return
        found = _caption_from_graphql_text_window(body, mid)
        if found:
            holder[0] = found
            return
        try:
            data = json.loads(body)
        except Exception:
            return
        found = _walk_graphql_json_for_caption(data, mid)
        if found:
            holder[0] = found

    return on_response, holder


def _fetch_reel_captions_batch(
    username: str,
    reels: list[dict[str, Any]],
    context: BrowserContext,
) -> dict[str, str]:
    """Captions via une seule session sur ``/{user}/reels/`` (évite les rechargements)."""
    uname = str(username or "").lstrip("@").strip()
    captions: dict[str, str] = {}
    if not uname or not reels:
        return captions

    page = context.new_page()
    try:
        if not open_reels_grid(page, uname):
            return captions

        for reel in reels:
            mid = str(reel.get("media_id") or "").strip()
            if not mid:
                continue
            existing = str(reel.get("caption") or "").strip()
            if existing:
                captions[mid] = existing
                continue

            handler, holder = _make_graphql_caption_listener(mid)
            page.on("response", handler)
            try:
                if not navigate_to_reel_page(
                    page, mid, uname, reels_grid_loaded=True
                ):
                    continue
                captions[mid] = _read_caption_from_loaded_reel_page(
                    page, mid, json_caption=holder[0]
                )
            finally:
                page.remove_listener("response", handler)
                return_to_reels_grid(page, uname)
    finally:
        page.close()

    return captions


def _visit_reel_page(
    media_id: str,
    context: BrowserContext,
    *,
    username: str = "",
    existing_caption: str = "",
    fetch_comments: bool = False,
) -> tuple[str, list[str]]:
    """Une visite /reel/ pour la caption ; commentaires via ``raw_comments.json``."""
    mid = str(media_id or "").strip()
    if not mid:
        return "", []

    caption = (existing_caption or "").strip()
    handler, holder = _make_graphql_caption_listener(mid)

    page = context.new_page()
    try:
        page.on("response", handler)
        if not navigate_to_reel_page(page, mid, username):
            return caption, []

        if not caption:
            caption = _read_caption_from_loaded_reel_page(
                page, mid, json_caption=holder[0]
            )

        if fetch_comments:
            reel_comments = _extract_comments_from_reel_page(
                page, username=username, media_id=mid
            )
            return caption, reel_comments
        return caption, []
    finally:
        page.close()


def _get_caption_from_reel_page(
    media_id: str,
    context: BrowserContext,
    *,
    username: str = "",
) -> str:
    """Récupère la caption d'un Reel (og + GraphQL) — préférer ``_visit_reel_page``."""
    caption, _ = _visit_reel_page(
        media_id, context, username=username, fetch_comments=False
    )
    return caption


def transcribe_audio(wav_path: Path | str | None) -> str:
    """Transcrit un WAV via faster-whisper (import lazy)."""
    if wav_path is None:
        return ""
    path = Path(wav_path)
    if not path.exists():
        return ""

    _quiet_whisper_logs()
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        _LOG.warning("faster-whisper indisponible (%s) — transcription ignorée.", e)
        return ""

    try:
        model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu")
        segments, _info = model.transcribe(str(path), language="fr")
        return " ".join(segment.text.strip() for segment in segments if segment.text).strip()
    except Exception as e:
        _LOG.warning("transcription échouée pour %s (%s).", path, e)
        return ""


def build_input_text(
    caption: str,
    hashtags: list[str] | str,
    transcript: str,
    comments: list[dict[str, Any]] | list[str],
    *,
    biography: str = "",
    niches: list[str] | None = None,
) -> str:
    """Assemble le texte unifié envoyé au modèle d'embedding LM Studio."""
    niches_str = ", ".join(str(n).strip() for n in (niches or ["humour"]) if str(n).strip())
    if not niches_str:
        niches_str = "humour"

    caption_text = (caption or "").strip() or "(vide)"
    if isinstance(hashtags, list):
        hashtags_text = " ".join(h.strip() for h in hashtags if str(h).strip())
    else:
        hashtags_text = str(hashtags or "").strip()
    hashtags_text = hashtags_text or "(vide)"
    transcript_text = (transcript or "").strip() or "(vide)"
    if comments:
        lines = [
            format_comment_for_embedding(c)
            for c in comments
            if format_comment_for_embedding(c)
        ]
        comments_text = "\n".join(lines) if lines else "(vide)"
    else:
        comments_text = "(vide)"

    parts = [f"[NICHES] {niches_str}"]
    bio = (biography or "").strip()
    if bio:
        parts.append(f"[BIOGRAPHY] {bio}")
    parts.extend(
        [
            f"[CAPTION] {caption_text}",
            f"[HASHTAGS] {hashtags_text}",
            f"[TRANSCRIPT] {transcript_text}",
            f"[COMMENTS_RECEIVED] {comments_text}",
        ]
    )
    return "\n".join(parts) + "\n"


def _validate_embedding_vector(vector: list[float]) -> list[float] | None:
    if any(math.isnan(x) for x in vector):
        _LOG.warning("embedding NaN détecté — entrée ignorée.")
        return None
    return vector


def embedding_config_ok() -> bool:
    """True si LM Studio est configuré pour les embeddings."""
    return bool(LM_STUDIO_URL and LM_STUDIO_EMBED_MODEL)


def embed_text(text: str) -> list[float] | None:
    """Embedding via LM Studio (``/v1/embeddings``)."""
    if not embedding_config_ok():
        _LOG.error(
            "LM Studio non configuré — définis LM_STUDIO_URL et "
            "LM_STUDIO_EMBED_MODEL dans .env (voir .env.example)."
        )
        return None

    payload_text = text[:8000]
    _LOG.debug(
        "embed_text : modèle=%s, %d caractères",
        LM_STUDIO_EMBED_MODEL,
        len(payload_text),
    )
    try:
        resp = requests.post(
            f"{LM_STUDIO_URL}/embeddings",
            json={"model": LM_STUDIO_EMBED_MODEL, "input": payload_text},
            timeout=120,
        )
        if resp.status_code == 400:
            _LOG.error(
                "LM Studio embed 400 — texte longueur=%d, début=%s",
                len(payload_text),
                payload_text[:100],
            )
            return None
        resp.raise_for_status()
        data = resp.json()
        vector = [float(x) for x in data["data"][0]["embedding"]]
        return _validate_embedding_vector(vector)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 400:
            _LOG.error(
                "LM Studio embed 400 — texte longueur=%d, début=%s",
                len(payload_text),
                payload_text[:100],
            )
        else:
            _LOG.error("LM Studio embed a échoué (%s).", e)
        return None
    except Exception as e:
        _LOG.error("LM Studio embed a échoué (%s).", e)
        return None


def _pca_input_dim(pca_model: Any) -> int | None:
    dim = getattr(pca_model, "n_features_in_", None)
    return int(dim) if dim is not None else None


def _infer_embedding_dim(vector_store: list[dict[str, Any]]) -> int | None:
    """Déduit la dimension des embeddings (store existant ou sonde LM Studio)."""
    for entry in vector_store:
        raw = entry.get("embedding_raw")
        if isinstance(raw, list) and raw:
            return len(raw)
    probe = embed_text("dimension probe")
    return len(probe) if probe else None


def project_to_named_axes(
    embedding: list[float],
    pca_model: Any | None,
) -> dict[str, float]:
    """Projette un embedding sur les 10 axes nommés via PCA."""
    if pca_model is None:
        _LOG.warning(
            "PCA non disponible — named_axes non calculés (besoin de 10+ comptes)"
        )
        return {axis: 0.0 for axis in NAMED_AXES}

    expected = _pca_input_dim(pca_model)
    if expected is not None and len(embedding) != expected:
        _LOG.warning(
            "PCA ignorée : embedding %dD vs pca_model.pkl %dD — named_axes à 0. "
            "Supprime data/pca_model.pkl ou relance avec ≥10 comptes à la même "
            "dimension pour recalculer la PCA.",
            len(embedding),
            expected,
        )
        return {axis: 0.0 for axis in NAMED_AXES}

    projected = pca_model.transform([embedding])[0]
    min_v = float(min(projected))
    max_v = float(max(projected))
    if math.isclose(max_v, min_v):
        normalized = [0.5 for _ in projected]
    else:
        normalized = [(float(v) - min_v) / (max_v - min_v) for v in projected]

    return {
        axis: float(value)
        for axis, value in zip(NAMED_AXES, normalized, strict=True)
    }


def fit_pca(
    vector_store: list[dict[str, Any]],
    *,
    target_dim: int | None = None,
) -> Any | None:
    """Ajuste une PCA 10D sur les ``embedding_raw`` (dimension homogène)."""
    by_dim: dict[int, list[list[float]]] = {}
    for entry in vector_store:
        raw = entry.get("embedding_raw")
        if isinstance(raw, list) and raw:
            dim = len(raw)
            by_dim.setdefault(dim, []).append([float(x) for x in raw])

    if not by_dim:
        return None

    if target_dim is None:
        target_dim = max(by_dim, key=lambda d: len(by_dim[d]))
    vectors = by_dim.get(target_dim, [])
    if len(by_dim) > 1:
        _LOG.warning(
            "fit_pca : dimensions mixtes %s — seuls les vecteurs %dD sont utilisés.",
            {d: len(v) for d, v in by_dim.items()},
            target_dim,
        )

    if len(vectors) < 10:
        _LOG.warning(
            "PCA non recalculée : %d compte(s) en %dD (10 minimum requis).",
            len(vectors),
            target_dim,
        )
        return None

    from sklearn.decomposition import PCA

    model = PCA(n_components=10)
    model.fit(vectors)

    pca_path = _resolve_path(PCA_MODEL_PATH)
    pca_path.parent.mkdir(parents=True, exist_ok=True)
    with pca_path.open("wb") as fh:
        pickle.dump(model, fh)
    return model


def load_pca(*, expected_dim: int | None = None) -> Any | None:
    """Charge le modèle PCA depuis le disque si compatible avec ``expected_dim``."""
    pca_path = _resolve_path(PCA_MODEL_PATH)
    if not pca_path.exists():
        return None
    with pca_path.open("rb") as fh:
        model = pickle.load(fh)
    if expected_dim is None:
        return model
    got = _pca_input_dim(model)
    if got is not None and got != expected_dim:
        _LOG.warning(
            "pca_model.pkl ignoré (%dD enregistré, %dD attendu). "
            "Recalcul auto après ≥10 comptes à la même dimension.",
            got,
            expected_dim,
        )
        return None
    return model


def load_vector_store(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Charge ``vector_store.json`` ou retourne une liste vide."""
    p = _resolve_path(Path(path) if path is not None else VECTOR_STORE_PATH)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    if isinstance(data, dict):
        entries = data.get("entries") or data.get("profiles") or []
        if isinstance(entries, list):
            return [entry for entry in entries if isinstance(entry, dict)]
    return []


def save_vector_store(entries: list[dict[str, Any]], path: Path | str | None = None) -> None:
    """Écrit ``vector_store.json`` de façon atomique."""
    p = _resolve_path(Path(path) if path is not None else VECTOR_STORE_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, p)


def collect_account_content(
    username: str,
    creator: dict[str, Any],
    context: BrowserContext,
    tmp_dir: Path,
) -> dict[str, Any] | None:
    """Collecte reels, captions, transcripts, commentaires et texte d'entrée."""
    uname = str(username or "").lstrip("@").strip()
    if not uname:
        return None

    reels = get_recent_reels(uname, context, max_reels=REELS_PER_ACCOUNT)
    if not reels:
        _LOG.warning("@%s : aucun Reel récupéré.", uname)
        return None

    profile_data = get_profile_data(uname, context)
    biography = str(profile_data.get("biography") or "").strip() if profile_data else ""
    niches_raw = creator.get("niches") or ["humour"]
    niches = niches_raw if isinstance(niches_raw, list) else [str(niches_raw)]

    transcripts: list[str] = []
    comments, comments_by_media, comments_source = ensure_comments_for_account(
        uname, reels, context, niches
    )
    if comments:
        _LOG.info(
            "@%s : %d commentaire(s) (%s) — %d reel(s) source(s) distinct(s).",
            uname,
            len(comments),
            comments_source or "?",
            len(comments_by_media),
        )

    reel_details: list[dict[str, Any]] = []
    captions_by_id = _fetch_reel_captions_batch(uname, reels, context)

    for reel in reels:
        media_id = str(reel.get("media_id") or "").strip()
        if not media_id:
            continue

        caption = captions_by_id.get(media_id) or str(reel.get("caption") or "").strip()
        if caption:
            reel["caption"] = caption
        else:
            _LOG.warning("@%s reel %s : caption non récupérée.", uname, media_id)

        wav_path = download_audio_from_reel(media_id, context, tmp_dir)
        transcript = transcribe_audio(wav_path)
        if transcript:
            transcripts.append(transcript)

        reel_comments = list(comments_by_media.get(media_id) or [])

        polite_sleep(1)

        reel_details.append(
            {
                "media_id": media_id,
                "caption": caption,
                "transcript": transcript,
                "comments": reel_comments,
                "view_count": reel.get("view_count"),
                "like_count": reel.get("like_count"),
                "comment_count": reel.get("comment_count"),
            }
        )

        for path in tmp_dir.glob("*"):
            if path.is_file():
                path.unlink(missing_ok=True)

    captions = [str(r.get("caption") or "").strip() for r in reels if r.get("caption")]
    caption_blob = "\n".join(captions)
    hashtags: list[str] = []
    for caption in captions:
        hashtags.extend(_HASHTAG_RE.findall(caption))
    hashtags = list(dict.fromkeys(hashtags))

    transcript_blob = "\n---\n".join(transcripts)
    input_text = build_input_text(
        caption_blob,
        hashtags,
        transcript_blob,
        comments,
        biography=biography,
        niches=niches,
    )
    return {
        "username": uname,
        "reels": reel_details,
        "biography": biography,
        "niches": niches,
        "captions": captions,
        "hashtags": hashtags,
        "transcripts": transcripts,
        "comments": comments,
        "input_text": input_text,
        "summary": {
            "reels_analyzed": len(reels),
            "captions_count": len(captions),
            "transcripts_count": len(transcripts),
            "comments_received_count": len(comments),
            "comments_source": comments_source,
            "hashtags_count": len(hashtags),
            "biography_present": bool(biography),
        },
    }


def _truncate_preview(text: str, limit: int = 280) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= limit:
        return t or "(vide)"
    return t[: limit - 3] + "..."


def print_verify_report(report: dict[str, Any]) -> None:
    """Affiche un rapport lisible de la collecte (sans embedding)."""
    u = report.get("username", "?")
    summary = report.get("summary") or {}
    _LOG.info("=== Vérification @%s ===", u)
    _LOG.info(
        "Résumé : reels=%d captions=%d transcripts=%d comments=%d hashtags=%d bio=%s",
        summary.get("reels_analyzed", 0),
        summary.get("captions_count", 0),
        summary.get("transcripts_count", 0),
        summary.get("comments_received_count", 0),
        summary.get("hashtags_count", 0),
        "oui" if summary.get("biography_present") else "non",
    )
    bio = str(report.get("biography") or "").strip()
    if bio:
        _LOG.info("Biographie : %s", _truncate_preview(bio, 400))
    niches = report.get("niches") or []
    if niches:
        _LOG.info("Niches : %s", ", ".join(str(n) for n in niches))

    for i, reel in enumerate(report.get("reels") or [], 1):
        mid = reel.get("media_id", "?")
        _LOG.info("--- Reel %d : %s ---", i, mid)
        _LOG.info("  caption : %s", _truncate_preview(str(reel.get("caption") or "")))
        _LOG.info("  transcript : %s", _truncate_preview(str(reel.get("transcript") or "")))
        rc = reel.get("comments") or []
        rc_lines = [
            format_comment_for_embedding(c) if isinstance(c, dict) else str(c)
            for c in rc
        ]
        _LOG.info(
            "  comments (%d) : %s",
            len(rc_lines),
            _truncate_preview("; ".join(rc_lines), 400),
        )

    _LOG.info("--- Texte envoyé à l'embedder (aperçu) ---")
    for line in str(report.get("input_text") or "").splitlines():
        _LOG.info("  %s", line)


def save_verify_report(
    report: dict[str, Any],
    path: Path | str | None = None,
) -> Path:
    """Sauvegarde le rapport JSON de vérification."""
    u = str(report.get("username") or "unknown").lstrip("@").strip().lower()
    p = _resolve_path(Path(path) if path is not None else Path(f"data/embed_verify_{u}.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


def process_account(
    username: str,
    creator: dict[str, Any],
    context: BrowserContext,
    pca_model: Any | None,
    tmp_dir: Path,
) -> dict[str, Any] | None:
    """Pipeline complet d'embedding pour un compte watchlisté (Playwright)."""
    collected = collect_account_content(username, creator, context, tmp_dir)
    if collected is None:
        return None

    embedding = embed_text(collected["input_text"])
    if embedding is None:
        _LOG.error(
            "embedding impossible pour @%s — compte ignoré.",
            collected["username"],
        )
        return None

    named_axes = project_to_named_axes(embedding, pca_model)
    return {
        "username": collected["username"],
        "updated_at": _utc_now_iso(),
        "embedding_raw": embedding,
        "named_axes": named_axes,
        "sources": collected["summary"],
    }


def _upsert_vector_store(
    store: list[dict[str, Any]],
    entry: dict[str, Any],
) -> list[dict[str, Any]]:
    username = str(entry.get("username") or "").lower()
    updated = [item for item in store if str(item.get("username") or "").lower() != username]
    updated.append(entry)
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embedding des comptes watchlistés.")
    parser.add_argument("--account", help="Traiter un seul compte (@username).")
    parser.add_argument(
        "--source",
        choices=("watchlist", "database"),
        default="watchlist",
        help="watchlist (défaut) : data/watchlist.json ; database : profils non archivés.",
    )
    parser.add_argument(
        "--tier",
        choices=("A", "B", "C"),
        default=None,
        help="Avec --source database : ne traiter que ce tier (défaut : tous non archivés).",
    )
    parser.add_argument(
        "--refit-pca",
        action="store_true",
        help="Recalcule la PCA après traitement.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche le plan sans appeler Whisper, LM Studio ni Instagram.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Collecte et affiche captions/transcripts/comments sans embedding.",
    )
    parser.add_argument(
        "--verify-out",
        metavar="PATH",
        help="Fichier JSON de sortie pour --verify (défaut : data/embed_verify_<user>.json).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _quiet_whisper_logs()

    if args.tier and args.source != "database":
        _LOG.error("--tier n'est utilisable qu'avec --source database.")
        return 1

    try:
        creators = load_creators(args.source, tier=args.tier)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.error("%s", exc)
        return 1

    if args.source == "database":
        tier_label = args.tier or "tous"
        _LOG.info(
            "Source : database.json — %d profils tier %s",
            len(creators),
            tier_label,
        )

    if args.account:
        target = args.account.lstrip("@").strip().lower()
        creators = [
            creator
            for creator in creators
            if str(creator.get("username") or "").lstrip("@").strip().lower() == target
        ]
        if not creators:
            source_label = "watchlist" if args.source == "watchlist" else "database"
            _LOG.error("Compte @%s introuvable dans la %s.", target, source_label)
            return 1

    vector_store = load_vector_store()
    embed_dim = _infer_embedding_dim(vector_store)
    pca_model = load_pca(expected_dim=embed_dim)
    processed = 0

    if not args.dry_run and not args.verify and not embedding_config_ok():
        _LOG.error(
            "Embedding impossible : configure LM_STUDIO_URL et "
            "LM_STUDIO_EMBED_MODEL dans .env."
        )
        return 1

    if not args.dry_run and not args.verify:
        _LOG.info(
            "Embeddings : LM Studio %s (modèle %s)",
            LM_STUDIO_URL,
            LM_STUDIO_EMBED_MODEL,
        )

    if args.dry_run:
        for creator in creators:
            username = str(creator.get("username") or "").lstrip("@").strip()
            _LOG.info("DRY-RUN : embedderait @%s", username)
        _LOG.info("=== Embedding terminé : %d comptes traités ===", len(creators))
        return 0

    if args.verify:
        if len(creators) > 1 and not args.account:
            _LOG.error("--verify : précise --account @username (un seul compte).")
            return 1
        playwright_instance = sync_playwright().start()
        context = get_browser_context(playwright_instance)
        try:
            for creator in creators:
                username = str(creator.get("username") or "").lstrip("@").strip()
                tmp_dir = Path(tempfile.mkdtemp(prefix="ait_verify_"))
                try:
                    report = collect_account_content(
                        username, creator, context, tmp_dir
                    )
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                if report is None:
                    return 1
                print_verify_report(report)
                out = save_verify_report(
                    report,
                    path=args.verify_out,
                )
                _LOG.info("Rapport JSON : %s", out)
        finally:
            context.close()
            br = context.browser
            if br:
                br.close()
            playwright_instance.stop()
        return 0

    playwright_instance = sync_playwright().start()
    context = get_browser_context(playwright_instance)
    try:
        for creator in creators:
            username = str(creator.get("username") or "").lstrip("@").strip()
            tmp_dir = Path(tempfile.mkdtemp(prefix="ait_embed_"))
            try:
                entry = process_account(username, creator, context, pca_model, tmp_dir)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            if entry is None:
                continue

            vector_store = _upsert_vector_store(vector_store, entry)
            processed += 1
            sources = entry.get("sources") or {}
            _LOG.info(
                "✓ @%s embeddé (reels=%d, transcripts=%d, captions=%d, comments=%d)",
                username,
                sources.get("reels_analyzed", 0),
                sources.get("transcripts_count", 0),
                sources.get("captions_count", 0),
                sources.get("comments_received_count", 0),
            )
            polite_sleep(seconds=3)
    finally:
        context.close()
        br = context.browser
        if br:
            br.close()
        playwright_instance.stop()

    n_current_dim = sum(
        1
        for entry in vector_store
        if isinstance(entry.get("embedding_raw"), list)
        and embed_dim is not None
        and len(entry["embedding_raw"]) == embed_dim
    )
    if args.refit_pca or n_current_dim >= 10:
        pca_model = fit_pca(vector_store, target_dim=embed_dim)
        if pca_model is not None:
            for entry in vector_store:
                raw = entry.get("embedding_raw")
                if (
                    isinstance(raw, list)
                    and raw
                    and embed_dim is not None
                    and len(raw) == embed_dim
                ):
                    entry["named_axes"] = project_to_named_axes(raw, pca_model)

    save_vector_store(vector_store)
    _LOG.info("=== Embedding terminé : %d comptes traités ===", processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
