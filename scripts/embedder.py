"""embedder.py — embeddings de profils watchlistés (bge-m3 + Whisper).

Script autonome : ne dépend ni de ``discovery.py`` ni de ``watcher.py``.
Pipeline hebdomadaire manuel — voir ``main()`` pour le CLI.
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

from scripts.instagram_browser import (
    get_browser_context,
    get_profile_data,
    get_recent_reels,
    get_reel_caption,
    parse_comments_from_dom_text,
    polite_sleep,
)

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "")
LM_STUDIO_EMBED_MODEL = os.environ.get("LM_STUDIO_EMBED_MODEL", "")
OLLAMA_EMBED_MODEL = "bge-m3"
WHISPER_MODEL_SIZE = "small"
REELS_PER_ACCOUNT = 5
COMMENTS_PER_REEL = 30
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


def _extract_comments_from_reel_page(page: Page) -> list[str]:
    """Ouvre le panneau commentaires et retourne les textes parsés."""
    try:
        comment_btn = page.locator(
            'svg[aria-label="Commenter"], svg[aria-label="Comment"]'
        )
        if comment_btn.count() == 0:
            return []
        comment_btn.first.click(timeout=10_000)
        page.wait_for_timeout(2000)
        panel_text = page.evaluate(
            """() => {
                const p = document.querySelector("div._aano") ||
                          document.querySelector("[role=dialog]");
                return p ? p.innerText : "";
            }"""
        )
    except Exception as e:
        _LOG.warning("extraction commentaires DOM échouée (%s).", e)
        return []

    parsed = parse_comments_from_dom_text(str(panel_text or ""))
    return [str(c.get("text") or "").strip() for c in parsed if c.get("text")]


def transcribe_audio(wav_path: Path | str | None) -> str:
    """Transcrit un WAV via faster-whisper (import lazy)."""
    if wav_path is None:
        return ""
    path = Path(wav_path)
    if not path.exists():
        return ""

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
    comments: list[str],
    *,
    biography: str = "",
    niches: list[str] | None = None,
) -> str:
    """Assemble le texte unifié envoyé à bge-m3."""
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
        comments_text = "\n".join(c.strip() for c in comments if str(c).strip())
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


def embed_text(text: str) -> list[float] | None:
    """Embedding 1024D via LM Studio (prioritaire) ou Ollama (fallback)."""
    if LM_STUDIO_URL and LM_STUDIO_EMBED_MODEL:
        text = text[:8000]
        _LOG.debug("embed_text : %d caractères", len(text))
        try:
            resp = requests.post(
                f"{LM_STUDIO_URL.rstrip('/')}/embeddings",
                json={"model": LM_STUDIO_EMBED_MODEL, "input": text},
                timeout=60,
            )
            if resp.status_code == 400:
                _LOG.error(
                    "LM Studio embed 400 — texte longueur=%d, début=%s",
                    len(text),
                    text[:100],
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
                    len(text),
                    text[:100],
                )
            else:
                _LOG.error("LM Studio embed a échoué (%s).", e)
            return None
        except Exception as e:
            _LOG.error("LM Studio embed a échoué (%s).", e)
            return None

    try:
        import ollama
    except ImportError as e:
        _LOG.error("ollama indisponible (%s).", e)
        return None

    try:
        response = ollama.embed(model=OLLAMA_EMBED_MODEL, input=text)
    except Exception as e:
        _LOG.error("Ollama embed a échoué (%s).", e)
        return None

    embeddings = getattr(response, "embeddings", None)
    if embeddings is None and isinstance(response, dict):
        embeddings = response.get("embeddings")
    if not embeddings:
        _LOG.error("Ollama embed : réponse sans embeddings.")
        return None

    vector = [float(x) for x in embeddings[0]]
    return _validate_embedding_vector(vector)


def project_to_named_axes(
    embedding_1024: list[float],
    pca_model: Any | None,
) -> dict[str, float]:
    """Projette un embedding 1024D sur les 10 axes nommés."""
    if pca_model is None:
        _LOG.warning(
            "PCA non disponible — named_axes non calculés (besoin de 10+ comptes)"
        )
        return {axis: 0.0 for axis in NAMED_AXES}

    projected = pca_model.transform([embedding_1024])[0]
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


def fit_pca(vector_store: list[dict[str, Any]]) -> Any | None:
    """Ajuste une PCA 10D sur les ``embedding_raw`` existants."""
    vectors: list[list[float]] = []
    for entry in vector_store:
        raw = entry.get("embedding_raw")
        if isinstance(raw, list) and raw:
            vectors.append([float(x) for x in raw])

    if len(vectors) < 10:
        return None

    from sklearn.decomposition import PCA

    model = PCA(n_components=10)
    model.fit(vectors)

    pca_path = _resolve_path(PCA_MODEL_PATH)
    pca_path.parent.mkdir(parents=True, exist_ok=True)
    with pca_path.open("wb") as fh:
        pickle.dump(model, fh)
    return model


def load_pca() -> Any | None:
    """Charge le modèle PCA depuis le disque, si présent."""
    pca_path = _resolve_path(PCA_MODEL_PATH)
    if not pca_path.exists():
        return None
    with pca_path.open("rb") as fh:
        return pickle.load(fh)


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


def process_account(
    username: str,
    creator: dict[str, Any],
    context: BrowserContext,
    pca_model: Any | None,
    tmp_dir: Path,
) -> dict[str, Any] | None:
    """Pipeline complet d'embedding pour un compte watchlisté (Playwright)."""
    uname = str(username or "").lstrip("@").strip()
    if not uname:
        return None

    reels = get_recent_reels(uname, context, max_reels=REELS_PER_ACCOUNT)
    if not reels:
        _LOG.warning("@%s : aucun Reel récupéré — skip embedding.", uname)
        return None

    for reel in reels:
        media_id = str(reel.get("media_id") or "").strip()
        if not media_id or str(reel.get("caption") or "").strip():
            continue
        reel["caption"] = get_reel_caption(media_id, context)
        polite_sleep(1)

    profile_data = get_profile_data(uname, context)
    biography = str(profile_data.get("biography") or "").strip() if profile_data else ""
    niches_raw = creator.get("niches") or ["humour"]
    niches = niches_raw if isinstance(niches_raw, list) else [str(niches_raw)]

    captions = [str(r.get("caption") or "").strip() for r in reels if r.get("caption")]
    caption_blob = "\n".join(captions)
    hashtags: list[str] = []
    for caption in captions:
        hashtags.extend(_HASHTAG_RE.findall(caption))
    hashtags = list(dict.fromkeys(hashtags))

    transcripts: list[str] = []
    comments: list[str] = []
    seen_comments: set[str] = set()

    for reel in reels:
        media_id = str(reel.get("media_id") or "").strip()
        if not media_id:
            continue

        wav_path = download_audio_from_reel(media_id, context, tmp_dir)
        transcript = transcribe_audio(wav_path)
        if transcript:
            transcripts.append(transcript)

        page = context.new_page()
        try:
            page.goto(
                f"https://www.instagram.com/reel/{media_id}/",
                wait_until="domcontentloaded",
            )
            page.wait_for_load_state("load")
            page.wait_for_timeout(2000)
            for text in _extract_comments_from_reel_page(page)[:COMMENTS_PER_REEL]:
                if text and text not in seen_comments:
                    seen_comments.add(text)
                    comments.append(text)
        finally:
            page.close()

        for path in tmp_dir.glob("*"):
            if path.is_file():
                path.unlink(missing_ok=True)

    transcript_blob = "\n---\n".join(transcripts)
    text = build_input_text(
        caption_blob,
        hashtags,
        transcript_blob,
        comments,
        biography=biography,
        niches=niches,
    )
    embedding_1024 = embed_text(text)
    if embedding_1024 is None:
        _LOG.error("embedding impossible pour @%s — compte ignoré.", uname)
        return None

    named_axes = project_to_named_axes(embedding_1024, pca_model)
    return {
        "username": uname,
        "updated_at": _utc_now_iso(),
        "embedding_raw": embedding_1024,
        "named_axes": named_axes,
        "sources": {
            "captions_count": len(captions),
            "transcripts_count": len(transcripts),
            "comments_received_count": len(comments),
            "reels_analyzed": len(reels),
        },
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
        help="Affiche le plan sans appeler Whisper, bge-m3 ni Instagram.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

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
    pca_model = load_pca()
    processed = 0

    if args.dry_run:
        for creator in creators:
            username = str(creator.get("username") or "").lstrip("@").strip()
            _LOG.info("DRY-RUN : embedderait @%s", username)
        _LOG.info("=== Embedding terminé : %d comptes traités ===", len(creators))
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
                "✓ @%s embeddé (reels=%d, transcripts=%d)",
                username,
                sources.get("reels_analyzed", 0),
                sources.get("transcripts_count", 0),
            )
            polite_sleep(seconds=3)
    finally:
        context.close()
        br = context.browser
        if br:
            br.close()
        playwright_instance.stop()

    if args.refit_pca or len(vector_store) >= 10:
        pca_model = fit_pca(vector_store)
        if pca_model is not None:
            for entry in vector_store:
                raw = entry.get("embedding_raw")
                if isinstance(raw, list) and raw:
                    entry["named_axes"] = project_to_named_axes(raw, pca_model)

    save_vector_store(vector_store)
    _LOG.info("=== Embedding terminé : %d comptes traités ===", processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
