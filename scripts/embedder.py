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

OLLAMA_EMBED_MODEL = "bge-m3"
WHISPER_MODEL_SIZE = "small"
REELS_PER_ACCOUNT = 5
COMMENTS_PER_REEL = 30
VECTOR_STORE_PATH = Path("data/vector_store.json")
WATCHLIST_PATH = Path("data/watchlist.json")
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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
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


def download_audio(media_pk: Any, client: Any, tmp_dir: Path) -> Path | None:
    """Télécharge un Reel et extrait l'audio en WAV mono 16 kHz."""
    try:
        video_path = client.video_download(str(media_pk), folder=str(tmp_dir))
    except Exception as e:
        _LOG.warning("video_download %s a échoué (%s) — skip audio.", media_pk, e)
        return None

    video = Path(video_path)
    if not video.exists():
        _LOG.warning("fichier vidéo absent après download (%s) — skip audio.", video_path)
        return None

    wav_path = tmp_dir / f"{video.stem}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "wav",
        str(wav_path),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        _LOG.warning("ffmpeg absent — transcription audio ignorée pour %s.", media_pk)
        return None
    except subprocess.CalledProcessError as e:
        _LOG.warning("ffmpeg a échoué pour %s (%s) — skip audio.", media_pk, e)
        return None

    if not wav_path.exists():
        _LOG.warning("fichier WAV absent après ffmpeg (%s).", wav_path)
        return None
    return wav_path


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
) -> str:
    """Assemble le texte unifié envoyé à bge-m3."""
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
    return (
        f"[CAPTION] {caption_text}\n"
        f"[HASHTAGS] {hashtags_text}\n"
        f"[TRANSCRIPT] {transcript_text}\n"
        f"[COMMENTS_RECEIVED] {comments_text}"
    )


def embed_text(text: str) -> list[float] | None:
    """Appelle Ollama ``embed`` et retourne le vecteur 1024D."""
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
    if any(math.isnan(x) for x in vector):
        _LOG.warning("embedding NaN détecté — entrée ignorée.")
        return None
    return vector


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


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _is_reel(media: Any) -> bool:
    media_type = str(_attr(media, "media_type", "") or "").lower()
    product_type = str(_attr(media, "product_type", "") or "").lower()
    return media_type in {"clip", "clips", "reel"} or product_type == "clips"


def _extract_hashtags(caption: str, media: Any) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for source in (_attr(media, "hashtags", None), caption):
        if isinstance(source, list):
            for item in source:
                tag = str(_attr(item, "name", item) or "").strip().lstrip("#")
                if tag and tag not in seen:
                    seen.add(tag)
                    tags.append(tag)
        elif isinstance(source, str):
            for match in _HASHTAG_RE.findall(source):
                if match not in seen:
                    seen.add(match)
                    tags.append(match)
    return tags


def _comment_text(comment: Any) -> str:
    text = _attr(comment, "text", None)
    if text is None:
        text = _attr(comment, "comment", "")
    return str(text or "").strip()


def _is_rate_limit_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return (
        "ratelimit" in name
        or "throttle" in name
        or "please wait" in message
        or "rate limit" in message
    )


def _with_instagram_retry(action, log: logging.Logger):
    try:
        return action()
    except Exception as exc:
        if not _is_rate_limit_error(exc):
            raise
        log.warning("rate limit Instagram (%s) — pause 60s puis retry.", exc)
        from instagram_client import polite_sleep

        polite_sleep(min_s=60, max_s=60)
        try:
            return action()
        except Exception as retry_exc:
            log.warning("retry Instagram échoué (%s) — skip.", retry_exc)
            return None


def process_account(
    username: str,
    creator: dict[str, Any],
    client: Any,
    pca_model: Any | None,
    tmp_dir: Path,
) -> dict[str, Any] | None:
    """Pipeline complet d'embedding pour un compte watchlisté."""
    del creator  # réservé aux extensions futures (niches, t_type, etc.)
    uname = str(username or "").lstrip("@").strip()
    if not uname:
        return None

    user_id = _with_instagram_retry(
        lambda: client.user_id_from_username(uname),
        _LOG,
    )
    if user_id is None:
        return None

    medias = _with_instagram_retry(
        lambda: client.user_medias(str(user_id), amount=REELS_PER_ACCOUNT),
        _LOG,
    )
    if medias is None:
        return None

    reels = [media for media in medias or [] if _is_reel(media)]
    captions: list[str] = []
    hashtags: list[str] = []
    transcripts: list[str] = []
    comments: list[str] = []
    seen_comments: set[str] = set()

    for media in reels:
        media_pk = _attr(media, "pk", None) or _attr(media, "id", None)
        caption = str(_attr(media, "caption_text", "") or _attr(media, "caption", "") or "").strip()
        if caption:
            captions.append(caption)
        hashtags.extend(_extract_hashtags(caption, media))

        wav_path = download_audio(media_pk, client, tmp_dir)
        transcript = transcribe_audio(wav_path)
        if transcript:
            transcripts.append(transcript)

        if media_pk is not None:
            raw_comments = _with_instagram_retry(
                lambda pk=media_pk: client.media_comments(
                    str(pk), amount=COMMENTS_PER_REEL
                ),
                _LOG,
            )
            if raw_comments is None:
                raw_comments = []
            for comment in raw_comments or []:
                text = _comment_text(comment)
                if text and text not in seen_comments:
                    seen_comments.add(text)
                    comments.append(text)

        for path in tmp_dir.glob("*"):
            if path.is_file():
                path.unlink(missing_ok=True)

    hashtags = list(dict.fromkeys(tag for tag in hashtags if tag))
    caption_blob = "\n".join(captions)
    transcript_blob = "\n---\n".join(transcripts)
    text = build_input_text(caption_blob, hashtags, transcript_blob, comments)
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

    try:
        creators = load_watchlist()
    except FileNotFoundError as exc:
        _LOG.error("%s", exc)
        return 1

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

    vector_store = load_vector_store()
    pca_model = load_pca()
    processed = 0

    if args.dry_run:
        for creator in creators:
            username = str(creator.get("username") or "").lstrip("@").strip()
            _LOG.info("DRY-RUN : embedderait @%s", username)
        _LOG.info("=== Embedding terminé : %d comptes traités ===", len(creators))
        return 0

    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from instagram_client import get_client, polite_sleep

    client = get_client()

    for creator in creators:
        username = str(creator.get("username") or "").lstrip("@").strip()
        tmp_dir = Path(tempfile.mkdtemp(prefix="ait_embed_"))
        try:
            entry = process_account(username, creator, client, pca_model, tmp_dir)
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
        polite_sleep(min_s=3, max_s=3)

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
