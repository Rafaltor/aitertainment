"""Téléchargement reel (yt-dlp) et transcription audio (faster-whisper)."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import BrowserContext

_LOG = logging.getLogger("aitertainment.reel_media")
_WHISPER_LOGGERS_QUIETED = False
_WHISPER_MODEL_SIZE = "small"


def _yt_dlp_executable() -> str:
    for base in (Path(sys.prefix) / "bin", Path(sys.executable).resolve().parent):
        candidate = base / "yt-dlp"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("yt-dlp") or "yt-dlp"


def _quiet_whisper_logs() -> None:
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


def download_reel_video(
    media_id: str, context: BrowserContext, tmp_dir: Path
) -> Path | None:
    """Télécharge la vidéo MP4 d'un Reel via yt-dlp et les cookies Playwright."""
    cookie_file = tmp_dir / "cookies.txt"
    export_playwright_cookies(context, cookie_file)

    media_id = str(media_id or "").strip()
    mp4_path = tmp_dir / f"{media_id}.mp4"
    try:
        result = subprocess.run(
            [
                _yt_dlp_executable(),
                "--cookies",
                str(cookie_file),
                "-f",
                "best[ext=mp4]/best",
                "-o",
                str(tmp_dir / "%(id)s.%(ext)s"),
                "--quiet",
                f"https://www.instagram.com/reel/{media_id}/",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
    except FileNotFoundError:
        _LOG.warning("yt-dlp absent — vidéo ignorée pour %s.", media_id)
        return None
    except subprocess.TimeoutExpired:
        _LOG.warning("yt-dlp timeout (vidéo) pour %s.", media_id)
        return None

    if result.returncode != 0:
        stderr = result.stderr or ""
        if "No video formats found" in stderr:
            _LOG.debug(
                "Reel %s : pas de vidéo (carousel/photo) — skip transcript.",
                media_id,
            )
            return None
        _LOG.warning("yt-dlp vidéo échoué pour %s : %s", media_id, stderr[:200])
        return None

    if mp4_path.exists():
        return mp4_path
    for candidate in sorted(tmp_dir.glob("*.mp4")):
        if candidate.stem == media_id or media_id in candidate.name:
            return candidate
    mp4_files = sorted(tmp_dir.glob("*.mp4"))
    if len(mp4_files) == 1:
        return mp4_files[0]
    if mp4_files:
        _LOG.warning(
            "plusieurs MP4 dans %s pour %s — fichier ambigu ignoré.",
            tmp_dir,
            media_id,
        )
    return None


def extract_wav_from_video(video_path: Path, tmp_dir: Path | None = None) -> Path | None:
    """Extrait un WAV mono 16 kHz depuis un MP4 (entrée Whisper)."""
    video_path = Path(video_path)
    if not video_path.exists():
        return None
    out_dir = Path(tmp_dir) if tmp_dir else video_path.parent
    wav_path = out_dir / f"{video_path.stem}.wav"
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(wav_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        _LOG.warning("ffmpeg absent — extraction audio ignorée pour %s.", video_path)
        return None
    except subprocess.TimeoutExpired:
        _LOG.warning("ffmpeg timeout pour %s.", video_path)
        return None

    if result.returncode != 0:
        stderr = (result.stderr or "")[:200]
        _LOG.warning("ffmpeg échoué pour %s : %s", video_path, stderr)
        return None
    return wav_path if wav_path.exists() else None


def transcribe_audio(wav_path: Path | str | None) -> str:
    """Transcrit un WAV via faster-whisper."""
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
        model = WhisperModel(_WHISPER_MODEL_SIZE, device="cpu")
        segments, _info = model.transcribe(str(path), language="fr")
        return " ".join(segment.text.strip() for segment in segments if segment.text).strip()
    except Exception as e:
        _LOG.warning("transcription échouée pour %s (%s).", path, e)
        return ""
