"""Enrichissement reel : téléchargement, Whisper, description visuelle (grille 2×2)."""

from __future__ import annotations

import base64
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import requests
from playwright.sync_api import BrowserContext

import config
from scripts.reel_media import (
    download_reel_video,
    extract_wav_from_video,
    transcribe_audio,
)

_LOG = logging.getLogger("aitertainment.reel_enrichment")

_GRID_CELL_W = 720
_GRID_CELL_H = 405
_GRID_PCTS = [0.12, 0.37, 0.62, 0.87]
_GRID_SCALE_VF = f"scale={_GRID_CELL_W}:{_GRID_CELL_H}"

_GRID_VISION_PROMPT = (
    "Cette image est une grille 2x2 montrant 4 moments d'un même reel Instagram "
    "humour français, lus chronologiquement de gauche à droite et de haut en bas "
    "(intro → développement → climax → chute). Décris en 3-5 phrases la "
    "progression de la vidéo : qui apparaît, ce qui se passe, le décor et le ton, "
    "tout texte visible."
)


def _make_frames_grid(mp4_path: Path, output_path: Path) -> Path | None:
    """Extrait 4 frames (12 %–87 %) et les assemble en grille 2×2 (720×405 par cellule)."""
    mp4_path = Path(mp4_path)
    output_path = Path(output_path)
    if not mp4_path.exists():
        return None

    duration = 10.0
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                str(mp4_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if probe.returncode == 0 and probe.stdout:
            data = json.loads(probe.stdout)
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    duration = float(stream.get("duration", 10))
                    break
    except Exception:
        pass

    work = output_path.parent
    frame_paths: list[Path] = []
    for i, pct in enumerate(_GRID_PCTS):
        fp = work / f"_grid_cell_{i}.jpg"
        t = max(0.0, duration * pct)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(t),
                "-i",
                str(mp4_path),
                "-frames:v",
                "1",
                "-vf",
                _GRID_SCALE_VF,
                "-q:v",
                "2",
                str(fp),
            ],
            capture_output=True,
            check=False,
        )
        if fp.exists() and fp.stat().st_size > 500:
            frame_paths.append(fp)

    if len(frame_paths) < 3:
        _LOG.warning(
            "Grille %s : seulement %d/4 frames extraites — abandon visuel.",
            mp4_path.name,
            len(frame_paths),
        )
        for fp in frame_paths:
            fp.unlink(missing_ok=True)
        return None

    while len(frame_paths) < 4:
        frame_paths.append(frame_paths[-1])

    inputs: list[str] = []
    for fp in frame_paths[:4]:
        inputs.extend(["-i", str(fp)])

    filter_complex = (
        "[0:v][1:v]hstack=inputs=2[r1];"
        "[2:v][3:v]hstack=inputs=2[r2];"
        "[r1][r2]vstack=inputs=2[out]"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-q:v",
            "2",
            str(output_path),
        ],
        capture_output=True,
        check=False,
    )
    for fp in frame_paths[:4]:
        fp.unlink(missing_ok=True)

    if result.returncode != 0 or not output_path.exists():
        _LOG.warning(
            "ffmpeg grille 2x2 échoué pour %s : %s",
            mp4_path,
            (result.stderr or b"").decode(errors="replace")[-300:],
        )
        return None
    return output_path


def _describe_grid(grid_path: Path) -> str:
    """Description visuelle via LM Studio (grille 2×2, ``config.LM_STUDIO_*``)."""
    grid_path = Path(grid_path)
    lm_url = (config.LM_STUDIO_URL or "").strip().rstrip("/")
    vision_model = (config.LM_STUDIO_VISION_MODEL or "").strip()
    if not grid_path.exists() or not lm_url or not vision_model:
        return ""

    img_b64 = base64.b64encode(grid_path.read_bytes()).decode()
    try:
        resp = requests.post(
            f"{lm_url}/chat/completions",
            json={
                "model": vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": _GRID_VISION_PROMPT,
                            },
                        ],
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.3,
            },
            timeout=90,
        )
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
        content = str(message.get("content") or "").strip()
        if not content:
            content = str(message.get("reasoning_content") or "").strip()
        if not content:
            _LOG.warning(
                "Vision grille : réponse vide (modèle=%s).",
                vision_model,
            )
        return content
    except Exception as e:
        _LOG.warning(
            "Vision grille échouée pour %s (modèle=%s, url=%s) : %s",
            grid_path,
            vision_model,
            lm_url,
            e,
        )
        return ""


def enrich_reel_with_transcript_and_visual(
    media_id: str,
    browser_context: BrowserContext,
    *,
    skip_transcript: bool = False,
    skip_visual: bool = False,
) -> tuple[str, str]:
    """Télécharge le reel, transcrit (Whisper), décrit le visuel (grille + vision).

    Retourne ``(transcript, visual_description)``. Nettoie le répertoire temporaire.
    """
    if skip_transcript and skip_visual:
        return "", ""

    media_id = str(media_id or "").strip()
    if not media_id:
        return "", ""

    tmp_dir = Path(tempfile.mkdtemp(prefix="ait_reel_enrich_"))
    transcript = ""
    visual = ""
    try:
        mp4_path = download_reel_video(media_id, browser_context, tmp_dir)
        if not mp4_path:
            return "", ""

        if not skip_transcript:
            wav_path = extract_wav_from_video(mp4_path, tmp_dir)
            if wav_path:
                transcript = (transcribe_audio(wav_path) or "")[:2000]

        if not skip_visual and config.LM_STUDIO_URL and config.LM_STUDIO_VISION_MODEL:
            grid_path = _make_frames_grid(mp4_path, tmp_dir / "grid.jpg")
            if grid_path:
                visual = (_describe_grid(grid_path) or "")[:1000]
            elif mp4_path:
                _LOG.warning(
                    "Reel %s : grille vision non générée (ffmpeg frames).",
                    media_id,
                )
    except Exception as e:
        _LOG.warning("Enrichissement échoué %s : %s", media_id, e)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return transcript, visual


def call_label_llm(
    user_prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.35,
    timeout: int = 60,
) -> str:
    """Appel chat/completions sur LABEL_LLM (LM Studio / OpenAI-compatible)."""
    api_url = (config.LABEL_LLM_URL or "").strip().rstrip("/")
    api_model = (config.LABEL_LLM_MODEL or "").strip()
    if not api_url or not api_model:
        return ""
    try:
        resp = requests.post(
            f"{api_url}/chat/completions",
            json={
                "model": api_model,
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
        return str(
            message.get("content") or message.get("reasoning_content") or ""
        ).strip()
    except Exception as exc:
        _LOG.warning("LABEL_LLM appel échoué : %s", exc)
        return ""


def fuse_transcript_visual_context(
    transcript: str,
    visual_description: str,
    caption: str = "",
) -> str:
    """Fusionne transcript + visuel (LABEL_LLM) — même entrée que le watcher."""
    if not (transcript or visual_description):
        return ""
    api_url = (config.LABEL_LLM_URL or "").strip().rstrip("/")
    api_model = (config.LABEL_LLM_MODEL or "").strip()
    if not api_url or not api_model:
        _LOG.warning("LABEL_LLM_* absent — fusion video_context ignorée.")
        return ""

    user_prompt = f"""Fusionne en 2-4 phrases chronologiques le transcript audio 
et la description visuelle de ce reel Instagram :

Caption : {str(caption or "")[:300]}
Transcript : {str(transcript or "")[:1000]}
Description visuelle : {str(visual_description or "")[:600]}

Réponds uniquement la fusion en français, pas d'explication ni de raisonnement."""

    raw = call_label_llm(user_prompt, max_tokens=500, temperature=0.3)
    return _clean_fused_context(raw) if raw else ""


def _clean_fused_context(raw: str) -> str:
    """Extrait la fusion française (ignore chain-of-thought des modèles reasoning)."""
    text = str(raw or "").strip()
    if not text:
        return ""
    if "thinking process" in text.lower() or text.lstrip().startswith("1."):
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith(("1.", "2.", "3.", "4.", "*", "-", "#")):
                continue
            if line.lower().startswith(("here's", "check constraints", "draft")):
                continue
            if len(line) > 40 and any(
                w in line.lower()
                for w in ("dans ", "il ", "elle ", "un jeune", "tandis", "salle")
            ):
                return line[:600]
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        for part in reversed(parts):
            if len(part) > 40 and not part.startswith(("1.", "2.", "*")):
                return part[:600]
    return text[:600]
