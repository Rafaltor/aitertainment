"""Notifications Telegram et logging pour le Watcher."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import requests

import config

_PROJECT_ROOT = Path(__file__).resolve().parent
_LOG_PATH = _PROJECT_ROOT / "logs" / "watcher.log"

_LOGGER_NAME = "aitertainment.watcher"
_log_initialized = False


def setup_watcher_logger() -> logging.Logger:
    """Configure (idempotent) le logger Watcher : ``logs/watcher.log`` + console."""
    global _log_initialized
    log = logging.getLogger(_LOGGER_NAME)
    if _log_initialized:
        return log
    log.setLevel(logging.INFO)
    log.handlers.clear()
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    # Console seulement en interactif — évite « suspended (tty output) » avec nohup/SSH.
    if sys.stdout.isatty():
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)
    log.propagate = False
    _log_initialized = True
    return log


def send_telegram_markdown(
    text: str,
    *,
    bot_token: str | None = None,
    chat_id: str | None = None,
    parse_mode: str = "Markdown",
) -> dict[str, Any]:
    """Envoie un message Telegram via l'API HTTP du bot."""
    token = (
        bot_token if bot_token is not None else (config.TELEGRAM_BOT_TOKEN or "")
    ).strip()
    chat = str(
        chat_id if chat_id is not None else (config.TELEGRAM_CHAT_ID or "")
    ).strip()
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN manquant : .env ou argument bot_token=")
    if not chat:
        raise ValueError("TELEGRAM_CHAT_ID manquant : .env ou argument chat_id=")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(
        url,
        json={
            "chat_id": chat,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": False,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API ok=false: {data}")
    return data
