"""Notifications Telegram (Bot API)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

from modules.detector import CreatorStats


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

TELEGRAM_API = "https://api.telegram.org"


def _telegram_markdown_escape(text: str) -> str:
    """Échappe les caractères spéciaux du Markdown classique Telegram."""
    out: list[str] = []
    for ch in text:
        if ch in ("\\", "_", "*", "[", "`"):
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _headline_video(creator: CreatorStats) -> dict[str, Any] | None:
    if not creator.recent_videos:
        return None
    return max(creator.recent_videos, key=lambda v: _ensure_utc(v["posted_at"]))


def _minutes_since_publication(
    video: dict[str, Any], *, now: datetime | None = None
) -> int:
    now_utc = _ensure_utc(now) if now else datetime.now(timezone.utc)
    posted = _ensure_utc(video["posted_at"])
    delta = now_utc - posted
    minutes = int(max(delta.total_seconds(), 0) // 60)
    return max(minutes, 0)


def _video_url(creator: CreatorStats, video: dict[str, Any] | None) -> str:
    if not video:
        return "non disponible"
    for key in ("url", "permalink", "video_url", "link"):
        u = video.get(key)
        if isinstance(u, str) and u.strip():
            return u.strip()
    return "non disponible"


def _alert_label(alert: str) -> str:
    a = (alert or "NONE").strip().upper()
    if a in ("CRITICAL", "ALERT"):
        return a
    if a == "NONE":
        return "AUCUNE"
    return a


class TelegramNotifier:
    """Envoie des alertes via l'API HTTP du Bot Telegram (Markdown)."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        if bot_token is None or chat_id is None:
            import config

            self.bot_token = (
                bot_token if bot_token is not None else (config.TELEGRAM_BOT_TOKEN or "")
            ).strip()
            self.chat_id = str(
                chat_id if chat_id is not None else (config.TELEGRAM_CHAT_ID or "")
            ).strip()
        else:
            self.bot_token = bot_token.strip()
            self.chat_id = str(chat_id).strip()
        if not self.bot_token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN manquant : .env ou argument bot_token="
            )
        if not self.chat_id:
            raise ValueError(
                "TELEGRAM_CHAT_ID manquant : .env ou argument chat_id="
            )

    def _send_raw(self, text: str, *, parse_mode: str = "Markdown") -> dict[str, Any]:
        url = f"{TELEGRAM_API}/bot{self.bot_token}/sendMessage"
        resp = requests.post(
            url,
            json={
                "chat_id": self.chat_id,
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

    def test_connection(self) -> dict[str, Any]:
        """Envoie un message court pour valider token + chat_id."""
        return self._send_raw("✅ AItertainment actif")

    def send_alert(
        self,
        creator: CreatorStats,
        signal_result: dict[str, Any],
        classification: dict[str, Any],
        comments: list[str],
    ) -> dict[str, Any]:
        """Formate et envoie l'alerte virale + contexte + suggestions de commentaires."""
        alert = _alert_label(str(signal_result.get("alert", "NONE")))
        score_viral = float(signal_result.get("score_viral", 0.0))
        groups = signal_result.get("groups") or {}
        s1 = float(groups.get("g1", {}).get("score", 0.0))
        s2 = float(groups.get("g2", {}).get("score", 0.0))
        s3 = float(groups.get("g3", {}).get("score", 0.0))
        s4 = float(groups.get("g4", {}).get("score", 0.0))

        ctype = str(classification.get("type", "?"))
        tone = str(classification.get("tone", ""))
        patterns_raw = classification.get("patterns", [])
        if isinstance(patterns_raw, list):
            patterns_str = " · ".join(str(p) for p in patterns_raw[:10]) or "—"
        else:
            patterns_str = str(patterns_raw) if patterns_raw else "—"

        headline = _headline_video(creator)
        minutes = _minutes_since_publication(headline, now=None) if headline else 0
        video_url = _video_url(creator, headline)

        padded = (list(comments) + ["—", "—", "—"])[:3]
        c1, c2, c3 = padded

        u = _telegram_markdown_escape(creator.username)
        plat = _telegram_markdown_escape(creator.platform)
        tone_e = _telegram_markdown_escape(tone)
        patterns_e = _telegram_markdown_escape(patterns_str)
        type_e = _telegram_markdown_escape(ctype)
        url_e = _telegram_markdown_escape(video_url)
        ce1, ce2, ce3 = (
            _telegram_markdown_escape(c1),
            _telegram_markdown_escape(c2),
            _telegram_markdown_escape(c3),
        )

        text = (
            f"🚨 *ALERTE VIRALE* - {alert}\n\n"
            f"👤 @{u} ({plat})\n"
            f"👥 {int(creator.followers)} followers\n\n"
            f"📊 Score: {score_viral:.2f}/1.0\n"
            f"├ Créateur: {s1:.2f}\n"
            f"├ Vélocité: {s2:.2f}\n"
            f"├ Contenu: {s3:.2f}\n"
            f"└ Communauté: {s4:.2f}\n\n"
            f"🎭 Type: {type_e} - {tone_e}\n"
            f"🔥 Patterns: {patterns_e}\n"
            f"⚡ Fenêtre: {minutes} min\n\n"
            f"💬 *Commentaires suggérés:*\n"
            f"1. {ce1}\n"
            f"2. {ce2}\n"
            f"3. {ce3}\n\n"
            f"🔗 {url_e}"
        )

        return self._send_raw(text, parse_mode="Markdown")
