"""Callbacks Telegram Watcher — publier un commentaire suggéré sur Instagram.

Boutons inline sur chaque alerte nouveau post (bot #1). Au clic :
1. Récupère le texte depuis ``data/watcher_pending_posts.json``
2. Publie via Playwright avec le **compte IG 1** (``data/instagram_cookies.json``)
3. Met à jour le clavier Telegram (bouton ✅)

Le long-polling tourne dans un thread daemon lancé par ``run_watcher``.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

import config
from config import ORDERED_T_TYPES
from modules.atomic_json import atomic_write_json, json_lock

_PROJECT_ROOT = Path(__file__).resolve().parent
PENDING_PATH = _PROJECT_ROOT / "data" / "watcher_pending_posts.json"
STATE_PATH = _PROJECT_ROOT / "data" / "watcher_bot_state.json"
PENDING_TTL_S = 48 * 3600
POLL_TIMEOUT_S = 25
HTTP_TIMEOUT_S = 35
_CALLBACK_PREFIX = "w:"
# Bouton permanent (texte publié tel quel sur Instagram).
FIXED_COMMENT_KEY = "lowtaper67"
FIXED_COMMENT_TEXT = "lowtaper67"
_IG_POST_LOCK = threading.Lock()

_LOG = logging.getLogger("aitertainment.watcher.telegram")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _telegram_api(
    method: str,
    payload: dict[str, Any],
    *,
    token: str,
) -> dict[str, Any] | None:
    url = f"https://api.telegram.org/bot{token}/{method}"
    try:
        resp = requests.post(url, json=payload, timeout=HTTP_TIMEOUT_S)
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        _LOG.warning("Telegram %s erreur : %s", method, e)
        return None
    if not data.get("ok"):
        _LOG.warning(
            "Telegram %s non-ok : %s",
            method,
            data.get("description") or data,
        )
    return data


def _ack_callback(callback_query_id: str, *, token: str, text: str | None = None) -> None:
    payload: dict[str, Any] = {"callback_query_id": callback_query_id}
    if text:
        payload["text"] = text[:200]
    _telegram_api("answerCallbackQuery", payload, token=token)


def _load_pending() -> dict[str, Any]:
    if not PENDING_PATH.is_file():
        return {"posts": {}}
    try:
        data = json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"posts": {}}
    if not isinstance(data, dict):
        return {"posts": {}}
    posts = data.get("posts")
    if not isinstance(posts, dict):
        return {"posts": {}}
    return data


def _save_pending(data: dict[str, Any], *, use_lock: bool = True) -> None:
    PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(PENDING_PATH, data, use_lock=use_lock)


def _prune_expired(posts: dict[str, Any]) -> None:
    cutoff = time.time() - PENDING_TTL_S
    for token in list(posts.keys()):
        entry = posts.get(token)
        if not isinstance(entry, dict):
            posts.pop(token, None)
            continue
        try:
            created = datetime.fromisoformat(
                str(entry.get("created_at") or "").replace("Z", "+00:00")
            ).timestamp()
        except (TypeError, ValueError):
            created = 0.0
        if created < cutoff:
            posts.pop(token, None)


def register_pending_post(
    *,
    creator: dict[str, Any],
    post: dict[str, Any],
    comments: dict[str, str],
) -> str:
    """Enregistre les commentaires suggérés ; retourne un token court pour callbacks."""
    token = secrets.token_urlsafe(6).replace("-", "x").replace("_", "y")[:10]
    with json_lock(PENDING_PATH):
        data = _load_pending()
        posts = data.setdefault("posts", {})
        _prune_expired(posts)
        stored_comments = {k: str(v or "") for k, v in comments.items()}
        stored_comments[FIXED_COMMENT_KEY] = FIXED_COMMENT_TEXT
        posts[token] = {
            "media_id": str(post.get("video_id") or ""),
            "creator_username": str(
                post.get("username") or creator.get("username") or ""
            ).lstrip("@"),
            "url": str(post.get("url") or ""),
            "comments": stored_comments,
            "posted": {},
            "created_at": _now_iso(),
        }
        _save_pending(data, use_lock=False)
    return token


def _callback_slots() -> frozenset[str]:
    return frozenset(ORDERED_T_TYPES) | {FIXED_COMMENT_KEY}


def _t_type_button_number(t_type: str) -> str:
    return str(ORDERED_T_TYPES.index(t_type) + 1)


def _button_label(slot: str, *, posted: bool, fixed: bool = False) -> str:
    prefix = "✅ " if posted else "📤 "
    if fixed:
        return f"{prefix}{FIXED_COMMENT_TEXT}"
    return f"{prefix}{_t_type_button_number(slot)}"


def build_comment_keyboard(
    token: str,
    comments: dict[str, str],
    *,
    posted: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Clavier inline : bouton fixe ``lowtaper67`` + un numéro par T-type."""
    posted = posted or {}
    rows: list[list[dict[str, str]]] = []

    fixed_cb = f"{_CALLBACK_PREFIX}{token}:{FIXED_COMMENT_KEY}"
    if len(fixed_cb.encode("utf-8")) <= 64:
        rows.append(
            [
                {
                    "text": _button_label(
                        FIXED_COMMENT_KEY,
                        posted=bool(posted.get(FIXED_COMMENT_KEY)),
                        fixed=True,
                    ),
                    "callback_data": fixed_cb,
                }
            ]
        )

    for t_type in ORDERED_T_TYPES:
        text = str(comments.get(t_type) or "").strip()
        if not text:
            continue
        cb = f"{_CALLBACK_PREFIX}{token}:{t_type}"
        if len(cb.encode("utf-8")) > 64:
            continue
        rows.append(
            [
                {
                    "text": _button_label(
                        t_type, posted=bool(posted.get(t_type))
                    ),
                    "callback_data": cb,
                }
            ]
        )
    return {"inline_keyboard": rows}


def send_watcher_post_alert(
    text: str,
    *,
    reply_markup: dict[str, Any] | None = None,
    bot_token: str | None = None,
    chat_id: str | None = None,
) -> dict[str, Any] | None:
    """Envoie l'alerte watcher (Markdown) avec clavier optionnel."""
    token = (bot_token or config.TELEGRAM_BOT_TOKEN or "").strip()
    chat = str(chat_id or config.TELEGRAM_CHAT_ID or "").strip()
    if not token or not chat:
        raise ValueError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID manquants")

    payload: dict[str, Any] = {
        "chat_id": chat,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": False,
    }
    if reply_markup and reply_markup.get("inline_keyboard"):
        payload["reply_markup"] = reply_markup
    data = _telegram_api("sendMessage", payload, token=token)
    if not data or not data.get("ok"):
        raise RuntimeError(f"Telegram sendMessage échoué : {data}")
    return data


def _edit_message_keyboard(
    *,
    chat_id: str,
    message_id: int,
    reply_markup: dict[str, Any],
    token: str,
) -> None:
    _telegram_api(
        "editMessageReplyMarkup",
        {
            "chat_id": chat_id,
            "message_id": message_id,
            "reply_markup": reply_markup,
        },
        token=token,
    )


def _parse_callback(data: str) -> tuple[str, str] | None:
    if not data.startswith(_CALLBACK_PREFIX):
        return None
    rest = data[len(_CALLBACK_PREFIX) :]
    if ":" not in rest:
        return None
    token, t_type = rest.split(":", 1)
    token = token.strip()
    t_type = t_type.strip()
    if not token or t_type not in _callback_slots():
        return None
    return token, t_type


def _post_comment_ig1(media_id: str, comment_text: str) -> tuple[bool, str]:
    """Publie via compte IG slot 0 (cookies principaux)."""
    from playwright.sync_api import sync_playwright

    from scripts.instagram_browser import (
        COOKIES_PATH,
        get_browser_context,
        post_reel_comment,
        session_ok,
    )

    with _IG_POST_LOCK:
        pw = sync_playwright().start()
        try:
            context = get_browser_context(pw, cookies_path=COOKIES_PATH)
            if not session_ok(context):
                return False, "Session Instagram compte 1 invalide — régénérer les cookies."
            try:
                return post_reel_comment(media_id, comment_text, context)
            finally:
                context.close()
                br = context.browser
                if br:
                    br.close()
        finally:
            pw.stop()


def handle_callback(
    callback_query: dict[str, Any],
    *,
    token: str | None = None,
    expected_chat_id: str | None = None,
    post_fn: Callable[[str, str], tuple[bool, str]] | None = None,
) -> str:
    """Traite un clic bouton « publier commentaire »."""
    bot_token = (token or config.TELEGRAM_BOT_TOKEN or "").strip()
    chat_expected = str(expected_chat_id or config.TELEGRAM_CHAT_ID or "").strip()
    post_fn = post_fn or _post_comment_ig1

    cq_id = str(callback_query.get("id") or "")
    data = str(callback_query.get("data") or "")
    msg = callback_query.get("message") or {}
    msg_chat = (msg.get("chat") or {}).get("id")
    msg_id = msg.get("message_id")

    if chat_expected and str(msg_chat) != chat_expected:
        _ack_callback(cq_id, token=bot_token, text="Chat non autorisé")
        return "callback: chat ignoré"

    parsed = _parse_callback(data)
    if parsed is None:
        _ack_callback(cq_id, token=bot_token, text="Action inconnue")
        return f"callback ignoré : {data!r}"

    pending_token, t_type = parsed

    with json_lock(PENDING_PATH):
        store = _load_pending()
        posts = store.get("posts") or {}
        entry = posts.get(pending_token)
        if not isinstance(entry, dict):
            _ack_callback(cq_id, token=bot_token, text="Alerte expirée")
            return f"pending absent : {pending_token}"

        if entry.get("posted", {}).get(t_type):
            _ack_callback(cq_id, token=bot_token, text="Déjà publié")
            return f"déjà posté {t_type}"

        media_id = str(entry.get("media_id") or "")
        comment_text = str((entry.get("comments") or {}).get(t_type) or "").strip()
        if not media_id or not comment_text:
            _ack_callback(cq_id, token=bot_token, text="Données invalides")
            return "media_id ou commentaire manquant"

    # Telegram exige answerCallbackQuery en ~30 s — Playwright IG peut prendre 1 min.
    _ack_callback(cq_id, token=bot_token, text="Publication en cours…")

    ok, err = post_fn(media_id, comment_text)

    creator_username = ""
    markup: dict[str, Any] | None = None
    with json_lock(PENDING_PATH):
        store = _load_pending()
        posts = store.get("posts") or {}
        entry = posts.get(pending_token)
        if isinstance(entry, dict):
            creator_username = str(entry.get("creator_username") or "")
            posted = entry.setdefault("posted", {})
            if ok:
                posted[t_type] = True
                _save_pending(store, use_lock=False)
                comments = entry.get("comments") or {}
                markup = build_comment_keyboard(
                    pending_token, comments, posted=posted
                )

    if ok:
        if markup and msg_id is not None and msg_chat is not None:
            _edit_message_keyboard(
                chat_id=str(msg_chat),
                message_id=int(msg_id),
                reply_markup=markup,
                token=bot_token,
            )
        return (
            f"post OK @{creator_username} {t_type} reel={media_id}"
        )

    if msg_chat is not None:
        _telegram_api(
            "sendMessage",
            {
                "chat_id": str(msg_chat),
                "text": f"❌ Publication {t_type} échouée : {err[:200]}",
                "reply_to_message_id": msg_id,
            },
            token=bot_token,
        )
    return f"post KO : {err}"


def _load_offset() -> int:
    if not STATE_PATH.is_file():
        return 0
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return int(data.get("last_update_id") or 0)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def _save_offset(offset: int) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(STATE_PATH, {"last_update_id": int(offset)})


def run_poller(
    *,
    stop_event: threading.Event | None = None,
    post_fn: Callable[[str, str], tuple[bool, str]] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Long-polling ``getUpdates`` pour les callbacks boutons commentaire."""
    token = (config.TELEGRAM_BOT_TOKEN or "").strip()
    chat_id = str(config.TELEGRAM_CHAT_ID or "").strip()
    if not token or not chat_id:
        _LOG.warning("Poller Telegram Watcher désactivé (token/chat_id manquants).")
        return

    offset = _load_offset()
    _LOG.info("Poller Telegram Watcher démarré (offset=%d, compte IG=1).", offset)

    while stop_event is None or not stop_event.is_set():
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params={"timeout": POLL_TIMEOUT_S, "offset": offset + 1},
                timeout=HTTP_TIMEOUT_S,
            )
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            _LOG.warning("getUpdates erreur : %s", e)
            sleep_fn(5)
            continue

        if not data.get("ok"):
            sleep_fn(5)
            continue

        updates = data.get("result") or []
        for upd in updates:
            try:
                offset = max(offset, int(upd.get("update_id") or 0))
            except (TypeError, ValueError):
                pass
            cq = upd.get("callback_query")
            if cq:
                try:
                    log_line = handle_callback(
                        cq, token=token, expected_chat_id=chat_id, post_fn=post_fn
                    )
                    _LOG.info("Callback : %s", log_line)
                except Exception as e:
                    _LOG.exception("Callback non géré : %s", e)

        if updates:
            _save_offset(offset)

    _save_offset(offset)
    _LOG.info("Poller Telegram Watcher arrêté (offset=%d).", offset)


def start_poller_thread(
    *,
    post_fn: Callable[[str, str], tuple[bool, str]] | None = None,
) -> tuple[threading.Thread, threading.Event]:
    """Lance le poller en thread daemon."""
    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_poller,
        kwargs={"stop_event": stop_event, "post_fn": post_fn},
        name="watcher-telegram-poller",
        daemon=True,
    )
    thread.start()
    return thread, stop_event
