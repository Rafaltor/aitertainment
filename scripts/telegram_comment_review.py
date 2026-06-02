"""Review humaine des commentaires training via Telegram (inline Garder / Supprimer).

Usage::

    # Envoyer le prochain lot (20 par défaut)
    python scripts/telegram_comment_review.py push --limit 20

    # Écouter les boutons (Ctrl-C pour arrêter)
    python scripts/telegram_comment_review.py run

    # Les deux en parallèle : push puis run dans un autre terminal

Utilise ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID`` (Bot #1 Watcher).
État : ``data/comment_review_state.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from modules.pipeline_state import comment_dedup_key

DEFAULT_TRAINING_PATH = _PROJECT_ROOT / "data" / "training_comments_viral.json"
DEFAULT_STATE_PATH = _PROJECT_ROOT / "data" / "comment_review_state.json"
DEFAULT_LOG_PATH = _PROJECT_ROOT / "logs" / "comment_review.log"

TELEGRAM_API = "https://api.telegram.org"
POLL_TIMEOUT_S = 10
HTTP_TIMEOUT_S = 20
PERSIST_DEBOUNCE_S = 0.25
CALLBACK_PREFIX = "tc"
ACTION_KEEP = "k"
ACTION_DELETE = "d"

_LOG = logging.getLogger("aitertainment.comment_review")


class CommentReviewError(RuntimeError):
    """Erreur configuration ou I/O."""


def setup_logger(*, log_path: Path | None = None) -> logging.Logger:
    log = logging.getLogger("aitertainment.comment_review")
    log.setLevel(logging.INFO)
    if any(getattr(h, "_aitertainment_comment_review", False) for h in log.handlers):
        return log
    target = log_path or DEFAULT_LOG_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(target, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        fh._aitertainment_comment_review = True  # type: ignore[attr-defined]
        log.addHandler(fh)
    except OSError:
        pass
    if not any(type(h) is logging.StreamHandler for h in log.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(sh)
    log.propagate = False
    return log


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_credentials() -> tuple[str, str]:
    token = (config.TELEGRAM_BOT_TOKEN or "").strip()
    chat_id = (config.TELEGRAM_CHAT_ID or "").strip()
    if not token or not chat_id:
        raise CommentReviewError(
            "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID manquants dans .env"
        )
    return token, chat_id


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _telegram_post(
    method: str,
    payload: dict[str, Any],
    *,
    token: str,
    timeout: int = 30,
    retries: int = 3,
) -> dict[str, Any] | None:
    url = f"{TELEGRAM_API}/bot{token}/{method}"
    last_err: str | None = None
    for attempt in range(max(retries, 1)):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            last_err = str(e)
            if attempt + 1 < retries:
                time.sleep(0.4 * (attempt + 1))
                continue
            _LOG.warning("Telegram %s : %s", method, e)
            return None
        if resp.status_code == 200 and data.get("ok"):
            return data
        desc = str(data.get("description") or resp.text[:400])
        last_err = desc
        # Callback expiré ou déjà ack — normal si redémarrage ou double-clic.
        if method == "answerCallbackQuery" and (
            "query is too old" in desc.lower()
            or "query id is invalid" in desc.lower()
        ):
            return data
        if attempt + 1 < retries and resp.status_code >= 500:
            time.sleep(0.4 * (attempt + 1))
            continue
        _LOG.warning("Telegram %s non-ok : %s", method, desc[:400])
        return data
    _LOG.warning("Telegram %s échec : %s", method, last_err)
    return None


def _ack_callback(
    callback_query_id: str,
    *,
    token: str,
    text: str | None = None,
) -> None:
    if not callback_query_id:
        return
    payload: dict[str, Any] = {"callback_query_id": callback_query_id}
    if text:
        payload["text"] = text
    _telegram_post("answerCallbackQuery", payload, token=token, timeout=10)


def _telegram_get(
    method: str, params: dict[str, Any], *, token: str, timeout: int = HTTP_TIMEOUT_S
) -> dict[str, Any] | None:
    url = f"{TELEGRAM_API}/bot{token}/{method}"
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        _LOG.warning("Telegram GET %s : %s", method, e)
        return None


def entry_dedup_key(entry: dict[str, Any]) -> str:
    return comment_dedup_key(
        str(entry.get("media_id") or ""),
        str(entry.get("text") or ""),
    )


def entry_short_id(entry: dict[str, Any]) -> str:
    return hashlib.sha256(entry_dedup_key(entry).encode()).hexdigest()[:12]


def load_training(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Retourne ``(entries, wrapper_meta)`` — wrapper_meta si racine objet."""
    if not path.exists():
        raise CommentReviewError(f"Fichier introuvable : {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)], None
    if isinstance(data, dict):
        entries = data.get("entries") or data.get("comments") or []
        if not isinstance(entries, list):
            raise CommentReviewError(f'"entries" invalide dans {path}')
        meta = {k: v for k, v in data.items() if k not in ("entries", "comments")}
        return [e for e in entries if isinstance(e, dict)], meta
    raise CommentReviewError(f"JSON inattendu dans {path}")


def save_training(
    path: Path,
    entries: list[dict[str, Any]],
    wrapper_meta: dict[str, Any] | None,
    *,
    fsync: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Any
    if wrapper_meta is not None:
        payload = {**wrapper_meta, "entries": entries}
    else:
        payload = entries
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        if fsync:
            os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


class DebouncedPersist:
    """Écriture disque différée — l'UI Telegram répond avant le flush JSON."""

    def __init__(
        self,
        *,
        state_path: Path,
        delay_s: float = PERSIST_DEBOUNCE_S,
    ) -> None:
        self.state_path = state_path
        self.delay_s = delay_s
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._cache: TrainingCache | None = None
        self._state: dict[str, Any] | None = None
        self._training_dirty = False
        self._state_dirty = False

    def note_training(self, cache: TrainingCache) -> None:
        with self._lock:
            self._cache = cache
            self._training_dirty = True
            self._arm_timer_locked()

    def note_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._state = state
            self._state_dirty = True
            self._arm_timer_locked()

    def _arm_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self.delay_s, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            cache = self._cache
            state = self._state
            training_dirty = self._training_dirty
            state_dirty = self._state_dirty
            self._training_dirty = False
            self._state_dirty = False
            self._timer = None
        try:
            if training_dirty and cache is not None:
                cache.save(fsync=False)
            if state_dirty and state is not None:
                save_state(self.state_path, state)
        except OSError as e:
            _LOG.warning("Flush disque échoué : %s", e)

    def flush_now(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self._flush()


def default_state() -> dict[str, Any]:
    return {
        "last_update_id": 0,
        "pending": {},
        "resolved": {},
        "reviewed": {},
        "stats": {"kept": 0, "removed": 0, "pushed": 0},
    }


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise CommentReviewError(f"État illisible {path} : {e}") from e
    if not isinstance(data, dict):
        return default_state()
    base = default_state()
    base.update(data)
    if not isinstance(base.get("pending"), dict):
        base["pending"] = {}
    if not isinstance(base.get("resolved"), dict):
        base["resolved"] = {}
    if not isinstance(base.get("reviewed"), dict):
        base["reviewed"] = {}
    if not isinstance(base.get("stats"), dict):
        base["stats"] = {"kept": 0, "removed": 0, "pushed": 0}
    return base


def save_state(path: Path, state: dict[str, Any]) -> None:
    _atomic_write_json(path, state)


def _truncate(text: str, n: int) -> str:
    s = (text or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def _format_hashtags(value: Any) -> str:
    if isinstance(value, list):
        tags = [str(h).strip() for h in value if str(h).strip()]
        return ", ".join(tags)
    return str(value or "").strip()


def format_review_message(
    entry: dict[str, Any],
    *,
    position: int,
    total: int,
    status: str | None = None,
) -> str:
    username = html.escape(str(entry.get("username") or "?").lstrip("@"))
    t_type = html.escape(str(entry.get("t_type") or "?"))
    t_profile = html.escape(str(entry.get("t_type_profile") or "?"))
    text = html.escape(_truncate(str(entry.get("text") or ""), 400))
    caption = html.escape(_truncate(str(entry.get("caption") or ""), 120))
    hashtags = html.escape(_format_hashtags(entry.get("hashtags")))
    niches = entry.get("niches") or []
    niche_str = html.escape(
        ", ".join(str(n).strip() for n in niches if str(n).strip()) or "?"
    )
    likes = int(entry.get("comment_likes") or 0)

    header = "📝 <b>Review training</b>"
    if status == "kept":
        header = "✅ <b>Gardé</b>"
    elif status == "removed":
        header = "❌ <b>Supprimé</b>"

    lines = [
        header,
        f"#{position}/{total}",
        f"👤 @{username} · label <b>{t_type}</b> · profil {t_profile}",
        f"🏷 {niche_str} · ❤️ {likes} likes",
        f"💬 <i>{text}</i>",
    ]
    if caption:
        lines.append(f"🎬 {caption}")
    if hashtags:
        lines.append(f"# {hashtags}")
    return "\n".join(lines)


def build_keyboard(short_id: str) -> dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "✅ Garder", "callback_data": f"{CALLBACK_PREFIX}:{ACTION_KEEP}:{short_id}"},
                {"text": "🗑 Supprimer", "callback_data": f"{CALLBACK_PREFIX}:{ACTION_DELETE}:{short_id}"},
            ]
        ]
    }


def _unreviewed_entries(
    entries: list[dict[str, Any]], state: dict[str, Any]
) -> list[dict[str, Any]]:
    reviewed = state.get("reviewed") or {}
    pending = state.get("pending") or {}
    pending_keys = {
        str(item.get("dedup_key") or "")
        for item in pending.values()
        if isinstance(item, dict)
    }
    out: list[dict[str, Any]] = []
    for entry in entries:
        key = entry_dedup_key(entry)
        if entry.get("human_validated") is True:
            if key not in reviewed:
                reviewed[key] = "kept"
            continue
        if key in reviewed or key in pending_keys:
            continue
        out.append(entry)
    state["reviewed"] = reviewed
    return out


def push_comments(
    *,
    training_path: Path,
    state_path: Path,
    limit: int,
    token: str,
    chat_id: str,
    sleep_s: float = 0.15,
) -> int:
    entries, wrapper_meta = load_training(training_path)
    state = load_state(state_path)
    queue = _unreviewed_entries(entries, state)
    total = len(entries)
    already = len(state.get("reviewed") or {}) + len(state.get("pending") or {})

    if not queue:
        _LOG.info("Rien à envoyer — %d/%d déjà traités ou en attente.", already, total)
        save_state(state_path, state)
        return 0

    batch = queue[: max(limit, 0)]
    sent = 0
    for i, entry in enumerate(batch, start=1):
        short_id = entry_short_id(entry)
        key = entry_dedup_key(entry)
        position = already + i
        text = format_review_message(entry, position=position, total=total)
        resp = _telegram_post(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
                "reply_markup": build_keyboard(short_id),
            },
            token=token,
        )
        if not resp:
            _LOG.warning("Échec envoi commentaire %s — stop batch.", short_id)
            break
        result = resp.get("result") or {}
        message_id = result.get("message_id")
        if message_id is None:
            _LOG.warning("Pas de message_id pour %s — skip.", short_id)
            continue
        state["pending"][short_id] = {
            "dedup_key": key,
            "message_id": int(message_id),
            "chat_id": str(chat_id),
            "pushed_at": _utc_now_iso(),
        }
        state["stats"]["pushed"] = int(state["stats"].get("pushed") or 0) + 1
        sent += 1
        if sleep_s > 0 and i < len(batch):
            time.sleep(sleep_s)

    save_state(state_path, state)
    _LOG.info(
        "Envoyé %d message(s) — %d restant(s) à reviewer.",
        sent,
        max(len(queue) - sent, 0),
    )
    return sent


def _find_entry_index(entries: list[dict[str, Any]], dedup_key: str) -> int | None:
    for i, entry in enumerate(entries):
        if entry_dedup_key(entry) == dedup_key:
            return i
    return None


class TrainingCache:
    """Cache mémoire pour éviter de relire training_comments_viral.json à chaque clic."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: list[dict[str, Any]] = []
        self.wrapper_meta: dict[str, Any] | None = None
        self._mtime_ns: int | None = None
        self.reload(force=True)

    def reload(self, *, force: bool = False) -> None:
        if not self.path.exists():
            self.entries = []
            self.wrapper_meta = None
            self._mtime_ns = None
            return
        mtime_ns = self.path.stat().st_mtime_ns
        if not force and self._mtime_ns == mtime_ns:
            return
        self.entries, self.wrapper_meta = load_training(self.path)
        self._mtime_ns = mtime_ns

    def save(self, *, fsync: bool = True) -> None:
        save_training(self.path, self.entries, self.wrapper_meta, fsync=fsync)
        if self.path.exists():
            self._mtime_ns = self.path.stat().st_mtime_ns

    def mark_dirty(self, persist: DebouncedPersist | None) -> None:
        if persist is not None:
            persist.note_training(self)
        else:
            self.save()


def apply_keep(
    *,
    cache: TrainingCache,
    state: dict[str, Any],
    short_id: str,
    persist: DebouncedPersist | None = None,
) -> tuple[dict[str, Any] | None, str, dict[str, Any] | None]:
    pending = state.get("pending") or {}
    item = pending.get(short_id)
    if not isinstance(item, dict):
        return None, "pending introuvable", None
    dedup = str(item.get("dedup_key") or "")
    entries = cache.entries
    idx = _find_entry_index(entries, dedup)
    entry_snapshot: dict[str, Any] | None = None
    if idx is None:
        state["reviewed"][dedup] = "kept"
        pending.pop(short_id, None)
        state["stats"]["kept"] = int(state["stats"].get("kept") or 0) + 1
        return item, "entrée déjà absente (marquée gardée)", None

    entry_snapshot = dict(entries[idx])
    entries[idx]["human_validated"] = True
    entries[idx]["human_validated_at"] = _utc_now_iso()
    entries[idx]["label_source"] = "human"
    cache.mark_dirty(persist)
    state["reviewed"][dedup] = "kept"
    _record_resolved(state, short_id=short_id, item=item, dedup=dedup, status="kept")
    pending.pop(short_id, None)
    state["stats"]["kept"] = int(state["stats"].get("kept") or 0) + 1
    return item, "gardé", entry_snapshot


def apply_delete(
    *,
    cache: TrainingCache,
    state: dict[str, Any],
    short_id: str,
    persist: DebouncedPersist | None = None,
) -> tuple[dict[str, Any] | None, str, dict[str, Any] | None]:
    pending = state.get("pending") or {}
    item = pending.get(short_id)
    if not isinstance(item, dict):
        return None, "pending introuvable", None
    dedup = str(item.get("dedup_key") or "")
    entries = cache.entries
    idx = _find_entry_index(entries, dedup)
    entry_snapshot: dict[str, Any] | None = None
    if idx is not None:
        entry_snapshot = dict(entries[idx])
        entries.pop(idx)
        cache.mark_dirty(persist)
    state["reviewed"][dedup] = "removed"
    _record_resolved(state, short_id=short_id, item=item, dedup=dedup, status="removed")
    pending.pop(short_id, None)
    state["stats"]["removed"] = int(state["stats"].get("removed") or 0) + 1
    return item, "supprimé", entry_snapshot


def _clear_message_buttons(
    chat_id: str | int,
    message_id: int | str,
    *,
    token: str,
) -> None:
    _telegram_post(
        "editMessageReplyMarkup",
        {
            "chat_id": chat_id,
            "message_id": message_id,
            "reply_markup": {"inline_keyboard": []},
        },
        token=token,
        timeout=8,
    )


def _record_resolved(
    state: dict[str, Any],
    *,
    short_id: str,
    item: dict[str, Any],
    dedup: str,
    status: str,
) -> None:
    resolved = state.setdefault("resolved", {})
    if not isinstance(resolved, dict):
        resolved = {}
        state["resolved"] = resolved
    resolved[short_id] = {
        "dedup_key": dedup,
        "status": status,
        "message_id": item.get("message_id"),
        "chat_id": item.get("chat_id"),
        "resolved_at": _utc_now_iso(),
    }


def _handle_stale_callback(
    *,
    short_id: str,
    cq_id: str,
    chat_id: str,
    message_id: int | str | None,
    state: dict[str, Any],
    token: str,
) -> str:
    """Callback sur un message déjà traité (double-clic ou re-clic)."""
    _ack_callback(cq_id, token=token, text="Déjà traité")
    resolved = state.get("resolved") or {}
    meta = resolved.get(short_id) if isinstance(resolved, dict) else None
    if message_id and meta:
        status = str(meta.get("status") or "")
        label = "✅ Déjà gardé" if status == "kept" else "❌ Déjà supprimé"
        _telegram_post(
            "editMessageText",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": f"{label} <i>(double-clic ignoré)</i>",
                "parse_mode": "HTML",
            },
            token=token,
            timeout=8,
        )
    elif message_id:
        _clear_message_buttons(chat_id, message_id, token=token)
    _LOG.debug("Callback stale ignoré : %s", short_id)
    return "déjà traité"


def _parse_callback(data: str) -> tuple[str, str] | None:
    parts = (data or "").split(":")
    if len(parts) != 3:
        return None
    prefix, action, short_id = parts
    if prefix != CALLBACK_PREFIX or action not in (ACTION_KEEP, ACTION_DELETE):
        return None
    if not short_id or len(short_id) > 32:
        return None
    return action, short_id


def handle_callback(
    callback_query: dict[str, Any],
    *,
    cache: TrainingCache,
    state: dict[str, Any],
    state_path: Path,
    token: str,
    expected_chat_id: str,
    persist: DebouncedPersist | None = None,
) -> str:
    cq_id = str(callback_query.get("id") or "")
    data = str(callback_query.get("data") or "")
    msg = callback_query.get("message") or {}
    chat_id = str((msg.get("chat") or {}).get("id") or "")
    message_id = msg.get("message_id")

    if expected_chat_id and chat_id and chat_id != str(expected_chat_id):
        _LOG.warning("Callback ignoré (chat %s)", chat_id)
        return "ignored"

    parsed = _parse_callback(data)
    if not parsed:
        _ack_callback(cq_id, token=token, text="?")
        return "unknown"

    action, short_id = parsed

    pending = state.get("pending") or {}
    if short_id not in pending:
        return _handle_stale_callback(
            short_id=short_id,
            cq_id=cq_id,
            chat_id=chat_id,
            message_id=message_id,
            state=state,
            token=token,
        )

    ack = "Gardé ✓" if action == ACTION_KEEP else "Supprimé"
    _ack_callback(cq_id, token=token, text=ack)

    if action == ACTION_KEEP:
        _, note, entry = apply_keep(
            cache=cache, state=state, short_id=short_id, persist=persist
        )
        status = "kept"
    else:
        _, note, entry = apply_delete(
            cache=cache, state=state, short_id=short_id, persist=persist
        )
        status = "removed"

    if message_id and entry:
        new_text = format_review_message(
            entry,
            position=0,
            total=len(cache.entries),
            status=status,
        )
        _telegram_post(
            "editMessageText",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": new_text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            token=token,
            timeout=8,
        )
    elif message_id:
        _telegram_post(
            "editMessageText",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": f"{'✅ Gardé' if status == 'kept' else '❌ Supprimé'} — {note}",
                "parse_mode": "HTML",
            },
            token=token,
            timeout=8,
        )

    if persist is not None:
        persist.note_state(state)
    else:
        save_state(state_path, state)

    _LOG.info("%s (%s) — %s", short_id, action, note)
    return note


def run_bot(
    *,
    training_path: Path,
    state_path: Path,
    poll_timeout_s: int = POLL_TIMEOUT_S,
    sleep_fn=time.sleep,
) -> None:
    token, chat_id = _resolve_credentials()
    cache = TrainingCache(training_path)
    state = load_state(state_path)
    persist = DebouncedPersist(state_path=state_path)
    offset = int(state.get("last_update_id") or 0)
    _LOG.info(
        "=== Comment review bot (offset=%d, training=%d entrées) ===",
        offset,
        len(cache.entries),
    )

    try:
        while True:
            data = _telegram_get(
                "getUpdates",
                {"timeout": poll_timeout_s, "offset": offset + 1},
                token=token,
                timeout=poll_timeout_s + 5,
            )
            if not data or not data.get("ok"):
                _LOG.warning("getUpdates échec — retry 5s")
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
                    handle_callback(
                        cq,
                        cache=cache,
                        state=state,
                        state_path=state_path,
                        token=token,
                        expected_chat_id=str(chat_id),
                        persist=persist,
                    )
            if updates:
                state["last_update_id"] = offset
                persist.note_state(state)
    except KeyboardInterrupt:
        _LOG.info("Arrêt (Ctrl-C).")
    finally:
        state["last_update_id"] = offset
        persist.note_state(state)
        persist.flush_now()


def cmd_status(training_path: Path, state_path: Path) -> None:
    entries, _ = load_training(training_path)
    state = load_state(state_path)
    reviewed = state.get("reviewed") or {}
    pending = state.get("pending") or {}
    kept_file = sum(1 for e in entries if e.get("human_validated"))
    _LOG.info("Training : %d entrées", len(entries))
    _LOG.info("Validés humain (fichier) : %d", kept_file)
    _LOG.info("Review state — pending: %d, reviewed: %d", len(pending), len(reviewed))
    _LOG.info("Stats : %s", state.get("stats"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Review Telegram des commentaires training (Garder / Supprimer).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_push = sub.add_parser("push", help="Envoie des commentaires à reviewer.")
    p_push.add_argument("--limit", type=int, default=20, help="Nb de messages (défaut 20).")
    p_push.add_argument("--training-path", type=Path, default=DEFAULT_TRAINING_PATH)
    p_push.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)

    p_run = sub.add_parser("run", help="Écoute les callbacks Telegram.")
    p_run.add_argument("--training-path", type=Path, default=DEFAULT_TRAINING_PATH)
    p_run.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)

    p_status = sub.add_parser("status", help="Affiche la progression.")
    p_status.add_argument("--training-path", type=Path, default=DEFAULT_TRAINING_PATH)
    p_status.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)

    args = parser.parse_args(argv)
    setup_logger()

    training_path = (
        args.training_path
        if args.training_path.is_absolute()
        else _PROJECT_ROOT / args.training_path
    )
    state_path = (
        args.state_path
        if args.state_path.is_absolute()
        else _PROJECT_ROOT / args.state_path
    )

    try:
        if args.command == "push":
            token, chat_id = _resolve_credentials()
            push_comments(
                training_path=training_path,
                state_path=state_path,
                limit=args.limit,
                token=token,
                chat_id=chat_id,
            )
        elif args.command == "run":
            run_bot(training_path=training_path, state_path=state_path)
        elif args.command == "status":
            cmd_status(training_path, state_path)
    except CommentReviewError as e:
        _LOG.error("%s", e)
        return 1
    except OSError as e:
        _LOG.error("I/O : %s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
