"""Bot Telegram #2 — validation humaine des candidats Discovery.

============================================================================
Architecture
============================================================================

Ce bot est **séparé** du Bot #1 (alertes Watcher) pour deux raisons :

1. **Hygiène du chat** : les alertes virales (Watcher, fréquentes, time-critical)
   ne se mélangent pas avec les revues humaines de candidats (Discovery, lentes,
   asynchrones).
2. **Sécurité** : le bot Discovery accepte des **callbacks** (validation,
   rejet, modification de T-type). On filtre les callbacks au ``chat_id`` du
   bot, ce qui isole les permissions.

Endpoints utilisés (``api.telegram.org/bot{token}``) :
    - ``sendMessage``        — push initial du candidat avec inline keyboard.
    - ``answerCallbackQuery``— ack rapide pour faire disparaître le spinner.
    - ``editMessageText``    — modifie le message d'origine après action.
    - ``getUpdates``         — long-polling pour récupérer les callbacks.

Callbacks ``callback_data`` (chaîne ≤ 64 octets, contrainte Telegram) :
    - ``v:{username}``         — validate
    - ``r:{username}``         — reject
    - ``m:{username}``         — open T-type menu
    - ``s:{T_TYPE}:{username}``— set t_type final (validate avec correction)
    - ``ev:{username}``        — afficher l'évolution (historique de scores)

Le candidat est récupéré dans ``data/candidates.json`` au moment du callback
(indexé par username). Pas d'état serveur en mémoire — tout est sur disque,
le bot redémarre proprement après crash.

============================================================================
Flux
============================================================================

1. ``discovery.explore_network`` détecte un candidat → appelle
   ``notify_candidate(candidate)``.
2. L'humain reçoit le message + boutons.
3. Action ``✅`` → ``add_to_watchlist`` + ``append_validation`` + retrait de
   ``candidates.json``.
4. Action ``❌`` → ``append_validation(action="rejected")`` + retrait de
   ``candidates.json`` (déjà blacklisté en Discovery).
5. Action ``✏️`` → menu T-types puis ``add_to_watchlist`` avec correction +
   ``append_validation`` + retrait de ``candidates.json``.
6. Action ``👁`` → ouvre directement l'URL Instagram (bouton URL inline,
   aucun callback côté bot).

``data/validations.json`` constitue le **dataset feedback loop** : c'est lui
qui servira à fine-tuner le classifieur T1–T5 dans une phase ultérieure.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import requests

import config
from config import VALID_T_TYPES

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_VALIDATIONS_PATH = DEFAULT_DATA_DIR / "validations.json"
DEFAULT_BOT_STATE_PATH = DEFAULT_DATA_DIR / "discovery_bot_state.json"
DEFAULT_LOG_PATH = _PROJECT_ROOT / "logs" / "discovery_bot.log"

T_TYPES_AVAILABLE: tuple[str, ...] = tuple(sorted(VALID_T_TYPES))
TREND_EMOJI = {"rising": "📈", "stable": "➡️", "declining": "📉"}

TELEGRAM_API_BASE = "https://api.telegram.org"

POLL_TIMEOUT_S = 25  # long-polling (Telegram serveur tient la connexion)
HTTP_TIMEOUT_S = 30   # > POLL_TIMEOUT_S pour ne pas couper avant Telegram

_LOG = logging.getLogger("aitertainment.discovery_bot")


class DiscoveryBotConfigError(RuntimeError):
    """Token / chat_id Discovery manquant."""


class DiscoveryBotIOError(RuntimeError):
    """I/O sur ``validations.json`` ou état du bot."""


# ---------------------------------------------------------------------------
# Logger dédié
# ---------------------------------------------------------------------------


def setup_bot_logger(
    *, log_path: Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    log = logging.getLogger("aitertainment.discovery_bot")
    log.setLevel(level)
    if any(getattr(h, "_aitertainment_discovery_bot", False) for h in log.handlers):
        return log
    target = Path(log_path) if log_path else DEFAULT_LOG_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(target, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        fh._aitertainment_discovery_bot = True  # type: ignore[attr-defined]
        log.addHandler(fh)
    except OSError:
        pass
    has_console = any(type(h) is logging.StreamHandler for h in log.handlers)
    if not has_console:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(sh)
    log.propagate = False
    return log


# ---------------------------------------------------------------------------
# Helpers I/O atomique
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise DiscoveryBotIOError(f"lecture {path} : {e}") from e


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
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
    except OSError as e:
        raise DiscoveryBotIOError(f"écriture {path} : {e}") from e


# ---------------------------------------------------------------------------
# Validations log (feedback loop)
# ---------------------------------------------------------------------------


def load_validations(*, path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_VALIDATIONS_PATH
    raw = _read_json(p, default={"validations": []})
    if not isinstance(raw, dict) or not isinstance(raw.get("validations"), list):
        raise DiscoveryBotIOError(f"{p} : schéma invalide.")
    return raw


def save_validations(data: dict[str, Any], *, path: Path | None = None) -> None:
    p = Path(path) if path else DEFAULT_VALIDATIONS_PATH
    _atomic_write_json(p, data)


def append_validation(
    entry: dict[str, Any], *, path: Path | None = None
) -> dict[str, Any]:
    """Append (atomique) à ``validations.json`` et retourne l'entrée écrite."""
    p = Path(path) if path else DEFAULT_VALIDATIONS_PATH
    try:
        data = load_validations(path=p)
    except DiscoveryBotIOError:
        data = {"validations": []}
    record = {**entry, "validated_at": entry.get("validated_at") or _now_iso()}
    data["validations"].append(record)
    save_validations(data, path=p)
    return record


# ---------------------------------------------------------------------------
# Récupération du candidat (depuis candidates.json)
# ---------------------------------------------------------------------------


def _find_candidate(
    username: str, *, candidates_path: Path | None = None
) -> dict[str, Any] | None:
    """Retrouve un candidat dans ``candidates.json`` par username (case-insensitive)."""
    from discovery import DEFAULT_CANDIDATES_PATH, load_candidates  # tardif

    p = Path(candidates_path) if candidates_path else DEFAULT_CANDIDATES_PATH
    try:
        data = load_candidates(path=p)
    except Exception:
        return None
    target = username.lstrip("@").strip().lower()
    for c in data.get("candidates", []):
        if not isinstance(c, dict):
            continue
        if str(c.get("username") or "").lstrip("@").lower() == target:
            return c
    return None


def _remove_candidate(
    username: str,
    *,
    candidates_path: Path | None = None,
) -> bool:
    """Retire un candidat de ``candidates.json`` après validation / rejet."""
    from discovery import (  # tardif
        DEFAULT_CANDIDATES_PATH,
        DiscoveryIOError,
        load_candidates,
        save_candidates,
    )

    p = Path(candidates_path) if candidates_path else DEFAULT_CANDIDATES_PATH
    target = username.lstrip("@").strip().lower()
    if not target:
        return False
    try:
        data = load_candidates(path=p)
    except DiscoveryIOError:
        return False
    arr = data.get("candidates", [])
    kept = [
        c
        for c in arr
        if isinstance(c, dict)
        and str(c.get("username") or "").lstrip("@").strip().lower() != target
    ]
    if len(kept) == len(arr):
        return False
    try:
        save_candidates({"candidates": kept}, path=p)
    except DiscoveryIOError as e:
        _LOG.warning(
            "remove_candidate : écriture impossible pour @%s (%s).", username, e
        )
        return False
    return True


def _domain_meta(
    domain_name: str, *, seeds_path: Path | None = None
) -> dict[str, Any] | None:
    """Retrouve le domaine dans ``seeds.json`` (utile pour `niche`)."""
    from discovery import load_seeds  # tardif

    try:
        data = load_seeds(path=seeds_path)
    except Exception:
        return None
    for d in data.get("domains", []):
        if isinstance(d, dict) and (d.get("name") or "").lower() == domain_name.lower():
            return d
    return None


# ---------------------------------------------------------------------------
# Construction du message
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int) -> str:
    s = (text or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def _format_score(value: Any) -> str:
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return "?"


def _format_ratio(value: Any) -> str:
    try:
        return f"{float(value):.2f}x"
    except (TypeError, ValueError):
        return "?"


def _build_candidate_text(candidate: dict[str, Any]) -> str:
    """Texte HTML du message Telegram (HTML > Markdown : pas d'échappement piégeux).

    Affiche les métriques par type (Reels / Posts) si disponibles, sinon retombe
    sur les anciennes clés ``ratio_median`` / ``ratio_trend`` pour rétro-compat.
    """
    username = str(candidate.get("username") or "?")
    followers = candidate.get("followers")
    score = _format_score(candidate.get("score"))
    t_dom = str(candidate.get("t_type_dominant") or "?")
    domain = str(candidate.get("domain") or "?")
    bio = _truncate(str(candidate.get("biography") or ""), 80)

    reels_n = int(candidate.get("reels_count") or 0)
    posts_n = int(candidate.get("posts_count") or 0)
    reel_w = candidate.get("reel_weight")
    post_w = candidate.get("post_weight")

    lines = [
        "🔍 <b>Nouveau candidat</b>",
        f"👤 @{username} · <b>{followers}</b> followers"
        if followers is not None
        else f"👤 @{username}",
        f"📊 Score : <b>{score}/1000</b>",
    ]

    if reels_n or posts_n:
        if reels_n:
            reel_ratio = _format_ratio(candidate.get("reel_ratio_median"))
            reel_trend = str(candidate.get("reel_trend") or "stable")
            trend_e = TREND_EMOJI.get(reel_trend, "➡️")
            w = f" · w={float(reel_w):.2f}" if reel_w is not None else ""
            lines.append(
                f"🎞 Reels {reels_n}{w} · ratio {reel_ratio} · {trend_e} {reel_trend}"
            )
        if posts_n:
            post_ratio = candidate.get("post_ratio_median")
            try:
                ratio_str = f"{float(post_ratio):.3f}" if post_ratio is not None else "?"
            except (TypeError, ValueError):
                ratio_str = "?"
            w = f" · w={float(post_w):.2f}" if post_w is not None else ""
            lines.append(
                f"🖼 Posts {posts_n}{w} · likes/follow {ratio_str}"
            )
    else:
        # rétro-compat : ancien schéma sans segmentation
        ratio_med = _format_ratio(candidate.get("ratio_median"))
        trend = str(candidate.get("ratio_trend") or "stable")
        trend_e = TREND_EMOJI.get(trend, "➡️")
        lines.append(f"📈 Ratio médian : {ratio_med} · Tendance : {trend_e} {trend}")

    lines.append(f"🎭 T-type estimé : <b>{t_dom}</b>")
    lines.append(f"🏷  Domaine : {domain}")
    if bio:
        lines.append(f"📝 Bio : {bio}")
    return "\n".join(lines)


def _normalize_username_for_url(username: str) -> str:
    """Nettoie un username pour le passer dans une URL Instagram.

    Instagram tolère uniquement ``[a-zA-Z0-9._]`` dans les usernames. On
    enlève tous les ``@`` initiaux (un caller peut envoyer ``"@@user"`` par
    erreur) puis on strip + lower. On ne ``urlquote`` PAS — un username
    contenant des caractères hors charset Instagram est de toute façon
    invalide, et l'URL résultante doit rester lisible (Telegram l'affiche
    en tooltip au survol sur desktop).
    """
    u = (username or "").strip()
    # Boucle sur ``lstrip("@")`` pour absorber ``"@@user"`` ou ``"@ @user"``.
    while u.startswith("@"):
        u = u[1:].lstrip()
    return u.strip().lower()


def _build_candidate_keyboard(username: str) -> dict[str, Any]:
    """Construit le clavier inline avec les **4 boutons obligatoires** + 1 bonus.

    Les 4 boutons exigés par le brief :

    * ``✅ Valider``         → callback ``v:{username}``
    * ``❌ Rejeter``         → callback ``r:{username}``
    * ``✏️ Modifier T-type`` → callback ``m:{username}``
    * ``👁 Voir profil``      → URL ``https://www.instagram.com/{username}/``

    Le 5e bouton (``📈 Voir évolution``, callback ``ev:{username}``) est
    conservé pour les profils ayant un ``scores_history`` ; il n'est pas
    requis par le brief mais utile au validateur.
    """
    u = _normalize_username_for_url(username)
    if not u:
        # On loggue mais on construit quand même un keyboard "vide" plutôt
        # que de retourner ``None`` — sinon le caller envoie un message sans
        # boutons silencieusement (régression difficile à diagnostiquer).
        _LOG.warning("_build_candidate_keyboard : username vide reçu — keyboard avec callbacks orphelins.")
    profile_url = f"https://www.instagram.com/{u}/"
    return {
        "inline_keyboard": [
            [
                {"text": "✅ Valider", "callback_data": f"v:{u}"},
                {"text": "❌ Rejeter", "callback_data": f"r:{u}"},
            ],
            [
                {"text": "✏️ Modifier T-type", "callback_data": f"m:{u}"},
                {"text": "👁 Voir profil", "url": profile_url},
            ],
            [
                {"text": "📈 Voir évolution", "callback_data": f"ev:{u}"},
            ],
        ]
    }


def _build_t_type_keyboard(username: str) -> dict[str, Any]:
    u = username.lstrip("@").strip()
    row1 = [
        {"text": t, "callback_data": f"s:{t}:{u}"}
        for t in T_TYPES_AVAILABLE[:4]
    ]
    row2 = [
        {"text": t, "callback_data": f"s:{t}:{u}"}
        for t in T_TYPES_AVAILABLE[4:]
    ]
    return {"inline_keyboard": [row1, row2]}


# ---------------------------------------------------------------------------
# HTTP Telegram (best-effort, jamais raise sur erreurs réseau)
# ---------------------------------------------------------------------------


def _resolve_credentials(
    *, token: str | None = None, chat_id: str | None = None
) -> tuple[str, str]:
    t = (token or config.TELEGRAM_DISCOVERY_TOKEN or "").strip()
    c = (chat_id or config.TELEGRAM_DISCOVERY_CHAT_ID or "").strip()
    if not t or not c:
        raise DiscoveryBotConfigError(
            "TELEGRAM_DISCOVERY_TOKEN / TELEGRAM_DISCOVERY_CHAT_ID manquants — éditez .env"
        )
    return t, c


def _telegram_post(
    method: str,
    payload: dict[str, Any],
    *,
    token: str,
    timeout: int = 10,
) -> dict[str, Any] | None:
    url = f"{TELEGRAM_API_BASE}/bot{token}/{method}"
    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as e:
        _LOG.warning("Telegram %s erreur réseau (%s)", method, e)
        return None
    try:
        data = r.json()
    except ValueError:
        _LOG.warning("Telegram %s : JSON invalide (status=%s)", method, r.status_code)
        return None
    if r.status_code != 200 or not data.get("ok"):
        # On loggue le **body brut** (tronqué à 500 chars pour ne pas
        # spammer les logs sur des erreurs verbeuses style ``description``
        # + ``parameters``). Le ``description`` reste utile en accès rapide
        # mais Telegram peut renvoyer des indices critiques uniquement dans
        # le corps complet (ex. ``parameters.retry_after``, champs
        # ``error_code`` détaillés, etc.).
        _LOG.warning(
            "Telegram %s non-ok : status=%s body=%s",
            method,
            r.status_code,
            r.text[:500],
        )
    return data


def _telegram_get(
    method: str, params: dict[str, Any], *, token: str, timeout: int = HTTP_TIMEOUT_S
) -> dict[str, Any] | None:
    url = f"{TELEGRAM_API_BASE}/bot{token}/{method}"
    try:
        r = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as e:
        _LOG.warning("Telegram %s GET erreur (%s)", method, e)
        return None
    try:
        return r.json()
    except ValueError:
        _LOG.warning("Telegram %s GET : JSON invalide (status=%s)", method, r.status_code)
        return None


# ---------------------------------------------------------------------------
# API publique : push & édition
# ---------------------------------------------------------------------------


def notify_candidate(
    candidate: dict[str, Any],
    *,
    token: str | None = None,
    chat_id: str | None = None,
    mock: bool = False,
) -> dict[str, Any] | None:
    """Pousse le candidat dans le chat Discovery avec les boutons inline.

    Retourne la réponse Telegram (dict) ou ``None`` (mock / erreur réseau).
    """
    setup_bot_logger()
    if mock:
        _LOG.info("[mock] notify_candidate @%s", candidate.get("username"))
        return None
    try:
        t, c = _resolve_credentials(token=token, chat_id=chat_id)
    except DiscoveryBotConfigError as e:
        _LOG.warning("notify_candidate skip : %s", e)
        return None

    username = _normalize_username_for_url(str(candidate.get("username") or ""))
    if not username:
        _LOG.warning("notify_candidate : username vide — skip.")
        return None

    reply_markup = _build_candidate_keyboard(username)
    payload = {
        "chat_id": c,
        "text": _build_candidate_text(candidate),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
        "reply_markup": reply_markup,
    }

    # Debug ciblé : quand le validateur signale "boutons absents", la 1re
    # vérification est de s'assurer que le keyboard est bien construit ET
    # bien sérialisé dans le payload envoyé. On loggue donc la liste des
    # boutons (text + callback ou URL) pour pouvoir corréler rapidement
    # avec l'absence visuelle côté Telegram.
    _log_reply_markup_debug(username, reply_markup)

    # Trace DEBUG du payload complet : utile quand Telegram refuse le push
    # (parse_mode HTML invalide, callback_data > 64 octets, reply_markup
    # mal sérialisé…). Désactivé par défaut (niveau DEBUG), ne pollue donc
    # pas les logs de prod.
    _LOG.debug(
        "notify_candidate payload: %s",
        json.dumps(payload, ensure_ascii=False),
    )

    response = _telegram_post("sendMessage", payload, token=t)
    if response and response.get("ok"):
        msg_id = (response.get("result") or {}).get("message_id")
        _LOG.info("notify_candidate @%s : envoyé (message_id=%s)", username, msg_id)
    elif response is not None:
        _LOG.warning(
            "notify_candidate @%s : Telegram a refusé (description=%s).",
            username,
            response.get("description"),
        )
    return response


def _log_reply_markup_debug(username: str, reply_markup: dict[str, Any] | None) -> None:
    """Inspecte ``reply_markup`` et loggue un résumé structuré.

    Émet un ``WARNING`` si le keyboard est ``None`` ou structurellement
    invalide (régression que ce log doit attraper en priorité), sinon un
    ``INFO`` listant les boutons par ligne. Tolère silencieusement les
    types inattendus pour ne **jamais** faire planter ``notify_candidate``
    à cause d'un log debug.
    """
    if reply_markup is None:
        _LOG.warning("notify_candidate @%s : reply_markup=None (boutons ABSENTS).", username)
        return
    rows = (reply_markup or {}).get("inline_keyboard")
    if not isinstance(rows, list) or not rows:
        _LOG.warning(
            "notify_candidate @%s : reply_markup invalide (inline_keyboard manquant ou vide) — %r",
            username,
            reply_markup,
        )
        return

    summary: list[str] = []
    total = 0
    for row in rows:
        if not isinstance(row, list):
            continue
        for btn in row:
            if not isinstance(btn, dict):
                continue
            total += 1
            text = str(btn.get("text") or "?")
            target = btn.get("callback_data") or btn.get("url") or "(sans cible)"
            summary.append(f"{text} → {target}")

    _LOG.info(
        "notify_candidate @%s : reply_markup = %d boutons sur %d ligne(s) [%s]",
        username,
        total,
        len(rows),
        " | ".join(summary),
    )


def _format_score_evolution_text(
    username: str,
    old_score: float,
    new_score: float,
    old_tier: str,
    new_tier: str,
) -> str:
    """Texte court pour ``notify_score_evolution``.

    Format : ``📈 @raikkonenaf : 448 → 745 (+66%) — Tier B→A``.

    L'emoji suit le **sens** de la variation (pas son amplitude). Si
    ``old_score`` est ≤ 0 ou ``None``, on affiche ``±0%`` plutôt que de
    faire planter le formattage.
    """
    u = (username or "").lstrip("@").strip()
    try:
        old_f = float(old_score) if old_score is not None else 0.0
    except (TypeError, ValueError):
        old_f = 0.0
    try:
        new_f = float(new_score) if new_score is not None else 0.0
    except (TypeError, ValueError):
        new_f = 0.0

    if old_f > 0:
        pct = (new_f - old_f) / old_f * 100.0
    else:
        pct = 0.0

    arrow = "📈" if new_f > old_f else "📉" if new_f < old_f else "➡️"
    sign = "+" if pct >= 0 else ""
    return (
        f"{arrow} @{u} : {old_f:.0f} → {new_f:.0f} "
        f"({sign}{pct:.0f}%) — Tier {old_tier or '?'}→{new_tier or '?'}"
    )


def notify_score_evolution(
    username: str,
    old_score: float,
    new_score: float,
    old_tier: str,
    new_tier: str,
    *,
    token: str | None = None,
    chat_id: str | None = None,
    mock: bool = False,
) -> dict[str, Any] | None:
    """Pousse une notif d'**évolution de score** (rescore) — info pure, pas de boutons.

    Distinct de ``notify_candidate`` (qui ouvre les actions de validation).
    Utilisé par ``rescore_scheduler.run_rescore_cycle`` quand un profil rescoré
    franchit ``±15 % / ±20 %``.
    """
    setup_bot_logger()
    text = _format_score_evolution_text(
        username, old_score, new_score, old_tier, new_tier
    )
    if mock:
        _LOG.info("[mock] notify_score_evolution : %s", text)
        return None
    try:
        t, c = _resolve_credentials(token=token, chat_id=chat_id)
    except DiscoveryBotConfigError as e:
        _LOG.warning("notify_score_evolution skip : %s", e)
        return None

    payload = {
        "chat_id": c,
        "text": text,
        "disable_web_page_preview": True,
    }
    return _telegram_post("sendMessage", payload, token=t)


def _edit_message(
    chat_id: int | str,
    message_id: int,
    text: str,
    *,
    token: str,
    reply_markup: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    return _telegram_post("editMessageText", payload, token=token)


def _ack_callback(
    callback_query_id: str, *, token: str, text: str | None = None
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {"callback_query_id": callback_query_id}
    if text:
        payload["text"] = text
    return _telegram_post("answerCallbackQuery", payload, token=token)


# ---------------------------------------------------------------------------
# Watchlist : promotion d'un candidat validé
# ---------------------------------------------------------------------------


def add_to_watchlist(
    candidate: dict[str, Any],
    *,
    t_type_final: str,
    watchlist_path: Path | None = None,
    seeds_path: Path | None = None,
) -> bool:
    """Ajoute le candidat à ``watchlist.json`` avec ``t_type_final``.

    Retourne ``True`` si ajouté, ``False`` si déjà présent.
    """
    from watcher import load_watchlist, save_watchlist  # tardif

    creators = load_watchlist(path=watchlist_path)
    target = (candidate.get("username") or "").lstrip("@").strip().lower()
    if not target:
        return False
    for c in creators:
        if isinstance(c, dict) and str(c.get("username") or "").lower() == target:
            return False  # déjà dans la watchlist

    # Niches (schéma 2026-05) : on lit en priorité la liste portée par le
    # candidate (sortie ``score_profile``) ; à défaut on retombe sur l'ancien
    # champ string ``niche``, et en dernier recours sur ``"humour"``. On ne
    # lit **plus** ``domain.get("niche")`` : la source de vérité est désormais
    # le candidate lui-même (qui propage ``_seed_niches`` du seed parent).
    niches_raw = candidate.get("niches") or [candidate.get("niche") or "humour"]
    niches: list[str] = [
        str(n).strip()
        for n in (niches_raw or [])
        if isinstance(n, str) and str(n).strip()
    ]
    if not niches:
        niches = ["humour"]

    # engagement_baseline : Reel-first (cible AItertainment), sinon Post,
    # sinon ancien champ ``engagement_median`` (rétro-compat).
    eng_baseline_raw = (
        candidate.get("reel_engagement_median")
        or candidate.get("post_engagement_median")
        or candidate.get("engagement_median")
        or 0.0
    )
    try:
        eng_baseline = float(eng_baseline_raw)
    except (TypeError, ValueError):
        eng_baseline = 0.0

    creators.append(
        {
            "username": target,
            "platform": str(candidate.get("platform") or "instagram"),
            "niches": list(niches),
            "t_type": t_type_final,
            "engagement_baseline": eng_baseline,
            "last_post_id": None,
            "added_at": _now_iso(),
        }
    )
    save_watchlist(creators, path=watchlist_path)
    return True


# ---------------------------------------------------------------------------
# Handlers de callback
# ---------------------------------------------------------------------------


def _validation_record(
    candidate: dict[str, Any],
    *,
    action: str,
    t_type_final: str | None,
) -> dict[str, Any]:
    return {
        "username": str(candidate.get("username") or "").lstrip("@").strip().lower(),
        "action": action,
        "t_type_original": str(candidate.get("t_type_dominant") or ""),
        "t_type_final": t_type_final or "",
        "score": candidate.get("score"),
        "domain": candidate.get("domain"),
        "validated_at": _now_iso(),
    }


def _persist_validation_in_db(
    username: str,
    t_type_final: str,
    *,
    db_path: Path | None,
) -> None:
    """Marque le profil ``validated=True`` + ``t_type_final`` dans ``database.json``.

    Best-effort : si le profil n'existe pas (validation manuelle d'un profil
    jamais scoré) ou si la DB est illisible, on logge juste un warning — on ne
    veut **jamais** bloquer la chaîne ``add_to_watchlist + append_validation``.
    """
    try:
        from database import (  # tardif : évite cycle d'import au chargement
            DatabaseIOError,
            load_db,
            save_db,
            validate_profile,
        )
    except ImportError as e:
        _LOG.warning("database module indisponible (%s) — skip DB validation.", e)
        return
    try:
        db = load_db(path=db_path)
    except DatabaseIOError as e:
        _LOG.warning("load_db a échoué (%s) — skip DB validation.", e)
        return
    try:
        validate_profile(db, username, t_type_final)
    except DatabaseIOError as e:
        _LOG.info("validate_profile @%s skip (%s).", username, e)
        return
    try:
        save_db(db, path=db_path)
    except DatabaseIOError as e:
        _LOG.warning("save_db a échoué (%s) après validation @%s.", e, username)


def _handle_validate(
    username: str,
    *,
    candidates_path: Path | None,
    validations_path: Path | None,
    watchlist_path: Path | None,
    seeds_path: Path | None,
    db_path: Path | None,
) -> str:
    cand = _find_candidate(username, candidates_path=candidates_path)
    if cand is None:
        return f"⚠️ @{username} introuvable dans candidates.json"
    t_final = str(cand.get("t_type_dominant") or "T?")
    added = add_to_watchlist(
        cand,
        t_type_final=t_final,
        watchlist_path=watchlist_path,
        seeds_path=seeds_path,
    )
    append_validation(
        _validation_record(cand, action="validated", t_type_final=t_final),
        path=validations_path,
    )
    _persist_validation_in_db(username, t_final, db_path=db_path)
    _remove_candidate(username, candidates_path=candidates_path)
    if added:
        return f"✅ @{username} ajouté à la watchlist (T-type {t_final})"
    return f"☑️ @{username} déjà dans la watchlist"


def _handle_reject(
    username: str,
    *,
    candidates_path: Path | None,
    validations_path: Path | None,
) -> str:
    cand = _find_candidate(username, candidates_path=candidates_path) or {
        "username": username
    }
    append_validation(
        _validation_record(cand, action="rejected", t_type_final=None),
        path=validations_path,
    )
    _remove_candidate(username, candidates_path=candidates_path)
    return f"❌ @{username} ignoré"


def _handle_set_ttype(
    username: str,
    new_t_type: str,
    *,
    candidates_path: Path | None,
    validations_path: Path | None,
    watchlist_path: Path | None,
    seeds_path: Path | None,
    db_path: Path | None,
) -> str:
    cand = _find_candidate(username, candidates_path=candidates_path)
    if cand is None:
        return f"⚠️ @{username} introuvable dans candidates.json"
    if new_t_type not in T_TYPES_AVAILABLE:
        return f"⚠️ T-type inconnu : {new_t_type}"

    t_original = str(cand.get("t_type_dominant") or "")
    added = add_to_watchlist(
        cand,
        t_type_final=new_t_type,
        watchlist_path=watchlist_path,
        seeds_path=seeds_path,
    )
    action = "corrected" if t_original and t_original != new_t_type else "validated"
    append_validation(
        _validation_record(cand, action=action, t_type_final=new_t_type),
        path=validations_path,
    )
    _persist_validation_in_db(username, new_t_type, db_path=db_path)
    _remove_candidate(username, candidates_path=candidates_path)
    if action == "corrected":
        suffix = f" (corrigé : {t_original} → {new_t_type})"
    else:
        suffix = f" (T-type {new_t_type})"
    if added:
        return f"✅ @{username} ajouté à la watchlist{suffix}"
    return f"☑️ @{username} déjà dans la watchlist{suffix}"


def _format_evolution(username: str, history: list[dict[str, Any]]) -> str:
    """Texte multiligne (HTML simple) résumant ``scores_history``.

    Format demandé ::

        📈 Évolution @username
        J0  08/05 → 448 (Tier B)
        J+7 15/05 → 612 (Tier A) +37%
        Tendance : ↑ rising
    """
    u = username.lstrip("@").strip().lower()
    if not history:
        return f"📈 Évolution @{u}\n(aucun historique)"
    if len(history) == 1:
        return f"📈 Évolution @{u}\nPas encore d'historique"

    try:
        from database import compute_tier  # tardif (évite cycle au chargement)
    except ImportError:
        def compute_tier(score: float) -> str:  # type: ignore[misc]
            return "?"

    parsed: list[tuple[datetime | None, dict[str, Any]]] = []
    for h in history:
        try:
            d: datetime | None = datetime.fromisoformat(str(h.get("date") or ""))
        except ValueError:
            d = None
        if d is not None and d.tzinfo is not None:
            d = d.astimezone(timezone.utc).replace(tzinfo=None)
        parsed.append((d, h))

    base_dt = parsed[0][0]
    lines = [f"📈 Évolution @{u}"]
    prev_score: float | None = None
    for i, (d, h) in enumerate(parsed):
        try:
            score = float(h.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        tier = compute_tier(score)
        date_str = d.strftime("%d/%m") if d else "??/??"
        if i == 0 or base_dt is None or d is None:
            label = "J0  "
        else:
            days = max(0, (d - base_dt).days)
            label = f"J+{days}"
        suffix = ""
        if prev_score is not None and prev_score > 0 and i > 0:
            pct = (score - prev_score) / prev_score * 100.0
            sign = "+" if pct >= 0 else ""
            suffix = f" {sign}{pct:.0f}%"
        lines.append(f"{label} {date_str} → {score:.0f} (Tier {tier}){suffix}")
        prev_score = score

    try:
        first_score = float(parsed[0][1].get("score") or 0.0)
        last_score = float(parsed[-1][1].get("score") or 0.0)
    except (TypeError, ValueError):
        first_score, last_score = 0.0, 0.0
    if first_score > 0:
        ratio = (last_score - first_score) / first_score
    else:
        ratio = 0.0
    if ratio > 0.10:
        trend_label = "↑ rising"
    elif ratio < -0.10:
        trend_label = "↓ declining"
    else:
        trend_label = "→ stable"
    lines.append(f"Tendance : {trend_label}")
    return "\n".join(lines)


def _handle_evolution(username: str, *, db_path: Path | None) -> str:
    """Charge ``database.json`` et formate l'évolution du profil."""
    try:
        from database import DatabaseIOError, load_db  # tardif
    except ImportError as e:
        _LOG.warning("database module indisponible (%s).", e)
        return f"📈 Évolution @{username}\n(database module indisponible)"
    try:
        db = load_db(path=db_path)
    except DatabaseIOError as e:
        _LOG.warning("load_db a échoué (%s).", e)
        return f"📈 Évolution @{username}\n(database illisible)"
    key = username.lstrip("@").strip().lower()
    profile = (db.get("profiles") or {}).get(key)
    if not profile:
        return f"📈 Évolution @{key}\n(profil pas encore en base)"
    history = profile.get("scores_history") or []
    return _format_evolution(key, history)


def _parse_callback_data(data: str) -> tuple[str, list[str]]:
    """Parse ``v:user`` / ``r:user`` / ``m:user`` / ``s:T2:user`` / ``ev:user``."""
    parts = (data or "").split(":")
    if not parts:
        return "", []
    return parts[0], parts[1:]


def handle_callback(
    callback_query: dict[str, Any],
    *,
    token: str | None = None,
    expected_chat_id: str | None = None,
    candidates_path: Path | None = None,
    validations_path: Path | None = None,
    watchlist_path: Path | None = None,
    seeds_path: Path | None = None,
    db_path: Path | None = None,
) -> str:
    """Traite un ``callback_query`` Telegram. Retourne le texte de réponse loggé.

    Le bot **filtre** sur ``expected_chat_id`` : seul le chat configuré peut
    valider — n'importe quel autre user est ignoré.
    """
    setup_bot_logger()
    cq_id = str(callback_query.get("id") or "")
    data = str(callback_query.get("data") or "")
    msg = callback_query.get("message") or {}
    chat = msg.get("chat") or {}
    chat_id = str(chat.get("id") or "")
    message_id = msg.get("message_id")

    expected = (
        expected_chat_id
        if expected_chat_id is not None
        else (config.TELEGRAM_DISCOVERY_CHAT_ID or "")
    )
    if expected and chat_id and chat_id != str(expected):
        _LOG.warning(
            "Callback ignoré : chat_id %s != attendu %s", chat_id, expected
        )
        return "ignored"

    bot_token = (token or config.TELEGRAM_DISCOVERY_TOKEN or "").strip()
    action, args = _parse_callback_data(data)

    if action == "v" and args:
        username = args[0]
        text = _handle_validate(
            username,
            candidates_path=candidates_path,
            validations_path=validations_path,
            watchlist_path=watchlist_path,
            seeds_path=seeds_path,
            db_path=db_path,
        )
        if bot_token and message_id:
            _edit_message(chat_id, message_id, text, token=bot_token, reply_markup=None)
        if bot_token and cq_id:
            _ack_callback(cq_id, token=bot_token, text="Validé")
        _LOG.info("validate @%s -> %s", username, text)
        return text

    if action == "r" and args:
        username = args[0]
        text = _handle_reject(
            username,
            candidates_path=candidates_path,
            validations_path=validations_path,
        )
        if bot_token and message_id:
            _edit_message(chat_id, message_id, text, token=bot_token, reply_markup=None)
        if bot_token and cq_id:
            _ack_callback(cq_id, token=bot_token, text="Rejeté")
        _LOG.info("reject @%s -> %s", username, text)
        return text

    if action == "m" and args:
        username = args[0]
        # Édite le message d'origine pour proposer le menu T-types
        cand = _find_candidate(username, candidates_path=candidates_path)
        prompt = (
            f"✏️ Choisis le T-type de <b>@{username}</b>"
            + (f" (estimé : {cand.get('t_type_dominant')})" if cand else "")
        )
        if bot_token and message_id:
            _edit_message(
                chat_id,
                message_id,
                prompt,
                token=bot_token,
                reply_markup=_build_t_type_keyboard(username),
            )
        if bot_token and cq_id:
            _ack_callback(cq_id, token=bot_token, text="Choisis un T-type")
        _LOG.info("modify @%s : menu T-types affiché", username)
        return f"menu T-types @{username}"

    if action == "s" and len(args) >= 2:
        new_t_type = args[0]
        username = args[1]
        text = _handle_set_ttype(
            username,
            new_t_type,
            candidates_path=candidates_path,
            validations_path=validations_path,
            watchlist_path=watchlist_path,
            seeds_path=seeds_path,
            db_path=db_path,
        )
        if bot_token and message_id:
            _edit_message(chat_id, message_id, text, token=bot_token, reply_markup=None)
        if bot_token and cq_id:
            _ack_callback(cq_id, token=bot_token, text=f"T-type {new_t_type}")
        _LOG.info("set t_type @%s -> %s", username, new_t_type)
        return text

    if action == "ev" and args:
        username = args[0]
        text = _handle_evolution(username, db_path=db_path)
        # Push un nouveau message (on ne touche pas au message du candidat
        # pour préserver les boutons de validation).
        if bot_token and chat_id:
            _telegram_post(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": text,
                    "disable_web_page_preview": True,
                },
                token=bot_token,
            )
        if bot_token and cq_id:
            _ack_callback(cq_id, token=bot_token, text="Évolution")
        _LOG.info("evolution @%s", username)
        return text

    _LOG.warning("Callback inconnu : data=%r", data)
    if bot_token and cq_id:
        _ack_callback(cq_id, token=bot_token, text="?")
    return "unknown"


# ---------------------------------------------------------------------------
# Long-polling (run_bot)
# ---------------------------------------------------------------------------


def _load_offset(*, path: Path | None = None) -> int:
    p = Path(path) if path else DEFAULT_BOT_STATE_PATH
    try:
        data = _read_json(p, default={})
    except DiscoveryBotIOError:
        return 0
    if not isinstance(data, dict):
        return 0
    try:
        return int(data.get("last_update_id") or 0)
    except (TypeError, ValueError):
        return 0


def _save_offset(offset: int, *, path: Path | None = None) -> None:
    p = Path(path) if path else DEFAULT_BOT_STATE_PATH
    try:
        _atomic_write_json(p, {"last_update_id": int(offset)})
    except DiscoveryBotIOError as e:
        _LOG.warning("save_offset a échoué : %s", e)


def run_bot(
    *,
    mock: bool = False,
    poll_timeout_s: int = POLL_TIMEOUT_S,
    state_path: Path | None = None,
    candidates_path: Path | None = None,
    validations_path: Path | None = None,
    watchlist_path: Path | None = None,
    seeds_path: Path | None = None,
    db_path: Path | None = None,
    sleep_fn=time.sleep,
) -> None:
    """Boucle de long-polling. Stoppe sur ``Ctrl-C``.

    En mode ``mock=True`` : pas d'appel réseau, juste un message loggé.
    """
    log = setup_bot_logger()
    if mock:
        log.info("[mock] run_bot — aucune connexion Telegram, sortie immédiate.")
        return

    try:
        token, chat_id = _resolve_credentials()
    except DiscoveryBotConfigError as e:
        log.error("run_bot : %s", e)
        return

    offset = _load_offset(path=state_path)
    log.info("=== Discovery Bot démarré (offset=%d) ===", offset)
    try:
        while True:
            params = {"timeout": poll_timeout_s, "offset": offset + 1}
            data = _telegram_get(
                "getUpdates", params, token=token, timeout=poll_timeout_s + 5
            )
            if not data or not data.get("ok"):
                log.warning("getUpdates non-ok — pause 5s et retry.")
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
                        token=token,
                        expected_chat_id=str(chat_id),
                        candidates_path=candidates_path,
                        validations_path=validations_path,
                        watchlist_path=watchlist_path,
                        seeds_path=seeds_path,
                        db_path=db_path,
                    )
            if updates:
                _save_offset(offset, path=state_path)
    except KeyboardInterrupt:
        log.info("Interruption clavier — arrêt Discovery Bot.")
    finally:
        _save_offset(offset, path=state_path)
        log.info("=== Discovery Bot arrêté (offset=%d) ===", offset)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Telegram Discovery Bot — validation humaine des candidats."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mode test : pas de connexion Telegram.",
    )
    parser.add_argument(
        "--notify",
        metavar="USERNAME",
        help="Pousse manuellement un candidat de candidates.json par username.",
    )
    args = parser.parse_args()
    setup_bot_logger()

    if args.notify:
        cand = _find_candidate(args.notify)
        if cand is None:
            _LOG.error("@%s introuvable dans candidates.json — abandon.", args.notify)
            return
        notify_candidate(cand, mock=args.mock)
        return

    run_bot(mock=args.mock)


if __name__ == "__main__":
    _main_cli()


__all__ = [
    "DEFAULT_BOT_STATE_PATH",
    "DEFAULT_VALIDATIONS_PATH",
    "DiscoveryBotConfigError",
    "DiscoveryBotIOError",
    "T_TYPES_AVAILABLE",
    "TREND_EMOJI",
    "add_to_watchlist",
    "append_validation",
    "handle_callback",
    "load_validations",
    "notify_candidate",
    "notify_score_evolution",
    "run_bot",
    "save_validations",
    "setup_bot_logger",
]
