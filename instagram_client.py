"""Client Instagram via ``instagrapi`` — sessions persistantes pour le Watcher.

============================================================================
POURQUOI instagrapi ET NON PLUS Apify
============================================================================

Apify est facturé à l'usage. Le Watcher polle ``check_new_post`` toutes les
quelques minutes sur N créateurs : à terme, le coût Apify devient bloquant.
On bascule donc vers ``instagrapi`` (API mobile privée d'Instagram) pour la
**détection** (read-only, lecture du dernier post + métadonnées vidéo).

Apify reste pertinent pour la phase **Discovery** (volume ponctuel hebdo,
historique + commentaires sur posts anciens) si le coût est marginal, ou
peut être remplacé par instagrapi aussi. Le module ``modules/scraper.py``
n'est pas supprimé : il sert de fallback / tests / mode mock.

============================================================================
RISQUES & PRÉCAUTIONS
============================================================================

instagrapi utilise l'API mobile **non-officielle** d'Instagram. Risques :
- bannissement du compte si comportement non-humain (rate limit, login
  fréquent depuis IP variable, scraping massif) ;
- challenge / 2FA forcé qui bloque les runs jusqu'à action humaine.

Mesures appliquées dans ce module :
- **Compte Instagram dédié** obligatoire (jamais le compte personnel).
- **Session persistante** (``session.json``) : on ne se reconnecte pas à
  chaque run, on réutilise les cookies device + UUIDs.
- ``delay_range = [1, 3]`` secondes entre actions (rate limit human-like).
- Validation de la session via ``get_timeline_feed`` au resume ; sur échec,
  fresh login automatique.
- En cas d'échec d'auth (mauvais mdp, 2FA, challenge, suspension) :
  ``InstagramAuthError`` levée, log dans ``logs/watcher.log``, **notification
  Telegram** envoyée pour alerte humaine immédiate.
"""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

import config

if TYPE_CHECKING:
    from instagrapi import Client

try:
    from instagrapi import Client
    from instagrapi.exceptions import (
        BadPassword,
        ChallengeRequired,
        ClientError,
        LoginRequired,
        PleaseWaitFewMinutes,
        TwoFactorRequired,
    )
except ImportError as e:
    raise ImportError(
        "instagrapi non installé. Lancez : pip install instagrapi"
    ) from e


_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SESSION_PATH = _PROJECT_ROOT / "session.json"
_LOG_PATH = _PROJECT_ROOT / "logs" / "watcher.log"

_LOGGER_NAME = "aitertainment.watcher"
_log_initialized = False

# Pause longue (s) après une perte de session (LoginRequired / PleaseWaitFewMinutes)
# avant de retenter un fresh login. Côté Instagram, retenter immédiatement
# aggrave le flag — 15 minutes laissent le système se calmer.
SESSION_RECOVERY_SLEEP_S = 900

# Liste de user-agents Instagram Android plausibles. Rotation **uniquement au
# fresh login** : changer l'UA au milieu d'une session existante (device fixé
# dans session.json) est un signal anti-bot fort côté IG.
ANDROID_USER_AGENTS: tuple[str, ...] = (
    "Instagram 312.0.0.32.110 Android (33/13; 420dpi; 1080x2274; samsung; "
    "SM-G991B; o1s; exynos2100; en_US; 545986395)",
    "Instagram 309.1.0.41.113 Android (32/12; 480dpi; 1080x2400; samsung; "
    "SM-A536B; a53x; s5e8825; en_US; 543812701)",
    "Instagram 305.0.0.34.111 Android (33/13; 420dpi; 1080x2340; Xiaomi; "
    "M2102J20SG; alioth; qcom; en_US; 537945543)",
    "Instagram 314.0.0.20.114 Android (34/14; 480dpi; 1080x2400; Google; "
    "Pixel 7; panther; gs101; en_US; 553519573)",
    "Instagram 308.0.0.36.109 Android (31/12; 420dpi; 1080x2280; OnePlus; "
    "OnePlus9; lemonade; qcom; en_US; 542316744)",
)


class InstagramAuthError(RuntimeError):
    """Authentification Instagram impossible (mauvais mdp, 2FA, challenge, suspension)."""


class WatcherStopRequested(RuntimeError):
    """Le Watcher doit s'arrêter proprement (session perdue, recovery KO).

    Levée par ``recover_from_session_loss`` quand le retry après les 15 min
    de pause échoue. La boucle ``run_watcher`` la capture pour sortir
    proprement (et notifier l'humain).
    """


def polite_sleep(min_s: float | None = None, max_s: float | None = None) -> None:
    """Attente aléatoire ``[IG_SLEEP_MIN, IG_SLEEP_MAX]`` avant un appel API.

    Appelée juste avant chaque ``client.user_medias`` (ou autre lecture
    sensible) pour rendre le rythme plus humain. Les bornes par défaut sont
    lues dans ``config.py`` (paramétrables via ``.env``).
    """
    lo = float(min_s if min_s is not None else config.IG_SLEEP_MIN)
    hi = float(max_s if max_s is not None else config.IG_SLEEP_MAX)
    if hi < lo:
        lo, hi = hi, lo
    time.sleep(random.uniform(lo, hi))


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
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
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


def _notify_telegram(text: str) -> None:
    """Envoie une notif Telegram en best-effort (silencieux si Telegram KO)."""
    log = logging.getLogger(_LOGGER_NAME)
    try:
        send_telegram_markdown(text, parse_mode="Markdown")
        log.info("Notification Telegram envoyée.")
    except Exception as e:
        log.warning("Notification Telegram échouée : %s", e)


def _read_credentials() -> tuple[str, str]:
    user = (config.IG_USERNAME or "").strip()
    pwd = (config.IG_PASSWORD or "").strip()
    if not user or not pwd:
        raise InstagramAuthError(
            "IG_USERNAME / IG_PASSWORD manquants — éditez .env "
            "(utilisez un compte Instagram dédié, pas votre compte personnel)."
        )
    return user, pwd


def _try_login_via_session(
    cl: Client,
    session_path: Path,
    username: str,
    password: str,
) -> bool:
    """Tente de réutiliser une session sauvegardée. ``False`` si invalide."""
    log = logging.getLogger(_LOGGER_NAME)
    if not session_path.exists():
        return False
    try:
        cl.load_settings(session_path)
    except Exception as e:
        log.warning("Session %s illisible (%s) — fresh login.", session_path.name, e)
        return False

    try:
        cl.login(username, password)
        cl.get_timeline_feed()
        log.info("Session Instagram réutilisée depuis %s", session_path.name)
        return True
    except LoginRequired:
        log.info("Session expirée pour @%s, fresh login requis.", username)
        return False
    except (BadPassword, TwoFactorRequired, ChallengeRequired):
        # Erreurs d'auth : on les fait remonter pour qu'elles soient traitées
        # par _fresh_login (notification Telegram cohérente).
        raise
    except ClientError as e:
        log.warning("Erreur Instagram pendant resume session (%s) — fresh login.", e)
        return False


def _pick_android_user_agent() -> str:
    return random.choice(ANDROID_USER_AGENTS)


def _fresh_login(
    cl: Client,
    username: str,
    password: str,
    session_path: Path,
) -> None:
    """Login frais + sauvegarde session. Notifie Telegram sur erreurs d'auth.

    Applique un user-agent Android **aléatoire** avant le login pour varier
    l'empreinte au démarrage (rotation contrôlée). Sur une session déjà
    établie (resume), on **ne change pas** d'UA pour préserver la cohérence
    device : c'est le rôle de ``_try_login_via_session`` de respecter les
    settings persistés.
    """
    log = logging.getLogger(_LOGGER_NAME)

    ua = _pick_android_user_agent()
    try:
        cl.set_user_agent(ua)
        log.info("Fresh login : user-agent rotaté (%s)", ua.split(" Android")[0])
    except Exception as e:
        log.debug("set_user_agent ignoré (%s)", e)

    try:
        cl.login(username, password)
    except BadPassword as e:
        msg = f"Auth refusée pour @{username} : mauvais mot de passe."
        log.error(msg)
        _notify_telegram(f"🛑 *Watcher Instagram* : {msg}")
        raise InstagramAuthError(msg) from e
    except TwoFactorRequired as e:
        msg = (
            f"Auth bloquée pour @{username} : 2FA requis. "
            "Désactive le 2FA sur le compte dédié ou implémente "
            "le flow `verification_code`."
        )
        log.error(msg)
        _notify_telegram(f"🔐 *Watcher Instagram* : {msg}")
        raise InstagramAuthError(msg) from e
    except ChallengeRequired as e:
        msg = (
            f"Auth bloquée pour @{username} : challenge / vérification "
            "Instagram (login suspect, suspension partielle). "
            "Connecte-toi manuellement à Instagram pour valider."
        )
        log.error(msg)
        _notify_telegram(f"🚧 *Watcher Instagram* : {msg}")
        raise InstagramAuthError(msg) from e
    except ClientError as e:
        msg = f"Erreur Instagram pendant le login pour @{username} : {e}"
        log.error(msg)
        _notify_telegram(f"⚠️ *Watcher Instagram* : {msg}")
        raise InstagramAuthError(msg) from e
    except Exception as e:
        msg = f"Erreur inattendue pendant le login pour @{username} : {e}"
        log.exception(msg)
        _notify_telegram(f"⚠️ *Watcher Instagram* : {msg}")
        raise InstagramAuthError(msg) from e

    session_path.parent.mkdir(parents=True, exist_ok=True)
    cl.dump_settings(session_path)
    log.info("Session Instagram sauvegardée → %s", session_path.name)


def get_client(
    *,
    session_path: Path | str | None = None,
) -> Client:
    """Retourne un ``Client`` instagrapi authentifié, session persistante.

    Stratégie :
    1. Si ``session.json`` existe → tentative de réutilisation
       (``load_settings`` + ``login`` pour rafraîchir le token + ``get_timeline_feed``
       pour valider).
    2. Sinon (ou session invalide) → fresh login + ``dump_settings``.
    3. Sur erreur d'auth (mauvais mdp, 2FA, challenge, suspension) :
       ``InstagramAuthError``, log dans ``logs/watcher.log``, notification
       Telegram envoyée si la config Telegram est présente.

    IMPORTANT : utiliser un compte Instagram **dédié**, jamais le compte
    personnel (risque de bannissement via API privée mobile).
    """
    log = setup_watcher_logger()
    session = Path(session_path) if session_path else DEFAULT_SESSION_PATH
    username, password = _read_credentials()

    cl = Client()
    cl.delay_range = [1, 3]  # rate limiting human-like

    try:
        if _try_login_via_session(cl, session, username, password):
            return cl
    except (BadPassword, TwoFactorRequired, ChallengeRequired):
        # Auth refusée pendant le resume : on ne tente pas un fresh login,
        # on délègue à _fresh_login pour la gestion uniforme + notif Telegram.
        log.info("Auth refusée pendant resume → fresh login pour traitement uniforme.")

    log.info("Fresh login Instagram pour @%s", username)
    _fresh_login(cl, username, password, session)
    return cl


def recover_from_session_loss(
    *,
    session_path: Path | str | None = None,
    sleep_s: float | None = None,
) -> Client:
    """Récupération sur ``LoginRequired`` / ``PleaseWaitFewMinutes``.

    1. Sleep ``SESSION_RECOVERY_SLEEP_S`` (15 min par défaut) — laisse le
       système IG se calmer, sinon le retry immédiat aggrave le flag.
    2. Tente un nouveau ``get_client()`` (avec rotation UA car fresh login).
    3. Sur succès : retourne le client.
    4. Sur échec : Telegram "session expirée, intervention requise" puis
       lève ``WatcherStopRequested`` pour que le Watcher s'arrête proprement.

    Pour les tests, ``sleep_s`` permet de raccourcir la pause.
    """
    log = setup_watcher_logger()
    pause = float(sleep_s if sleep_s is not None else SESSION_RECOVERY_SLEEP_S)
    log.warning(
        "Session Instagram perdue — pause %.0fs avant retry login.",
        pause,
    )
    time.sleep(pause)

    log.info("Tentative de reconnexion Instagram après pause anti-flag...")
    try:
        return get_client(session_path=session_path)
    except InstagramAuthError as e:
        msg = (
            f"Session expirée et retry login échoué pour @{config.IG_USERNAME} : "
            f"{e}. Intervention humaine requise (vérifier l'app, valider le "
            "challenge, désactiver 2FA si nouveau, etc.). Watcher arrêté."
        )
        log.error(msg)
        _notify_telegram(f"🛑 *Watcher arrêté* : {msg}")
        raise WatcherStopRequested(msg) from e
    except Exception as e:
        msg = f"Reconnexion Instagram impossible (erreur inattendue) : {e}. Watcher arrêté."
        log.exception(msg)
        _notify_telegram(f"🛑 *Watcher arrêté* : {msg}")
        raise WatcherStopRequested(msg) from e


__all__ = [
    "ANDROID_USER_AGENTS",
    "DEFAULT_SESSION_PATH",
    "SESSION_RECOVERY_SLEEP_S",
    "InstagramAuthError",
    "WatcherStopRequested",
    "get_client",
    "polite_sleep",
    "recover_from_session_loss",
    "setup_watcher_logger",
]
