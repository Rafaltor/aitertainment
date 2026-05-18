"""rescore_scheduler.py — rescore périodique des profils en database.json.

============================================================================
Architecture
============================================================================

Tier A → rescore tous les 7 jours.
Tier B → rescore tous les 30 jours.
Tier C → archivé, pas de rescore.

Ce module **ne décide pas** de la fréquence : c'est ``next_rescore_at`` (calculé
par ``database.upsert_profile``) qui pilote. Le scheduler se contente d'itérer
sur ``get_profiles_due_for_rescore(db)`` et de relancer ``score_and_persist``
sur chacun.

Particularités :

- Entre chaque profil, ``sleep`` aléatoire de **5–15 minutes** (rescore lent
  et non urgent — on évite de stresser instagrapi).
- En **mock**, ni sleep ni Telegram réel.
- Compare ``last_history_score`` (avant le rescore) au nouveau score :

    * variation ≥ +15 % → push « rise » via ``notify_score_evolution``.
    * variation ≤ −20 % → push « drop ».
    * sinon → silencieux.

Exécuter 1×/jour (cron / Task Scheduler) suffit largement : la logique
``next_rescore_at`` filtre toute seule ce qui est réellement dû.

============================================================================
CLI
============================================================================

::

    python rescore_scheduler.py            # cycle complet
    python rescore_scheduler.py --mock     # test sans Instagram ni Telegram
    python rescore_scheduler.py --due      # liste seulement les profils dûs

============================================================================
Cron (Linux) — exemple
============================================================================

::

    # tous les jours à 09:30, env Python projet activé
    30 9 * * * cd /opt/aitertainment && /opt/aitertainment/venv/bin/python \\
              rescore_scheduler.py >> logs/rescore.log 2>&1

============================================================================
Task Scheduler (Windows) — exemple
============================================================================

Action : ``python.exe``
Arguments : ``rescore_scheduler.py``
Démarrer dans : chemin du projet
Déclencheur : quotidien à 09:30
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable

from database import (
    DatabaseIOError,
    compute_tier,
    get_profiles_due_for_rescore,
    load_db,
)

_LOG = logging.getLogger("aitertainment.rescore")

# Seuils relatifs de variation pour notification (fraction, pas %).
# +15 % → notif "📈" ; -20 % → notif "📉" ; entre les deux → silence.
RESCORE_RISE_THRESHOLD = 0.15
RESCORE_DROP_THRESHOLD = -0.20

# Sleep aléatoire entre deux profils (rescore non urgent).
RESCORE_SLEEP_MIN_S = 5 * 60
RESCORE_SLEEP_MAX_S = 15 * 60


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------


def _last_score(profile: dict[str, Any]) -> tuple[float | None, str | None]:
    """Lit le dernier point de ``scores_history`` et déduit son tier.

    Retourne ``(None, None)`` si l'historique est vide ou cassé — dans ce cas
    on ne peut pas calculer une variation, donc pas de notification.
    """
    history = profile.get("scores_history") or []
    if not isinstance(history, list) or not history:
        return None, None
    last = history[-1]
    if not isinstance(last, dict):
        return None, None
    raw = last.get("score")
    if raw is None:
        return None, None
    try:
        s = float(raw)
    except (TypeError, ValueError):
        return None, None
    return s, compute_tier(s)


def _pct_change(old: float | None, new: float) -> float:
    """Variation relative ``(new − old) / old``.

    ``old`` ``None`` ou ``≤ 0`` → ``0.0`` (pas de ``ZeroDivisionError`` ; on
    refuse de notifier sur un baseline absurde).
    """
    if old is None:
        return 0.0
    try:
        old_f = float(old)
    except (TypeError, ValueError):
        return 0.0
    if old_f <= 0:
        return 0.0
    return (float(new) - old_f) / old_f


def _format_due_summary(profiles: list[dict[str, Any]]) -> str:
    if not profiles:
        return "Aucun profil dû pour rescore."
    lines = [f"{len(profiles)} profil(s) à rescorer :"]
    for p in profiles:
        u = p.get("username") or "?"
        tier = p.get("tier") or "?"
        nxt = p.get("next_rescore_at") or "?"
        lines.append(f"  - @{u}  tier={tier}  next_rescore_at={nxt}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cycle principal
# ---------------------------------------------------------------------------


def run_rescore_cycle(
    *,
    mock: bool = False,
    db_path: Path | None = None,
    seeds_path: Path | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    rng: random.Random | None = None,
    score_fn: Callable[..., dict[str, Any] | None] | None = None,
    notify_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Itère sur les profils dûs et les rescore via ``score_and_persist``.

    Pour chaque profil :

    1. On capture ``(old_score, old_tier)`` depuis le **dernier** point de
       ``scores_history`` *avant* le rescore — ``score_and_persist`` va
       ajouter le nouveau point, donc lire après serait inutile.
    2. ``score_and_persist(username, added_via='rescore', ...)`` — qui
       ré-écrit ``database.json`` (tier, next_rescore_at, history).
    3. Si variation hors seuils → push Telegram via ``notify_fn``.
    4. ``sleep`` aléatoire avant le profil suivant (sauf pour le dernier).

    Les exceptions inattendues sur un profil ne stoppent **jamais** le cycle :
    elles incrémentent ``stats['errors']`` et on passe au profil suivant.

    Parameters
    ----------
    mock
        ``True`` → pas de réseau Instagram (``score_and_persist(mock=True)``)
        ni de sleep réel ; on garde la logique de variation/notif (en mode
        mock, ``notify_fn`` est appelée mais ``notify_score_evolution`` no-op).
    db_path, seeds_path
        Délégués tels quels à ``score_and_persist`` et ``load_db``.
    sleep_fn, rng, score_fn, notify_fn
        Hooks d'injection pour les tests (defaults : ``time.sleep``,
        ``random.Random()``, ``discovery.score_and_persist``,
        ``telegram_discovery_bot.notify_score_evolution``).

    Returns
    -------
    dict
        ``{"due_count", "processed", "notified_rise", "notified_drop",
        "errors"}``.
    """
    rnd = rng or random.Random()
    sleep = sleep_fn if sleep_fn is not None else time.sleep

    stats: dict[str, Any] = {
        "due_count": 0,
        "processed": 0,
        "notified_rise": 0,
        "notified_drop": 0,
        "errors": 0,
    }

    try:
        db = load_db(path=db_path)
    except DatabaseIOError as e:
        _LOG.error("rescore : load_db a échoué (%s) — abort.", e)
        stats["errors"] = 1
        return stats

    due = get_profiles_due_for_rescore(db)
    stats["due_count"] = len(due)
    _LOG.info("rescore cycle : %d profil(s) dû(s).", len(due))

    score_fn_injected = score_fn is not None
    if due and score_fn is None:
        from discovery import score_and_persist as _default_score_fn  # tardif (cycle)
        score_fn = _default_score_fn
    if due and notify_fn is None:
        from telegram_discovery_bot import (  # tardif
            notify_score_evolution as _default_notify_fn,
        )
        notify_fn = _default_notify_fn

    def _process_due(context: Any | None = None) -> None:
        for i, profile in enumerate(due):
            username = str(profile.get("username") or "").lstrip("@").strip().lower()
            if not username:
                continue

            old_score, old_tier = _last_score(profile)

            score_kwargs: dict[str, Any] = {
                "added_via": "rescore",
                "db_path": db_path,
                "seeds_path": seeds_path,
                "mock": mock,
            }
            if context is not None:
                score_kwargs["context"] = context

            try:
                summary = score_fn(username, **score_kwargs)
            except Exception as e:  # filet large : un profil ne stoppe pas le cycle
                _LOG.warning(
                    "rescore @%s : score_and_persist a levé (%s).", username, e
                )
                stats["errors"] += 1
                summary = None

            if summary is not None:
                stats["processed"] += 1
                res = summary.get("score_result") or {}
                try:
                    new_score = float(res.get("score") or 0.0)
                except (TypeError, ValueError):
                    new_score = 0.0
                new_tier = str(summary.get("tier") or "?")

                if old_score is None or old_score <= 0:
                    _LOG.info(
                        "rescore @%s : pas de baseline (history vide ou score nul) — pas de notif.",
                        username,
                    )
                else:
                    variation = _pct_change(old_score, new_score)
                    if variation >= RESCORE_RISE_THRESHOLD:
                        try:
                            notify_fn(
                                username=username,
                                old_score=old_score,
                                new_score=new_score,
                                old_tier=old_tier or "?",
                                new_tier=new_tier,
                                mock=mock,
                            )
                            stats["notified_rise"] += 1
                        except Exception as e:
                            _LOG.warning(
                                "rescore @%s : notify rise a échoué (%s).",
                                username,
                                e,
                            )
                    elif variation <= RESCORE_DROP_THRESHOLD:
                        try:
                            notify_fn(
                                username=username,
                                old_score=old_score,
                                new_score=new_score,
                                old_tier=old_tier or "?",
                                new_tier=new_tier,
                                mock=mock,
                            )
                            stats["notified_drop"] += 1
                        except Exception as e:
                            _LOG.warning(
                                "rescore @%s : notify drop a échoué (%s).",
                                username,
                                e,
                            )
                    else:
                        _LOG.info(
                            "rescore @%s : variation %+.1f%% — sous seuils, silence.",
                            username,
                            variation * 100.0,
                        )
            else:
                _LOG.info(
                    "rescore @%s : score_and_persist a renvoyé None (filtré).",
                    username,
                )

            is_last = i == len(due) - 1
            if not is_last and not mock:
                wait = rnd.uniform(RESCORE_SLEEP_MIN_S, RESCORE_SLEEP_MAX_S)
                _LOG.info(
                    "rescore : pause %.0fs avant @%s suivant.",
                    wait,
                    due[i + 1].get("username") or "?",
                )
                sleep(wait)

    if mock or not due or score_fn_injected:
        _process_due()
    else:
        from playwright.sync_api import sync_playwright

        from scripts.instagram_browser import get_browser_context

        with sync_playwright() as pw:
            context = get_browser_context(pw)
            try:
                _process_due(context)
            finally:
                context.close()

    _LOG.info(
        "rescore cycle terminé : processed=%d rise=%d drop=%d errors=%d",
        stats["processed"],
        stats["notified_rise"],
        stats["notified_drop"],
        stats["errors"],
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main_cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rescore scheduler — rebalaye les profils dont "
            "next_rescore_at est dépassé."
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Test sans Instagram ni Telegram (et sans sleep entre profils).",
    )
    parser.add_argument(
        "--due",
        action="store_true",
        help="Affiche seulement les profils dûs (lecture, aucune action).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Chemin alternatif vers database.json (debug).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.due:
        try:
            db = load_db(path=args.db_path)
        except DatabaseIOError as e:
            print(f"❌ load_db : {e}")
            return
        print(_format_due_summary(get_profiles_due_for_rescore(db)))
        return

    stats = run_rescore_cycle(mock=args.mock, db_path=args.db_path)
    print(
        f"✅ Rescore terminé — dûs={stats['due_count']} | "
        f"processed={stats['processed']} | "
        f"📈={stats['notified_rise']} | 📉={stats['notified_drop']} | "
        f"errors={stats['errors']}"
    )


if __name__ == "__main__":
    _main_cli()


__all__ = [
    "RESCORE_DROP_THRESHOLD",
    "RESCORE_RISE_THRESHOLD",
    "RESCORE_SLEEP_MAX_S",
    "RESCORE_SLEEP_MIN_S",
    "run_rescore_cycle",
]
