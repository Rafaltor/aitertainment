"""Point d'entrée du pipeline AItertainment."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import config
from modules.classifier import (
    ClassificationError,
    CommentClassifier,
    generate_comments,
)
from modules.detector import CreatorStats, SignalDetector, load_creators_csv
from modules.notifier import TelegramNotifier
from modules.scraper import fetch_creator_stats_mock, mock_recent_comments

_LOGGER_NAME = "aitertainment"
_log_initialized = False


def setup_logging() -> logging.Logger:
    """Fichier logs/pipeline.log + console."""
    global _log_initialized
    log = logging.getLogger(_LOGGER_NAME)
    if _log_initialized:
        return log
    log.setLevel(logging.INFO)
    log.handlers.clear()
    Path("logs").mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler("logs/pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    _log_initialized = True
    return log


def default_niche() -> str:
    return (os.environ.get("NICHE") or "streetwear").strip()


def _process_stats_pipeline(
    stats: CreatorStats,
    niche: str,
    log: logging.Logger,
    comments: list[str],
    *,
    detect_now=None,
    skip_score_gate: bool = False,
    send_telegram: bool = True,
    ollama_post_fn=None,
) -> None:
    """SignalDetector -> (optionnel si score) classify -> generate -> Telegram."""
    print(f"\n--- @{stats.username} ({stats.platform}) ---")
    log.info("Pipeline stats %s @%s", stats.platform, stats.username)

    print("  [2/5] SignalDetector...")
    signal_result = SignalDetector(stats).detect(
        now=detect_now,
        likes_at_30m_ago=None,
    )
    score = float(signal_result.get("score_viral", 0.0))
    alert = str(signal_result.get("alert", "NONE"))
    print(f"  -> score_viral={score:.3f}, alert={alert}")
    log.info(
        "Signal %s score=%.3f alert=%s",
        stats.creator_id,
        score,
        alert,
    )

    if not skip_score_gate and score <= 0.75:
        print("  [STOP] Score <= 0.75, pas d'alerte ni LLM.")
        log.info("Sous seuil pour %s", stats.creator_id)
        return

    print("  [3/5] CommentClassifier...")
    llm_ctx = (
        patch("modules.classifier._http_post", side_effect=ollama_post_fn)
        if ollama_post_fn is not None
        else nullcontext()
    )
    with llm_ctx:
        try:
            clf = CommentClassifier()
            classification = clf.classify(comments, niche=niche)
        except ValueError as e:
            print(f"  [ERREUR] Ollama / classification : {e}")
            log.warning("CommentClassifier indisponible: %s", e)
            return
        except ClassificationError as e:
            print(f"  [ERREUR] Reponse classification invalide : {e}")
            log.warning("ClassificationError: %s", e, exc_info=True)
            return

        ctype = str(classification.get("type", "?"))
        conf = classification.get("confidence", 0)
        print(f"  -> type={ctype}, confidence={conf}")
        log.info("Classification %s: %s conf=%s", stats.creator_id, ctype, conf)

        print("  [4/5] Generation des 3 commentaires...")
        if ctype in ("T1", "T3a"):
            suggestions = ["—", "—", "—"]
            print(f"  -> type {ctype} : pas de generate_comments() (positionnement).")
            log.info("Pas de generation (T1/T3a) pour %s", stats.creator_id)
        else:
            suggestions = generate_comments(classification, comments, niche=niche)
            print(f"  -> {len(suggestions)} suggestion(s) generees.")
            log.info("Suggestions generees pour %s", stats.creator_id)

    if not send_telegram:
        print("  [SKIP] Telegram (mode mock / sans envoi).")
        log.info("Telegram skip pour %s", stats.creator_id)
        return

    print("  [5/5] TelegramNotifier...")
    try:
        notifier = TelegramNotifier()
        notifier.send_alert(stats, signal_result, classification, suggestions)
    except ValueError as e:
        print(f"  [ERREUR] Telegram : {e}")
        log.warning("TelegramNotifier: %s", e)
        return
    except Exception as e:
        print(f"  [ERREUR] Envoi Telegram : {e}")
        log.exception("Echec envoi Telegram pour %s", stats.creator_id)
        return

    print("  [OK] Alerte envoyee sur Telegram.")
    log.info("Alerte Telegram OK pour %s", stats.creator_id)


def _process_creator_row(
    row: CreatorStats,
    niche: str,
    log: logging.Logger,
) -> None:
    """Scrape mock puis pipeline complet."""
    print(f"\n--- Createur @{row.username} ({row.platform}) ---")
    log.info("Debut traitement %s @%s", row.platform, row.username)

    print("  [1/5] Chargement stats (mock Apify)...")
    stats = fetch_creator_stats_mock(row)
    log.info("Stats mock chargees pour %s", stats.creator_id)

    comments = mock_recent_comments(stats.username, 30)
    _process_stats_pipeline(
        stats,
        niche,
        log,
        comments,
        detect_now=None,
        skip_score_gate=False,
        send_telegram=True,
        ollama_post_fn=None,
    )


def run_pipeline() -> None:
    """Charge creators.csv, traite chaque ligne, log dans logs/pipeline.log."""
    log = setup_logging()
    print("\n========== PIPELINE RUN ==========")
    log.info("=== run_pipeline demarre ===")

    print("[0] Lecture de data/creators.csv...")
    creators = load_creators_csv()
    if not creators:
        print("  [AVERTISSEMENT] Aucun createur (CSV vide ou schema legacy).")
        log.warning("Aucun createur a traiter")
        print("========== PIPELINE FIN ==========\n")
        return

    niche = default_niche()
    print(f"  -> {len(creators)} createur(s), niche={niche!r}")
    log.info("%d createur(s), niche=%r", len(creators), niche)

    for row in creators:
        try:
            _process_creator_row(row, niche, log)
        except Exception as e:
            print(f"  [ERREUR] Non gere pour @{row.username}: {e}")
            log.exception("Erreur sur %s", row.creator_id)

    print("\n========== PIPELINE FIN ==========\n")
    log.info("=== run_pipeline termine ===")


def run_mock_scenarios(*, send_telegram: bool = False) -> None:
    """Lance les 3 jeux de donnees mock et verifie detection + classification (Ollama simule)."""
    from tests.mock_data import (
        SCENARIOS,
        assert_signal_expectations,
        generate_mock_creator,
        ollama_post_double_stub,
    )

    log = setup_logging()
    print("\n========== MOCK : 3 scenarios ==========")
    log.info("run_mock_scenarios")
    failures: list[str] = []

    for scenario in SCENARIOS:
        print(f"\n>>> Scenario: {scenario}")
        scenario_ok = False
        try:
            bundle = generate_mock_creator(scenario)
            signal_result = SignalDetector(bundle.stats).detect(now=bundle.detect_now)
            assert_signal_expectations(bundle, signal_result)
            print(
                f"    [OK] Signal score={signal_result['score_viral']:.3f} "
                f"(attente virale={bundle.expect_score_above_075})"
            )

            stub = ollama_post_double_stub(
                bundle.simulated_classification,
                bundle.simulated_generate,
            )
            suggestions = ["—", "—", "—"]
            out: dict = {}
            with patch("modules.classifier._http_post", side_effect=stub):
                clf = CommentClassifier()
                out = clf.classify(bundle.comments, niche=bundle.niche)
                got = str(out.get("type", ""))
                if got != bundle.expected_type:
                    msg = (
                        f"[{scenario}] type attendu {bundle.expected_type!r}, "
                        f"obtenu {got!r}"
                    )
                    print(f"    [ECHEC] {msg}")
                    failures.append(msg)
                    log.error(msg)
                else:
                    print(
                        f"    [OK] Classification type={got} "
                        f"(attendu {bundle.expected_type})"
                    )
                    gen_ok = True
                    if (
                        bundle.expect_score_above_075
                        and got not in ("T1", "T3a")
                    ):
                        gen = generate_comments(
                            out, bundle.comments, niche=bundle.niche
                        )
                        if len(gen) != 3:
                            failures.append(
                                f"[{scenario}] generate_comments: {len(gen)} != 3"
                            )
                            gen_ok = False
                        else:
                            print("    [OK] generate_comments: 3 suggestions")
                            suggestions = gen
                    if gen_ok:
                        scenario_ok = True
            if scenario_ok and send_telegram:
                print("    [Telegram] Envoi force (--send-telegram)...")
                try:
                    notifier = TelegramNotifier()
                    notifier.send_alert(
                        bundle.stats,
                        signal_result,
                        out,
                        suggestions,
                    )
                    print("    [OK] Telegram envoye.")
                    log.info("Telegram mock scenario %s OK", scenario)
                except ValueError as e:
                    print(f"    [ERREUR] Telegram : {e}")
                    log.warning("TelegramNotifier: %s", e)
                except Exception as e:
                    print(f"    [ERREUR] Envoi Telegram : {e}")
                    log.exception("Telegram mock %s", scenario)
        except AssertionError as e:
            print(f"    [ECHEC] {e}")
            failures.append(str(e))
            log.error("Assertion mock: %s", e)

    print("\n========== MOCK : synthese ==========")
    if failures:
        for f in failures:
            print(f"  ECHEC: {f}")
        log.error("Mock scenarios: %d echec(s)", len(failures))
        sys.exit(4)
    print("  Tous les scenarios: OK")
    log.info("run_mock_scenarios OK")
    print("========== MOCK FIN ==========\n")


def run_once(
    username: str,
    platform: str,
    *,
    mock: bool = False,
    scenario: str = "T2_serie",
    send_telegram: bool = False,
) -> None:
    """Un createur CSV, ou donnees mock si mock=True (ignore le CSV)."""
    from tests.mock_data import generate_mock_creator, ollama_post_double_stub

    log = setup_logging()
    p = platform.strip().lower()
    if p not in ("instagram", "tiktok"):
        print(f"[ERREUR] plateforme invalide: {platform}")
        sys.exit(1)

    if mock:
        print(f"\n========== RUN ONCE MOCK scenario={scenario} ==========")
        log.info("run_once MOCK scenario=%s", scenario)
        try:
            bundle = generate_mock_creator(scenario)
        except ValueError as e:
            print(f"  [ERREUR] {e}")
            sys.exit(2)
        stub = ollama_post_double_stub(
            bundle.simulated_classification,
            bundle.simulated_generate,
        )
        print(f"  (pseudo CLI @{username} ignore ; stats mock @{bundle.stats.username})")
        _process_stats_pipeline(
            bundle.stats,
            bundle.niche,
            log,
            bundle.comments,
            detect_now=bundle.detect_now,
            skip_score_gate=True,
            send_telegram=send_telegram,
            ollama_post_fn=stub,
        )
        print("========== RUN ONCE MOCK FIN ==========\n")
        return

    u = username.lstrip("@").strip()
    print(f"\n========== RUN ONCE @{u} ({p}) ==========")
    log.info("run_once @%s %s", u, p)

    print("[0] Lecture de data/creators.csv...")
    creators = load_creators_csv()
    match = [
        c
        for c in creators
        if c.username.strip().lower() == u.lower() and c.platform.strip().lower() == p
    ]
    if not match:
        print(f"  [ERREUR] Aucune ligne pour @{u} + {p} dans creators.csv")
        log.error("Createur introuvable: @%s %s", u, p)
        sys.exit(2)

    _process_creator_row(match[0], default_niche(), log)
    print("========== RUN ONCE FIN ==========\n")


def main_scheduler() -> None:
    """Execute le pipeline tout de suite puis toutes les 6 heures."""
    try:
        import schedule
    except ImportError:
        print(
            "[ERREUR] Le package 'schedule' est requis pour le mode planificateur.\n"
            "  Installez : pip install schedule\n"
            "  Ou lancez un createur ponctuel : python main.py --creator @x --platform instagram"
        )
        sys.exit(3)

    setup_logging()
    print("\n[schedule] Demarrage : premier run immediat, puis toutes les 6 h.")
    print("[schedule] Ctrl+C pour arreter.\n")
    run_pipeline()
    schedule.every(6).hours.do(run_pipeline)
    while True:
        schedule.run_pending()
        time.sleep(30)


def main() -> None:
    """CLI : sans argument -> planificateur ; --mock -> 3 scenarios ; --creator -> run ponctuel."""
    _ = (
        config.APIFY_TOKEN,
        config.ANTHROPIC_API_KEY,
        config.TELEGRAM_BOT_TOKEN,
        config.TELEGRAM_CHAT_ID,
    )
    parser = argparse.ArgumentParser(description="AItertainment — pipeline culturel")
    parser.add_argument(
        "--creator",
        type=str,
        default=None,
        help="Pseudo (avec ou sans @), ex. @marque",
    )
    parser.add_argument(
        "--platform",
        choices=["instagram", "tiktok"],
        default=None,
        help="Requis avec --creator (sauf avec --mock seul)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Donnees tests/mock_data.py ; sans --creator : 3 scenarios ; avec --creator : 1 scenario (--scenario)",
    )
    parser.add_argument(
        "--scenario",
        choices=["T2_serie", "T3b_hater", "T1_gros"],
        default=None,
        help="Avec --mock et --creator : quel jeu de donnees (defaut: T2_serie)",
    )
    parser.add_argument(
        "--send-telegram",
        action="store_true",
        help="Avec --mock : envoyer quand meme la notification Telegram (3 scenarios ou run-once)",
    )
    args = parser.parse_args()

    if args.send_telegram and not args.mock:
        parser.error("--send-telegram requiert --mock")

    if args.mock and not args.creator:
        run_mock_scenarios(send_telegram=args.send_telegram)
        return

    if args.mock and args.creator:
        if not args.platform:
            parser.error("--platform est requis avec --creator et --mock")
        sc = args.scenario or "T2_serie"
        run_once(
            args.creator,
            args.platform,
            mock=True,
            scenario=sc,
            send_telegram=args.send_telegram,
        )
        return

    if args.creator:
        if not args.platform:
            parser.error("--platform est requis avec --creator (instagram|tiktok)")
        run_once(args.creator, args.platform, mock=False)
        return

    main_scheduler()


if __name__ == "__main__":
    main()
