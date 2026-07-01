#!/usr/bin/env python3
"""Collecte des commentaires viraux (fort engagement) depuis Instagram.

Mode par défaut : **fil Reels aléatoire** (``/reels/``) — indépendant de la
watchlist / Discovery. Alternative : grilles profil ``--from profiles``.

Usage::

    .venv/bin/python scripts/scrape_viral_comments.py
    .venv/bin/python scripts/scrape_viral_comments.py --feed-scrolls 50 --min-likes 1000
    .venv/bin/python scripts/scrape_viral_comments.py --from profiles --from watchlist --limit 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from playwright.sync_api import BrowserContext, sync_playwright

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from discovery import _seed_niches, _seed_username, load_seeds
from modules.reel_enrichment import enrich_reel_with_transcript_and_visual
from scripts.instagram_browser import (
    VIRAL_COMMENTS_PATH,
    collect_viral_comments,
    collect_viral_comments_from_feed,
    get_browser_context,
    get_recent_reels,
    load_viral_comments_file,
    polite_sleep,
    save_viral_comments_file,
    session_ok,
)
from scripts.label_comments import TRAINING_COMMENTS_PATH, load_training_comments
from watcher import load_watchlist

_LOG = logging.getLogger(__name__)

# Rétro-compat (tests / imports externes).
_enrich_reel_with_transcript_and_visual = enrich_reel_with_transcript_and_visual


def _count_entries(path: Path) -> int:
    entries, _ = load_viral_comments_file(path)
    return len(entries)


def _pool_stats(viral_path: Path) -> tuple[int, int]:
    """``(non labellisés dans viral, déjà dans training_comments_viral)``."""
    viral_n = _count_entries(viral_path)
    training_n = 0
    if TRAINING_COMMENTS_PATH.exists():
        training_entries, _ = load_training_comments(TRAINING_COMMENTS_PATH)
        training_n = len(training_entries)
    return viral_n, training_n


def _accounts_from_watchlist(limit: int | None) -> list[tuple[str, list[str]]]:
    rows = [
        (
            str(c.get("username") or "").lstrip("@").strip(),
            list(c.get("niches") or []),
        )
        for c in load_watchlist()
        if str(c.get("username") or "").strip()
    ]
    return rows[:limit] if limit else rows


def _accounts_from_seeds(limit: int | None) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for domain in load_seeds().get("domains", []):
        if not isinstance(domain, dict):
            continue
        for seed in domain.get("seeds") or []:
            username = _seed_username(seed)
            if not username or username in seen:
                continue
            seen.add(username)
            rows.append((username, _seed_niches(seed, domain)))
    return rows[:limit] if limit else rows


def _accounts_from_csv(accounts: str) -> list[tuple[str, list[str]]]:
    return [
        (chunk.strip().lstrip("@"), ["humour"])
        for chunk in accounts.split(",")
        if chunk.strip().lstrip("@")
    ]


def _run_profiles_mode(args: argparse.Namespace) -> dict[str, int]:
    limit = args.limit if args.limit > 0 else None
    if args.source == "watchlist":
        account_rows = _accounts_from_watchlist(limit)
    elif args.source == "seeds":
        account_rows = _accounts_from_seeds(limit)
    else:
        if not args.accounts.strip():
            raise SystemExit("Erreur : --accounts requis avec --from profiles accounts.")
        account_rows = _accounts_from_csv(args.accounts)
        if limit:
            account_rows = account_rows[:limit]

    if not account_rows:
        raise SystemExit("Aucun compte à traiter.")

    totals = {"collected": 0, "accounts": 0}
    top_per_reel = args.top_per_reel if args.top_per_reel > 0 else None

    with sync_playwright() as p:
        context = get_browser_context(p)
        if not session_ok(context):
            raise SystemExit("Session Instagram invalide — data/instagram_cookies.json")

        for idx, (username, niches) in enumerate(account_rows, start=1):
            _LOG.info("Profil %d/%d : @%s", idx, len(account_rows), username)
            reels = get_recent_reels(username, context, max_reels=args.max_reels)
            if not reels:
                _LOG.warning("@%s : aucun reel.", username)
                continue
            stats = collect_viral_comments(
                username,
                context,
                reels,
                niches,
                min_likes=args.min_likes,
                max_reels=args.max_reels,
                scroll_rounds=args.scroll_rounds,
                top_per_reel=top_per_reel,
                output_path=args.output,
                between_reels_min_s=config.DISCOVERY_BETWEEN_POSTS_MIN_S,
                between_reels_max_s=config.DISCOVERY_BETWEEN_POSTS_MAX_S,
                french_only=not args.include_english,
                skip_transcript=args.skip_transcript,
                skip_visual=args.skip_visual,
                logger=_LOG,
            )
            totals["collected"] += stats["collected"]
            totals["accounts"] += 1
            if idx < len(account_rows):
                polite_sleep(
                    min_s=config.DISCOVERY_BETWEEN_PROFILES_MIN_S,
                    max_s=config.DISCOVERY_BETWEEN_PROFILES_MAX_S,
                )
    return totals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scrape commentaires viraux depuis le fil Reels ou des profils."
    )
    parser.add_argument(
        "--mode",
        choices=("feed", "profiles"),
        default="feed",
        help="feed = scroll /reels/ aléatoire (défaut) ; profiles = grilles créateurs.",
    )
    parser.add_argument(
        "--from",
        dest="source",
        choices=("watchlist", "seeds", "accounts"),
        default="watchlist",
        help="Source des comptes (mode profiles uniquement).",
    )
    parser.add_argument("--accounts", default="", help="Comptes CSV (mode profiles).")
    parser.add_argument("--limit", type=int, default=0, help="Max comptes (profiles).")
    parser.add_argument("--min-likes", type=int, default=1000)
    parser.add_argument("--max-reels", type=int, default=30)
    parser.add_argument(
        "--feed-scrolls",
        type=int,
        default=config.FEED_SCROLL_STEPS_DEFAULT,
        help="Scrolls du fil /reels/ (mode feed, phase 1 complète).",
    )
    parser.add_argument(
        "--fr-watch-s",
        type=float,
        default=config.FEED_FR_REEL_WATCH_MIN_S,
        help="Secondes sur un reel dont la caption est FR (signal algo IG).",
    )
    parser.add_argument(
        "--feed-en-skip-ms",
        type=int,
        default=config.FEED_EN_REEL_SKIP_MS,
        help="Pause courte avant scroll si caption non-FR (ms).",
    )
    parser.add_argument("--scroll-rounds", type=int, default=8, help="Scroll panneau commentaires.")
    parser.add_argument(
        "--between-min",
        type=float,
        default=config.DISCOVERY_BETWEEN_PROFILES_MIN_S,
        help="Pause min entre profils phase 3 (s) — défaut DISCOVERY_BETWEEN_PROFILES_MIN_S.",
    )
    parser.add_argument(
        "--between-max",
        type=float,
        default=config.DISCOVERY_BETWEEN_PROFILES_MAX_S,
        help="Pause max entre profils phase 3 (s) — défaut DISCOVERY_BETWEEN_PROFILES_MAX_S.",
    )
    parser.add_argument("--top-per-reel", type=int, default=0)
    parser.add_argument("--output", type=Path, default=VIRAL_COMMENTS_PATH)
    parser.add_argument(
        "--include-english",
        action="store_true",
        help="Ne pas filtrer les commentaires anglais.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=0,
        help="Enchaîne des sessions feed jusqu'à N commentaires uniques dans --output.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Max sessions en mode --target (0 = auto ~500).",
    )
    parser.add_argument(
        "--session-between-min",
        type=float,
        default=60.0,
        help="Pause min entre sessions (mode --target).",
    )
    parser.add_argument(
        "--session-between-max",
        type=float,
        default=120.0,
        help="Pause max entre sessions (mode --target).",
    )
    parser.add_argument(
        "--no-feed-reset",
        action="store_true",
        help="Pas de reload explore/reels entre sessions (--target).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-transcript",
        action="store_true",
        help="Skip Whisper (gain temps).",
    )
    parser.add_argument(
        "--skip-visual",
        action="store_true",
        help="Skip vision LM Studio (gain temps).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.dry_run:
        if args.mode == "feed":
            current, labeled = _pool_stats(args.output)
            target_note = (
                f", objectif {args.target} (reste {max(0, args.target - current)})"
                if args.target
                else ""
            )
            print(
                f"Mode feed : {args.feed_scrolls} scrolls, max {args.max_reels} reels, "
                f"min_likes={args.min_likes}, fr_watch={args.fr_watch_s}s{target_note} → {args.output}"
            )
            print(
                f"  Pool viral (non labellisés) : {current} entrée(s) "
                f"(+ {labeled} déjà dans {TRAINING_COMMENTS_PATH.name})"
            )
        else:
            rows = (
                _accounts_from_watchlist(args.limit or None)
                if args.source == "watchlist"
                else _accounts_from_seeds(args.limit or None)
                if args.source == "seeds"
                else _accounts_from_csv(args.accounts)
            )
            for username, niches in rows[: args.limit or len(rows)]:
                print(f"  @{username}  niches={','.join(niches) or '(none)'}")
        return 0

    if args.mode == "feed":
        top_per_reel = args.top_per_reel if args.top_per_reel > 0 else None
        french_only = not args.include_english
        start_count, labeled_count = _pool_stats(args.output)
        target = max(0, int(args.target or 0))
        max_runs = int(args.max_runs or 0)
        if target and max_runs <= 0:
            max_runs = max(500, (target - start_count) // 15 + 20)
        _LOG.info(
            "Pool viral : %d non labellisé(s), %d déjà dans %s.",
            start_count,
            labeled_count,
            TRAINING_COMMENTS_PATH.name,
        )
        if target:
            _LOG.info(
                "Objectif pool viral : %d (actuel : %d, reste : %d).",
                target,
                start_count,
                max(0, target - start_count),
            )
        else:
            max_runs = 1

        session_added = 0
        stale_runs = 0

        with sync_playwright() as p:
            context = get_browser_context(p)
            if not session_ok(context):
                print("Session Instagram invalide.", file=sys.stderr)
                return 1

            for run_idx in range(1, max_runs + 1):
                current = _count_entries(args.output)
                if target and current >= target:
                    _LOG.info("Objectif atteint : %d/%d commentaires.", current, target)
                    break

                if target:
                    _LOG.info(
                        "=== Session %d/%d — BDD %d/%d ===",
                        run_idx,
                        max_runs,
                        current,
                        target,
                    )
                else:
                    _LOG.info(
                        "Fil Reels : scrolls=%d, max_reels=%d, min_likes=%d → %s",
                        args.feed_scrolls,
                        args.max_reels,
                        args.min_likes,
                        args.output,
                    )

                before_run = current
                fresh_feed = bool(target and run_idx > 1 and not args.no_feed_reset)
                stats = collect_viral_comments_from_feed(
                    context,
                    min_likes=args.min_likes,
                    max_reels=args.max_reels,
                    scroll_steps=args.feed_scrolls,
                    scroll_rounds=args.scroll_rounds,
                    top_per_reel=top_per_reel,
                    output_path=args.output,
                    between_reels_min_s=args.between_min,
                    between_reels_max_s=args.between_max,
                    french_only=french_only,
                    fr_reel_watch_s=args.fr_watch_s,
                    feed_en_skip_ms=args.feed_en_skip_ms,
                    fresh_feed=fresh_feed,
                    skip_transcript=args.skip_transcript,
                    skip_visual=args.skip_visual,
                    logger=_LOG,
                )
                after_run = _count_entries(args.output)
                added = after_run - before_run
                session_added += added

                if target:
                    _LOG.info(
                        "Session %d : +%d (total %d/%d).",
                        run_idx,
                        added,
                        after_run,
                        target,
                    )

                if not target:
                    break
                if after_run >= target:
                    _LOG.info("Objectif atteint : %d/%d commentaires.", after_run, target)
                    break
                if added == 0:
                    stale_runs += 1
                    if stale_runs >= 5:
                        _LOG.warning(
                            "5 sessions consécutives sans nouveau commentaire — arrêt "
                            "(%d/%d).",
                            after_run,
                            target,
                        )
                        break
                else:
                    stale_runs = 0

                if run_idx < max_runs and after_run < target:
                    polite_sleep(
                        min_s=args.session_between_min,
                        max_s=args.session_between_max,
                    )

        totals = {
            "collected": session_added,
            "total": _count_entries(args.output),
            "accounts": run_idx if target else 1,
        }
    else:
        totals = _run_profiles_mode(args)

    _LOG.info(
        "Terminé : +%d cette exécution, %d au total dans %s.",
        totals["collected"],
        totals.get("total", totals["collected"]),
        args.output,
    )
    if args.target and totals.get("total", 0) < args.target:
        _LOG.info(
            "Relance pour continuer : python scripts/scrape_viral_comments.py "
            "--target %d --max-reels 50",
            args.target,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
