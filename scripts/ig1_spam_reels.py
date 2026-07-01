#!/usr/bin/env python3
"""IG1 — commentaires Ollama sur le fil Reels (+ file watcher).

Pipeline par commentaire : enrichissement reel → 1 base Ollama (filtrée qualité)
→ 1 weave (remplace un verbe / sujet / nom par ``SPAM_COMMENT_KEYWORD``).

Le watcher (IG2/IG3) pousse les nouveaux posts dans ``data/ig1_comment_queue.json``.
Ce script tourne en boucle sur IG1 (``data/instagram_cookies.json``) :

1. Vide la file watcher en priorité (re-queue si échec, max ``IG1_QUEUE_MAX_RETRIES``).
2. Scroll le fil ``/reels/`` (même mécanique que ``scrape_viral_comments``).
3. Option ``--hashtag`` : parcourt une page tag puis commente chaque reel.

Le rythme naturel (lecture ~60s FR + génération Ollama) limite le débit sans pause
artificielle ni plafond configuré.

Usage::

    .venv/bin/python scripts/ig1_spam_reels.py --loop
    .venv/bin/python scripts/ig1_spam_reels.py --login
    .venv/bin/python scripts/ig1_spam_reels.py --hashtag humour --limit 30
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from playwright.sync_api import BrowserContext, Page, sync_playwright

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from ig1_comment_queue import (
    mark_commented,
    pop_ig1_comment,
    queue_length,
    requeue_ig1_comment,
    was_recently_commented,
)
from modules.classifier import ClassificationError
from modules.ig1_spam_generator import (
    generate_ig1_spam_comment,
    normalize_for_instagram,
    spam_keyword,
)
from modules.reel_enrichment import enrich_reel_with_transcript_and_visual
from scripts.instagram_browser import (
    BASE_URL,
    COOKIES_PATH,
    _REQUEST_TIMEOUT_MS,
    _VIEWPORT,
    _advance_reels_feed,
    _attach_graphql_metrics_listener,
    _extract_reel_code_from_page,
    _focus_reels_feed_player,
    _is_valid_reel_code,
    _metrics_bucket_for_dom_media_id,
    _owner_username_from_bucket,
    _wait_for_feed_reel_change,
    engage_feed_reel_for_algo,
    get_browser_context,
    login_instagram_interactive,
    post_reel_comment,
    refresh_reels_feed,
    session_ok,
    should_boost_french_reel_on_feed,
)

_LOG = logging.getLogger("aitertainment.ig1_spam")


@dataclass
class SpamSession:
    """État d'une session spam : mot-clé + dédup commentaires générés."""

    keyword: str = field(default_factory=spam_keyword)
    seen: set[str] = field(default_factory=set)

    def build_comment(
        self,
        *,
        caption: str = "",
        media_id: str = "",
        username: str = "",
        browser_context: BrowserContext | None = None,
    ) -> str:
        transcript = ""
        visual_description = ""
        if browser_context and str(media_id or "").strip():
            skip_transcript = bool(getattr(config, "IG1_SPAM_SKIP_TRANSCRIPT", False))
            skip_visual = bool(getattr(config, "IG1_SPAM_SKIP_VISUAL", False))
            transcript, visual_description = enrich_reel_with_transcript_and_visual(
                media_id,
                browser_context,
                skip_transcript=skip_transcript,
                skip_visual=skip_visual,
            )
            _LOG.info(
                "IG1 enrich reel=%s : transcript=%d chars, visuel=%d chars",
                media_id,
                len(transcript),
                len(visual_description),
            )
        raw = generate_ig1_spam_comment(
            caption=caption,
            transcript=transcript,
            visual_description=visual_description,
            media_id=media_id,
            username=username,
            keyword=self.keyword,
            seen=self.seen,
        )
        return normalize_for_instagram(raw, keyword=self.keyword)


def _post_once(
    context: BrowserContext,
    media_id: str,
    session: SpamSession,
    *,
    caption: str = "",
    username: str = "",
    source: str,
) -> bool:
    if was_recently_commented(media_id):
        _LOG.debug("reel %s déjà commenté récemment — skip.", media_id)
        return False
    try:
        comment_text = session.build_comment(
            caption=caption,
            media_id=media_id,
            username=username,
            browser_context=context,
        )
    except ClassificationError as exc:
        _LOG.warning("Génération Ollama KO (%s) reel=%s : %s", source, media_id, exc)
        return False
    if not str(comment_text or "").strip():
        _LOG.warning(
            "Pas de commentaire valide (%s) reel=%s — skip post.",
            source,
            media_id,
        )
        return False
    ok, err = post_reel_comment(media_id, comment_text, context)
    if ok:
        mark_commented(media_id)
        _LOG.info(
            "Commentaire OK (%s) reel=%s → %r",
            source,
            media_id,
            comment_text[:120],
        )
        return True
    _LOG.warning(
        "Commentaire KO (%s) reel=%s : %s (texte=%r)",
        source,
        media_id,
        err,
        comment_text[:80],
    )
    return False


def _drain_queue(
    context: BrowserContext,
    session: SpamSession,
) -> int:
    posted = 0
    while True:
        item = pop_ig1_comment()
        if not item:
            break
        mid = str(item.get("media_id") or "").strip()
        if not mid:
            continue
        uname = str(item.get("username") or "").lstrip("@")
        if _post_once(
            context,
            mid,
            session,
            username=uname,
            source=f"queue @{uname or '?'}",
        ):
            posted += 1
        elif requeue_ig1_comment(item):
            attempts = int(item.get("attempts") or 0) + 1
            _LOG.info(
                "Reel %s remis en file (tentative %d/%d).",
                mid,
                attempts,
                config.IG1_QUEUE_MAX_RETRIES,
            )
            break
        else:
            _LOG.warning(
                "Reel %s abandonné après %d tentative(s) — file ou max retries.",
                mid,
                int(item.get("attempts") or 0) + 1,
            )
    return posted


def _collect_hashtag_codes(
    page: Page,
    hashtag: str,
    *,
    limit: int,
) -> list[str]:
    tag = hashtag.lstrip("#").strip()
    if not tag:
        return []
    url = f"{BASE_URL}/explore/tags/{tag}/"
    page.goto(url, timeout=_REQUEST_TIMEOUT_MS, wait_until="domcontentloaded")
    page.wait_for_load_state("load")
    page.wait_for_timeout(2500)
    codes: list[str] = []
    seen: set[str] = set()
    stagnant = 0
    while len(codes) < limit and stagnant < 8:
        html = page.content()
        found = 0
        for m in re.finditer(r"/reel/([A-Za-z0-9_-]{8,20})/", html):
            code = m.group(1)
            if _is_valid_reel_code(code) and code not in seen:
                seen.add(code)
                codes.append(code)
                found += 1
                if len(codes) >= limit:
                    break
        if found == 0:
            stagnant += 1
        else:
            stagnant = 0
        page.mouse.wheel(0, 1200)
        page.wait_for_timeout(int(random.uniform(1200, 2200)))
    return codes[:limit]


def _run_hashtag_pass(
    context: BrowserContext,
    page: Page,
    hashtag: str,
    *,
    limit: int,
    session: SpamSession,
) -> int:
    codes = _collect_hashtag_codes(page, hashtag, limit=limit)
    _LOG.info("Hashtag #%s : %d reel(s) à traiter.", hashtag.lstrip("#"), len(codes))
    posted = 0
    for code in codes:
        _drain_queue(context, session)
        if _post_once(
            context,
            code,
            session,
            source=f"hashtag #{hashtag}",
        ):
            posted += 1
    return posted


def _run_feed_loop(
    context: BrowserContext,
    page: Page,
    *,
    session: SpamSession,
    scroll_steps: int,
    french_only: bool = True,
    fr_watch_s: float | None = None,
    en_skip_ms: int | None = None,
    scroll_wait_ms: int = 800,
) -> None:
    """Scroll fil Reels avec comportement humain (aligné ``scrape_viral_comments``).

    - Reel caption FR → regarder ~60s (engagement algo) puis commenter (Ollama).
    - Reel non-FR / sans caption → skip rapide, pas de commentaire.
    """
    fr_watch_s = float(
        fr_watch_s if fr_watch_s is not None else config.FEED_FR_REEL_WATCH_MIN_S
    )
    en_skip_ms = int(
        en_skip_ms if en_skip_ms is not None else config.FEED_EN_REEL_SKIP_MS
    )

    metrics_by_pk, metrics_by_code = _attach_graphql_metrics_listener(page)
    refresh_reels_feed(page, logger=_LOG)
    page.set_viewport_size(_VIEWPORT)
    _focus_reels_feed_player(page)

    stats = {"fr_watched": 0, "fr_commented": 0, "en_skipped": 0, "watch_s": 0.0}
    stagnant = 0
    step = 0

    _LOG.info(
        "Fil Reels : mot-clé=%r, engagement FR=%.0fs, skip EN=%dms, french_only=%s.",
        session.keyword,
        fr_watch_s,
        en_skip_ms,
        french_only,
    )

    while scroll_steps <= 0 or step < scroll_steps:
        q = queue_length()
        if q:
            _LOG.info("File watcher : %d reel(s) en attente.", q)
        _drain_queue(context, session)

        code = _extract_reel_code_from_page(page)
        if not code or not _is_valid_reel_code(code):
            stagnant += 1
            _advance_reels_feed(page)
            page.wait_for_timeout(en_skip_ms)
            step += 1
            continue

        bucket = _metrics_bucket_for_dom_media_id(
            code, metrics_by_pk, metrics_by_code
        )
        caption = str(bucket.get("caption") or "")
        owner = _owner_username_from_bucket(bucket)

        if french_only and not should_boost_french_reel_on_feed(
            caption, french_only=True
        ):
            stats["en_skipped"] += 1
            _LOG.debug(
                "Fil skip (non-FR ou sans caption) reel=%s caption=%r",
                code,
                caption[:60],
            )
            _advance_reels_feed(page)
            page.wait_for_timeout(en_skip_ms)
            _wait_for_feed_reel_change(page, code)
            stagnant = 0
            step += 1
            continue

        if french_only:
            watched = engage_feed_reel_for_algo(
                page, fr_watch_s, logger=_LOG
            )
            stats["fr_watched"] += 1
            stats["watch_s"] += watched

        if not was_recently_commented(code):
            if _post_once(
                context,
                code,
                session,
                caption=caption,
                username=owner,
                source="feed",
            ):
                stats["fr_commented"] += 1
                stagnant = 0
            else:
                stagnant += 1
        else:
            stagnant += 1

        _advance_reels_feed(page)
        page.wait_for_timeout(scroll_wait_ms)
        _wait_for_feed_reel_change(page, code)
        step += 1

        if stagnant >= 15:
            _LOG.info(
                "Fil stagnant — refresh (stats FR=%d commentés=%d, EN skip=%d, ~%.0fs regardés).",
                stats["fr_watched"],
                stats["fr_commented"],
                stats["en_skipped"],
                stats["watch_s"],
            )
            metrics_by_pk.clear()
            metrics_by_code.clear()
            metrics_by_pk, metrics_by_code = _attach_graphql_metrics_listener(page)
            refresh_reels_feed(page, logger=_LOG)
            stagnant = 0
            step = 0
            stats = {"fr_watched": 0, "fr_commented": 0, "en_skipped": 0, "watch_s": 0.0}


def login_ig1_spam(*, wait_after_submit_s: float = 300) -> None:
    """Connexion interactive IG1 (2FA) → ``data/instagram_cookies.json``."""
    login_user = (config.IG_USERNAME or "").strip()
    login_pass = config.IG_PASSWORD or ""
    if not login_user or not login_pass:
        _LOG.error(
            "Identifiants requis dans .env pour --login (IG_USERNAME / IG_PASSWORD)."
        )
        raise SystemExit(1)

    cookies_path = Path(COOKIES_PATH)
    _LOG.info("=== Login interactif IG1 @%s → %s ===", login_user, cookies_path)
    _LOG.info("Fenêtre Chromium — compléter 2FA si demandé.")

    pw = sync_playwright().start()
    try:
        login_instagram_interactive(
            pw,
            login_user,
            login_pass,
            save_cookies_path=cookies_path,
            wait_after_submit_s=wait_after_submit_s,
        )
        _LOG.info("Login IG1 terminé — relancer ig1_spam_reels.py --loop.")
    except Exception as exc:
        _LOG.error("Login IG1 échoué : %s", exc)
        raise SystemExit(1) from exc
    finally:
        pw.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="IG1 spam fil Reels — commentaires Ollama avec mot-clé obligatoire.",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Login interactif IG1 (2FA) → data/instagram_cookies.json puis quitter.",
    )
    parser.add_argument(
        "--hashtag",
        help="Passe unique sur un hashtag (ex. humour) puis retour fil si --loop.",
    )
    parser.add_argument("--limit", type=int, default=25, help="Max reels par passe hashtag.")
    parser.add_argument(
        "--scroll-steps",
        type=int,
        default=0,
        help="Steps fil Reels par cycle (0 = infini).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Boucle infinie fil Reels (défaut si pas de --hashtag seul).",
    )
    parser.add_argument(
        "--no-french-filter",
        action="store_true",
        help="Désactive le filtre FR (commente tous les reels, sans engagement long).",
    )
    parser.add_argument(
        "--fr-watch-s",
        type=float,
        default=None,
        help=f"Secondes sur un reel FR (défaut: FEED_FR_REEL_WATCH_MIN_S={config.FEED_FR_REEL_WATCH_MIN_S}).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.login:
        login_ig1_spam()
        return 0

    if not (config.OLLAMA_GENERATOR_MODEL or "").strip():
        _LOG.error(
            "OLLAMA_GENERATOR_MODEL absent — définir le modèle fine-tuné dans .env."
        )
        return 1

    session = SpamSession()

    infinite = args.loop or (not args.hashtag and args.scroll_steps == 0)

    pw = sync_playwright().start()
    context = get_browser_context(pw, cookies_path=COOKIES_PATH)
    if not session_ok(context):
        _LOG.error(
            "Session IG1 invalide — régénérer %s (python scripts/ig1_spam_reels.py --login).",
            COOKIES_PATH,
        )
        context.close()
        br = context.browser
        if br:
            br.close()
        pw.stop()
        return 1

    page = context.new_page()
    _LOG.info(
        "IG1 spam démarré — mot-clé=%r, modèle=%s, queue=%d, hashtag=%s",
        session.keyword,
        config.OLLAMA_GENERATOR_MODEL,
        queue_length(),
        args.hashtag or "(fil)",
    )

    try:
        while True:
            if args.hashtag:
                _run_hashtag_pass(
                    context,
                    page,
                    args.hashtag,
                    limit=args.limit,
                    session=session,
                )
                if not infinite:
                    break
            _run_feed_loop(
                context,
                page,
                session=session,
                scroll_steps=0 if infinite else args.scroll_steps,
                french_only=not args.no_french_filter,
                fr_watch_s=args.fr_watch_s,
            )
            if not infinite:
                break
            _LOG.info("Cycle fil terminé — nouveau cycle.")
    except KeyboardInterrupt:
        _LOG.info("Arrêt demandé (Ctrl+C).")
    finally:
        page.close()
        context.close()
        br = context.browser
        if br:
            br.close()
        pw.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
