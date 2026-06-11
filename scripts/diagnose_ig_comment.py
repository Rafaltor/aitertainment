#!/usr/bin/env python3
"""Diagnostic pas-à-pas : session IG + publication commentaire reel.

Usage:
  .venv/bin/python scripts/diagnose_ig_comment.py --reel DYFcxWjOkbX
  .venv/bin/python scripts/diagnose_ig_comment.py --reel DYFcxWjOkbX --visible
  .venv/bin/python scripts/diagnose_ig_comment.py --reel DYFcxWjOkbX --post
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright

from scripts import instagram_browser as ig


def _ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def _ko(msg: str) -> None:
    print(f"  ❌ {msg}")


def _step(n: int, title: str) -> None:
    print(f"\n[{n}] {title}")


def run_diagnostic(
    *,
    media_id: str,
    visible: bool,
    do_post: bool,
    comment_text: str,
) -> int:
    cookies = ig.COOKIES_PATH
    print("=== Diagnostic commentaire Instagram (compte IG1) ===")
    print(f"  reel     : {media_id}")
    print(f"  cookies  : {cookies} ({'OK' if cookies.is_file() else 'MANQUANT'})")
    print(f"  headless : {not visible}")
    print(f"  mode     : {'POST réel' if do_post else 'dry-run (pas de publication)'}")

    failures = 0

    _step(1, "Fichier cookies")
    if not cookies.is_file():
        _ko(f"{cookies} introuvable — régénérer la session IG1.")
        return 1
    _ok(f"cookies présents ({cookies.stat().st_size} octets)")

    with sync_playwright() as pw:
        if visible:
            browser = pw.chromium.launch(headless=False)
            context = browser.new_context(viewport=ig._VIEWPORT, locale="fr-FR")
            raw = ig.load_instagram_cookies(cookies)
            if raw:
                context.add_cookies(raw)
        else:
            context = ig.get_browser_context(pw, cookies_path=cookies)

        try:
            _step(2, "Session Instagram")
            if ig.session_ok(context):
                _ok("session_ok → connecté")
            else:
                _ko("session expirée ou login requis — régénérer data/instagram_cookies.json")
                failures += 1
                return 1

            page = context.new_page()
            url = f"{ig.BASE_URL}/p/{media_id}/"
            _step(3, f"Chargement post {url}")
            try:
                page.goto(url, timeout=ig._REQUEST_TIMEOUT_MS, wait_until="domcontentloaded")
                page.wait_for_load_state("load")
                page.wait_for_timeout(2000)
                _ok(f"page chargée — titre={page.title()[:60]!r}")
            except Exception as e:
                _ko(f"navigation échouée : {e}")
                return 1

            blocked = ig._instagram_page_blocked(page)
            _step(4, "Page bloquée / login")
            if blocked:
                _ko(f"page bloquée : {blocked}")
                failures += 1
            else:
                _ok("pas de mur login / challenge détecté")

            _step(5, "Bouton commentaire")
            has_ta = page.locator(
                'textarea[placeholder*="commentaire" i], textarea[placeholder*="comment" i]'
            ).count() > 0
            if has_ta:
                _ok("textarea commentaire visible (/p/)")
            else:
                clicked = ig.click_reel_comment_button(page)
                if clicked:
                    _ok(f"clic commentaire OK (label={clicked!r})")
                else:
                    labels = ig._list_reel_page_aria_labels(page)
                    _ko(f"textarea/bouton introuvable — aria-labels={labels[:12]}")
                    failures += 1
                    page.close()
                    return 1
                page.wait_for_timeout(1500)

            _step(6, "Champ commentaire (remplissage test)")
            probe = "[test diagnostic — ne pas publier]"
            filled = ig._fill_reel_comment_field(page, probe)
            if filled:
                _ok("champ textbox trouvé et rempli")
            else:
                labels = ig._list_reel_page_aria_labels(page)
                _ko(f"champ introuvable — aria-labels={labels[:12]}")
                failures += 1
                page.close()
                return 1

            if not do_post:
                _step(7, "Publication (ignorée — dry-run)")
                _ok("dry-run terminé — relancer avec --post pour publier")
                page.close()
                return failures

            page.close()

            _step(7, f"Publication réelle via post_reel_comment")
            t0 = time.monotonic()
            ok, err = ig.post_reel_comment(media_id, comment_text, context)
            elapsed = time.monotonic() - t0
            if ok:
                _ok(f"commentaire publié et vérifié en {elapsed:.1f}s")
                _ok("compte IG1 — vérifier sur IG que le commentaire est bien visible")
            else:
                _ko(f"échec ({elapsed:.1f}s) : {err}")
                failures += 1

        finally:
            context.close()
            br = context.browser
            if br:
                br.close()

    print(f"\n=== Résultat : {failures} échec(s) ===")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostic commentaire Instagram reel")
    parser.add_argument("--reel", default="DYFcxWjOkbX", help="media_id / shortcode reel")
    parser.add_argument(
        "--visible",
        action="store_true",
        help="navigateur visible (désactive headless)",
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="publier réellement (sinon dry-run)",
    )
    parser.add_argument(
        "--text",
        default="[AIT] test diagnostic commentaire",
        help="texte si --post",
    )
    args = parser.parse_args()
    return run_diagnostic(
        media_id=str(args.reel).strip(),
        visible=bool(args.visible),
        do_post=bool(args.post),
        comment_text=str(args.text).strip(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
