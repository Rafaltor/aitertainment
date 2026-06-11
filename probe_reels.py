#!/usr/bin/env python3
"""Sonde de diagnostic v4 — trouve QUELLE page/flux sert les reels du créateur.

Teste, pour le créateur donné, deux pages source :
  1. la page profil principale   /username/
  2. l'onglet reels              /username/reels/
Pour chacune : clés de connexion GraphQL vues, médias groupés par propriétaire,
et présence des médias du créateur. Scanne aussi le HTML du document pour des
données embarquées (Instagram injecte parfois les posts dans un <script>).

Usage :
    python probe_reels.py marrant_club
    IG_HEADLESS=0 python probe_reels.py marrant_club

Sort dans ./debug/. Ne modifie rien.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from playwright.sync_api import Response, sync_playwright

from scripts.instagram_browser import BASE_URL, get_browser_context

DEBUG_DIR = Path("debug")
DEBUG_DIR.mkdir(exist_ok=True)


def media_nodes(node, owner_inherited=None):
    """Parcourt le JSON et yield un dict par nœud média (code non nul)."""
    if isinstance(node, dict):
        u_obj = node.get("user") or node.get("owner")
        owner = (u_obj.get("username") if isinstance(u_obj, dict) else None) or owner_inherited
        code = node.get("code") or node.get("shortcode")
        if code:
            yield {
                "code": code,
                "owner": owner,
                "play_count": node.get("play_count"),
                "view_count": node.get("view_count"),
                "ig_play_count": node.get("ig_play_count"),
                "like_count": node.get("like_count"),
                "taken_at": node.get("taken_at"),
                "product_type": node.get("product_type"),
            }
        for v in node.values():
            yield from media_nodes(v, owner)
    elif isinstance(node, list):
        for v in node:
            yield from media_nodes(v, owner_inherited)


def analyze(page, user, label, captured):
    target = user.lower()
    print("\n" + "=" * 60 + "\n  PAGE : " + label + "\n" + "=" * 60)
    print(f"→ URL finale : {page.url}")

    login_form = (
        page.locator('input[name="username"]').count() > 0
        and page.locator('input[name="password"]').count() > 0
    )
    print(f"→ formulaire login présent : {login_form}")

    connection_keys = set()
    owner_tally = {}
    target_nodes = []
    for c in captured:
        body = c.get("body") or ""
        if '"code"' not in body and '"shortcode"' not in body:
            continue
        try:
            data = json.loads(body)
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("data"), dict):
            for k in data["data"]:
                connection_keys.add(k)
        for m in media_nodes(data):
            o = (m["owner"] or "").lower()
            owner_tally[o or "(inconnu)"] = owner_tally.get(o or "(inconnu)", 0) + 1
            if o == target:
                target_nodes.append((c["url"], m))

    print("→ clés de connexion vues :", sorted(connection_keys) or "(aucune)")
    print("→ propriétaires des médias captés :", json.dumps(owner_tally, ensure_ascii=False))

    if target_nodes:
        print(f"\n→ ✅ {len(target_nodes)} média(s) de @{user} dans le FLUX réseau :")
        seen = set()
        for url, m in target_nodes:
            if m["code"] in seen:
                continue
            seen.add(m["code"])
            metrics = {k: m[k] for k in ("play_count", "view_count", "ig_play_count",
                                         "like_count", "taken_at", "product_type")
                       if m[k] is not None}
            print(f"     {m['code']:14} {json.dumps(metrics, ensure_ascii=False)}")
            print(f"        ↳ via {url}")
    else:
        print(f"\n→ ⚠️  Aucun média de @{user} dans le FLUX réseau pour cette page.")

    html = page.content()
    (DEBUG_DIR / f"{user}_{label.replace('/', '_')}.html").write_text(html, encoding="utf-8")
    n_code = len(re.findall(r'"code"\s*:\s*"[A-Za-z0-9_-]{8,15}"', html))
    n_clips = html.count('"product_type":"clips"')
    n_play = html.count('"play_count"')
    has_user_timeline = "user_timeline" in html or "feed__user" in html
    has_clips_user = "clips__user" in html or "clips_user" in html
    print("\n→ HTML embarqué :")
    print(f"     codes courts trouvés      : {n_code}")
    print(f'     "product_type":"clips"     : {n_clips}')
    print(f'     "play_count"               : {n_play}')
    print(f"     mentionne user_timeline    : {has_user_timeline}")
    print(f"     mentionne clips_user       : {has_clips_user}")
    if n_code:
        sample = re.findall(r'"code"\s*:\s*"([A-Za-z0-9_-]{8,15})"', html)[:8]
        print(f"     échantillon de codes       : {sample}")


def main():
    if len(sys.argv) < 2:
        print("usage: python probe_reels.py <username>")
        raise SystemExit(2)
    user = sys.argv[1].lstrip("@").strip()

    pages_to_test = [
        (f"{BASE_URL}/{user}/", f"{user}"),
        (f"{BASE_URL}/{user}/reels/", f"{user}/reels"),
    ]

    with sync_playwright() as pw:
        for target_url, label in pages_to_test:
            context = get_browser_context(pw)
            captured = []
            page = context.new_page()

            def on_response(resp, _cap=captured):
                url = resp.url or ""
                if "graphql" not in url and "/api/v1/" not in url:
                    return
                body = ""
                try:
                    body = resp.text()
                except Exception:
                    pass
                _cap.append({"url": url[:120], "body": body})

            page.on("response", on_response)
            print(f"\n→ goto {target_url}")
            try:
                page.goto(target_url, timeout=30_000, wait_until="domcontentloaded")
                page.wait_for_load_state("load")
                page.wait_for_timeout(3_000)
                for _ in range(4):
                    page.mouse.wheel(0, 4000)
                    page.wait_for_timeout(1500)
                page.wait_for_timeout(1500)
                analyze(page, user, label, captured)
                page.screenshot(
                    path=str(DEBUG_DIR / f"{user}_{label.replace('/', '_')}.png"),
                    full_page=True,
                )
            except Exception as e:
                print(f"  erreur sur {label} : {e}")
            finally:
                context.close()
                br = context.browser
                if br:
                    br.close()


if __name__ == "__main__":
    main()
