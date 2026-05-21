#!/usr/bin/env python3
"""instagram_browser.py — navigation Instagram via Playwright (Discovery + Watcher)."""

from __future__ import annotations

import ast
import base64
import json
import logging
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from playwright.sync_api import BrowserContext, Page, Playwright, Response

COOKIES_PATH = Path("data/instagram_cookies.json")
HEADLESS = True
BASE_URL = "https://www.instagram.com"
_VIEWPORT = {"width": 1920, "height": 1080}
_REELS_GRID_COLUMNS = 5
_REELS_GRID_ROW_HEIGHT_PX = 430
_REELS_SCROLL_WAIT_MS = 2_000
_REQUEST_TIMEOUT_MS = 15_000
RAW_COMMENTS_PATH = Path("data/raw_comments.json")
COMMENTS_COLLECT_REELS_MAX = 3
COMMENTS_PANEL_SCROLL_ROUNDS = 6
_GRAPHQL_METRIC_KEYS = (
    "view_count",
    "play_count",
    "video_view_count",
    "like_count",
    "comment_count",
    "share_count",
)
LM_STUDIO_CHAT_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_VISION_MODEL = "google/gemma-4-e4b"

log = logging.getLogger(__name__)

_VISION_FALLBACK: dict[str, Any] = {
    "followers": 0,
    "following": 0,
    "posts_count": 0,
    "full_name": "",
    "biography": "",
    "is_private": False,
}

_IG_RESERVED_USERNAMES = frozenset(
    {
        "explore",
        "reels",
        "stories",
        "p",
        "tv",
        "accounts",
        "direct",
        "legal",
        "about",
        "help",
    }
)


def polite_sleep(
    seconds: float = 2,
    min_s: float | None = None,
    max_s: float | None = None,
) -> None:
    """Pause entre requêtes Playwright."""
    if min_s is not None and max_s is not None:
        time.sleep(random.uniform(min_s, max_s))
    else:
        time.sleep(seconds + random.uniform(0.5, 1.5))


_COMMENTS_UI_NOISE = (
    "Ne pas suggérer",
    "Masquer temporairement",
    "Cette publication me met mal",
    "suggérées dans le fil",
    "Répondre",
    "Voir les",
    "Pour vous",
    "Commentaires",
    "Ajouter un commentaire",
)
_TIMESTAMP_RE = re.compile(r"^\d+\s*[jhdmywsJHDMYWS]")
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9._]{2,30}$")

# Icônes commentaire reel (FR prioritaire — context Playwright en locale fr-FR).
_REEL_COMMENT_ARIA_LABELS = (
    "Commentaire",
    "Commenter",
    "Comment",
    "Comments",
    "Commentaires",
)

_REEL_COMMENTS_PANEL_SELECTORS = (
    'div[role="dialog"] ul',
    "div._aano",
    '[role="dialog"]',
    "section ul",
)

_COMMENT_META_RE = re.compile(
    r"\d+\s*(?:sem|j|h|min|mois|s\b)|Répondre|J.aime|like",
    re.IGNORECASE,
)


def _reel_page_has_shell(page: Page) -> bool:
    """True si la page reel a un DOM Instagram (pas le shell vide /reel/ direct)."""
    try:
        if len(page.content()) < 5000:
            return False
        return page.locator("svg").count() > 3
    except Exception:
        return False


def _reel_link_locator(page: Page, media_id: str):
    mid = str(media_id or "").strip()
    return page.locator(
        f'a[href*="/reel/{mid}"], a[href*="/p/{mid}"], a[href*="{mid}"]'
    )


def _scroll_reels_grid_to_find_link(page: Page, media_id: str, *, max_rounds: int = 10) -> bool:
    """Scroll la grille /reels/ jusqu'à trouver un lien vers ``media_id``."""
    mid = str(media_id or "").strip()
    if not mid:
        return False
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(800)
    for _ in range(max_rounds):
        if _reel_link_locator(page, mid).count() > 0:
            return True
        page.evaluate("window.scrollBy(0, 450)")
        page.wait_for_timeout(1200)
    return _reel_link_locator(page, mid).count() > 0


def navigate_to_reel_page(
    page: Page,
    media_id: str,
    username: str = "",
    *,
    timeout_ms: int = _REQUEST_TIMEOUT_MS,
    reels_grid_loaded: bool = False,
) -> bool:
    """Charge un reel. Préfère la grille ``/{user}/reels/`` (``/reel/{id}/`` seul est souvent vide)."""
    mid = str(media_id or "").strip()
    if not mid:
        return False

    page.set_viewport_size(_VIEWPORT)
    u = str(username or "").lstrip("@").strip()

    if u and not reels_grid_loaded:
        try:
            page.goto(
                f"{BASE_URL}/{u}/reels/",
                timeout=timeout_ms,
                wait_until="domcontentloaded",
            )
            page.wait_for_load_state("load")
            page.wait_for_timeout(2500)
        except Exception as e:
            log.warning("reel %s : accès @%s/reels/ échoué (%s).", mid, u, e)
            return False

    if u:
        try:
            if not _scroll_reels_grid_to_find_link(page, mid):
                log.warning(
                    "reel %s : lien absent sur @%s/reels/ (même après scroll).",
                    mid,
                    u,
                )
                return False
            _reel_link_locator(page, mid).first.click(timeout=10_000)
            page.wait_for_timeout(4000)
            if _reel_page_has_shell(page):
                return True
            log.warning("reel %s : clic grille @%s mais DOM toujours vide.", mid, u)
            return False
        except Exception as e:
            log.warning("reel %s : ouverture via grille @%s échouée (%s).", mid, u, e)
            return False

    try:
        page.goto(
            f"{BASE_URL}/reel/{mid}/",
            timeout=timeout_ms,
            wait_until="domcontentloaded",
        )
        page.wait_for_load_state("load")
        page.wait_for_timeout(3000)
        if _reel_page_has_shell(page):
            return True
    except Exception:
        pass

    log.warning("reel %s : page vide (pas de username pour la grille).", mid)
    return False


def open_reels_grid(page: Page, username: str, *, timeout_ms: int = _REQUEST_TIMEOUT_MS) -> bool:
    """Ouvre ``/{username}/reels/`` (à réutiliser pour plusieurs reels)."""
    u = str(username or "").lstrip("@").strip()
    if not u:
        return False
    page.set_viewport_size(_VIEWPORT)
    try:
        page.goto(
            f"{BASE_URL}/{u}/reels/",
            timeout=timeout_ms,
            wait_until="domcontentloaded",
        )
        page.wait_for_load_state("load")
        page.wait_for_timeout(2500)
        return True
    except Exception as e:
        log.warning("grille @%s/reels/ inaccessible (%s).", u, e)
        return False


def return_to_reels_grid(page: Page, username: str) -> bool:
    """Revenir à la grille reels après avoir ouvert un reel."""
    u = str(username or "").lstrip("@").strip()
    if not u:
        return False
    try:
        if f"/{u}/reels" in (page.url or ""):
            return True
        page.go_back(wait_until="domcontentloaded", timeout=15_000)
        page.wait_for_timeout(2000)
        if f"/{u}/reels" in (page.url or ""):
            return True
    except Exception:
        pass
    return open_reels_grid(page, u)


def build_comment_dedup_key(media_id: str, text: str) -> str:
    return f"{media_id}||{text.strip().lower()}"


def load_raw_comments_file(
    path: Path | str | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Charge ``raw_comments.json`` et retourne ``(entries, clés dédup)``."""
    p = Path(path) if path is not None else RAW_COMMENTS_PATH
    if not p.exists():
        return [], set()
    data = json.loads(p.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else []
    keys = {
        build_comment_dedup_key(str(e["media_id"]), str(e["text"]))
        for e in entries
        if isinstance(e, dict) and e.get("media_id") and e.get("text")
    }
    return entries, keys


def save_raw_comments_file(entries: list[dict[str, Any]], path: Path | str | None = None) -> None:
    p = Path(path) if path is not None else RAW_COMMENTS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)


def collect_top_comments(
    username: str,
    context: BrowserContext,
    reels: list[dict[str, Any]],
    niches: list[str] | str,
    *,
    raw_comments_path: Path | str | None = None,
    classify: bool = True,
    logger: logging.Logger | None = None,
) -> int:
    """Collecte tous les commentaires visibles sur les 3 reels les plus commentés."""
    log_cb = logger or log
    u = (username or "").lstrip("@").strip()
    reels_sorted = sorted(
        [r for r in (reels or []) if str(r.get("media_id") or "").strip()],
        key=lambda r: int(r.get("comment_count") or 0),
        reverse=True,
    )
    reels_to_visit = reels_sorted[:COMMENTS_COLLECT_REELS_MAX]
    if not u or not reels_to_visit:
        return 0

    entries, dedup_keys = load_raw_comments_file(raw_comments_path)
    classifier = None
    if classify:
        try:
            from modules.classifier import CommentClassifier

            classifier = CommentClassifier()
        except Exception as e:
            log_cb.warning(
                "Collecte commentaires @%s : classifier indisponible (%s) — "
                "commentaires non classifiés.",
                u,
                e,
            )
            classify = False

    niches_list = list(niches) if isinstance(niches, list) else [str(niches)]
    candidates: list[dict[str, Any]] = []
    page = context.new_page()

    try:
        if not open_reels_grid(page, u):
            return 0

        for reel in reels_to_visit:
            media_id = str(reel.get("media_id") or "").strip()
            view_count = int(reel.get("view_count") or reel.get("views") or 0)
            try:
                if not navigate_to_reel_page(
                    page, media_id, u, reels_grid_loaded=True
                ):
                    log_cb.warning(
                        "Collecte commentaires @%s reel %s : page non chargée.",
                        u,
                        media_id,
                    )
                    continue

                clicked = click_reel_comment_button(page)
                if not clicked:
                    log_cb.warning(
                        "Collecte commentaires @%s reel %s : bouton absent. "
                        "aria-labels : %s",
                        u,
                        media_id,
                        _list_reel_page_aria_labels(page)[:25] or "(aucun)",
                    )
                    return_to_reels_grid(page, u)
                    continue

                page.wait_for_timeout(2000)
                panel_text = extract_reel_comments_panel_text(page)
                parsed = parse_comments_from_dom_text(str(panel_text or ""))
                for comment in parsed:
                    text = str(comment.get("text") or "").strip()
                    if not text:
                        continue
                    candidates.append(
                        {
                            "media_id": media_id,
                            "views": view_count,
                            "text": text,
                            "like_count": int(comment.get("like_count") or 0),
                        }
                    )
            except Exception as e:
                log_cb.warning(
                    "Collecte commentaires @%s reel %s : erreur (%s).",
                    u,
                    media_id,
                    e,
                )
            finally:
                return_to_reels_grid(page, u)
            polite_sleep(seconds=1)

        collected = 0

        for comment in candidates:
            media_id = str(comment.get("media_id") or "").strip()
            comment_text = str(comment.get("text") or "").strip()
            view_count = int(comment.get("views") or 0)

            t_type = ""
            llm_validated = False
            if classify and classifier is not None:
                try:
                    from modules.classifier import ClassificationError

                    clf = classifier.classify([comment_text], niches=niches_list)
                    t_type = str(clf.get("type") or "")
                    llm_validated = True
                except (ValueError, ClassificationError) as e:
                    log_cb.warning(
                        "Collecte commentaires @%s reel %s : classify KO (%s).",
                        u,
                        media_id,
                        e,
                    )
                    continue

            dedup_key = build_comment_dedup_key(media_id, comment_text)
            if dedup_key in dedup_keys:
                continue
            dedup_keys.add(dedup_key)
            entries.append(
                {
                    "media_id": media_id,
                    "username": u,
                    "niches": niches_list,
                    "text": comment_text,
                    "comment_likes": int(comment.get("like_count") or 0),
                    "views": view_count,
                    "comment_to_like_ratio": 0.0,
                    "caption": "",
                    "hashtags": [],
                    "audio_id": "",
                    "t_type": t_type or None,
                    "t_type_profile": None,
                    "llm_validated": llm_validated,
                    "collected_at": datetime.now(timezone.utc)
                    .replace(microsecond=0)
                    .isoformat(),
                }
            )
            collected += 1
    finally:
        page.close()

    if collected:
        save_raw_comments_file(entries, raw_comments_path)
    if candidates:
        reels_with_data = len({c.get("media_id") for c in candidates if c.get("media_id")})
        log_cb.info(
            "Collecte commentaires @%s : %d nouveau(x) enregistré(s) "
            "(%d parsé(s) sur %d reel(s)).",
            u,
            collected,
            len(candidates),
            reels_with_data,
        )
    return collected


def _list_reel_page_aria_labels(page: Page, limit: int = 40) -> list[str]:
    """Debug : aria-labels visibles sur la page reel (diagnostic sélecteurs)."""
    try:
        raw = page.evaluate(
            """(limit) => {
                const out = [];
                document.querySelectorAll("[aria-label]").forEach((el) => {
                    const a = el.getAttribute("aria-label");
                    if (a && !out.includes(a)) out.push(a);
                });
                return out.slice(0, limit);
            }""",
            limit,
        )
        return [str(x) for x in raw] if isinstance(raw, list) else []
    except Exception:
        return []


def click_reel_comment_button(page: Page) -> str | None:
    """Ouvre le panneau commentaires. Retourne le aria-label cliqué ou None."""
    for label in _REEL_COMMENT_ARIA_LABELS:
        for selector in (
            f'button:has(svg[aria-label="{label}"])',
            f'div[role="button"]:has(svg[aria-label="{label}"])',
            f'svg[aria-label="{label}"]',
            f'[aria-label="{label}"]',
        ):
            loc = page.locator(selector)
            if loc.count() == 0:
                continue
            try:
                loc.first.click(timeout=10_000)
                return label
            except Exception:
                continue

    try:
        page.get_by_role(
            "button",
            name=re.compile(r"comment", re.IGNORECASE),
        ).first.click(timeout=10_000)
        return "role=button(name~/comment/i)"
    except Exception:
        pass

    try:
        clicked_label = page.evaluate(
            """() => {
                const labels = %s;
                for (const label of labels) {
                    const svg = document.querySelector(
                        'svg[aria-label="' + label + '"]'
                    );
                    if (!svg) continue;
                    const btn = svg.closest('div[role="button"]')
                        || svg.closest('button')
                        || svg.parentElement;
                    if (btn) { btn.click(); return label; }
                }
                return null;
            }"""
            % json.dumps(list(_REEL_COMMENT_ARIA_LABELS))
        )
        return str(clicked_label) if clicked_label else None
    except Exception:
        return None


def extract_reel_comments_panel_text(
    page: Page,
    *,
    scroll_rounds: int = COMMENTS_PANEL_SCROLL_ROUNDS,
) -> str:
    """Texte brut du panneau commentaires (scroll pour charger plus de lignes)."""
    try:
        for _ in range(max(0, scroll_rounds)):
            page.evaluate(
                """() => {
                    const dialog = document.querySelector('[role="dialog"]');
                    if (!dialog) return;
                    const scrollable = dialog.querySelector('ul')
                        || dialog.querySelector('div[style*="overflow"]')
                        || dialog;
                    scrollable.scrollTop = scrollable.scrollHeight;
                }"""
            )
            page.wait_for_timeout(700)
        return str(
            page.evaluate(
                """(selectors) => {
                    for (const sel of selectors) {
                        const el = document.querySelector(sel);
                        if (el && el.innerText && el.innerText.trim().length > 20) {
                            return el.innerText;
                        }
                    }
                    return "";
                }""",
                list(_REEL_COMMENTS_PANEL_SELECTORS),
            )
            or ""
        )
    except Exception:
        return ""


def extract_reel_caption_from_dom(page: Page) -> str:
    """Repli caption depuis le DOM visible (dialog reel ou article)."""
    try:
        dialog_lines = page.evaluate(
            """() => {
                const ul = document.querySelector('div[role="dialog"] ul');
                if (!ul) return [];
                return (ul.innerText || "").split("\\n").map((s) => s.trim());
            }"""
        )
        if isinstance(dialog_lines, list):
            for line in dialog_lines[1:10]:
                s = str(line or "").strip()
                if len(s) < 8 or len(s) > 2200:
                    continue
                if _TIMESTAMP_RE.match(s) or _COMMENT_META_RE.search(s):
                    continue
                if any(
                    x in s
                    for x in (
                        "Voir la traduction",
                        "Suivre",
                        "Audio d'origine",
                        "Aimé par",
                    )
                ):
                    continue
                if "@" in s or len(s.split()) >= 4:
                    return s
    except Exception:
        pass

    try:
        text = page.evaluate(
            """() => {
                const article = document.querySelector("article");
                if (!article) return "";
                const skip = /likes?|comment|J'aime|partager|enregistr|écouter|audio/i;
                const nodes = article.querySelectorAll(
                    'h1, span[dir="auto"], div[dir="auto"]'
                );
                for (const el of nodes) {
                    const t = (el.innerText || "").trim();
                    if (t.length < 8 || t.length > 2200) continue;
                    if (skip.test(t)) continue;
                    if (/^@[\\w.]+$/.test(t)) continue;
                    return t;
                }
                return "";
            }"""
        )
        return str(text or "").strip()
    except Exception:
        return ""


def _parse_dom_comment_likes(likes_line: str) -> int:
    line = str(likes_line or "")
    m = re.search(
        r"(\d[\d\s\u202f\xa0.,]*)\s*J['\u2019]?aime",
        line,
        re.IGNORECASE,
    )
    if m:
        return _parse_count(m.group(1))
    likes_clean = re.sub(r"[^\d]", "", line.split("J")[0])
    if not likes_clean:
        return 0
    try:
        return int(likes_clean)
    except ValueError:
        return 0


def _parse_comments_dom_flexible(text: str) -> list[dict[str, Any]]:
    """Parse le panneau commentaires layout actuel (lignes collées type ``7 semRépondre``)."""
    lines = [ln.strip() for ln in str(text).split("\n") if ln.strip()]
    if not lines:
        return []

    start = 0
    for idx, ln in enumerate(lines):
        if "Voir la traduction" in ln:
            start = idx + 1
            break

    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    i = start
    while i < len(lines):
        user_line = lines[i]
        if not _USERNAME_RE.match(user_line):
            i += 1
            continue

        commenter = user_line.lstrip("@").lower()
        i += 1
        comment_parts: list[str] = []
        like_count = 0

        while i < len(lines):
            line = lines[i]
            if _COMMENT_META_RE.search(line):
                if re.search(r"J.aime|like", line, re.IGNORECASE):
                    like_count = _parse_dom_comment_likes(line)
                i += 1
                break
            if not any(noise in line for noise in _COMMENTS_UI_NOISE):
                comment_parts.append(line)
            i += 1

        comment_line = " ".join(comment_parts).strip()
        if not comment_line or any(noise in comment_line for noise in _COMMENTS_UI_NOISE):
            continue
        lower = comment_line.lower()
        if any(x in lower for x in ("http", "www", ".com")):
            continue
        if len(comment_line) < 2:
            continue

        key = (commenter, comment_line.lower())
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "username_commenter": commenter,
                "text": comment_line,
                "like_count": like_count,
            }
        )

    return results


def parse_comments_from_dom_text(text: str) -> list[dict[str, Any]]:
    """Parse le panneau commentaires Instagram (texte DOM) en entrées structurées.

    Structure legacy par bloc (6 lignes) ou layout reel dialog (flexible).
    """
    if not text or not str(text).strip():
        return []

    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    lines = str(text).split("\n")
    i = 0
    while i < len(lines) - 5:
        username_line = lines[i].strip()
        space_line = lines[i + 1]
        timestamp_line = lines[i + 2].strip() if i + 2 < len(lines) else ""
        comment_line = lines[i + 3].strip() if i + 3 < len(lines) else ""
        likes_line = lines[i + 4].strip() if i + 4 < len(lines) else ""

        is_username = bool(_USERNAME_RE.match(username_line))
        is_space = space_line.strip() == "" or space_line in (" ", "\xa0", "\u00a0")
        is_timestamp = bool(_TIMESTAMP_RE.match(timestamp_line))
        is_likes = bool(re.search(r"J.aime|like", likes_line, re.IGNORECASE))

        if is_username and is_space and is_timestamp and is_likes:
            like_count = _parse_dom_comment_likes(likes_line)
            if not any(noise in comment_line for noise in _COMMENTS_UI_NOISE):
                words = re.findall(r"[a-zA-ZÀ-ÿ]{2,}", comment_line)
                lower = comment_line.lower()
                has_link = any(x in lower for x in ("http", "www", ".com"))
                if len(comment_line.split()) >= 2 and not has_link and len(words) >= 1:
                    commenter = username_line.lstrip("@").lower()
                    key = (commenter, comment_line.lower())
                    if key not in seen:
                        seen.add(key)
                        results.append(
                            {
                                "username_commenter": commenter,
                                "text": comment_line,
                                "like_count": like_count,
                            }
                        )
            i += 6
        else:
            i += 1

    if results:
        return results
    return _parse_comments_dom_flexible(text)


def _parse_count(text: str) -> int:
    """Parse un compteur Instagram (FR/EN : k/K, m/M, virgule décimale, espaces milliers).

    Règles :
    1. Extraire nombre + suffixe (k/K/m/M)
    2. Virgule + 1 chiffre (ou ≤2 avec suffixe) après → décimale (``73,7 k`` → 73700)
    3. Virgule + 3 chiffres après → séparateur milliers (``1,234`` → 1234)
    4. Espaces → séparateur milliers (``1 234`` → 1234)
    5. k/K ×1000, m/M ×1_000_000
    """
    if not text or not str(text).strip():
        return 0
    s0 = str(text).strip().replace("\u202f", " ").replace("\xa0", " ")
    m = re.search(r"([\d\s,.]+)\s*([kKmMbB])?(?:\b|$)", s0, re.I)
    if not m:
        return 0

    num_raw = m.group(1).strip()
    suf = (m.group(2) or "").lower()
    mult = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(suf, 1)
    has_suffix = bool(suf)

    if not num_raw:
        return 0

    # 4. Espaces comme séparateurs de milliers
    if " " in num_raw:
        num_raw = "".join(num_raw.split())

    val = _parse_count_number(num_raw, has_suffix=has_suffix)
    return int(round(val * mult))


def _parse_count_number(num_raw: str, *, has_suffix: bool) -> float:
    """Convertit la partie numérique (sans suffixe k/m) en float."""
    if not num_raw:
        return 0.0

    if "," in num_raw and "." in num_raw:
        if num_raw.rfind(",") > num_raw.rfind("."):
            return float(num_raw.replace(".", "").replace(",", "."))
        return float(num_raw.replace(",", ""))

    if "," in num_raw:
        parts = num_raw.split(",")
        if len(parts) == 2:
            after = parts[1]
            # 3. Virgule + 3 chiffres → milliers (sans suffixe compact type k)
            if len(after) == 3 and not has_suffix:
                return float(parts[0] + after)
            # 2. Virgule + 1 chiffre, ou ≤2 avec suffixe → décimale FR
            if len(after) == 1 or (len(after) <= 2 and has_suffix):
                return float(f"{parts[0]}.{after}")
            if len(after) <= 2:
                return float(f"{parts[0]}.{after}")
            return float(num_raw.replace(",", ""))
        if len(parts) >= 2 and all(len(p) == 3 for p in parts[1:]):
            return float("".join(parts))
        return float(num_raw.replace(",", ""))

    if "." in num_raw:
        parts = num_raw.split(".")
        if len(parts) == 2:
            after = parts[1]
            if len(after) == 3 and not has_suffix:
                return float(parts[0] + after)
            if len(after) <= 2:
                return float(num_raw)
            return float(num_raw.replace(".", ""))
        if len(parts) >= 2 and all(len(p) == 3 for p in parts[1:]):
            return float("".join(parts))
        return float(num_raw.replace(".", ""))

    return float(num_raw)


def _parse_compact_number(s: str) -> int:
    """Alias pour les autres scrapers (reels, etc.)."""
    return _parse_count(s)


def _strip_model_json_fence(raw: str) -> str:
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _parse_model_json(raw: str) -> Any | None:
    text = _strip_model_json_fence(raw)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None


def _empty_reel_metrics() -> dict[str, int]:
    return {
        "view_count": 0,
        "like_count": 0,
        "comment_count": 0,
        "share_count": 0,
    }


def _merge_metric_bucket(bucket: dict[str, int], patch: dict[str, int]) -> None:
    for key, value in patch.items():
        if value > 0:
            bucket[key] = value
        elif key not in bucket:
            bucket[key] = 0


def _normalize_metric_bucket(raw: dict[str, Any]) -> dict[str, int]:
    view_count = int(raw.get("view_count") or 0)
    if not view_count:
        view_count = int(raw.get("play_count") or 0)
    if not view_count:
        view_count = int(raw.get("video_view_count") or 0)
    return {
        "view_count": view_count,
        "like_count": int(raw.get("like_count") or 0),
        "comment_count": int(raw.get("comment_count") or 0),
        "share_count": int(raw.get("share_count") or 0),
    }


def _unescape_json_string_fragment(fragment: str) -> str:
    try:
        return str(json.loads(f'"{fragment}"')).strip()
    except json.JSONDecodeError:
        return fragment.replace("\\n", "\n").replace('\\"', '"').strip()


def _extract_caption_from_node(node: dict[str, Any]) -> str:
    caption_obj = node.get("caption")
    if isinstance(caption_obj, dict):
        text = caption_obj.get("text")
        if text is not None:
            return str(text).strip()
    caption_text = node.get("caption_text")
    if caption_text is not None:
        return str(caption_text).strip()
    return ""


def _extract_caption_from_graphql_window(window: str) -> str:
    caption_match = re.search(
        r'"caption"\s*:\s*\{\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"',
        window,
    )
    if caption_match:
        return _unescape_json_string_fragment(caption_match.group(1))
    caption_text_match = re.search(
        r'"caption_text"\s*:\s*"((?:[^"\\]|\\.)*)"',
        window,
    )
    if caption_text_match:
        return _unescape_json_string_fragment(caption_text_match.group(1))
    return ""


def _merge_caption_into_bucket(bucket: dict[str, Any], caption: str) -> None:
    text = (caption or "").strip()
    if not text:
        return
    existing = str(bucket.get("caption") or "").strip()
    if not existing or len(text) > len(existing):
        bucket["caption"] = text


def _is_pinned_from_clips_tab_ids(raw: Any) -> bool:
    return bool(isinstance(raw, list) and len(raw) > 0)


def _merge_pinned_into_bucket(bucket: dict[str, Any], is_pinned: bool | None) -> None:
    if is_pinned is None:
        return
    bucket["is_pinned"] = is_pinned


def _extract_pinned_from_graphql_window(window: str) -> bool | None:
    match = re.search(
        r'"clips_tab_pinned_user_ids"\s*:\s*(\[[^\]]*\])',
        window,
    )
    if not match:
        return None
    try:
        ids = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return _is_pinned_from_clips_tab_ids(ids)


def _ingest_graphql_media_node(
    node: dict[str, Any],
    metrics_by_pk: dict[str, dict[str, Any]],
    metrics_by_code: dict[str, dict[str, Any]],
) -> None:
    pk: str | None = None
    for key in ("pk", "id"):
        value = node.get(key)
        if value is not None and str(value).strip().isdigit():
            pk = str(value).strip()
            break

    code: str | None = None
    for key in ("code", "shortcode"):
        value = node.get(key)
        if value is not None and str(value).strip():
            code = str(value).strip()
            break

    patch: dict[str, int] = {}
    for key in _GRAPHQL_METRIC_KEYS:
        value = node.get(key)
        if isinstance(value, (int, float)):
            patch[key] = int(value)

    caption = _extract_caption_from_node(node)
    has_pinned_field = "clips_tab_pinned_user_ids" in node
    is_pinned: bool | None = None
    if has_pinned_field:
        is_pinned = _is_pinned_from_clips_tab_ids(node.get("clips_tab_pinned_user_ids"))

    if not patch and not caption and not has_pinned_field:
        return

    normalized = _normalize_metric_bucket(patch) if patch else {}
    if pk:
        bucket = metrics_by_pk.setdefault(pk, {})
        if patch:
            _merge_metric_bucket(bucket, normalized)
        if caption:
            _merge_caption_into_bucket(bucket, caption)
        if has_pinned_field:
            _merge_pinned_into_bucket(bucket, is_pinned)
        if code:
            metrics_by_code[code] = dict(bucket)
    elif code:
        bucket = metrics_by_code.setdefault(code, {})
        if patch:
            _merge_metric_bucket(bucket, normalized)
        if caption:
            _merge_caption_into_bucket(bucket, caption)
        if has_pinned_field:
            _merge_pinned_into_bucket(bucket, is_pinned)


def _walk_graphql_metrics(
    data: Any,
    metrics_by_pk: dict[str, dict[str, Any]],
    metrics_by_code: dict[str, dict[str, Any]],
) -> None:
    if isinstance(data, dict):
        _ingest_graphql_media_node(data, metrics_by_pk, metrics_by_code)
        for value in data.values():
            _walk_graphql_metrics(value, metrics_by_pk, metrics_by_code)
    elif isinstance(data, list):
        for item in data:
            _walk_graphql_metrics(item, metrics_by_pk, metrics_by_code)


def _ingest_metrics_from_graphql_text(
    text: str,
    metrics_by_pk: dict[str, dict[str, Any]],
    metrics_by_code: dict[str, dict[str, Any]],
) -> None:
    """Regex sur le JSON sérialisé : associe pk/id + code court aux métriques."""
    block_window = 800

    def _apply_pk_metric(pattern: str, metric_key: str, *, only_if_missing: bool = False) -> None:
        for media_pk, raw_value in re.findall(pattern, text):
            bucket = metrics_by_pk.setdefault(media_pk, {})
            value = int(raw_value)
            if only_if_missing and bucket.get(metric_key, 0) > 0:
                continue
            bucket[metric_key] = value

    _apply_pk_metric(
        rf'"(?:pk|id)"\s*:\s*"?(\d+)"?[^}}]{{0,{block_window}}}"view_count"\s*:\s*(\d+)',
        "view_count",
    )
    _apply_pk_metric(
        rf'"(?:pk|id)"\s*:\s*"?(\d+)"?[^}}]{{0,{block_window}}}"play_count"\s*:\s*(\d+)',
        "view_count",
        only_if_missing=True,
    )
    _apply_pk_metric(
        rf'"(?:pk|id)"\s*:\s*"?(\d+)"?[^}}]{{0,{block_window}}}"video_view_count"\s*:\s*(\d+)',
        "view_count",
        only_if_missing=True,
    )
    _apply_pk_metric(
        rf'"(?:pk|id)"\s*:\s*"?(\d+)"?[^}}]{{0,{block_window}}}"like_count"\s*:\s*(\d+)',
        "like_count",
    )
    _apply_pk_metric(
        rf'"(?:pk|id)"\s*:\s*"?(\d+)"?[^}}]{{0,{block_window}}}"comment_count"\s*:\s*(\d+)',
        "comment_count",
    )
    _apply_pk_metric(
        rf'"(?:pk|id)"\s*:\s*"?(\d+)"?[^}}]{{0,{block_window}}}"share_count"\s*:\s*(\d+)',
        "share_count",
    )

    code_re = re.compile(r'"(?:code|shortcode)"\s*:\s*"([A-Za-z0-9_-]+)"')
    pk_re = re.compile(r'"(?:pk|id)"\s*:\s*"?(\d+)"?')
    pinned_ids = re.findall(
        r'"clips_tab_pinned_user_ids"\s*:\s*(\[[^\]]*\])',
        text,
    )

    for match in code_re.finditer(text):
        code = match.group(1)
        window_start = max(0, match.start() - block_window)
        window = text[window_start : match.start() + block_window]
        caption = _extract_caption_from_graphql_window(window)
        is_pinned = _extract_pinned_from_graphql_window(window)
        if code in metrics_by_code:
            if caption:
                _merge_caption_into_bucket(metrics_by_code[code], caption)
            if is_pinned is not None:
                _merge_pinned_into_bucket(metrics_by_code[code], is_pinned)
            continue
        pk_match = pk_re.search(window)
        if pk_match and pk_match.group(1) in metrics_by_pk:
            metrics_by_code[code] = dict(metrics_by_pk[pk_match.group(1)])
            if caption:
                _merge_caption_into_bucket(metrics_by_code[code], caption)
            if is_pinned is not None:
                _merge_pinned_into_bucket(metrics_by_code[code], is_pinned)
            continue
        bucket = metrics_by_code.setdefault(code, {})
        patch: dict[str, int] = {}
        for key in _GRAPHQL_METRIC_KEYS:
            metric_match = re.search(rf'"{key}"\s*:\s*(\d+)', window)
            if metric_match:
                patch[key] = int(metric_match.group(1))
        if patch:
            _merge_metric_bucket(bucket, _normalize_metric_bucket(patch))
        if caption:
            _merge_caption_into_bucket(bucket, caption)
        if is_pinned is not None:
            _merge_pinned_into_bucket(bucket, is_pinned)


def _metrics_bucket_for_dom_media_id(
    media_id: str,
    metrics_by_pk: dict[str, dict[str, Any]],
    metrics_by_code: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    mid = media_id.strip()
    if not mid:
        return {}
    if mid in metrics_by_code:
        return metrics_by_code[mid]
    if mid in metrics_by_pk:
        return metrics_by_pk[mid]
    return {}


def _metrics_for_dom_media_id(
    media_id: str,
    metrics_by_pk: dict[str, dict[str, Any]],
    metrics_by_code: dict[str, dict[str, Any]],
) -> dict[str, int]:
    bucket = _metrics_bucket_for_dom_media_id(media_id, metrics_by_pk, metrics_by_code)
    if not bucket:
        return _empty_reel_metrics()
    return _normalize_metric_bucket(bucket)


def _attach_graphql_metrics_listener(
    page: Page,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Écoute GraphQL et indexe métriques + captions par pk numérique et code court."""
    metrics_by_pk: dict[str, dict[str, Any]] = {}
    metrics_by_code: dict[str, dict[str, Any]] = {}

    def on_response(response: Response) -> None:
        if "graphql" not in response.url:
            return
        try:
            data = response.json()
        except Exception:
            try:
                text = response.text()
            except Exception:
                return
            _ingest_metrics_from_graphql_text(text, metrics_by_pk, metrics_by_code)
            return
        _walk_graphql_metrics(data, metrics_by_pk, metrics_by_code)
        _ingest_metrics_from_graphql_text(json.dumps(data), metrics_by_pk, metrics_by_code)

    page.on("response", on_response)
    return metrics_by_pk, metrics_by_code


def _lm_studio_vision_text(screenshot_path: Path | str, prompt: str) -> str:
    path = Path(screenshot_path)
    if not path.is_file():
        return ""
    try:
        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return ""
    payload = {
        "model": LM_STUDIO_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    try:
        r = requests.post(
            LM_STUDIO_CHAT_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return ""


_SUGGESTION_RESERVED_USERNAMES = frozenset(
    {
        "instagram",
        "meta",
        "facebook",
        "threads",
        "explore",
        "reels",
        "stories",
        "direct",
        *_IG_RESERVED_USERNAMES,
    }
)


def _ingest_suggestion_usernames_from_graphql_text(
    text: str,
    target_username: str,
    seen: set[str],
    suggested_usernames: list[str],
) -> None:
    """Extrait les usernames d'une réponse GraphQL (sans filtre suggest/recommend)."""
    for raw in re.findall(r'"username"\s*:\s*"([^"]+)"', text):
        u = raw.strip().lstrip("@").lower()
        if u == target_username:
            continue
        if u in _SUGGESTION_RESERVED_USERNAMES:
            continue
        if len(u) < 2 or ("." not in u and len(u) < 3):
            continue
        if u in seen:
            continue
        seen.add(u)
        suggested_usernames.append(u)


def _attach_graphql_suggestions_listener(
    page: Page, username: str
) -> list[str]:
    """Écoute GraphQL et accumule les usernames depuis toutes les réponses."""
    suggested_usernames: list[str] = []
    seen: set[str] = set()
    target_username = username.lstrip("@").strip().lower()

    def on_response(response: Response) -> None:
        if "graphql" not in response.url:
            return
        try:
            text = response.text()
        except Exception:
            return
        _ingest_suggestion_usernames_from_graphql_text(
            text, target_username, seen, suggested_usernames
        )

    page.on("response", on_response)
    return suggested_usernames


def _vision_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    try:
        return int(float(str(value).replace(",", "").replace(" ", "").replace("\u202f", "")))
    except (TypeError, ValueError):
        return 0


def _vision_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "oui")
    return bool(value)


_PROFILE_STATS_VISION_PROMPT = """Analyse ce screenshot d'un profil Instagram.
Extrais exactement ces données en JSON :
{
  "followers": int,
  "following": int,
  "posts_count": int,
  "full_name": string,
  "biography": string,
  "is_private": false
}
Réponds UNIQUEMENT avec le JSON, rien d'autre."""


def extract_stats_from_screenshot(screenshot_path: Path | str) -> dict[str, Any]:
    """Envoie le screenshot à LM Studio (vision) et retourne un dict de stats (fallback si échec)."""
    out = dict(_VISION_FALLBACK)
    raw_text = _lm_studio_vision_text(screenshot_path, _PROFILE_STATS_VISION_PROMPT)
    parsed = _parse_model_json(raw_text)
    if not isinstance(parsed, dict):
        return dict(_VISION_FALLBACK)
    out["followers"] = _vision_int(parsed.get("followers"))
    out["following"] = _vision_int(parsed.get("following"))
    out["posts_count"] = _vision_int(parsed.get("posts_count"))
    out["full_name"] = str(parsed.get("full_name") or "").strip()
    out["biography"] = str(parsed.get("biography") or "").strip()
    out["is_private"] = _vision_bool(parsed.get("is_private"))
    return out


def _normalize_playwright_cookies(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict) and "cookies" in raw:
        raw = raw["cookies"]
    if not isinstance(raw, list):
        return []

    out: list[dict[str, Any]] = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        name, value = c.get("name"), c.get("value")
        if not name or value is None:
            continue
        entry: dict[str, Any] = {
            "name": str(name),
            "value": str(value),
            "domain": str(c.get("domain") or ".instagram.com"),
            "path": str(c.get("path") or "/"),
        }
        exp = c.get("expires")
        if exp is not None and exp != 0 and exp != -1:
            try:
                entry["expires"] = int(float(exp))
            except (TypeError, ValueError):
                pass
        if "httpOnly" in c:
            entry["httpOnly"] = bool(c["httpOnly"])
        if "secure" in c:
            entry["secure"] = bool(c["secure"])
        ss = c.get("sameSite")
        if ss in ("Lax", "Strict", "None"):
            entry["sameSite"] = ss
        elif isinstance(ss, str):
            low = ss.lower()
            if low == "lax":
                entry["sameSite"] = "Lax"
            elif low == "strict":
                entry["sameSite"] = "Strict"
            elif low in ("none", "no_restriction"):
                entry["sameSite"] = "None"
        out.append(entry)
    return out


def get_browser_context(playwright: Playwright) -> BrowserContext:
    """Lance Chromium, injecte les cookies Instagram depuis ``COOKIES_PATH``."""
    if not COOKIES_PATH.is_file():
        raise FileNotFoundError(f"Fichier cookies introuvable : {COOKIES_PATH.resolve()}")

    cookies = _normalize_playwright_cookies(
        json.loads(COOKIES_PATH.read_text(encoding="utf-8"))
    )
    browser = playwright.chromium.launch(headless=HEADLESS)
    context = browser.new_context(
        viewport=_VIEWPORT,
        locale="fr-FR",
    )
    if cookies:
        context.add_cookies(cookies)
    return context


def _session_ok(context: BrowserContext) -> bool:
    """True si la page d'accueil ne montre pas le formulaire de connexion."""
    page = context.new_page()
    try:
        page.goto(
            f"{BASE_URL}/",
            timeout=_REQUEST_TIMEOUT_MS,
            wait_until="domcontentloaded",
        )
        page.wait_for_timeout(800)
        url = page.url or ""
        if "/accounts/login" in url:
            return False
        user_in = page.locator('input[name="username"]')
        pass_in = page.locator('input[name="password"]')
        if user_in.count() > 0 and pass_in.count() > 0:
            try:
                if user_in.first.is_visible(timeout=1_500) and pass_in.first.is_visible(timeout=500):
                    return False
            except Exception:
                pass
        if page.locator('svg[aria-label="Accueil"]').count() > 0:
            return True
        if page.locator('svg[aria-label="Home"]').count() > 0:
            return True
        if page.locator('a[href*="/direct/inbox/"]').count() > 0:
            return True
        return "/accounts/login" not in url
    except Exception:
        return False
    finally:
        page.close()


def test_session() -> bool:
    """Vérifie la session (cookies) sans argument : lance Playwright localement."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        context = get_browser_context(p)
        try:
            return _session_ok(context)
        finally:
            br = context.browser
            if br:
                br.close()


def _merge_profile_graphql_text(
    text: str, target_username: str, profile_data: dict[str, Any]
) -> None:
    """Extrait les champs profil depuis un fragment GraphQL (fenêtre autour du username)."""
    username_match = re.search(
        rf'"username"\s*:\s*"{re.escape(target_username)}"',
        text,
        re.I,
    )
    if username_match:
        start = max(0, username_match.start() - 400)
        window = text[start : username_match.start() + 3000]
    else:
        window = text

    followers = re.findall(r'"follower_count"\s*:\s*(\d+)', window)
    following = re.findall(r'"following_count"\s*:\s*(\d+)', window)
    biography = re.findall(r'"biography"\s*:\s*"([^"]{0,300})"', window)
    full_name = re.findall(r'"full_name"\s*:\s*"([^"]{0,100})"', window)
    media_count = re.findall(r'"media_count"\s*:\s*(\d+)', window)
    is_private = re.findall(r'"is_private"\s*:\s*(true|false)', window)

    if followers:
        profile_data["followers"] = int(followers[0])
    if following:
        profile_data["following"] = int(following[0])
    if biography:
        profile_data["biography"] = biography[0]
    if full_name:
        profile_data["full_name"] = full_name[0]
    if media_count:
        profile_data["posts_count"] = int(media_count[0])
    if is_private:
        profile_data["is_private"] = is_private[0] == "true"


def _merge_profile_graphql_node(
    node: dict[str, Any], target_username: str, profile_data: dict[str, Any]
) -> None:
    username = node.get("username")
    if not username or str(username).strip().lower() != target_username:
        return

    if "follower_count" in node:
        profile_data["followers"] = int(node["follower_count"])
    edge_followers = node.get("edge_followed_by")
    if isinstance(edge_followers, dict) and "count" in edge_followers:
        profile_data["followers"] = int(edge_followers["count"])

    if "following_count" in node:
        profile_data["following"] = int(node["following_count"])
    edge_following = node.get("edge_follow")
    if isinstance(edge_following, dict) and "count" in edge_following:
        profile_data["following"] = int(edge_following["count"])

    if "media_count" in node:
        profile_data["posts_count"] = int(node["media_count"])
    edge_media = node.get("edge_owner_to_timeline_media")
    if isinstance(edge_media, dict) and "count" in edge_media:
        profile_data["posts_count"] = int(edge_media["count"])

    if "biography" in node and node["biography"] is not None:
        profile_data["biography"] = str(node["biography"])
    if "full_name" in node and node["full_name"] is not None:
        profile_data["full_name"] = str(node["full_name"]).strip()
    if "is_private" in node:
        profile_data["is_private"] = bool(node["is_private"])


def _walk_profile_graphql(
    data: Any, target_username: str, profile_data: dict[str, Any]
) -> None:
    if isinstance(data, dict):
        _merge_profile_graphql_node(data, target_username, profile_data)
        for value in data.values():
            _walk_profile_graphql(value, target_username, profile_data)
    elif isinstance(data, list):
        for item in data:
            _walk_profile_graphql(item, target_username, profile_data)


def _attach_graphql_profile_listener(
    page: Page, username: str
) -> dict[str, Any]:
    """Écoute GraphQL et accumule les champs du profil cible."""
    profile_data: dict[str, Any] = {}
    target = username.lstrip("@").strip().lower()

    def on_response(response: Response) -> None:
        if "graphql" not in response.url:
            return
        try:
            text = response.text()
        except Exception:
            return
        _merge_profile_graphql_text(text, target, profile_data)
        try:
            data = response.json()
        except Exception:
            return
        _walk_profile_graphql(data, target, profile_data)
        _merge_profile_graphql_text(json.dumps(data), target, profile_data)

    page.on("response", on_response)
    return profile_data


def get_profile_data(username: str, context: BrowserContext) -> dict[str, Any] | None:
    """Charge la page profil et extrait métriques + bio via interception GraphQL."""
    u = username.lstrip("@").strip()
    if not u:
        return None

    page = context.new_page()
    try:
        profile_data = _attach_graphql_profile_listener(page, u)

        page.goto(
            f"{BASE_URL}/{u}/",
            timeout=_REQUEST_TIMEOUT_MS,
            wait_until="domcontentloaded",
        )
        page.wait_for_load_state("load")
        page.wait_for_timeout(4000)

        body_preview = (page.inner_text("body") or "")[:8000]
        if any(
            x in body_preview
            for x in (
                "Sorry, this page isn't available",
                "Désolé, cette page n'est pas disponible",
                "Page introuvable",
            )
        ):
            return None

        if not profile_data or "followers" not in profile_data:
            return None

        profile_data.setdefault("followers", 0)
        profile_data.setdefault("following", 0)
        profile_data.setdefault("biography", "")
        profile_data.setdefault("full_name", u)
        profile_data.setdefault("posts_count", 0)
        profile_data.setdefault("is_private", False)
        profile_data["username"] = u

        return profile_data
    except Exception:
        return None
    finally:
        page.close()


def debug_profile(username: str, context: BrowserContext) -> None:
    """Navigation debug : dump HTML + extraits de texte pouvant être des stats."""
    page = context.new_page()
    try:
        page.goto(
            f"{BASE_URL}/{username}/",
            timeout=_REQUEST_TIMEOUT_MS,
            wait_until="domcontentloaded",
        )
        page.wait_for_load_state("load", timeout=15_000)
        page.wait_for_timeout(3000)
        content = page.content()
        Path("/tmp/instagram_debug.html").write_text(content, encoding="utf-8")
        snippets = page.evaluate(
            """() => {
              const texts = [];
              document.querySelectorAll('*').forEach(el => {
                if (el.children.length === 0 && el.textContent.match(/\\d/)) {
                  texts.push(el.textContent.trim());
                }
              });
              return texts.filter(t => t.length < 20).slice(0, 50);
            }"""
        )
        print("[debug_profile] HTML écrit : /tmp/instagram_debug.html")
        print("[debug_profile] Extraits (max 50, len < 20) :")
        if isinstance(snippets, list):
            for i, t in enumerate(snippets):
                print(f"  {i + 1}: {t!r}")
        else:
            print(f"  {snippets!r}")
    finally:
        page.close()


def _collect_media_ids_from_grid(page: Page, max_reels: int) -> list[dict[str, str]]:
    """Collecte media_id + thumbnail depuis la grille /reels/ (ordre DOM)."""
    items = page.evaluate(
        """(max) => {
          const seen = new Set();
          const rows = [];
          document.querySelectorAll('a[href*="/reel/"]').forEach(a => {
            if (rows.length >= max) return;
            const href = a.getAttribute("href") || "";
            const m = href.match(/\\/reel\\/([^/?#]+)/);
            if (!m) return;
            const id = m[1];
            if (seen.has(id)) return;
            seen.add(id);
            let thumb = "";
            const img = a.querySelector("img");
            if (img) thumb = img.getAttribute("src") || "";
            rows.push({ media_id: id, thumbnail_url: thumb });
          });
          return rows;
        }""",
        max_reels,
    )
    out: list[dict[str, str]] = []
    if not isinstance(items, list):
        return out
    for row in items:
        if not isinstance(row, dict):
            continue
        media_id = str(row.get("media_id") or "").strip()
        if not media_id:
            continue
        out.append(
            {
                "media_id": media_id,
                "thumbnail_url": str(row.get("thumbnail_url") or ""),
            }
        )
    return out


def _reels_grid_rows_needed(max_reels: int) -> int:
    """Nombre de lignes de grille à couvrir (5 colonnes par ligne)."""
    return max(1, (max_reels + _REELS_GRID_COLUMNS - 1) // _REELS_GRID_COLUMNS)


def _scroll_reels_grid_until_loaded(
    page: Page, max_reels: int, grid_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Scroll progressif pour charger jusqu'à max_reels (ex. 20 = 4 lignes × 5)."""
    if max_reels <= 5:
        return grid_rows

    rows_target = _reels_grid_rows_needed(max_reels)
    max_rounds = rows_target * 3 + 4
    rounds = 0

    while len(grid_rows) < max_reels and rounds < max_rounds:
        previous_count = len(grid_rows)
        current_rows = (len(grid_rows) + _REELS_GRID_COLUMNS - 1) // _REELS_GRID_COLUMNS
        rows_to_advance = max(1, min(2, rows_target - current_rows))
        scroll_y = _REELS_GRID_ROW_HEIGHT_PX * rows_to_advance
        page.evaluate("(y) => window.scrollBy(0, y)", scroll_y)
        page.wait_for_timeout(_REELS_SCROLL_WAIT_MS)
        grid_rows = _collect_media_ids_from_grid(page, max_reels)
        rounds += 1

        if len(grid_rows) >= max_reels:
            break
        if len(grid_rows) == previous_count:
            remaining_rows = rows_target - current_rows
            if remaining_rows > 0:
                boost_y = _REELS_GRID_ROW_HEIGHT_PX * max(remaining_rows, 2)
                page.evaluate("(y) => window.scrollBy(0, y)", boost_y)
                page.wait_for_timeout(_REELS_SCROLL_WAIT_MS)
                grid_rows = _collect_media_ids_from_grid(page, max_reels)
            if len(grid_rows) == previous_count:
                break

    page.wait_for_timeout(3000)
    return grid_rows


def get_reel_caption(media_id: str, context: BrowserContext) -> str:
    """Récupère la caption d'un Reel via interception GraphQL sur /reel/{id}/."""
    mid = str(media_id or "").strip()
    if not mid:
        return ""

    caption = ""

    def capture(response: Response) -> None:
        nonlocal caption
        if "graphql" not in response.url or caption:
            return
        try:
            text = response.text()
            if mid not in text:
                return
            idx = text.find(mid)
            if idx < 0:
                return
            window = text[idx : idx + 2000]
            matches = re.findall(
                r'"(?:caption_text|text)"\s*:\s*"((?:[^"\\]|\\.){5,500})"',
                window,
            )
            for raw in matches:
                decoded = _unescape_json_string_fragment(raw)
                if len(decoded.split()) >= 3 and "Ne pas suggérer" not in decoded:
                    caption = decoded
                    break
        except Exception:
            pass

    page = context.new_page()
    try:
        page.on("response", capture)
        page.goto(
            f"{BASE_URL}/reel/{mid}/",
            timeout=_REQUEST_TIMEOUT_MS,
            wait_until="domcontentloaded",
        )
        page.wait_for_load_state("load")
        page.wait_for_timeout(3000)
    except Exception:
        pass
    finally:
        page.close()

    return caption


def get_recent_reels(
    username: str, context: BrowserContext, max_reels: int = 5
) -> list[dict[str, Any]]:
    """Reels récents : media_ids (DOM) + métriques (interception GraphQL).

    Les captions manquantes ne sont pas récupérées ici (pas de visite
    /reel/{id}/ — trop lent pour le scoring discovery). Utiliser
    ``get_reel_caption()`` depuis l'embedder (1x/semaine).
    """
    page = context.new_page()
    out: list[dict[str, Any]] = []
    try:
        metrics_by_pk, metrics_by_code = _attach_graphql_metrics_listener(page)

        page.goto(
            f"{BASE_URL}/{username}/reels/",
            timeout=_REQUEST_TIMEOUT_MS,
            wait_until="domcontentloaded",
        )
        page.set_viewport_size(_VIEWPORT)
        page.wait_for_load_state("load")

        deadline = time.time() + 5
        while time.time() < deadline:
            if len(metrics_by_code) >= max_reels:
                break
            page.wait_for_timeout(200)

        grid_rows = _collect_media_ids_from_grid(page, max_reels)

        if max_reels > 5:
            grid_rows = _scroll_reels_grid_until_loaded(page, max_reels, grid_rows)

        for row in grid_rows:
            media_id = row["media_id"]
            metrics = _metrics_for_dom_media_id(media_id, metrics_by_pk, metrics_by_code)
            entry = _metrics_bucket_for_dom_media_id(
                media_id, metrics_by_pk, metrics_by_code
            )
            caption = str(entry.get("caption") or "")
            is_pinned = bool(entry.get("is_pinned", False))
            out.append(
                {
                    "media_id": media_id,
                    "thumbnail_url": row["thumbnail_url"],
                    "view_count": metrics["view_count"],
                    "like_count": metrics["like_count"],
                    "comment_count": metrics["comment_count"],
                    "share_count": metrics["share_count"],
                    "reshare_count": 0,
                    "caption": caption,
                    "is_pinned": is_pinned,
                }
            )
        # caption vide : pas de get_reel_caption() ici (réservé à l'embedder).
    except Exception:
        return []
    finally:
        page.close()

    return out


def get_suggested_accounts(
    username: str, context: BrowserContext, max_results: int = 30
) -> list[str]:
    """Suggestions de comptes similaires via interception GraphQL (sans follow)."""
    u = username.lstrip("@").strip()
    if not u:
        return []

    page = context.new_page()
    try:
        suggested_usernames = _attach_graphql_suggestions_listener(page, u)

        page.goto(
            f"{BASE_URL}/{u}/",
            timeout=_REQUEST_TIMEOUT_MS,
            wait_until="domcontentloaded",
        )
        page.wait_for_load_state("load")
        page.wait_for_timeout(4000)

        log.info(
            "get_suggested_accounts @%s : %d usernames GraphQL capturés",
            u,
            len(suggested_usernames),
        )
        return suggested_usernames[:max_results]
    except Exception:
        log.info("get_suggested_accounts @%s : 0 usernames GraphQL capturés", u)
        return []
    finally:
        page.close()


def _run_parse_comments_dom_self_tests() -> None:
    sample = (
        "alice\n"
        "\xa0\n"
        "2 j\n"
        "Super sketch de fou rire\n"
        "1\u202f234\xa0J\u2019aime\n"
        "Répondre\n"
        "bob\n"
        "\xa0\n"
        "1 sem\n"
        "ok\n"
        "12\xa0J'aime\n"
        "Répondre\n"
        "spam\n"
        "\xa0\n"
        "1 j\n"
        "http://evil.com scam\n"
        "99\xa0J'aime\n"
        "Répondre\n"
    )
    parsed = parse_comments_from_dom_text(sample)
    if len(parsed) != 1 or parsed[0]["like_count"] != 1234:
        print(f"_parse_comments_dom : ÉCHEC → {parsed}", file=sys.stderr)
        sys.exit(1)
    print("_parse_comments_dom : OK")


def _run_suggestions_graphql_self_tests() -> None:
    sample = json.dumps(
        {
            "data": {
                "user": {
                    "username": "seed_one",
                    "edge_suggested_users": {
                        "edges": [
                            {"node": {"username": "suggest_a"}},
                            {"node": {"username": "suggest_b"}},
                        ]
                    },
                }
            }
        }
    )
    suggested: list[str] = []
    seen: set[str] = set()
    _ingest_suggestion_usernames_from_graphql_text(sample, "seed_one", seen, suggested)
    if suggested != ["suggest_a", "suggest_b"]:
        print(f"_suggestions_graphql : ÉCHEC → {suggested}", file=sys.stderr)
        sys.exit(1)
    print("_suggestions_graphql : OK")


def _run_profile_graphql_self_tests() -> None:
    sample = json.dumps(
        {
            "data": {
                "user": {
                    "username": "recrutestagiaire",
                    "full_name": "Test User",
                    "biography": "Bio test",
                    "follower_count": 48000,
                    "following_count": 120,
                    "media_count": 42,
                    "is_private": False,
                }
            }
        }
    )
    profile_data: dict[str, Any] = {}
    _merge_profile_graphql_text(sample, "recrutestagiaire", profile_data)
    _walk_profile_graphql(json.loads(sample), "recrutestagiaire", profile_data)
    if profile_data.get("followers") != 48000 or profile_data.get("posts_count") != 42:
        print(f"_profile_graphql : ÉCHEC → {profile_data}", file=sys.stderr)
        sys.exit(1)
    print("_profile_graphql : OK")


def _run_reels_grid_scroll_self_tests() -> None:
    if _reels_grid_rows_needed(20) != 4:
        print("_reels_grid_rows : ÉCHEC (20 reels → 4 lignes)", file=sys.stderr)
        sys.exit(1)
    if _reels_grid_rows_needed(5) != 1:
        print("_reels_grid_rows : ÉCHEC (5 reels → 1 ligne)", file=sys.stderr)
        sys.exit(1)
    print("_reels_grid_rows : OK (20 reels = 4×5)")


def _run_graphql_metrics_self_tests() -> None:
    """Vérifie l'association pk/code ↔ métriques (pas par index)."""
    sample = json.dumps(
        {
            "items": [
                {
                    "pk": "3893326453836395926",
                    "code": "DYcbkPMM1cR",
                    "view_count": 87300,
                    "like_count": 1063,
                    "comment_count": 12,
                    "caption": {"text": "Premier reel caption"},
                },
                {
                    "pk": "3893326453836395927",
                    "code": "DYaMNJqMckX",
                    "play_count": 171000,
                    "like_count": 748,
                    "caption_text": "Deuxième via caption_text",
                },
            ]
        }
    )
    by_pk: dict[str, dict[str, Any]] = {}
    by_code: dict[str, dict[str, Any]] = {}
    _ingest_metrics_from_graphql_text(sample, by_pk, by_code)
    _walk_graphql_metrics(json.loads(sample), by_pk, by_code)

    m1 = _metrics_for_dom_media_id("DYcbkPMM1cR", by_pk, by_code)
    m2 = _metrics_for_dom_media_id("DYaMNJqMckX", by_pk, by_code)
    c1 = str(by_code.get("DYcbkPMM1cR", {}).get("caption") or "")
    c2 = str(by_code.get("DYaMNJqMckX", {}).get("caption") or "")
    failed: list[str] = []
    if m1["view_count"] != 87300 or m1["like_count"] != 1063:
        failed.append(f"DYcbkPMM1cR → {m1}, attendu views=87300 likes=1063")
    if m2["view_count"] != 171000 or m2["like_count"] != 748:
        failed.append(f"DYaMNJqMckX → {m2}, attendu views=171000 likes=748")
    if c1 != "Premier reel caption":
        failed.append(f"DYcbkPMM1cR caption → {c1!r}")
    if c2 != "Deuxième via caption_text":
        failed.append(f"DYaMNJqMckX caption → {c2!r}")
    if failed:
        print("_graphql_metrics : ÉCHEC", file=sys.stderr)
        for line in failed:
            print(line, file=sys.stderr)
        sys.exit(1)
    print("_graphql_metrics : OK (2 reels par code court + captions)")


def _run_graphql_pinned_self_tests() -> None:
    """Vérifie clips_tab_pinned_user_ids → is_pinned par code court."""
    sample = json.dumps(
        {
            "items": [
                {
                    "code": "DSDvH57CDnT",
                    "view_count": 12000,
                    "clips_tab_pinned_user_ids": ["17841400000000000"],
                },
                {
                    "code": "UNPINNED01",
                    "view_count": 5000,
                    "clips_tab_pinned_user_ids": [],
                },
            ]
        }
    )
    by_pk: dict[str, dict[str, Any]] = {}
    by_code: dict[str, dict[str, Any]] = {}
    _ingest_metrics_from_graphql_text(sample, by_pk, by_code)
    _walk_graphql_metrics(json.loads(sample), by_pk, by_code)

    failed: list[str] = []
    if not by_code.get("DSDvH57CDnT", {}).get("is_pinned"):
        failed.append("DSDvH57CDnT devrait être is_pinned=True")
    if by_code.get("UNPINNED01", {}).get("is_pinned") is not False:
        failed.append(f"UNPINNED01 → is_pinned={by_code.get('UNPINNED01', {}).get('is_pinned')!r}")
    if failed:
        print("_graphql_pinned : ÉCHEC", file=sys.stderr)
        for line in failed:
            print(line, file=sys.stderr)
        sys.exit(1)
    print("_graphql_pinned : OK (DSDvH57CDnT épinglé)")


def _run_raikkonenaf_pinned_integration_test(context: BrowserContext) -> None:
    """Test live : DSDvH57CDnT doit être is_pinned sur @raikkonenaf."""
    reels = get_recent_reels("raikkonenaf", context, max_reels=20)
    by_id = {str(r.get("media_id") or ""): r for r in reels}
    target = "DSDvH57CDnT"
    if target not in by_id:
        print(
            f"_raikkonenaf_pinned : ÉCHEC — {target} absent des {len(reels)} reels récupérés",
            file=sys.stderr,
        )
        sys.exit(1)
    if not by_id[target].get("is_pinned"):
        print(
            f"_raikkonenaf_pinned : ÉCHEC — {target} is_pinned="
            f"{by_id[target].get('is_pinned')!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"_raikkonenaf_pinned : OK ({target} is_pinned=True)")


def _run_parse_count_self_tests() -> None:
    """Tests rapides pour _parse_count (ex. ``73,7 k`` → 73700)."""
    cases: list[tuple[str, int]] = [
        ("73,7 k", 73_700),
        ("1,2 M", 1_200_000),
        ("974 k", 974_000),
        ("27,7 k", 27_700),
        ("304,4 k", 304_400),
        ("1 234", 1_234),
        ("29,3 k", 29_300),
    ]
    failed: list[str] = []
    for raw, expected in cases:
        got = _parse_count(raw)
        if got != expected:
            failed.append(f"  {raw!r} → {got}, attendu {expected}")
    if failed:
        print("_parse_count : ÉCHEC", file=sys.stderr)
        for line in failed:
            print(line, file=sys.stderr)
        sys.exit(1)
    print(f"_parse_count : OK ({len(cases)} cas)")


if __name__ == "__main__":
    _run_parse_count_self_tests()
    _run_parse_comments_dom_self_tests()
    _run_suggestions_graphql_self_tests()
    _run_profile_graphql_self_tests()
    _run_reels_grid_scroll_self_tests()
    _run_graphql_metrics_self_tests()
    _run_graphql_pinned_self_tests()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        context = get_browser_context(p)
        try:
            if not _session_ok(context):
                print("Session Instagram invalide ou non connectée.", file=sys.stderr)
                sys.exit(1)
            _run_raikkonenaf_pinned_integration_test(context)
            debug_profile("recrutestagiaire", context)
            data = get_profile_data("recrutestagiaire", context)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        finally:
            br = context.browser
            if br:
                br.close()
