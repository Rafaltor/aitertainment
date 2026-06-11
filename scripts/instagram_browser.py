#!/usr/bin/env python3
"""instagram_browser.py — navigation Instagram via Playwright (Discovery + Watcher)."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.sync_api import BrowserContext, Page, Playwright, Response

COOKIES_PATH = Path("data/instagram_cookies.json")
_DEFAULT_SPA_WAIT_MS = 2500
# IG_HEADLESS=0 → navigateur visible (debug / réduit l'empreinte automation).
HEADLESS = os.environ.get("IG_HEADLESS", "1").strip() != "0"
BASE_URL = "https://www.instagram.com"
_VIEWPORT = {"width": 1920, "height": 1080}
_REELS_GRID_COLUMNS = 5
_REELS_GRID_ROW_HEIGHT_PX = 430
_REELS_SCROLL_WAIT_MS = 2_000
_REQUEST_TIMEOUT_MS = 15_000
_PROFILE_GOTO_TIMEOUT_MS = 30_000
VIRAL_COMMENTS_PATH = Path("data/viral_comments.json")
COMMENTS_COLLECT_REELS_MAX = 3
COMMENTS_PANEL_SCROLL_ROUNDS = 6
MIN_COMMENT_COUNT_TO_SCRAPE = 20
_FEED_REEL_STABLE_WAIT_MS = 500
_FEED_AFTER_PANEL_CLOSE_MS = 1500
_IG_WEB_APP_ID = "936619743392459"
_GRAPHQL_METRIC_KEYS = (
    "view_count",
    "play_count",
    "video_view_count",
    "like_count",
    "comment_count",
    "share_count",
)

log = logging.getLogger(__name__)

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
    "Commentaires",
    "Comment",
    "Comments",
    "Voir les commentaires",
    "View comments",
)

_REEL_COMMENTS_PANEL_SELECTORS = (
    'div[role="dialog"] ul',
    "div._aano",
    '[role="dialog"]',
    "section ul",
    "article ul",
    "main ul",
    'div[role="presentation"] ul',
)

_COMMENT_META_RE = re.compile(
    r"\d+\s*(?:sem|j|h|min|mois|s\b)|Répondre|J.aime|like",
    re.IGNORECASE,
)


def _looks_like_profile_not_reel(page: Page) -> bool:
    """True si on est sur le profil (highlights) et pas sur l'overlay reel."""
    try:
        url = (page.url or "").rstrip("/")
        if re.search(r"instagram\.com/[^/?#]+/?$", url):
            return True
        labels = _list_reel_page_aria_labels(page)
        if any("à la une" in (lab or "").lower() for lab in labels):
            return True
    except Exception:
        pass
    return False


def _reel_page_has_shell(page: Page, media_id: str = "") -> bool:
    """True si l'overlay reel est ouvert (URL reel ou bouton Commentaire visible)."""
    mid = str(media_id or "").strip()
    try:
        if len(page.content()) < 5000:
            return False
        url = page.url or ""
        if mid and (f"/reel/{mid}" in url or f"/p/{mid}" in url):
            return True
        if "/reel/" in url or re.search(r"/p/[A-Za-z0-9_-]+", url):
            return True
        for label in _REEL_COMMENT_ARIA_LABELS:
            if page.locator(f'svg[aria-label="{label}"]').count() > 0:
                return True
        return False
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
    direct_only: bool = False,
) -> bool:
    """Charge un reel. Préfère la grille ``/{user}/reels/`` (``/reel/{id}/`` seul est souvent vide)."""
    mid = str(media_id or "").strip()
    if not mid:
        return False

    page.set_viewport_size(_VIEWPORT)
    u = str(username or "").lstrip("@").strip()

    if direct_only:
        try:
            page.goto(
                f"{BASE_URL}/reel/{mid}/",
                timeout=timeout_ms,
                wait_until="domcontentloaded",
            )
            page.wait_for_load_state("load")
            page.wait_for_timeout(3000)
            if _reel_page_has_shell(page, mid):
                return True
            blocked = _instagram_page_blocked(page)
            if blocked:
                log.warning("reel %s : URL directe bloquée (%s).", mid, blocked)
        except Exception as e:
            log.warning("reel %s : URL directe échouée (%s).", mid, e)
        return False

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
            if _reel_page_has_shell(page, mid):
                return True
            log.warning(
                "reel %s : clic grille @%s sans overlay reel — repli /reel/%s/.",
                mid,
                u,
                mid,
            )
        except Exception as e:
            log.warning("reel %s : ouverture via grille @%s échouée (%s).", mid, u, e)

    try:
        page.goto(
            f"{BASE_URL}/reel/{mid}/",
            timeout=timeout_ms,
            wait_until="domcontentloaded",
        )
        page.wait_for_load_state("load")
        page.wait_for_timeout(3000)
        if _reel_page_has_shell(page, mid):
            return True
        blocked = _instagram_page_blocked(page)
        if blocked:
            log.warning("reel %s : /reel/ bloqué (%s).", mid, blocked)
    except Exception as e:
        log.warning("reel %s : /reel/ échoué (%s).", mid, e)

    log.warning("reel %s : overlay reel introuvable.", mid)
    return False


def _instagram_page_blocked(page: Page) -> str | None:
    """``login`` | ``private`` | ``unavailable`` si la page n'est pas exploitable."""
    url = (page.url or "").lower()
    if "/accounts/login" in url:
        return "login"
    try:
        if page.locator('input[name="username"]').count() > 0 and page.locator(
            'input[name="password"]'
        ).count() > 0:
            return "login"
    except Exception:
        pass
    try:
        snippet = (page.inner_text("body", timeout=4_000) or "")[:3000].lower()
    except Exception:
        snippet = ""
    try:
        # Ne déclarer "login" que si le DOM contient réellement le formulaire,
        # pas sur la simple présence de "log in" dans du texte de page.
        has_login_form = (
            page.locator('input[name="username"]').count() > 0
            and page.locator('input[name="password"]').count() > 0
        )
    except Exception:
        has_login_form = False
    if has_login_form:
        return "login"
    if "this account is private" in snippet or "compte est privé" in snippet:
        return "private"
    if "page isn't available" in snippet or "n'est pas disponible" in snippet:
        return "unavailable"
    if "sorry" in snippet and "available" in snippet:
        return "unavailable"
    return None


def session_ok(context: BrowserContext) -> bool:
    """True si les cookies Instagram permettent de naviguer (pas de mur login)."""
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


def open_reels_grid(
    page: Page,
    username: str,
    *,
    timeout_ms: int = _REQUEST_TIMEOUT_MS,
    spa_wait_ms: int | None = None,
) -> bool:
    """Ouvre le profil créateur ``/{username}/`` (session + csrftoken).

    La grille Reels est ensuite lue via ``/api/v1/clips/user/`` ; la timeline
    profil (carrousels) et l'onglet ``/reels/`` ne suffisent plus seuls.
    """
    u = str(username or "").lstrip("@").strip()
    if not u:
        return False
    wait_ms = _DEFAULT_SPA_WAIT_MS if spa_wait_ms is None else max(0, int(spa_wait_ms))
    page.set_viewport_size(_VIEWPORT)
    try:
        page.goto(
            f"{BASE_URL}/{u}/",
            timeout=timeout_ms,
            wait_until="domcontentloaded",
        )
        page.wait_for_load_state("load")
        page.wait_for_timeout(wait_ms)
        blocked = _instagram_page_blocked(page)
        if blocked:
            log.warning("grille @%s : page bloquée (%s).", u, blocked)
            return False
        url = (page.url or "").lower()
        if f"/{u.lower()}" not in url:
            log.warning(
                "grille @%s : URL inattendue (%s) — données grille ignorées.",
                u,
                page.url,
            )
            return False
        return True
    except Exception as e:
        log.warning("grille @%s inaccessible (%s).", u, e)
        return False


def return_to_reels_grid(page: Page, username: str) -> bool:
    """Revenir à la grille reels après avoir ouvert un reel."""
    u = str(username or "").lstrip("@").strip()
    if not u:
        return False
    try:
        url = page.url or ""
        if f"/{u}/" in url and "/reel/" not in url.split(f"/{u}/")[-1][:20]:
            return True
        page.go_back(wait_until="domcontentloaded", timeout=15_000)
        page.wait_for_timeout(2000)
        url = page.url or ""
        if f"/{u}/" in url:
            return True
    except Exception:
        pass
    return open_reels_grid(page, u)


def build_comment_dedup_key(media_id: str, text: str) -> str:
    return f"{media_id}||{text.strip().lower()}"


class ViralCommentsIOError(RuntimeError):
    """Lecture / écriture pool viral refusée pour éviter perte de données."""


def _parse_viral_comments_payload(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    if isinstance(data, dict):
        raw_entries = data.get("entries") or data.get("comments") or []
        if isinstance(raw_entries, list):
            return [e for e in raw_entries if isinstance(e, dict)]
    return []


def _dedup_keys_for_entries(entries: list[dict[str, Any]]) -> set[str]:
    return {
        build_comment_dedup_key(str(e["media_id"]), str(e["text"]))
        for e in entries
        if isinstance(e, dict) and e.get("media_id") and e.get("text")
    }


def _load_viral_comments_unlocked(
    path: Path,
    *,
    retries: int = 5,
    retry_delay_s: float = 0.15,
) -> tuple[list[dict[str, Any]], set[str]]:
    import time

    if not path.exists():
        return [], set()
    last_err: json.JSONDecodeError | None = None
    for attempt in range(max(1, retries)):
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                return [], set()
            data = json.loads(raw)
            entries = _parse_viral_comments_payload(data)
            return entries, _dedup_keys_for_entries(entries)
        except json.JSONDecodeError as e:
            last_err = e
            if attempt + 1 < retries:
                time.sleep(retry_delay_s)
                continue
            break
    raise ViralCommentsIOError(
        f"JSON invalide dans {path} après {retries} tentative(s) — "
        "arrêt pour ne pas écraser le pool viral."
    ) from last_err


def load_viral_comments_file(
    path: Path | str,
    *,
    retries: int = 5,
    retry_delay_s: float = 0.15,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Charge un fichier commentaires JSON → ``(entries, clés dédup)``.

    Formats acceptés (comme ``load_viral_comments``) :

    * ``[{...}, ...]`` (liste racine, écriture scrape)
    * ``{"entries": [...]}`` ou ``{"comments": [...]}``

    En cas de JSON illisible (souvent lecture pendant un ``save`` concurrent),
    réessaie puis lève ``ViralCommentsIOError`` — **ne repart jamais silencieusement
    de zéro**, ce qui provoquait des écrasements du pool à ``[]``.
    """
    p = Path(path)
    return _load_viral_comments_unlocked(
        p, retries=retries, retry_delay_s=retry_delay_s
    )


def save_viral_comments_file(
    entries: list[dict[str, Any]],
    path: Path | str,
    *,
    merge: bool = True,
    allow_shrink: bool = False,
    allow_empty: bool = False,
) -> None:
    """Persiste le pool viral.

    * ``merge=True`` (défaut scrape) : recharge le disque sous verrou et fusionne
      par clé ``media_id||text`` avant écriture.
    * ``merge=False`` (ex. clean) : remplace le fichier ; refuse une chute brutale
      sauf si ``allow_shrink=True``.
    """
    from modules.atomic_json import json_lock

    p = Path(path)
    with json_lock(p):
        on_disk, _ = _load_viral_comments_unlocked(p, retries=5)
        if merge:
            merged: dict[str, dict[str, Any]] = {}
            for entry in on_disk + list(entries):
                if not isinstance(entry, dict):
                    continue
                media_id = str(entry.get("media_id") or "")
                text = str(entry.get("text") or "")
                if not media_id or not text:
                    continue
                merged[build_comment_dedup_key(media_id, text)] = entry
            final = list(merged.values())
        else:
            final = list(entries)

        if (
            not allow_shrink
            and len(on_disk) >= 50
            and len(final) < len(on_disk) * 0.5
        ):
            raise ViralCommentsIOError(
                f"Refus d'écrire {len(final)} entrée(s) (fichier en avait {len(on_disk)}). "
                "Utilisez --force sur clean_comments ou corrigez la cause."
            )

        if len(final) == 0 and not allow_empty:
            log.warning(
                "Refus d'écrire %s vide (%d entrée(s) sur disque) — fichier inchangé. "
                "Fermez l'onglet éditeur sur ce fichier (autosave = []). "
                "Restaurez %s.autobak si besoin.",
                p.name,
                len(on_disk),
                p.name,
            )
            return

        if len(final) == 0 and allow_empty and len(on_disk) >= 50:
            log.warning(
                "Écriture %s vide autorisée explicitement (avant : %d entrée(s)).",
                p.name,
                len(on_disk),
            )

        if len(on_disk) >= 50:
            autobak = p.with_suffix(p.suffix + ".autobak")
            try:
                import shutil

                shutil.copy2(p, autobak)
            except OSError as e:
                log.warning("Backup auto %s impossible : %s", autobak.name, e)

        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps(final, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(p)


def scrape_profile_comments(
    username: str,
    context: BrowserContext,
    reels: list[dict[str, Any]],
    niches: list[str] | str,
    *,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """Scrape les commentaires visibles sur les reels profil (sans persistance).

    Retourne une liste de dicts ``media_id``, ``text``, ``comment_likes``, ``views``,
  ``username``, ``niches``. Utilisé par l'embedder (mémoire seulement).
    """
    log_cb = logger or log
    u = (username or "").lstrip("@").strip()
    reels_sorted = sorted(
        [r for r in (reels or []) if str(r.get("media_id") or "").strip()],
        key=lambda r: int(r.get("comment_count") or 0),
        reverse=True,
    )
    reels_to_visit = reels_sorted[:COMMENTS_COLLECT_REELS_MAX]
    if not u or not reels_to_visit:
        return []

    niches_list = list(niches) if isinstance(niches, list) else [str(niches)]
    candidates: list[dict[str, Any]] = []
    page = context.new_page()

    try:
        grid_ok = open_reels_grid(page, u)
        if not grid_ok:
            log_cb.warning(
                "Collecte commentaires @%s : grille /reels/ KO — repli /reel/{{id}}/ direct.",
                u,
            )

        for reel in reels_to_visit:
            media_id = str(reel.get("media_id") or "").strip()
            view_count = int(reel.get("view_count") or reel.get("views") or 0)
            try:
                if grid_ok:
                    loaded = navigate_to_reel_page(
                        page, media_id, u, reels_grid_loaded=True
                    )
                else:
                    loaded = navigate_to_reel_page(
                        page, media_id, u, direct_only=True
                    )
                if loaded and _looks_like_profile_not_reel(page):
                    log_cb.info(
                        "Collecte @%s reel %s : profil/highlights détecté — /reel/ direct.",
                        u,
                        media_id,
                    )
                    loaded = navigate_to_reel_page(
                        page, media_id, u, direct_only=True
                    )
                if not loaded:
                    log_cb.warning(
                        "Collecte commentaires @%s reel %s : page non chargée (url=%s).",
                        u,
                        media_id,
                        (page.url or "")[:80],
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
                    if grid_ok:
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
                            "username": u,
                            "niches": niches_list,
                            "views": view_count,
                            "text": text,
                            "comment_likes": int(comment.get("like_count") or 0),
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
                if grid_ok:
                    return_to_reels_grid(page, u)
            polite_sleep(seconds=1)
    finally:
        page.close()

    if candidates:
        reels_with_data = len({c.get("media_id") for c in candidates if c.get("media_id")})
        log_cb.info(
            "Scrape commentaires @%s : %d commentaire(s) sur %d reel(s) (non persistés).",
            u,
            len(candidates),
            reels_with_data,
        )
    return candidates


def collect_viral_comments(
    username: str,
    context: BrowserContext,
    reels: list[dict[str, Any]],
    niches: list[str] | str,
    *,
    min_likes: int = 1000,
    max_reels: int = 30,
    scroll_rounds: int = 25,
    top_per_reel: int | None = None,
    output_path: Path | str | None = None,
    between_reels_min_s: float = 20,
    between_reels_max_s: float = 45,
    french_only: bool = True,
    creator_index: dict[str, dict[str, Any]] | None = None,
    skip_transcript: bool = False,
    skip_visual: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, int]:
    """Collecte les commentaires à fort engagement sur plusieurs reels d'un compte.

    Parcourt la grille ``/{user}/reels/`` (jusqu'à ``max_reels``), ouvre le
    panneau commentaires avec un scroll profond, ne retient que les commentaires
    dont ``like_count >= min_likes``.
    """
    from modules.comment_quality import is_french_comment, is_french_reel_caption
    from modules.creator_registry import build_creator_index, resolve_creator_fields

    log_cb = logger or log
    u = (username or "").lstrip("@").strip()
    reels_sorted = sorted(
        [r for r in (reels or []) if str(r.get("media_id") or "").strip()],
        key=lambda r: int(r.get("comment_count") or 0),
        reverse=True,
    )
    reels_to_visit = reels_sorted[: max(1, max_reels)]
    stats = {
        "collected": 0,
        "parsed": 0,
        "reels_visited": 0,
        "kept": 0,
        "skipped_english": 0,
    }
    if not u or not reels_to_visit or min_likes < 0:
        return stats

    idx = creator_index if creator_index is not None else build_creator_index()
    creator_meta = resolve_creator_fields(u, index=idx)
    niches_list = list(creator_meta["niches"])

    out_path = Path(output_path) if output_path is not None else VIRAL_COMMENTS_PATH
    entries, dedup_keys = load_viral_comments_file(out_path)
    page = context.new_page()

    try:
        grid_ok = open_reels_grid(page, u)
        if not grid_ok:
            log_cb.warning(
                "Viral comments @%s : grille /reels/ KO — repli /reel/{{id}}/ direct.",
                u,
            )

        for reel in reels_to_visit:
            media_id = str(reel.get("media_id") or "").strip()
            view_count = int(reel.get("view_count") or reel.get("views") or 0)
            caption = str(reel.get("caption") or "").strip()
            stats["reels_visited"] += 1
            reel_hits: list[dict[str, Any]] = []
            try:
                if grid_ok:
                    loaded = navigate_to_reel_page(
                        page, media_id, u, reels_grid_loaded=True
                    )
                else:
                    loaded = navigate_to_reel_page(
                        page, media_id, u, direct_only=True
                    )
                if loaded and _looks_like_profile_not_reel(page):
                    loaded = navigate_to_reel_page(
                        page, media_id, u, direct_only=True
                    )
                if not loaded:
                    log_cb.warning(
                        "Viral comments @%s reel %s : page non chargée.",
                        u,
                        media_id,
                    )
                    continue

                clicked = click_reel_comment_button(page)
                if not clicked:
                    log_cb.warning(
                        "Viral comments @%s reel %s : bouton commentaire absent.",
                        u,
                        media_id,
                    )
                    continue

                page.wait_for_timeout(2000)
                panel_text = extract_reel_comments_panel_text(
                    page, scroll_rounds=scroll_rounds
                )
                parsed = parse_comments_from_dom_text(str(panel_text or ""))
                stats["parsed"] += len(parsed)
                for comment in parsed:
                    text = str(comment.get("text") or "").strip()
                    likes = int(comment.get("like_count") or 0)
                    if not text or likes < min_likes:
                        continue
                    if french_only and not is_french_comment(text):
                        stats["skipped_english"] += 1
                        continue
                    reel_hits.append(
                        {
                            "media_id": media_id,
                            "views": view_count,
                            "caption": caption,
                            "text": text,
                            "like_count": likes,
                        }
                    )
            except Exception as e:
                log_cb.warning(
                    "Viral comments @%s reel %s : erreur (%s).",
                    u,
                    media_id,
                    e,
                )
            finally:
                if grid_ok:
                    return_to_reels_grid(page, u)

            if top_per_reel is not None and top_per_reel > 0:
                reel_hits.sort(key=lambda c: int(c.get("like_count") or 0), reverse=True)
                reel_hits = reel_hits[:top_per_reel]

            transcript = ""
            visual_description = ""
            if reel_hits:
                from scripts.scrape_viral_comments import (
                    _enrich_reel_with_transcript_and_visual,
                )

                transcript, visual_description = _enrich_reel_with_transcript_and_visual(
                    media_id,
                    context,
                    skip_transcript=skip_transcript,
                    skip_visual=skip_visual,
                )
                log_cb.info(
                    "Viral @%s reel %s : enrichi (transcript=%d chars, visuel=%d chars)",
                    u,
                    media_id,
                    len(transcript),
                    len(visual_description),
                )
            else:
                log_cb.debug(
                    "Viral @%s reel %s : 0 commentaire viral, skip enrichissement",
                    u,
                    media_id,
                )

            for comment in reel_hits:
                media_id = str(comment.get("media_id") or "").strip()
                comment_text = str(comment.get("text") or "").strip()
                dedup_key = build_comment_dedup_key(media_id, comment_text)
                if dedup_key in dedup_keys:
                    continue
                dedup_keys.add(dedup_key)
                stats["kept"] += 1
                entries.append(
                    {
                        "media_id": media_id,
                        "username": u,
                        "niches": niches_list,
                        "text": comment_text,
                        "comment_likes": int(comment.get("like_count") or 0),
                        "views": int(comment.get("views") or 0),
                        "comment_to_like_ratio": 0.0,
                        "caption": str(comment.get("caption") or ""),
                        "hashtags": [],
                        "audio_id": "",
                        "transcript": transcript,
                        "visual_description": visual_description,
                        "t_type": None,
                        "t_type_profile": creator_meta.get("t_type_profile"),
                        "llm_validated": False,
                        "source": "viral_scrape",
                        "min_likes_threshold": min_likes,
                        "needs_embed": bool(creator_meta.get("needs_embed")),
                        "collected_at": datetime.now(timezone.utc)
                        .replace(microsecond=0)
                        .isoformat(),
                    }
                )
                stats["collected"] += 1

            polite_sleep(min_s=between_reels_min_s, max_s=between_reels_max_s)
    finally:
        page.close()

    if stats["collected"]:
        save_viral_comments_file(entries, out_path)
    log_cb.info(
        "Viral comments @%s : %d nouveau(x) (≥%d likes) — %d parsé(s), "
        "%d reel(s) visité(s), fichier %s.",
        u,
        stats["collected"],
        min_likes,
        stats["parsed"],
        stats["reels_visited"],
        out_path,
    )
    return stats


def _is_valid_reel_code(code: str) -> bool:
    mid = str(code or "").strip()
    if len(mid) < 8 or len(mid) > 20:
        return False
    if mid.lower() in _IG_RESERVED_USERNAMES or mid.lower() in {"audio", "reels", "explore"}:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", mid))


def _extract_reel_code_from_page(page: Page) -> str:
    """Code reel courant depuis l'URL ou le lien DOM le plus visible (fil /reels/)."""
    try:
        url = page.url or ""
        match = re.search(r"/reels?/([A-Za-z0-9_-]{8,20})/?", url)
        if match:
            code = match.group(1)
            if _is_valid_reel_code(code):
                return code
        code = page.evaluate(
            """() => {
                const pickCode = (href) => {
                    const m = (href || '').match(/\\/reel\\/([A-Za-z0-9_-]{8,20})/);
                    return m ? m[1] : '';
                };
                let best = '';
                let bestArea = 0;
                for (const video of document.querySelectorAll('video')) {
                    const rect = video.getBoundingClientRect();
                    const area = rect.width * rect.height;
                    if (area <= bestArea) continue;
                    let el = video.parentElement;
                    for (let d = 0; d < 14 && el; d++) {
                        for (const a of el.querySelectorAll('a[href*="/reel/"]')) {
                            const code = pickCode(a.getAttribute('href') || a.href || '');
                            if (code) {
                                best = code;
                                bestArea = area;
                                break;
                            }
                        }
                        if (best) break;
                        el = el.parentElement;
                    }
                }
                if (best) return best;
                for (const a of document.querySelectorAll('a[href*="/reel/"]')) {
                    const code = pickCode(a.getAttribute('href') || a.href || '');
                    if (code) return code;
                }
                return '';
            }"""
        )
        if code and _is_valid_reel_code(str(code)):
            return str(code)
    except Exception:
        pass
    return ""


def _wait_for_feed_reel_change(
    page: Page,
    previous_code: str,
    *,
    timeout_ms: int = 4000,
) -> str:
    """Attend qu'un nouveau code reel apparaisse après scroll."""
    deadline = time.time() + timeout_ms / 1000.0
    prev = str(previous_code or "").strip()
    while time.time() < deadline:
        code = _extract_reel_code_from_page(page)
        if code and code != prev and _is_valid_reel_code(code):
            return code
        page.wait_for_timeout(200)
    return _extract_reel_code_from_page(page) or ""


def _merge_owner_username_into_bucket(bucket: dict[str, Any], username: str) -> None:
    user = str(username or "").lstrip("@").strip().lower()
    if not user or user in _IG_RESERVED_USERNAMES:
        return
    existing = str(bucket.get("username") or "").lstrip("@").strip().lower()
    if not existing:
        bucket["username"] = user


def _extract_owner_username_from_graphql_window(window: str) -> str:
    for pattern in (
        r'"owner"\s*:\s*\{[^}]{0,400}?"username"\s*:\s*"([^"]+)"',
        r'"user"\s*:\s*\{[^}]{0,400}?"username"\s*:\s*"([^"]+)"',
        r'"coauthor_producers"\s*:\s*\[\s*\{\s*"username"\s*:\s*"([^"]+)"',
    ):
        match = re.search(pattern, window)
        if match:
            user = str(match.group(1)).lstrip("@").strip().lower()
            if user and user not in _IG_RESERVED_USERNAMES:
                return user
    return ""


def _owner_username_from_bucket(bucket: dict[str, Any]) -> str:
    return str(bucket.get("username") or "").lstrip("@").strip().lower()


def _extract_reel_owner_username(page: Page, *, media_id: str = "") -> str:
    """Username du créateur depuis la page reel (lien proche de la vidéo, pas la nav)."""
    mid = str(media_id or "").strip()
    try:
        owner = page.evaluate(
            """({ reserved, mediaId }) => {
                const reservedSet = new Set(reserved);
                const pick = (href) => {
                    const m = (href || '').match(/^\\/([\\w.]{2,30})\\/$/);
                    if (!m || reservedSet.has(m[1])) return '';
                    return m[1];
                };
                if (mediaId) {
                    for (const a of document.querySelectorAll('a[href*="/reel/"], a[href*="/p/"]')) {
                        const href = a.getAttribute('href') || '';
                        if (!href.includes(mediaId)) continue;
                        let el = a.parentElement;
                        for (let depth = 0; depth < 10 && el; depth++) {
                            for (const link of el.querySelectorAll('a[href^="/"]')) {
                                const user = pick(link.getAttribute('href') || '');
                                if (user) return user;
                            }
                            el = el.parentElement;
                        }
                    }
                }
                for (const video of document.querySelectorAll('video')) {
                    let el = video.parentElement;
                    for (let depth = 0; depth < 12 && el; depth++) {
                        for (const link of el.querySelectorAll('a[href^="/"]')) {
                            const user = pick(link.getAttribute('href') || '');
                            if (user) return user;
                        }
                        el = el.parentElement;
                    }
                }
                const main = document.querySelector('main') || document.body;
                for (const a of main.querySelectorAll('header a[href], nav a[href]')) {
                    a.setAttribute('data-skip-owner', '1');
                }
                for (const a of main.querySelectorAll('a[href^="/"]')) {
                    if (a.getAttribute('data-skip-owner')) continue;
                    const user = pick(a.getAttribute('href') || '');
                    if (user) return user;
                }
                return '';
            }""",
            {"reserved": sorted(_IG_RESERVED_USERNAMES), "mediaId": mid},
        )
        if owner:
            return str(owner).lstrip("@").strip()
    except Exception:
        pass
    return ""


def refresh_reels_feed(page: Page, *, logger: logging.Logger | None = None) -> None:
    """Reset fil Reels entre sessions (Explore → /reels/ → reload)."""
    log_cb = logger or log
    page.set_viewport_size(_VIEWPORT)
    try:
        page.goto(
            f"{BASE_URL}/explore/",
            timeout=_REQUEST_TIMEOUT_MS,
            wait_until="domcontentloaded",
        )
        page.wait_for_load_state("load")
        page.wait_for_timeout(int(random.uniform(2000, 4000)))
    except Exception as e:
        log_cb.warning("Refresh fil : explore (%s).", e)
    try:
        page.goto(
            f"{BASE_URL}/reels/",
            timeout=_REQUEST_TIMEOUT_MS,
            wait_until="domcontentloaded",
        )
        page.wait_for_load_state("load")
        page.wait_for_timeout(1500)
        page.reload(wait_until="domcontentloaded")
        page.wait_for_load_state("load")
        page.wait_for_timeout(int(random.uniform(2000, 3500)))
        _focus_reels_feed_player(page)
        log_cb.info("Fil Reels : feed rechargé (explore → reload).")
    except Exception as e:
        log_cb.warning("Refresh fil : reels (%s).", e)


def _focus_reels_feed_player(page: Page) -> None:
    """Focus le lecteur Reels (desktop) pour que ArrowDown change de reel."""
    try:
        page.mouse.click(_VIEWPORT["width"] // 2, _VIEWPORT["height"] // 2)
        page.wait_for_timeout(400)
    except Exception:
        pass


def _advance_reels_feed(page: Page) -> None:
    """Passe au reel suivant dans le fil (ArrowDown >> molette sur desktop IG)."""
    _focus_reels_feed_player(page)
    try:
        page.keyboard.press("ArrowDown")
    except Exception:
        pass
    page.wait_for_timeout(200)


def should_boost_french_reel_on_feed(caption: str, *, french_only: bool = True) -> bool:
    """Caption non vide et clairement FR → engagement long sur le fil (signal algo)."""
    if not french_only:
        return False
    from modules.comment_quality import is_french_comment

    cap = str(caption or "").strip()
    if not cap:
        return False
    return is_french_comment(cap)


def feed_watch_duration_s(
    watch_min_s: float,
    *,
    watch_jitter_s: float = 5.0,
    rng: random.Random | None = None,
) -> float:
    """Durée d'engagement fil (respecte ``watch_min_s``, pas de plancher artificiel)."""
    base = max(0.0, float(watch_min_s))
    jitter = min(float(watch_jitter_s), max(0.5, base * 0.25))
    r = rng or random
    return max(0.5, base + r.uniform(-jitter, jitter))


def engage_feed_reel_for_algo(
    page: Page,
    watch_min_s: float,
    *,
    watch_jitter_s: float = 5.0,
    logger: logging.Logger | None = None,
) -> float:
    """Reste sur le reel affiché (lecteur focus) pour influencer le fil."""
    log_cb = logger or log
    duration = feed_watch_duration_s(watch_min_s, watch_jitter_s=watch_jitter_s)
    _focus_reels_feed_player(page)
    log_cb.info("Fil Reels : engagement algo %.1fs sur le reel affiché.", duration)
    deadline = time.time() + duration
    while time.time() < deadline:
        remaining = deadline - time.time()
        chunk_ms = min(2500, max(200, int(remaining * 1000)))
        page.wait_for_timeout(chunk_ms)
        if random.random() < 0.2:
            _focus_reels_feed_player(page)
    return duration


def _filter_reels_for_comment_scrape(
    candidates: list[dict[str, Any]],
    *,
    max_reels: int,
    min_comment_count: int = MIN_COMMENT_COUNT_TO_SCRAPE,
) -> list[dict[str, Any]]:
    """Phase 2 : filtre local sans réseau."""
    filtered = [
        c
        for c in candidates
        if int(c.get("comment_count") or 0) > 0
        and int(c.get("comment_count") or 0) >= min_comment_count
    ]
    filtered.sort(key=lambda c: int(c.get("comment_count") or 0), reverse=True)
    return filtered[:max_reels]


def collect_viral_comments_from_feed(
    context: BrowserContext,
    *,
    min_likes: int = 1000,
    max_reels: int = 30,
    scroll_steps: int = 80,
    scroll_wait_ms: int = 800,
    scroll_rounds: int = 8,
    panel_scroll_wait_ms: int = 400,
    top_per_reel: int | None = None,
    output_path: Path | str | None = None,
    between_reels_min_s: float = 90,
    between_reels_max_s: float = 180,
    french_only: bool = True,
    fr_reel_watch_s: float = 60.0,
    feed_en_skip_ms: int = 600,
    fresh_feed: bool = False,
    creator_index: dict[str, dict[str, Any]] | None = None,
    skip_transcript: bool = False,
    skip_visual: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, int]:
    """Découverte passive sur /reels/, puis scrape commentaires via grille ``/@user/reels/``."""
    from modules.comment_quality import is_french_comment, is_french_reel_caption
    from modules.creator_registry import build_creator_index, resolve_creator_fields

    log_cb = logger or log
    stats: dict[str, int] = {
        "collected": 0,
        "parsed": 0,
        "reels_visited": 0,
        "kept": 0,
        "skipped_english": 0,
        "skipped_english_reel": 0,
        "skipped_no_creator": 0,
        "skipped_low_engagement": 0,
        "french_reels_watched": 0,
        "feed_watch_s": 0,
    }
    idx = creator_index if creator_index is not None else build_creator_index()
    out_path = Path(output_path) if output_path is not None else VIRAL_COMMENTS_PATH
    entries, dedup_keys = load_viral_comments_file(out_path)
    new_since_save = 0

    # —— Phase 1 : découverte feed passive (GraphQL, pas de panneau commentaires) ——
    feed_page = context.new_page()
    metrics_by_pk, metrics_by_code = _attach_graphql_metrics_listener(feed_page)
    candidates: dict[str, dict[str, Any]] = {}
    stagnant = 0
    pool_target = max(max_reels * 3, scroll_steps)
    max_steps = max(1, int(scroll_steps))

    try:
        if fresh_feed:
            refresh_reels_feed(feed_page, logger=log_cb)
        else:
            feed_page.goto(
                f"{BASE_URL}/reels/",
                timeout=_REQUEST_TIMEOUT_MS,
                wait_until="domcontentloaded",
            )
        feed_page.set_viewport_size(_VIEWPORT)
        feed_page.wait_for_load_state("load")
        feed_page.wait_for_timeout(3000)
        feed_page.mouse.click(_VIEWPORT["width"] // 2, _VIEWPORT["height"] // 2)
        feed_page.wait_for_timeout(400)

        log_cb.info(
            "Fil Reels phase 1 : %d scrolls (engagement FR=%ds, skip EN=%dms).",
            max_steps,
            int(fr_reel_watch_s),
            feed_en_skip_ms,
        )
        for step_idx in range(max_steps):
            code = _extract_reel_code_from_page(feed_page)
            if not code or not _is_valid_reel_code(code):
                stagnant += 1
                if stagnant >= 10:
                    log_cb.info("Fil Reels : scroll stagnant après %d step(s).", step_idx + 1)
                    break
                _advance_reels_feed(feed_page)
                feed_page.wait_for_timeout(feed_en_skip_ms)
                continue

            if code not in candidates and len(candidates) < pool_target:
                stagnant = 0
                bucket = _metrics_bucket_for_dom_media_id(
                    code, metrics_by_pk, metrics_by_code
                )
                metrics = _metrics_for_dom_media_id(
                    code, metrics_by_pk, metrics_by_code
                )
                caption = str(bucket.get("caption") or "")
                candidates[code] = {
                    "media_id": code,
                    "username": _owner_username_from_bucket(bucket),
                    "comment_count": int(metrics.get("comment_count") or 0),
                    "view_count": int(metrics.get("view_count") or 0),
                    "caption": caption,
                }
            else:
                bucket = _metrics_bucket_for_dom_media_id(
                    code, metrics_by_pk, metrics_by_code
                )
                caption = str(bucket.get("caption") or "")

            if should_boost_french_reel_on_feed(caption, french_only=french_only):
                stats["french_reels_watched"] += 1
                watched = engage_feed_reel_for_algo(
                    feed_page,
                    fr_reel_watch_s,
                    logger=log_cb,
                )
                stats["feed_watch_s"] += int(watched)
            else:
                feed_page.wait_for_timeout(feed_en_skip_ms)

            _advance_reels_feed(feed_page)
            feed_page.wait_for_timeout(scroll_wait_ms)
            _wait_for_feed_reel_change(feed_page, code)
    finally:
        feed_page.close()

    log_cb.info(
        "Fil Reels phase 1 terminée : %d candidat(s), %d reel(s) FR regardé(s) (~%ds).",
        len(candidates),
        stats["french_reels_watched"],
        stats["feed_watch_s"],
    )

    # —— Phase 2 : filtre local ——
    to_scrape: list[dict[str, Any]] = []
    for c in candidates.values():
        if not c.get("username") or len(str(c["username"])) < 4:
            continue
        if str(c["username"]).lower() in _IG_RESERVED_USERNAMES:
            continue
        if int(c.get("comment_count") or 0) < MIN_COMMENT_COUNT_TO_SCRAPE:
            continue
        if french_only and not is_french_reel_caption(str(c.get("caption") or "")):
            stats["skipped_english_reel"] += 1
            log_cb.debug(
                "Fil Reels %s @%s : caption non-FR — skip reel (%s).",
                c.get("media_id"),
                c.get("username"),
                str(c.get("caption") or "")[:80],
            )
            continue
        to_scrape.append(c)
    to_scrape.sort(key=lambda c: int(c.get("comment_count") or 0), reverse=True)
    to_scrape = to_scrape[:max_reels]
    log_cb.info(
        "Phase 1 : %d découverts → Phase 2 : %d retenus "
        "(comment_count >= %d, reels caption EN skippés=%d).",
        len(candidates),
        len(to_scrape),
        MIN_COMMENT_COUNT_TO_SCRAPE,
        stats["skipped_english_reel"],
    )

    if not to_scrape:
        log_cb.info("Fil Reels : aucun reel à scraper après filtrage.")
        return stats

    # —— Phase 3 : scrape via grille /@user/reels/ (pas /reel/{id}/ direct) ——
    grid_page = context.new_page()
    grid_page.set_viewport_size(_VIEWPORT)

    try:
        for visit_idx, reel in enumerate(to_scrape, start=1):
            media_id = str(reel.get("media_id") or "").strip()
            username = str(reel.get("username") or "").lstrip("@").strip()
            view_count = int(reel.get("view_count") or 0)
            caption = str(reel.get("caption") or "")
            stats["reels_visited"] += 1
            creator_meta = resolve_creator_fields(username, index=idx)
            grid_ok = False

            try:
                grid_ok = open_reels_grid(grid_page, username)
                if not grid_ok:
                    log_cb.warning(
                        "Fil Reels @%s : grille /reels/ inaccessible — skip reel %s.",
                        username,
                        media_id,
                    )
                    stats["skipped_no_creator"] += 1
                    continue

                loaded = navigate_to_reel_page(
                    grid_page,
                    media_id,
                    username,
                    reels_grid_loaded=True,
                )
                if not loaded:
                    log_cb.warning(
                        "Fil Reels %s introuvable dans grille @%s/reels/ — skip.",
                        media_id,
                        username,
                    )
                    stats["skipped_no_creator"] += 1
                    continue

                clicked = click_reel_comment_button(grid_page)
                if not clicked:
                    log_cb.warning(
                        "Fil Reels %s @%s : bouton commentaire absent.",
                        media_id,
                        username,
                    )
                    if grid_ok:
                        return_to_reels_grid(grid_page, username)
                    continue

                grid_page.wait_for_timeout(2000)
                panel_text = extract_reel_comments_panel_text(
                    grid_page,
                    scroll_rounds=scroll_rounds,
                    scroll_wait_ms=panel_scroll_wait_ms,
                )
                if len(str(panel_text or "").strip()) < 40:
                    click_reel_comment_button(grid_page)
                    grid_page.wait_for_timeout(1500)
                    panel_text = extract_reel_comments_panel_text(
                        grid_page,
                        scroll_rounds=scroll_rounds,
                        scroll_wait_ms=panel_scroll_wait_ms,
                    )
                parsed = parse_comments_from_dom_text(str(panel_text or ""))
                stats["parsed"] += len(parsed)

                reel_hits: list[dict[str, Any]] = []
                for comment in parsed:
                    text = str(comment.get("text") or "").strip()
                    likes = int(comment.get("like_count") or 0)
                    if not text or likes < min_likes:
                        continue
                    if french_only and not is_french_comment(text):
                        stats["skipped_english"] += 1
                        continue
                    reel_hits.append(
                        {
                            "media_id": media_id,
                            "views": view_count,
                            "caption": caption,
                            "text": text,
                            "like_count": likes,
                            "username": creator_meta["username"],
                            "niches": creator_meta["niches"],
                            "t_type_profile": creator_meta["t_type_profile"],
                            "needs_embed": creator_meta["needs_embed"],
                        }
                    )

                if top_per_reel is not None and top_per_reel > 0:
                    reel_hits.sort(key=lambda c: int(c.get("like_count") or 0), reverse=True)
                    reel_hits = reel_hits[:top_per_reel]

                transcript = ""
                visual_description = ""
                if reel_hits:
                    from scripts.scrape_viral_comments import (
                        _enrich_reel_with_transcript_and_visual,
                    )

                    transcript, visual_description = (
                        _enrich_reel_with_transcript_and_visual(
                            media_id,
                            context,
                            skip_transcript=skip_transcript,
                            skip_visual=skip_visual,
                        )
                    )
                    log_cb.info(
                        "Fil Reels %s : enrichi (transcript=%d chars, visuel=%d chars)",
                        media_id,
                        len(transcript),
                        len(visual_description),
                    )
                else:
                    log_cb.debug(
                        "Fil Reels %s : 0 commentaire viral, skip enrichissement",
                        media_id,
                    )

                new_on_reel = 0
                for comment in reel_hits:
                    comment_text = str(comment.get("text") or "").strip()
                    dedup_key = build_comment_dedup_key(media_id, comment_text)
                    if dedup_key in dedup_keys:
                        continue
                    dedup_keys.add(dedup_key)
                    stats["kept"] += 1
                    new_on_reel += 1
                    entries.append(
                        {
                            "media_id": media_id,
                            "username": creator_meta["username"],
                            "niches": list(creator_meta.get("niches") or ["humour"]),
                            "text": comment_text,
                            "comment_likes": int(comment.get("like_count") or 0),
                            "views": view_count,
                            "comment_to_like_ratio": 0.0,
                            "caption": caption,
                            "hashtags": [],
                            "audio_id": "",
                            "transcript": transcript,
                            "visual_description": visual_description,
                            "t_type": None,
                            "t_type_profile": creator_meta.get("t_type_profile"),
                            "llm_validated": False,
                            "source": "reels_feed",
                            "min_likes_threshold": min_likes,
                            "needs_embed": bool(creator_meta.get("needs_embed")),
                            "collected_at": datetime.now(timezone.utc)
                            .replace(microsecond=0)
                            .isoformat(),
                        }
                    )
                    stats["collected"] += 1

                if new_on_reel > 0:
                    save_viral_comments_file(entries, out_path)
                    new_since_save += new_on_reel

                log_cb.info(
                    "Fil Reels [%d/%d] %s @%s — %d gardé(s) / %d parsés (total %d).",
                    visit_idx,
                    len(to_scrape),
                    media_id,
                    username,
                    new_on_reel,
                    len(parsed),
                    stats["collected"],
                )

            except Exception as e:
                log_cb.warning("Fil Reels @%s reel %s : erreur (%s).", username, media_id, e)
            finally:
                if grid_ok:
                    return_to_reels_grid(grid_page, username)

            if visit_idx < len(to_scrape):
                polite_sleep(min_s=between_reels_min_s, max_s=between_reels_max_s)
    finally:
        grid_page.close()

    if stats["collected"] and new_since_save == 0:
        save_viral_comments_file(entries, out_path)
    log_cb.info(
        "Fil Reels : %d commentaire(s) viral(aux) (≥%d likes) → %s.",
        stats["collected"],
        min_likes,
        out_path,
    )
    return stats


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


def _reel_comment_post_urls(media_id: str) -> list[str]:
    """URLs post classique puis reel (la vue ``/reels/`` immersive n'a pas de vrai champ)."""
    mid = str(media_id or "").strip()
    return [f"{BASE_URL}/p/{mid}/", f"{BASE_URL}/reel/{mid}/"]


def _fill_reel_comment_field(page: Page, text: str) -> bool:
    """Remplit le champ commentaire (``/p/`` textarea ou contenteditable reel)."""
    payload = str(text or "").strip()
    if not payload:
        return False

    textarea_selectors = (
        'textarea[placeholder*="commentaire" i]',
        'textarea[placeholder*="comment" i]',
        "form textarea",
    )
    for selector in textarea_selectors:
        loc = page.locator(selector)
        for idx in range(min(loc.count(), 3)):
            target = loc.nth(idx)
            try:
                if not target.is_visible(timeout=1_500):
                    continue
                target.click(timeout=5_000)
                target.press_sequentially(payload, delay=15, timeout=15_000)
                return True
            except Exception:
                continue

    candidates = [
        page.get_by_placeholder(re.compile(r"ajouter.*commentaire", re.I)),
        page.get_by_placeholder(re.compile(r"add.*comment", re.I)),
        page.locator('div[contenteditable="true"][role="textbox"]'),
        page.locator('div[contenteditable="true"]'),
        page.get_by_role("textbox"),
    ]
    for loc in candidates:
        if loc.count() == 0:
            continue
        try:
            target = loc.first
            if not target.is_visible(timeout=2_000):
                continue
            target.click(timeout=5_000)
            target.press_sequentially(payload, delay=15, timeout=15_000)
            return True
        except Exception:
            continue
    return False


def _submit_reel_comment_field(page: Page) -> bool:
    """Soumet le commentaire (Entrée sur textarea, sinon bouton Publier)."""
    for selector in (
        'textarea[placeholder*="commentaire" i]',
        'textarea[placeholder*="comment" i]',
        "form textarea",
    ):
        loc = page.locator(selector)
        for idx in range(min(loc.count(), 3)):
            target = loc.nth(idx)
            try:
                if not target.is_visible(timeout=1_000):
                    continue
                target.press("Enter", timeout=5_000)
                page.wait_for_timeout(2_500)
                return True
            except Exception:
                continue

    for label in ("Publier", "Poster", "Post", "Publish"):
        for selector in (
            f'div[role="button"]:has-text("{label}")',
            f'button:has-text("{label}")',
        ):
            loc = page.locator(selector)
            if loc.count() == 0:
                continue
            try:
                loc.first.click(timeout=4_000)
                page.wait_for_timeout(2_500)
                return True
            except Exception:
                continue

    try:
        page.keyboard.press("Enter")
        page.wait_for_timeout(2_500)
        return True
    except Exception:
        return False


def _verify_comment_published(page: Page, text: str, media_id: str) -> bool:
    """Vérifie que le texte apparaît dans les commentaires après publication."""
    snippet = str(text or "").strip()
    mid = str(media_id or "").strip()
    if not snippet or not mid:
        return False

    def _visible() -> bool:
        try:
            return bool(
                page.evaluate(
                    "(needle) => document.body.innerText.includes(needle)",
                    snippet,
                )
            )
        except Exception:
            return False

    if _visible():
        return True

    try:
        page.goto(
            f"{BASE_URL}/p/{mid}/",
            timeout=_REQUEST_TIMEOUT_MS,
            wait_until="domcontentloaded",
        )
        page.wait_for_load_state("load")
        page.wait_for_timeout(2_500)
        click_reel_comment_button(page)
        page.wait_for_timeout(2_000)
    except Exception:
        pass

    return _visible()


def post_reel_comment(
    media_id: str,
    comment_text: str,
    context: BrowserContext,
) -> tuple[bool, str]:
    """Publie un commentaire sur un post/reel (session connectée).

    Utilise ``/p/{id}/`` en priorité : la vue ``/reel/`` redirige souvent vers
    ``/reels/`` immersive sans champ de saisie fonctionnel.

    Retourne ``(ok, message_erreur)``.
    """
    mid = str(media_id or "").strip()
    text = str(comment_text or "").strip()
    if not mid:
        return False, "media_id vide"
    if not text:
        return False, "commentaire vide"

    page = context.new_page()
    last_err = "champ commentaire introuvable"
    try:
        for url in _reel_comment_post_urls(mid):
            page.goto(url, timeout=_REQUEST_TIMEOUT_MS, wait_until="domcontentloaded")
            page.wait_for_load_state("load")
            page.wait_for_timeout(2_500)

            blocked = _instagram_page_blocked(page)
            if blocked:
                last_err = f"page bloquée ({blocked})"
                continue

            has_textarea = page.locator(
                'textarea[placeholder*="commentaire" i], textarea[placeholder*="comment" i]'
            ).count() > 0
            if not has_textarea and not click_reel_comment_button(page):
                labels = _list_reel_page_aria_labels(page)
                last_err = f"bouton commentaire introuvable (labels={labels[:8]})"
                continue

            page.wait_for_timeout(1_500)

            if not _fill_reel_comment_field(page, text):
                labels = _list_reel_page_aria_labels(page)
                last_err = f"champ commentaire introuvable (labels={labels[:8]})"
                continue

            page.wait_for_timeout(400)

            if not _submit_reel_comment_field(page):
                last_err = "soumission échouée (Entrée / Publier)"
                continue

            page.wait_for_timeout(1_500)
            if _verify_comment_published(page, text, mid):
                return True, ""

            last_err = "commentaire non visible après soumission (faux positif / modération)"

        return False, last_err
    except Exception as e:
        return False, str(e)
    finally:
        page.close()


def click_reel_comment_button(page: Page) -> str | None:
    """Ouvre le panneau commentaires. Retourne le aria-label cliqué ou None."""
    for label in _REEL_COMMENT_ARIA_LABELS:
        for selector in (
            f'button:has(svg[aria-label="{label}"])',
            f'div[role="button"]:has(svg[aria-label="{label}"])',
            f'span[role="button"]:has(svg[aria-label="{label}"])',
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

    for pattern in (
        r"comment",
        r"commentaire",
    ):
        try:
            page.get_by_role("button", name=re.compile(pattern, re.IGNORECASE)).first.click(
                timeout=8_000
            )
            return f"role=button({pattern})"
        except Exception:
            pass

    try:
        link = page.locator('a[href*="/comments/"]').first
        if link.count() > 0:
            link.click(timeout=8_000)
            return "href=/comments/"
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
                        || svg.closest('span[role="button"]')
                        || svg.parentElement;
                    if (btn) { btn.click(); return label; }
                }
                const links = document.querySelectorAll('a[href*="/comments/"]');
                if (links.length) { links[0].click(); return 'comments-link'; }
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
    scroll_wait_ms: int = 700,
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
            page.wait_for_timeout(scroll_wait_ms)
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
    m = re.search(
        r"(\d[\d\s\u202f\xa0.,]*)\s*(?:likes?)\b",
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
    owner_username = ""
    for user_key in ("user", "owner"):
        user_obj = node.get(user_key)
        if isinstance(user_obj, dict):
            raw_user = user_obj.get("username")
            if raw_user:
                owner_username = str(raw_user).lstrip("@").strip().lower()
                break
    has_pinned_field = "clips_tab_pinned_user_ids" in node
    is_pinned: bool | None = None
    if has_pinned_field:
        is_pinned = _is_pinned_from_clips_tab_ids(node.get("clips_tab_pinned_user_ids"))

    product_type = node.get("product_type")
    product_type_str = str(product_type).strip() if product_type is not None else ""

    taken_at: int | None = None
    for ts_key in ("taken_at", "taken_at_timestamp", "device_timestamp"):
        ts_val = node.get(ts_key)
        if isinstance(ts_val, (int, float)) and int(ts_val) > 1_000_000_000:
            taken_at = int(ts_val)
            break

    if (
        not patch
        and not caption
        and not has_pinned_field
        and not owner_username
        and not product_type_str
        and taken_at is None
    ):
        return

    normalized = _normalize_metric_bucket(patch) if patch else {}
    if pk:
        bucket = metrics_by_pk.setdefault(pk, {})
        if patch:
            _merge_metric_bucket(bucket, normalized)
        if caption:
            _merge_caption_into_bucket(bucket, caption)
        if owner_username:
            _merge_owner_username_into_bucket(bucket, owner_username)
        if has_pinned_field:
            _merge_pinned_into_bucket(bucket, is_pinned)
        if product_type_str:
            bucket["product_type"] = product_type_str
        if taken_at is not None:
            bucket["taken_at"] = taken_at
        if code:
            metrics_by_code[code] = dict(bucket)
    elif code:
        bucket = metrics_by_code.setdefault(code, {})
        if patch:
            _merge_metric_bucket(bucket, normalized)
        if caption:
            _merge_caption_into_bucket(bucket, caption)
        if owner_username:
            _merge_owner_username_into_bucket(bucket, owner_username)
        if has_pinned_field:
            _merge_pinned_into_bucket(bucket, is_pinned)
        if product_type_str:
            bucket["product_type"] = product_type_str
        if taken_at is not None:
            bucket["taken_at"] = taken_at


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
        owner_username = _extract_owner_username_from_graphql_window(window)
        is_pinned = _extract_pinned_from_graphql_window(window)
        if code in metrics_by_code:
            if caption:
                _merge_caption_into_bucket(metrics_by_code[code], caption)
            if owner_username:
                _merge_owner_username_into_bucket(metrics_by_code[code], owner_username)
            if is_pinned is not None:
                _merge_pinned_into_bucket(metrics_by_code[code], is_pinned)
            continue
        pk_match = pk_re.search(window)
        if pk_match and pk_match.group(1) in metrics_by_pk:
            metrics_by_code[code] = dict(metrics_by_pk[pk_match.group(1)])
            if caption:
                _merge_caption_into_bucket(metrics_by_code[code], caption)
            if owner_username:
                _merge_owner_username_into_bucket(metrics_by_code[code], owner_username)
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
        if owner_username:
            _merge_owner_username_into_bucket(bucket, owner_username)
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
        url = response.url or ""
        # IG sert les données médias via /graphql/query OU /api/v1/ (clips/user,
        # feed/user…) selon les rollouts — intercepter les deux.
        if "graphql" not in url and "/api/v1/" not in url:
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

    previous = getattr(page, _GRAPHQL_LISTENER_ATTR, None)
    if previous is not None:
        try:
            page.remove_listener("response", previous)
        except Exception:
            pass
    page.on("response", on_response)
    setattr(page, _GRAPHQL_LISTENER_ATTR, on_response)
    return metrics_by_pk, metrics_by_code


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


_GRAPHQL_LISTENER_ATTR = "_ait_graphql_response_handler"


def load_instagram_cookies(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Charge et normalise les cookies Instagram depuis ``path`` (défaut : ``COOKIES_PATH``)."""
    p = Path(path) if path is not None else COOKIES_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Fichier cookies introuvable : {p.resolve()}")
    return _normalize_playwright_cookies(json.loads(p.read_text(encoding="utf-8")))


def save_instagram_cookies(context: BrowserContext, path: Path | str) -> None:
    """Persiste les cookies Playwright au format ``{"cookies": [...]}``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cookies": context.cookies()}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Cookies Instagram sauvegardés → %s", p.resolve())


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
        if exp is None:
            exp = c.get("expirationDate")
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


def get_browser_context(
    playwright: Playwright,
    *,
    cookies_path: Path | str | None = None,
) -> BrowserContext:
    """Lance Chromium, injecte les cookies Instagram depuis ``cookies_path``."""
    cookies = load_instagram_cookies(cookies_path)
    browser = playwright.chromium.launch(
        headless=HEADLESS,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
    )
    context = browser.new_context(
        viewport=_VIEWPORT,
        locale="fr-FR",
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    )
    # Masque navigator.webdriver (détection automation la plus courante).
    context.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
    )
    if cookies:
        context.add_cookies(cookies)
    return context


def _dismiss_post_login_dialogs(page: Page) -> None:
    """Ferme les popups « enregistrer infos » / notifications après login."""
    for label in (
        "Plus tard",
        "Not Now",
        "Pas maintenant",
        "Not now",
        "Plus tard",
    ):
        try:
            btn = page.get_by_role("button", name=label)
            if btn.count() > 0 and btn.first.is_visible(timeout=800):
                btn.first.click(timeout=2_000)
                page.wait_for_timeout(600)
        except Exception:
            pass


def _fill_instagram_login_form(page: Page, username: str, password: str) -> None:
    page.goto(
        f"{BASE_URL}/accounts/login/",
        timeout=_REQUEST_TIMEOUT_MS,
        wait_until="domcontentloaded",
    )
    page.wait_for_timeout(1200)
    page.locator('input[name="username"]').first.fill(username, timeout=10_000)
    page.locator('input[name="password"]').first.fill(password, timeout=10_000)
    page.locator('button[type="submit"]').first.click(timeout=10_000)
    page.wait_for_load_state("load", timeout=30_000)


def login_instagram_interactive(
    playwright: Playwright,
    username: str,
    password: str,
    *,
    save_cookies_path: Path | str,
    wait_after_submit_s: float = 300,
) -> BrowserContext:
    """Connexion avec navigateur visible : l'utilisateur complète 2FA/challenge à la main."""
    u = str(username or "").strip()
    pwd = str(password or "")
    if not u or not pwd:
        raise ValueError("username et password requis pour login_instagram_interactive")

    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context(viewport=_VIEWPORT, locale="fr-FR")
    page = context.new_page()
    try:
        log.info(
            "Ouverture navigateur pour @%s — compléter 2FA/challenge si demandé "
            "(timeout %ds).",
            u,
            int(wait_after_submit_s),
        )
        _fill_instagram_login_form(page, u, pwd)
        deadline = time.time() + wait_after_submit_s
        while time.time() < deadline:
            _dismiss_post_login_dialogs(page)
            if session_ok(context):
                save_instagram_cookies(context, save_cookies_path)
                log.info("Session @%s OK — cookies sauvegardés.", u)
                return context
            page.wait_for_timeout(2_000)

        raise TimeoutError(
            f"Login interactif expiré pour @{u} après {int(wait_after_submit_s)}s."
        )
    finally:
        page.close()


def ensure_watcher_browser_context(playwright: Playwright, *, slot: int) -> BrowserContext:
    """Contexte IG pour le watcher : slot 0 = cookies principaux, slot 1 = 2e compte."""
    import config

    if slot == 0:
        return get_browser_context(playwright, cookies_path=COOKIES_PATH)

    cookies_path = Path(config.WATCHER_IG2_COOKIES_PATH)
    if not cookies_path.is_file():
        raise FileNotFoundError(
            f"Cookies 2e compte absents ({cookies_path.resolve()}). "
            "Lancer : python watcher.py --login-ig2"
        )

    context = get_browser_context(playwright, cookies_path=cookies_path)
    if session_ok(context):
        return context

    br = context.browser
    if br:
        br.close()
    raise RuntimeError(
        f"Session 2e compte expirée ({cookies_path}). "
        "Relancer : python watcher.py --login-ig2"
    )


def test_session() -> bool:
    """Vérifie la session (cookies) sans argument : lance Playwright localement."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        context = get_browser_context(p)
        try:
            return session_ok(context)
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


def _collect_media_ids_from_grid(
    page: Page, max_reels: int, *, include_p_links: bool | None = None
) -> list[dict[str, str]]:
    """Collecte media_id + thumbnail depuis la grille /reels/ (ordre DOM).

    ``include_p_links`` : sur l'onglet ``/reels/`` tout est un Reel, donc on
    accepte aussi les ancres ``/p/<code>`` (rollouts IG 2025+ unifiant les
    URLs). Hors onglet reels (redirection profil racine), on reste strict sur
    ``/reel/`` pour ne pas ramasser des photos. ``None`` = auto via l'URL.
    """
    if include_p_links is None:
        # Profil créateur : reels et posts partagent souvent /p/ et /reel/.
        include_p_links = True
    items = page.evaluate(
        """({max, includeP}) => {
          const seen = new Set();
          const rows = [];
          const selector = includeP
            ? 'a[href*="/reel/"], a[href*="/p/"]'
            : 'a[href*="/reel/"]';
          const pattern = includeP
            ? /\\/(?:reel|p)\\/([^/?#]+)/
            : /\\/reel\\/([^/?#]+)/;
          document.querySelectorAll(selector).forEach(a => {
            if (rows.length >= max) return;
            const href = a.getAttribute("href") || "";
            const m = href.match(pattern);
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
        {"max": max_reels, "includeP": bool(include_p_links)},
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


def _code_has_rich_metrics(bucket: Any) -> bool:
    """True si le bucket porte des données de reel réelles (pas un code nu)."""
    if not isinstance(bucket, dict):
        return False
    if isinstance(bucket.get("taken_at"), (int, float)) and bucket["taken_at"] > 0:
        return True
    for key in ("view_count", "play_count", "like_count", "comment_count"):
        if isinstance(bucket.get(key), (int, float)) and bucket[key] > 0:
            return True
    return False


def _is_clips_media_bucket(bucket: Any) -> bool:
    """True si le bucket décrit un Reel vidéo (exclut carrousels / posts photo)."""
    if not isinstance(bucket, dict):
        return False
    product_type = str(bucket.get("product_type") or "").strip().lower()
    if product_type == "clips":
        return True
    if product_type in ("carousel_container", "carousel", "feed", "sidecar", "igtv"):
        return False
    for key in ("view_count", "play_count", "ig_play_count"):
        value = bucket.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return True
    return False


def _count_rich_graphql_codes(metrics_by_code: dict[str, dict[str, Any]]) -> int:
    """Nombre de codes porteurs de métriques (sert à temporiser l'attente)."""
    return sum(1 for b in metrics_by_code.values() if _code_has_rich_metrics(b))


def _grid_rows_from_graphql_codes(
    metrics_by_code: dict[str, dict[str, Any]],
    max_reels: int,
    *,
    expected_owner: str = "",
) -> list[dict[str, str]]:
    """Reels depuis l'interception GraphQL/REST, ordonnés par ``taken_at`` desc.

    Source PRIMAIRE depuis le markup IG 2026 (le DOM ne porte plus les codes).
    On ne garde que les buckets PORTEURS de métriques réelles — les codes nus
    issus de petites réponses (suggestions, audio…) sont du bruit et écartés —
    filtrés par propriétaire et product_type, puis triés du plus récent au plus
    ancien pour que la tête de liste soit bien le dernier post.
    """
    expected = str(expected_owner or "").lstrip("@").strip().lower()
    candidates: list[tuple[float, str]] = []
    for code, bucket in metrics_by_code.items():
        mid = str(code or "").strip()
        if not mid or not _is_valid_reel_code(mid):
            continue
        if not _code_has_rich_metrics(bucket):
            continue
        b = bucket if isinstance(bucket, dict) else {}
        owner = _owner_username_from_bucket(b)
        if expected and owner != expected:
            continue
        product_type = str(b.get("product_type") or "").strip().lower()
        if product_type and product_type != "clips":
            continue
        taken = b.get("taken_at")
        sort_key = float(taken) if isinstance(taken, (int, float)) else 0.0
        candidates.append((sort_key, mid))

    # taken_at desc ; codes sans taken_at (sort_key=0) repoussés en fin de liste.
    candidates.sort(key=lambda t: t[0], reverse=True)

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, mid in candidates:
        if mid in seen:
            continue
        seen.add(mid)
        rows.append({"media_id": mid, "thumbnail_url": ""})
        if len(rows) >= max_reels:
            break
    return rows


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


def get_reel_page_metadata(media_id: str, context: BrowserContext) -> dict[str, str]:
    """Caption + propriétaire depuis une seule visite ``/reel/{id}/``."""
    mid = str(media_id or "").strip()
    if not mid:
        return {"caption": "", "owner_username": ""}

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

    owner_username = ""
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
        if not caption:
            caption = extract_reel_caption_from_dom(page)
        owner_username = _extract_reel_owner_username(page, media_id=mid)
    except Exception:
        pass
    finally:
        page.close()

    return {
        "caption": str(caption or "").strip(),
        "owner_username": str(owner_username or "").lstrip("@").strip().lower(),
    }


def _csrf_token_from_page_context(page: Page) -> str:
    for cookie in page.context.cookies():
        if cookie.get("name") == "csrftoken":
            return str(cookie.get("value") or "")
    return ""


def _resolve_instagram_user_pk(page: Page, username: str) -> str | None:
    """PK numérique du créateur via ``/api/v1/users/web_profile_info/``."""
    u = str(username or "").lstrip("@").strip().lower()
    if not u:
        return None
    try:
        payload = page.evaluate(
            """async ({ user, appId }) => {
              const r = await fetch(
                'https://www.instagram.com/api/v1/users/web_profile_info/?username='
                  + encodeURIComponent(user),
                {
                  credentials: 'include',
                  headers: {
                    'X-IG-App-ID': appId,
                    'X-Requested-With': 'XMLHttpRequest',
                  },
                }
              );
              if (!r.ok) return null;
              const data = await r.json();
              const userObj = data?.data?.user;
              const pk = userObj?.id || userObj?.pk || '';
              return pk ? String(pk) : null;
            }""",
            {"user": u, "appId": _IG_WEB_APP_ID},
        )
        pk = str(payload or "").strip()
        return pk if pk.isdigit() else None
    except Exception as e:
        log.debug("pk @%s via web_profile_info : %s", u, e)
        return None


def _thumbnail_from_clips_media(media: dict[str, Any]) -> str:
    image_versions = media.get("image_versions2")
    if isinstance(image_versions, dict):
        candidates = image_versions.get("candidates")
        if isinstance(candidates, list) and candidates:
            first = candidates[0]
            if isinstance(first, dict):
                return str(first.get("url") or "")
    return ""


def _reel_from_clips_api_media(
    media: dict[str, Any],
    *,
    expected_owner: str,
) -> dict[str, Any] | None:
    """Convertit un nœud ``items[].media`` de ``/api/v1/clips/user/`` en reel watcher."""
    if not isinstance(media, dict):
        return None
    code = str(media.get("code") or "").strip()
    if not code or not _is_valid_reel_code(code):
        return None
    product_type = str(media.get("product_type") or "").strip().lower()
    if product_type and product_type != "clips":
        return None
    user_obj = media.get("user") if isinstance(media.get("user"), dict) else {}
    owner = str(user_obj.get("username") or expected_owner or "").lstrip("@").strip().lower()
    expected = str(expected_owner or "").lstrip("@").strip().lower()
    if expected and owner and owner != expected:
        return None
    metrics = _normalize_metric_bucket(media)
    taken_at = media.get("taken_at")
    return {
        "media_id": code,
        "thumbnail_url": _thumbnail_from_clips_media(media),
        "view_count": metrics["view_count"],
        "like_count": metrics["like_count"],
        "comment_count": metrics["comment_count"],
        "share_count": metrics["share_count"],
        "reshare_count": 0,
        "caption": _extract_caption_from_node(media),
        "is_pinned": _is_pinned_from_clips_tab_ids(media.get("clips_tab_pinned_user_ids")),
        "product_type": product_type or "clips",
        "owner_username": owner or expected,
        "taken_at": int(taken_at) if isinstance(taken_at, (int, float)) else None,
        "row_source": "api_clips",
    }


def _fetch_creator_clips_from_api(
    page: Page,
    username: str,
    *,
    max_reels: int,
) -> list[dict[str, Any]] | None:
    """Grille Reels officielle via ``POST /api/v1/clips/user/`` (onglet Reels profil).

    Indispensable pour les comptes dont la timeline profil ne mélange que des
    carrousels (``user_timeline``) alors que les Reels vivent dans l'onglet
    Reels séparé (ex. @marrant_club).
    """
    u = str(username or "").lstrip("@").strip()
    if not u:
        return None
    pk = _resolve_instagram_user_pk(page, u)
    if not pk:
        log.debug("@%s : pk introuvable (web_profile_info).", u)
        return None
    csrf = _csrf_token_from_page_context(page)
    if not csrf:
        log.debug("@%s : csrftoken manquant pour /clips/user/.", u)
        return None
    page_size = min(50, max(12, max_reels + 4))
    try:
        raw = page.evaluate(
            """async ({ pk, csrf, appId, pageSize }) => {
              const r = await fetch('https://www.instagram.com/api/v1/clips/user/', {
                method: 'POST',
                credentials: 'include',
                headers: {
                  'X-IG-App-ID': appId,
                  'X-Requested-With': 'XMLHttpRequest',
                  'Content-Type': 'application/x-www-form-urlencoded',
                  'X-CSRFToken': csrf,
                },
                body: new URLSearchParams({
                  target_user_id: String(pk),
                  page_size: String(pageSize),
                }),
              });
              if (!r.ok) {
                return { ok: false, status: r.status, body: '' };
              }
              return { ok: true, status: r.status, body: await r.text() };
            }""",
            {"pk": pk, "csrf": csrf, "appId": _IG_WEB_APP_ID, "pageSize": page_size},
        )
    except Exception as e:
        log.debug("@%s : clips/user fetch error (%s)", u, e)
        return None
    if not isinstance(raw, dict) or not raw.get("ok"):
        status = raw.get("status") if isinstance(raw, dict) else "?"
        log.info("@%s : /api/v1/clips/user/ HTTP %s", u, status)
        return None
    try:
        data = json.loads(str(raw.get("body") or ""))
    except json.JSONDecodeError:
        return None
    if str(data.get("status") or "").lower() == "fail":
        return None
    items = data.get("items")
    if not isinstance(items, list) or not items:
        return None
    reels: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        media = item.get("media")
        if not isinstance(media, dict):
            continue
        reel = _reel_from_clips_api_media(media, expected_owner=u)
        if reel:
            reels.append(reel)
    if not reels:
        return None
    # Ordre grille IG : épinglés en tête puis récents. Ne pas re-trier par
    # taken_at — le watcher saute les épinglés et prend le premier restant.
    out = reels[:max_reels]
    log.info("@%s : %d reel(s) via /api/v1/clips/user/ (pk=%s).", u, len(out), pk)
    return out


def verify_reel_on_profile_grid(
    page: Page,
    media_id: str,
    *,
    max_reels: int = 8,
) -> bool:
    """True si ``media_id`` est visible dans la grille DOM actuelle (anti-cache GraphQL)."""
    mid = str(media_id or "").strip()
    if not mid:
        return False
    try:
        rows = _collect_media_ids_from_grid(page, max_reels)
    except Exception:
        return False
    return any(str(row.get("media_id") or "") == mid for row in rows)


def get_recent_reels_on_page(
    page: Page,
    username: str,
    max_reels: int = 5,
    *,
    spa_wait_ms: int | None = None,
    dom_only: bool = False,
) -> list[dict[str, Any]]:
    """Reels récents en réutilisant une page Playwright (pool watcher parallèle).

    Utilise ``open_reels_grid`` puis ``/api/v1/clips/user/`` en priorité ; repli
    interception GraphQL / grille DOM si l'API ne répond pas.
    """
    out: list[dict[str, Any]] = []
    u = str(username or "").lstrip("@").strip()
    if not u:
        return out
    try:
        metrics_by_pk, metrics_by_code = _attach_graphql_metrics_listener(page)
        metrics_by_pk.clear()
        metrics_by_code.clear()

        if not open_reels_grid(page, u, spa_wait_ms=spa_wait_ms):
            return out

        if not dom_only:
            api_reels = _fetch_creator_clips_from_api(page, u, max_reels=max_reels)
            if api_reels:
                return api_reels

        # Attendre des codes PORTEURS de métriques (view_count/taken_at), pas
        # de simples codes nus : la grosse réponse clips (~100 Ko) arrive en
        # dernier, après plusieurs petites réponses ne contenant que des codes.
        # Sortir trop tôt donnait des reels à 0 vue / mauvais ordre. On sort dès
        # que le gros payload est là (seuil atteint) OU que le nombre de codes
        # riches s'est stabilisé (comptes à peu de reels), plafond 10 s.
        deadline = time.time() + 10
        target = min(max_reels, 6)
        last_rich = 0
        stable_since: float | None = None
        while time.time() < deadline:
            rich = _count_rich_graphql_codes(metrics_by_code)
            if rich >= target:
                break
            if rich > 0 and rich == last_rich:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since > 1.5:
                    break
            else:
                stable_since = None
            last_rich = rich
            page.wait_for_timeout(250)

        # Source PRIMAIRE = GraphQL ordonné par taken_at desc. Depuis 2026 la
        # grille DOM rend les tuiles en <a role="link"> sans code dans le href
        # (navigation JS), donc le DOM ne fournit plus de media_id. On tente
        # tout de même le DOM d'abord (comptes encore servis à l'ancienne).
        dom_rows = _collect_media_ids_from_grid(page, max_reels)
        if max_reels > 5 and dom_rows:
            dom_rows = _scroll_reels_grid_until_loaded(page, max_reels, dom_rows)

        graphql_rows = _grid_rows_from_graphql_codes(
            metrics_by_code, max_reels, expected_owner=u
        )

        owned_dom_clip_rows: list[dict[str, str]] = []
        for row in dom_rows:
            mid = str(row.get("media_id") or "")
            bucket = _metrics_bucket_for_dom_media_id(
                mid, metrics_by_pk, metrics_by_code
            )
            owner = _owner_username_from_bucket(bucket)
            if owner and owner != u.lower():
                continue
            if not _is_clips_media_bucket(bucket):
                continue
            owned_dom_clip_rows.append(row)

        if owned_dom_clip_rows:
            grid_rows = owned_dom_clip_rows
            row_source = "dom"
        elif graphql_rows and not dom_only:
            log.info(
                "@%s : grille DOM sans reel vidéo du créateur — source GraphQL "
                "(%d reel(s) porteurs de métriques).",
                u,
                len(graphql_rows),
            )
            grid_rows = graphql_rows
            row_source = "graphql"
        else:
            grid_rows = []
            row_source = "none"

        if not grid_rows:
            log.warning(
                "@%s : aucun reel exploitable (DOM sans code%s, GraphQL=%d).",
                u,
                ", dom_only" if dom_only else "",
                len(graphql_rows),
            )
            return out

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
                    "product_type": str(entry.get("product_type") or "") or "clips",
                    "owner_username": _owner_username_from_bucket(entry),
                    "taken_at": entry.get("taken_at"),
                    "row_source": row_source,
                }
            )
    except Exception as e:
        log.warning("get_recent_reels_on_page @%s : %s", u, e)
        return out

    return out


def get_recent_reels(
    username: str,
    context: BrowserContext,
    max_reels: int = 5,
    *,
    spa_wait_ms: int | None = None,
    dom_only: bool = False,
) -> list[dict[str, Any]]:
    """Reels récents : media_ids (DOM) + métriques (interception GraphQL).

    Les captions manquantes restent vides ici (pas de visite /reel/{id}/ —
    trop lent pour le scoring discovery). L'embedder complète si besoin.
    """
    page = context.new_page()
    try:
        return get_recent_reels_on_page(
            page,
            username,
            max_reels=max_reels,
            spa_wait_ms=spa_wait_ms,
            dom_only=dom_only,
        )
    finally:
        page.close()


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
