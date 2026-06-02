"""Configuration chargée depuis l'environnement et un fichier .env local."""

import logging
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Bot Telegram #2 — dédié à la phase Discovery (validation humaine des candidats).
# Si non défini, on retombe sur TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.
TELEGRAM_DISCOVERY_TOKEN = (
    os.environ.get("TELEGRAM_DISCOVERY_TOKEN")
    or os.environ.get("TELEGRAM_BOT_TOKEN_DISCOVERY")
    or TELEGRAM_BOT_TOKEN
)
TELEGRAM_DISCOVERY_CHAT_ID = (
    os.environ.get("TELEGRAM_DISCOVERY_CHAT_ID")
    or os.environ.get("TELEGRAM_CHAT_ID_DISCOVERY")
    or TELEGRAM_CHAT_ID
)

# Anti-détection Instagram (Playwright) — Watcher
MAX_ACCOUNTS_PER_SESSION = int(os.environ.get("MAX_ACCOUNTS_PER_SESSION", "50"))
# Pause longue tous les N comptes vérifiés (secondes)
WATCHER_ACCOUNTS_PAUSE_S = int(os.environ.get("WATCHER_ACCOUNTS_PAUSE_S", "600"))
# Délai entre deux créateurs dans un cycle (secondes)
WATCHER_SLEEP_BETWEEN_CREATORS_S = int(
    os.environ.get("WATCHER_SLEEP_BETWEEN_CREATORS_S", "3")
)
# Intervalle entre deux cycles selon l'heure locale (secondes)
WATCHER_PRIME_INTERVAL_S = int(os.environ.get("WATCHER_PRIME_INTERVAL_S", "300"))
WATCHER_DAY_INTERVAL_S = int(os.environ.get("WATCHER_DAY_INTERVAL_S", "600"))
WATCHER_NIGHT_INTERVAL_S = int(os.environ.get("WATCHER_NIGHT_INTERVAL_S", "1800"))

# ----------------------------------------------------------------------------
# Discovery (Layer 0) — valeurs MODE TEST par défaut.
# Override possible via .env / variables d'environnement pour passer en prod.
# ----------------------------------------------------------------------------
# Quota humain quotidien : nb de profils scorés/jour (seed inclus si auto-scoré).
# Défaut production : 40. Pour les runs CI / tests rapides : MAX_PROFILES_PER_DAY=5 dans .env.
MAX_PROFILES_PER_DAY = int(os.environ.get("MAX_PROFILES_PER_DAY", "40"))
# Sleep aléatoire entre deux posts pendant le scrape de commentaires.
DISCOVERY_BETWEEN_POSTS_MIN_S = float(
    os.environ.get("DISCOVERY_BETWEEN_POSTS_MIN_S", "20")
)
DISCOVERY_BETWEEN_POSTS_MAX_S = float(
    os.environ.get("DISCOVERY_BETWEEN_POSTS_MAX_S", "45")
)
# Sleep aléatoire entre deux profils dans explore_network.
DISCOVERY_BETWEEN_PROFILES_MIN_S = float(
    os.environ.get("DISCOVERY_BETWEEN_PROFILES_MIN_S", "90")
)
DISCOVERY_BETWEEN_PROFILES_MAX_S = float(
    os.environ.get("DISCOVERY_BETWEEN_PROFILES_MAX_S", "180")
)
# Mode test : ignore les fenêtres humaines (nuit 23h-8h, déjeuner 12h-14h,
# pause de burst toutes les 2h). À mettre à True pour pouvoir tester à tout
# moment sans attendre la prochaine fenêtre active.
DISABLE_HUMAN_SCHEDULE = (
    os.environ.get("DISABLE_HUMAN_SCHEDULE", "false").strip().lower()
    in ("1", "true", "yes", "on")
)

# Seuil de score au-dessus duquel ``score_and_persist`` envoie une notif
# Telegram au validateur humain. Tout score est de toute façon upserté dans
# ``database.json`` — la notif est juste un signal pour un examen prioritaire.
# 350 = sweet spot empirique : assez bas pour capter les profils corrects en
# milieu de tier B, assez haut pour ne pas spammer le bot sur du tier C.
DISCOVERY_NOTIFY_THRESHOLD = float(
    os.environ.get("DISCOVERY_NOTIFY_THRESHOLD", "350")
)
# Ollama (labélisation T-type + generate_comments) — OLLAMA_GENERATE_URL reste accepté en repli
OLLAMA_URL = (
    os.environ.get("OLLAMA_URL")
    or os.environ.get("OLLAMA_GENERATE_URL")
    or "http://localhost:11434/api/generate"
).strip()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b").strip()

# Modèle Ollama fine-tuné pour ``generate_comments`` (format Alpaca).
# Requis pour ``generate_comments`` (Watcher). Ex. après ``ollama create`` (cf. deploy/).
OLLAMA_GENERATOR_MODEL = os.environ.get("OLLAMA_GENERATOR_MODEL", "").strip()

# LM Studio — embeddings profils (``scripts/embedder.py`` uniquement)
LM_STUDIO_URL = (
    os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1") or ""
).strip().rstrip("/")
LM_STUDIO_EMBED_MODEL = os.environ.get("LM_STUDIO_EMBED_MODEL", "").strip()

VALID_T_TYPES = frozenset({"T1", "T2", "T2b", "T3a", "T3b", "T4", "T5"})

# ----------------------------------------------------------------------------
# Niches éditoriales — vocabulaire fermé pour la classification de profils
# ----------------------------------------------------------------------------
#
# Cette liste est volontairement **fermée** (pas d'override env) pour deux
# raisons :
#
# 1. Les niches sont des **labels d'apprentissage** : si Discovery commence à
#    en inventer (« humour_noir_paris » ou autre), les datasets de fine-tuning
#    se fragmentent et les T-types par niche perdent leur stabilité.
# 2. Les valeurs sont co-référencées par les seeds (``data/seeds.json``), les
#    profils en base (``data/database.json:niches``), et les prompts du
#    classifier (``modules/classifier.py``). Une dérive doit être un acte
#    explicite (modifier ce code), pas une coquille dans un fichier de config.
#
# Ordre = thématique (humour → lifestyle → savoir → mode → sport → niches
# spécifiques). Pas alphabétique pour faciliter la lecture humaine.
VALID_NICHES: frozenset[str] = frozenset({
    # Humour
    "humour", "sketch", "stand_up", "imitation", "réaction", "brainrot",
    "trend", "prank", "POV", "relatable", "dark_humor", "cringe", "absurde",
    # Lifestyle / personnel
    "lifestyle", "vlog", "parentalité", "couple", "routine", "travel",
    # Savoir / tech
    "ai", "vulgarisation", "review_tech", "dev",
    # Mode / beauté
    "makeup", "skincare", "fashion", "thrift",
    # Sport
    "fitness", "sport_pro",
    # Pop culture / divertissement
    "gaming", "musique", "danse", "animaux",
    # Verticales sérieuses
    "education", "finance", "psychologie",
    # Cuisine
    "cuisine", "food_review",
    # Méta-formats
    "viral", "storytelling", "faceless", "collab",
})


_NICHE_LOG = logging.getLogger("aitertainment.config.niches")


def validate_niches(niches: list[str]) -> list[str]:
    """Filtre ``niches`` pour ne garder que les labels présents dans ``VALID_NICHES``.

    Comportement :

    * Tout label inconnu est **logué** en ``WARNING`` via le logger standard
      ``aitertainment.config.niches`` (un warning par label inconnu, pour
      rendre la grep-trace exploitable côté ops).
    * La liste retournée préserve l'ordre d'entrée et déduplique
      silencieusement (utile : seeds.json contient parfois deux fois la même
      niche par copy-paste).
    * **Jamais vide** : si tout est filtré (ou si l'entrée est vide / non
      itérable), on retourne ``["humour"]`` — c'est la niche par défaut du
      projet, présente dans ``VALID_NICHES``, et qui correspond au domaine
      principal de Discovery.

    Tolère ``None`` et les types non-string en entrée sans lever (chaque
    élément est passé par ``str(...)``) — on est appelés depuis du JSON
    brut, où une niche absente peut être ``None`` ou un nombre par accident.
    """
    if not isinstance(niches, list):
        _NICHE_LOG.warning(
            "validate_niches : entrée non-liste (%s) — fallback sur ['humour'].",
            type(niches).__name__,
        )
        return ["humour"]

    seen: set[str] = set()
    kept: list[str] = []
    for raw in niches:
        if raw is None:
            continue
        label = str(raw).strip()
        if not label:
            continue
        if label not in VALID_NICHES:
            _NICHE_LOG.warning(
                "validate_niches : niche inconnue rejetée %r (utilise une "
                "valeur de VALID_NICHES).",
                label,
            )
            continue
        if label in seen:
            continue
        seen.add(label)
        kept.append(label)

    if not kept:
        return ["humour"]
    return kept
