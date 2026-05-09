"""Configuration chargée depuis l'environnement et un fichier .env local."""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

APIFY_TOKEN = os.environ.get("APIFY_TOKEN", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
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

# Instagram (instagrapi) — utiliser un compte dédié, JAMAIS le compte personnel
IG_USERNAME = os.environ.get("IG_USERNAME", "")
IG_PASSWORD = os.environ.get("IG_PASSWORD", "")

# Anti-détection Instagram (instagrapi)
# Délai aléatoire (s) entre deux appels read sensibles (user_medias, etc.)
IG_SLEEP_MIN = float(os.environ.get("IG_SLEEP_MIN", "1.5"))
IG_SLEEP_MAX = float(os.environ.get("IG_SLEEP_MAX", "4.0"))
# Nombre max de comptes vérifiés avant longue pause anti-flag
MAX_ACCOUNTS_PER_SESSION = int(os.environ.get("MAX_ACCOUNTS_PER_SESSION", "50"))

# ----------------------------------------------------------------------------
# Discovery (Layer 0) — valeurs MODE TEST par défaut.
# Override possible via .env / variables d'environnement pour passer en prod.
# ----------------------------------------------------------------------------
# Quota humain quotidien : nb de profils scorés/jour. Mode test = 5, prod ≈ 40.
MAX_PROFILES_PER_DAY = int(os.environ.get("MAX_PROFILES_PER_DAY", "5"))
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
    os.environ.get("DISABLE_HUMAN_SCHEDULE", "true").strip().lower()
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

# Ollama (classifier / generate_comments) — OLLAMA_GENERATE_URL reste accepté en repli
OLLAMA_URL = (
    os.environ.get("OLLAMA_URL")
    or os.environ.get("OLLAMA_GENERATE_URL")
    or "http://localhost:11434/api/generate"
).strip()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b").strip()
