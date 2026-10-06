# AItertainment

Deux phases. La découverte lit l'historique des reels d'un créateur, classe le type d'audience et enregistre le profil. La veille détecte un nouveau reel et prépare un commentaire. Un humain valide dans Telegram avant que ça parte.

L'installation d'une machine dédiée est dans `README_SETUP.md`.

## Lancer

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Renseigner `.env` à partir de `.env.example`. Ne pas commiter `.env`.

## Fichiers

- `discovery.py` — profilage, lancé de temps en temps
- `watcher.py` — détection d'un nouveau reel
- `telegram_discovery_bot.py` — revue humaine
