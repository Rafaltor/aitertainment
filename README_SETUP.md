# AItertainment — Setup Mac Mini M4

Guide complet pour faire tourner le pipeline en autonome sur un Mac Mini Apple Silicon (M4 ou M4 Pro). Toutes les commandes sont à exécuter dans Terminal.

> **Cible** : machine dédiée allumée 24/7, daemons Watcher + bot Discovery lancés au boot via `launchd` (équivalent macOS de cron + systemd).

**Checklist prod `.env`** : `DISABLE_HUMAN_SCHEDULE=false`, `OLLAMA_GENERATOR_MODEL` défini, `data/instagram_cookies.json` présent. Remplir `data/seeds.json` (comptes seed) pour lancer Discovery.

---

## 1. Prérequis système

### 1.1 Mises à jour macOS

```bash
softwareupdate --install --all --restart
```

### 1.2 Xcode Command Line Tools

Nécessaire pour compiler certaines dépendances Python natives.

```bash
xcode-select --install
```

### 1.3 Homebrew

Installeur de packages Mac. La commande officielle (vérifie sur [brew.sh](https://brew.sh)) :

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Sur Apple Silicon, ajouter Homebrew au `PATH` :

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Vérifier :

```bash
brew --version   # doit afficher une version ≥ 4.x
```

---

## 2. Python 3.11+

`config.py` et plusieurs modules utilisent des features 3.11+ (notamment `from __future__ import annotations` + des type hints modernes).

```bash
brew install python@3.12
```

Vérifier :

```bash
python3.12 --version   # Python 3.12.x
which python3.12       # /opt/homebrew/bin/python3.12
```

---

## 3. Ollama (LLM local pour classifier + generate_comments)

[Ollama](https://ollama.com/) tourne en daemon local — accédé par `requests` sur `http://localhost:11434`.

### 3.1 Installation

```bash
brew install ollama
brew services start ollama
```

`brew services` enregistre Ollama comme daemon launchd → démarrage auto au boot.

### 3.2 Pull du modèle par défaut

```bash
ollama pull qwen2.5:7b
```

⚠️ ~4 Go de téléchargement. Ce modèle tient en RAM sur M4 8 Go (charge à la demande, déchargé après 5 min d'inactivité).

### 3.3 Test

```bash
ollama run qwen2.5:7b "réponds en un mot : OK"
```

---

## 4. Cloner le projet

```bash
mkdir -p ~/Apps && cd ~/Apps
git clone <URL_DE_TON_REPO> aitertainment
cd aitertainment
```

Si tu transfères depuis ton poste de dev (pas de Git distant), un simple `rsync` fait l'affaire :

```bash
# depuis le poste de dev :
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
      --exclude='logs/' --exclude='data/' --exclude='session_*.json' \
      ./aitertainment/ user@macmini.local:~/Apps/aitertainment/
```

---

## 5. Environnement Python isolé

```bash
cd ~/Apps/aitertainment
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Vérifier les versions installées :

```bash
python -c "from importlib.metadata import version; \
  [print(p, '==', version(p)) for p in ['playwright', 'requests', 'python-dotenv']]"
```

---

## 6. Configuration `.env`

```bash
cp .env.example .env
nano .env   # ou ton éditeur préféré
```

Champs **obligatoires** :

| Variable | Comment l'obtenir |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Crée un bot via [@BotFather](https://t.me/BotFather) → `/newbot` |
| `TELEGRAM_CHAT_ID` | Démarre une conversation avec ton bot, puis : `curl https://api.telegram.org/bot<TOKEN>/getUpdates` → champ `chat.id` |
| `TELEGRAM_DISCOVERY_TOKEN` | Idem, mais bot **séparé** pour Discovery |
| `TELEGRAM_DISCOVERY_CHAT_ID` | Idem (souvent même chat_id que le bot #1, juste deux bots distincts) |
| `data/instagram_cookies.json` | Cookies Playwright du compte Instagram **dédié** (export navigateur). **JAMAIS** le compte perso. Voir §6.1. |

Champs **optionnels** : quotas Discovery, `MAX_ACCOUNTS_PER_SESSION`, modèle Ollama → valeurs par défaut dans `.env.example`.

### 6.1 Cookies Instagram (Playwright)

Le scraping passe par Chromium + cookies persistés (pas d'API mobile privée).

1. Connecte-toi à Instagram dans Chrome avec le **compte dédié**.
2. Exporte les cookies au format Playwright (extension type « EditThisCookie » / export DevTools, ou script maison) vers `data/instagram_cookies.json`.
3. Format accepté : liste JSON `[{ "name": "sessionid", "value": "...", "domain": ".instagram.com", ... }]` ou objet `{"cookies": [...]}`.
4. Vérifie la session :

```bash
python -c "from scripts.instagram_browser import test_session; print('OK' if test_session() else 'session expirée')"
```

Renouvelle les cookies si la commande échoue ou si Discovery/Watcher loguent « session expirée ».

---

## 7. Smoke test

### 7.1 Watcher en mode mock (sans Instagram, sans Telegram réel)

```bash
source .venv/bin/activate
python watcher.py --mock --max-cycles 1
```

Doit afficher quelque chose comme :

```
... [INFO] === Watcher AItertainment démarré (mode mock=True) ===
... [INFO] [mock] post synthétique pour test
... [INFO] max_cycles=1 atteint, arrêt.
```

### 7.2 Tests unitaires

```bash
python -m unittest discover -s tests
# Doit afficher : Ran 330+ tests -> OK
```

### 7.3 Discovery — score d'un profil unique (mock)

```bash
python discovery.py --score @raikkonenaf --mock
```

### 7.4 Pipeline viral (commentaires → générateur)

```bash
.venv/bin/python scripts/scrape_viral_comments.py --target 5000 --min-likes 200
.venv/bin/python scripts/clean_comments.py --viral
.venv/bin/python scripts/label_comments.py --limit 100
.venv/bin/python scripts/clean_comments.py --training
.venv/bin/python scripts/prepare_dataset.py
```

Puis fine-tune Colab (`notebooks/finetune_generator.ipynb`) et déploiement Ollama (`scripts/deploy_generator.py`).

Si tout est vert : tu peux passer en mode prod.

---

## 8. Lancement automatique au boot — `launchd`

`launchd` est l'équivalent macOS de `cron` + `systemd`. Chaque service est décrit par un fichier `plist` dans `~/Library/LaunchAgents/`.

> **Convention** : on suppose que le projet est dans `/Users/paulm/Apps/aitertainment` et le venv dans `.venv/`. Adapte les chemins ci-dessous à ton arborescence (`pwd` dans le dossier projet, `which python` dans le venv).

### 8.1 Watcher — daemon permanent (poll toutes les N minutes en interne)

Crée `~/Library/LaunchAgents/com.aitertainment.watcher.plist` :

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.aitertainment.watcher</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/paulm/Apps/aitertainment/.venv/bin/python</string>
        <string>/Users/paulm/Apps/aitertainment/watcher.py</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/paulm/Apps/aitertainment</string>

    <!-- Démarrage au login + relance auto si crash -->
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>

    <!-- Throttle anti-spin : si le process re-crashe en moins de 30s,
         attendre avant de relancer. -->
    <key>ThrottleInterval</key>
    <integer>30</integer>

    <!-- Logs séparés (stdout / stderr) -->
    <key>StandardOutPath</key>
    <string>/Users/paulm/Apps/aitertainment/logs/launchd_watcher.out.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/paulm/Apps/aitertainment/logs/launchd_watcher.err.log</string>

    <!-- UTF-8 et logs non bufferisés -->
    <key>EnvironmentVariables</key>
    <dict>
        <key>LANG</key>
        <string>fr_FR.UTF-8</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
</dict>
</plist>
```

Activer :

```bash
mkdir -p ~/Apps/aitertainment/logs
launchctl load -w ~/Library/LaunchAgents/com.aitertainment.watcher.plist
```

Vérifier qu'il tourne :

```bash
launchctl list | grep aitertainment
# → 0   <pid>   com.aitertainment.watcher

tail -f ~/Apps/aitertainment/logs/launchd_watcher.out.log
```

Pour arrêter / recharger après modification :

```bash
launchctl unload ~/Library/LaunchAgents/com.aitertainment.watcher.plist
launchctl load -w ~/Library/LaunchAgents/com.aitertainment.watcher.plist
```

### 8.2 Telegram Discovery Bot — daemon permanent

Mêmes principes pour `telegram_discovery_bot.py` (long-polling Telegram). Crée `~/Library/LaunchAgents/com.aitertainment.discovery_bot.plist` en remplaçant :

```xml
    <key>Label</key>
    <string>com.aitertainment.discovery_bot</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/paulm/Apps/aitertainment/.venv/bin/python</string>
        <string>/Users/paulm/Apps/aitertainment/telegram_discovery_bot.py</string>
    </array>

    <!-- ... reste identique, juste les StandardOutPath/StandardErrorPath
         pointent vers logs/launchd_discovery_bot.{out,err}.log ... -->
```

```bash
launchctl load -w ~/Library/LaunchAgents/com.aitertainment.discovery_bot.plist
```

### 8.3 Désactiver complètement un service

```bash
launchctl unload -w ~/Library/LaunchAgents/com.aitertainment.<label>.plist
```

---

## 9. Maintenance

### 9.1 Logs

```bash
cd ~/Apps/aitertainment/logs
ls -lh
tail -f watcher.log              # logs applicatifs (rotation gérée côté code)
tail -f launchd_watcher.err.log  # crashs Python qui remontent jusqu'à launchd
```

### 9.2 Mise à jour des dépendances

```bash
cd ~/Apps/aitertainment
source .venv/bin/activate
pip install --upgrade -r requirements.txt
launchctl unload ~/Library/LaunchAgents/com.aitertainment.watcher.plist
launchctl load   ~/Library/LaunchAgents/com.aitertainment.watcher.plist
```

### 9.3 Mise à jour du code (depuis poste de dev)

```bash
# depuis le poste de dev
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='logs/' \
         --exclude='data/' --exclude='session_*.json' --exclude='.env' \
         ./aitertainment/ paulm@macmini.local:~/Apps/aitertainment/

# sur le Mac Mini
ssh paulm@macmini.local
cd ~/Apps/aitertainment
launchctl unload ~/Library/LaunchAgents/com.aitertainment.watcher.plist
launchctl load   ~/Library/LaunchAgents/com.aitertainment.watcher.plist
launchctl unload ~/Library/LaunchAgents/com.aitertainment.discovery_bot.plist
launchctl load   ~/Library/LaunchAgents/com.aitertainment.discovery_bot.plist
```

`.env`, `data/` et `logs/` sont préservés (exclus du rsync). Les services se relancent proprement après unload/load.

### 9.4 Économie d'énergie

Sur Mac Mini, désactiver la mise en veille pour éviter que `launchd` ne soit suspendu :

```bash
sudo pmset -a sleep 0
sudo pmset -a disksleep 0
sudo pmset -a hibernatemode 0
```

Et activer le démarrage auto après coupure de courant :
```bash
sudo pmset -a autorestart 1
```

---

## 10. Récap des fichiers persistés

| Fichier | Géré par | Remarque |
|---|---|---|
| `data/database.json` | `database.py` | Profils + scores_history + tier |
| `data/candidates.json` | `discovery.py` | Candidats Discovery (en attente de validation Telegram) |
| `data/viral_comments.json` | `scrape_viral_comments.py` | Pool de commentaires non labellisés (fil Reels) |
| `data/training_comments_viral.json` | `label_comments.py` | Commentaires labellisés (T-types) pour le générateur |
| `data/generator_dataset.json` | `prepare_dataset.py` | Export Alpaca pour fine-tune Colab |
| `data/discovery_bot_state.json` | `telegram_discovery_bot.py` | Offset Telegram pour `getUpdates` |
| `data/watchlist.json` | `watcher.py`, `scripts/embedder.py` | Créateurs surveillés (watcher + embeddings) |
| `data/vector_store.json` | `scripts/embedder.py` | Embeddings + axes 32D |
| `data/dataset_generator.jsonl` | `scripts/prepare_dataset.py` | Export JSONL (fine-tune) |
| `data/instagram_cookies.json` | `scripts/instagram_browser.py` | Session Playwright (non versionné) |
| `data/seeds.json` | `discovery.py` | Domaines + comptes seed pour exploration |

Écritures **atomiques** + verrou fichier (`modules/atomic_json.py`) sur `database.json`, `watchlist.json`, `candidates.json`, `seeds.json` et état bot.

**Pool viral** : `label_comments` retire du viral ce qui part en training — un viral « bas » est normal si tu as beaucoup labellisé. `clean_comments --viral` réécrit le fichier (backup auto sauf `--no-backup`).

### Scripts (`scripts/`)

| Script | Rôle |
|---|---|
| `scrape_viral_comments.py` | Pool `viral_comments.json` (fil Reels) |
| `clean_comments.py` | Nettoyage viral (`--viral`) et training (`--training`) |
| `label_comments.py` | Labélisation T-types → `training_comments_viral.json` |
| `prepare_dataset.py` | Export `generator_dataset.json` + `dataset_generator.jsonl` |
| `train_from_viral.py` | Orchestrateur clean → label → prepare |
| `embedder.py` | Embeddings + `vector_store.json` |
| `deploy_generator.py` / `merge_generator_lora.py` | Post-Colab → Ollama |
| `instagram_browser.py` | Couche Playwright partagée |

Dépendances merge/deploy : `pip install -r requirements-ml.txt` (hors runtime quotidien).

---

## 11. Dépannage

| Symptôme | Cause probable | Fix |
|---|---|---|
| `Fichier cookies introuvable` | `data/instagram_cookies.json` absent | Exporter les cookies (cf. §6.1) |
| `Erreur HTTP Ollama: Connection refused` | daemon Ollama down | `brew services restart ollama` |
| `TELEGRAM_DISCOVERY_TOKEN manquant` | `.env` pas chargé | Vérifier que `.env` est à la racine du projet, pas dans un sous-dossier |
| Watcher tourne mais ne notifie rien | chat_id incorrect | `curl https://api.telegram.org/bot<TOKEN>/getUpdates` après avoir parlé au bot |
| `launchctl: status 78` | `python` du venv introuvable | Vérifier le chemin absolu dans `ProgramArguments` |
| Compte Instagram bloqué (challenge) | Pattern trop régulier | Baisser `MAX_PROFILES_PER_DAY`, augmenter les pauses Discovery, renouveler les cookies, attendre 24-48h |

Pour aller plus loin : `man launchd.plist`, `man launchctl`.

---

## 12. Roadmap audit (juin 2026)

| Priorité | Sujet | Statut |
|---|---|---|
| P0 | Vestiges instagrapi / `telegram_notify` | Fait |
| P0 | Suppression `rescore_scheduler`, `sync_discovery_state`, `curate_*` | Fait |
| P0 | Fusion `clean_comments.py` | Fait |
| P0 | Verrous JSON cross-process (`atomic_json`) | Fait |
| P0 | `DISABLE_HUMAN_SCHEDULE` défaut prod (`false`) | Fait |
| P0 | `seeds.json` vide sur ta machine | **À remplir** (comptes seed) |
| P1 | Unifier loaders watchlist (`embedder` / `label` / `watcher`) | À faire |
| P1 | Découper `instagram_browser.py` | À faire |
| P1 | Label/prepare incrémental (gros JSON) | À faire |
| P2 | `Makefile` / cibles smoke | Optionnel |

**Parcours opérationnel type**

```bash
# Services Mac Mini
launchctl load com.aitertainment.watcher.plist
launchctl load com.aitertainment.discovery_bot.plist

# Discovery (après seeds remplis)
.venv/bin/python discovery.py --domain humour

# Pipeline générateur (manuel, hors heures de pointe)
.venv/bin/python scripts/train_from_viral.py --label-limit 200
# Colab → models/incoming_lora/ → deploy_generator.py

.venv/bin/python watcher.py --once
```
