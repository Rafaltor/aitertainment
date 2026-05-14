# AItertainment — Setup Mac Mini M4

Guide complet pour faire tourner le pipeline en autonome sur un Mac Mini Apple Silicon (M4 ou M4 Pro). Toutes les commandes sont à exécuter dans Terminal.

> **Cible** : machine dédiée allumée 24/7, daemons Watcher + scheduler de rescore lancés au boot via `launchd` (équivalent macOS de cron + systemd).

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
  [print(p, '==', version(p)) for p in ['instagrapi', 'requests', 'python-dotenv']]"
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
| `IG_USERNAME` / `IG_PASSWORD` | Compte Instagram **dédié**. **JAMAIS** le compte perso. |

Champs **optionnels** : sleep ranges, quotas Discovery, modèle Ollama → tous ont des valeurs par défaut sensées (cf. `.env.example`).

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
# Doit afficher : Ran 242 tests in ~30s -> OK
```

### 7.3 Discovery — score d'un profil unique (mock)

```bash
python discovery.py --score @raikkonenaf --mock
```

### 7.4 Scheduler de rescore — mode dry-run

```bash
python rescore_scheduler.py --due    # liste les profils dûs
python rescore_scheduler.py --mock   # lance un cycle sans réseau
```

### 7.5 Dataset builder — collectes différées

```bash
python dataset_builder.py --pending
```

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

    <!-- Pour qu'instagrapi & dotenv trouvent les UTF-8 et le HOME -->
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

### 8.3 Rescore scheduler — 1×/jour à 09:30

Crée `~/Library/LaunchAgents/com.aitertainment.rescore.plist` :

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.aitertainment.rescore</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/paulm/Apps/aitertainment/.venv/bin/python</string>
        <string>/Users/paulm/Apps/aitertainment/rescore_scheduler.py</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/paulm/Apps/aitertainment</string>

    <!-- One-shot quotidien : pas KeepAlive, juste StartCalendarInterval -->
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>30</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>/Users/paulm/Apps/aitertainment/logs/launchd_rescore.out.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/paulm/Apps/aitertainment/logs/launchd_rescore.err.log</string>

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

```bash
launchctl load -w ~/Library/LaunchAgents/com.aitertainment.rescore.plist
```

Le scheduler tournera à 09:30 chaque jour, traitera les profils dûs (next_rescore_at ≤ now), pousera les notifs `📈 / 📉` sur Telegram en cas de variation ≥ ±15-20%, et balayera les `pending_collection.json` de `dataset_builder` au passage.

### 8.4 Forcer un déclenchement immédiat (debug)

```bash
launchctl start com.aitertainment.rescore
```

### 8.5 Désactiver complètement un service

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
| `data/database.json` | `database.py` | Profils + scores_history + tier + planning rescore |
| `data/candidates.json` | `discovery.py` | Candidats Discovery (en attente de validation Telegram) |
| `data/training_comments.json` | `dataset_builder.py` | Dataset GENERATOR + CLASSIFIER (deux schémas dans un fichier) |
| `data/pending_collection.json` | `dataset_builder.py` | Profils validés sans Reels ≥ 7 jours, à recollecter plus tard |
| `data/discovery_session.json` | `discovery.py` | Compteur de quota humain quotidien |
| `data/validations.json` | `telegram_discovery_bot.py` | Feedback humain (validate / reject / corrected) |
| `data/discovery_bot_state.json` | `telegram_discovery_bot.py` | Offset Telegram pour `getUpdates` |
| `watchlist.json` (racine) | `watcher.py` | Créateurs surveillés en temps réel par le Watcher |
| `seeds.json` (racine) | `discovery.py` | Domaines + comptes seed pour exploration |

Toutes les écritures sont **atomiques** (`tempfile + replace`), donc safe en cas de coupure brutale (panne de courant Mac Mini, kill -9, etc.).

---

## 11. Dépannage

| Symptôme | Cause probable | Fix |
|---|---|---|
| `instagrapi non installé` | venv pas activé | `source .venv/bin/activate` |
| `Erreur HTTP Ollama: Connection refused` | daemon Ollama down | `brew services restart ollama` |
| `TELEGRAM_DISCOVERY_TOKEN manquant` | `.env` pas chargé | Vérifier que `.env` est à la racine du projet, pas dans un sous-dossier |
| Watcher tourne mais ne notifie rien | chat_id incorrect | `curl https://api.telegram.org/bot<TOKEN>/getUpdates` après avoir parlé au bot |
| `launchctl: status 78` | `python` du venv introuvable | Vérifier le chemin absolu dans `ProgramArguments` |
| Compte Instagram bloqué (challenge) | Pattern trop régulier | Augmenter `IG_SLEEP_MIN/MAX`, baisser `MAX_PROFILES_PER_DAY`, attendre 24-48h |

Pour aller plus loin : `man launchd.plist`, `man launchctl`.
