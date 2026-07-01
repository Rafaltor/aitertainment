# Generator fine-tune → Ollama → Watcher

## 1. Dataset (déjà fait)

```bash
.venv/bin/python scripts/prepare_dataset.py
```

Produit :

- `data/dataset_generator.jsonl` — 4937 lignes
- `data/generator_dataset.json` — upload Colab / RunPod

## 2. Fine-tune (GPU cloud)

1. Ouvrir `notebooks/finetune_generator.ipynb` sur **Colab Pro** ou **RunPod A100**.
2. Uploader `data/generator_dataset.json` à côté du notebook.
3. Renseigner `HF_REPO` + token HuggingFace (`notebook_login()`).
4. Exécuter toutes les cellules → adapters sur HF.
5. **Merger** : `model.save_pretrained_merged("./qwen_generator_merged", ...)`.
6. Convertir en **GGUF Q4_K_M** (`llama.cpp`, cf. cellule 8 du notebook).

## 3. Ollama sur Mac Mini

```bash
# Copier le .gguf dans le repo ou ~/models/
ollama create aitertainment-generator -f deploy/Modelfile.generator
```

Dans `.env` :

```env
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_GENERATOR_MODEL=aitertainment-generator
```

Sans `OLLAMA_GENERATOR_MODEL`, le Watcher utilise l’ancien chemin (prompt T-type + JSON via `OLLAMA_MODEL`).

## 4. Test générateur

```bash
.venv/bin/python scripts/test_generator.py --t-type T2b --caption "pov tu découvres le reel"
```

## 5. Watcher

Prérequis : `data/watchlist.json`, `data/instagram_cookies.json`, Telegram (optionnel).

```bash
# Dry-run (pas d’Instagram, pas de Telegram)
.venv/bin/python watcher.py --mock

# Un cycle réel
.venv/bin/python watcher.py --once
```

Le Watcher appelle `generate_comments` avec le `t_type` du créateur et la caption / contexte vidéo du nouveau reel.

## 6. Après un nouveau discover

```bash
.venv/bin/python scripts/label_comments.py
.venv/bin/python scripts/prepare_dataset.py
# Re-fine-tune generator si le dataset a beaucoup changé
```
