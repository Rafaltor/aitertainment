#!/usr/bin/env python3
"""Déploie un export Colab (LoRA) → merge → GGUF → Ollama → test.

Place le dossier téléchargé depuis Colab (``qwen_generator_finetuned`` ou
``checkpoint-600``) dans ``models/incoming_lora/``, puis :

    .venv/bin/python scripts/deploy_generator.py
    .venv/bin/python scripts/deploy_generator.py --lora-dir models/incoming_lora/checkpoint-600
    .venv/bin/python scripts/deploy_generator.py --skip-gguf --skip-ollama  # merge seulement
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

PYTHON = sys.executable
DEFAULT_INCOMING = _PROJECT_ROOT / "models" / "incoming_lora"
DEFAULT_LORA = _PROJECT_ROOT / "models" / "aitertainment-generator-lora"
DEFAULT_MERGED = _PROJECT_ROOT / "models" / "aitertainment-generator-merged"
GGUF_F16 = _PROJECT_ROOT / "models" / "aitertainment-generator-f16.gguf"
GGUF_Q4 = _PROJECT_ROOT / "models" / "aitertainment-generator-q4km.gguf"
LLAMA_CPP = _PROJECT_ROOT / "llama.cpp"
MODELFILE = _PROJECT_ROOT / "deploy" / "Modelfile.generator"


def _run(cmd: list[str], *, cwd: Path | None = None) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=cwd or _PROJECT_ROOT)


def _resolve_lora_dir(path: Path) -> Path:
    from scripts.merge_generator_lora import _has_adapter_weights, _resolve_lora_dir

    return _resolve_lora_dir(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge LoRA Colab → GGUF → Ollama.")
    parser.add_argument(
        "--lora-dir",
        type=Path,
        default=None,
        help=f"Dossier LoRA (défaut : {DEFAULT_INCOMING} ou checkpoint le plus récent).",
    )
    parser.add_argument(
        "--checkpoint",
        default="600",
        help="Nom checkpoint sous incoming_lora (ex. 600 → checkpoint-600).",
    )
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--hf-repo",
        default="Rafaltor/aitertainment-generator",
        help="Télécharge les adapters depuis Hugging Face (--hf-download).",
    )
    parser.add_argument(
        "--hf-download",
        action="store_true",
        help="Télécharge checkpoint-600 depuis --hf-repo vers incoming_lora/.",
    )
    parser.add_argument(
        "--hf-checkpoint",
        default="600",
        help="Numéro checkpoint HF (ex. 600).",
    )
    args = parser.parse_args(argv)

    if args.hf_download:
        from huggingface_hub import snapshot_download

        import os

        token = os.environ.get("HF_TOKEN", "").strip() or None
        ck = f"checkpoint-{args.hf_checkpoint}"
        patterns = [
            f"{ck}/adapter_model.safetensors",
            f"{ck}/adapter_config.json",
            f"{ck}/tokenizer.json",
            f"{ck}/tokenizer_config.json",
            f"{ck}/chat_template.jinja",
        ]
        print(f"Téléchargement HF {args.hf_repo} ({ck})…")
        snapshot_download(
            repo_id=args.hf_repo,
            allow_patterns=patterns,
            local_dir=str(DEFAULT_INCOMING),
            token=token,
        )
        print(f"✓ Écrit sous {DEFAULT_INCOMING / ck}")

    if args.lora_dir:
        lora_src = args.lora_dir if args.lora_dir.is_absolute() else _PROJECT_ROOT / args.lora_dir
    elif (DEFAULT_INCOMING / f"checkpoint-{args.checkpoint}").exists():
        lora_src = DEFAULT_INCOMING / f"checkpoint-{args.checkpoint}"
    elif DEFAULT_INCOMING.exists() and any(DEFAULT_INCOMING.iterdir()):
        lora_src = DEFAULT_INCOMING
    else:
        lora_src = DEFAULT_LORA

    lora_src = _resolve_lora_dir(lora_src)
    print(f"LoRA source : {lora_src}")

    if not args.skip_merge:
        code = _run(
            [
                PYTHON,
                "scripts/merge_generator_lora.py",
                "--lora-dir",
                str(lora_src),
                "--out-dir",
                str(DEFAULT_MERGED),
            ]
        )
        if code != 0:
            return code

    if not args.skip_gguf:
        convert = LLAMA_CPP / "convert_hf_to_gguf.py"
        if not convert.exists():
            print(f"Erreur : {convert} introuvable.", file=sys.stderr)
            return 1
        code = _run(
            [
                sys.executable,
                str(convert),
                str(DEFAULT_MERGED),
                "--outfile",
                str(GGUF_F16),
                "--outtype",
                "f16",
            ],
            cwd=LLAMA_CPP,
        )
        if code != 0:
            return code
        for candidate in (
            LLAMA_CPP / "build" / "bin" / "llama-quantize",
            LLAMA_CPP / "llama-quantize",
            LLAMA_CPP / "quantize",
        ):
            if candidate.exists():
                quantize_bin = candidate
                break
        else:
            quantize_bin = None
        if quantize_bin is None:
            print(
                "Compile llama.cpp : cd llama.cpp && cmake -B build && cmake --build build",
                file=sys.stderr,
            )
            return 1
        code = _run([str(quantize_bin), str(GGUF_F16), str(GGUF_Q4), "Q4_K_M"])
        if code != 0:
            return code

    if not args.skip_ollama:
        code = _run(
            [
                "ollama",
                "create",
                "aitertainment-generator",
                "-f",
                str(MODELFILE),
            ]
        )
        if code != 0:
            return code

    if not args.skip_test:
        code = _run(
            [
                PYTHON,
                "scripts/test_generator.py",
                "--t-type",
                "T3b",
                "--caption",
                "La recale qu'il s'est pris à la fin mdrr",
            ]
        )
        if code != 0:
            return code

    print("\n✓ Déploiement terminé. OLLAMA_GENERATOR_MODEL=aitertainment-generator dans .env")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
