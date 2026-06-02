#!/usr/bin/env python3
"""Fusionne les adapters LoRA HF avec Qwen2.5-7B-Instruct (Mac Mini / local).

Usage::

    .venv/bin/python scripts/merge_generator_lora.py \\
      --lora-dir models/aitertainment-generator-lora \\
      --out-dir models/aitertainment-generator-merged

Ensuite : conversion GGUF (llama.cpp) → ``ollama create`` (cf. docs/GENERATOR_WATCHER.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_ADAPTER_WEIGHT_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model.safetensors.index.json",
)


def _has_adapter_weights(path: Path) -> bool:
    return any((path / name).exists() for name in _ADAPTER_WEIGHT_NAMES)


def _resolve_lora_dir(lora_dir: Path) -> Path:
    """Trouve un dossier avec les poids LoRA (racine ou dernier checkpoint)."""
    lora_dir = lora_dir.resolve()
    if _has_adapter_weights(lora_dir):
        return lora_dir

    checkpoints = sorted(
        (p for p in lora_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-", 1)[1]) if p.name.split("-", 1)[1].isdigit() else 0,
    )
    for ckpt in reversed(checkpoints):
        if _has_adapter_weights(ckpt):
            print(f"Poids LoRA trouvés dans {ckpt.name}/")
            return ckpt

    return lora_dir


def _print_missing_weights_help(lora_dir: Path, hf_repo: str | None) -> None:
    print(
        f"\nErreur : aucun poids LoRA dans {lora_dir}\n"
        "  (adapter_model.safetensors manquant — le download --include a sans doute "
        "téléchargé seulement adapter_config.json).\n",
        file=sys.stderr,
    )
    if hf_repo:
        print(
            "Télécharge les poids :\n"
            f"  huggingface-cli download {hf_repo} adapter_model.safetensors \\\n"
            f"    --local-dir {lora_dir}\n",
            file=sys.stderr,
        )
    else:
        print(
            "Télécharge les poids :\n"
            f"  huggingface-cli download TON_USERNAME/aitertainment-generator "
            f"adapter_model.safetensors --local-dir {lora_dir}\n",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base Qwen2.5-7B.")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Modèle HuggingFace de base.",
    )
    parser.add_argument(
        "--lora-dir",
        type=Path,
        default=_PROJECT_ROOT / "models" / "aitertainment-generator-lora",
        help="Dossier adapters (download HF option 2).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJECT_ROOT / "models" / "aitertainment-generator-merged",
        help="Sortie modèle fusionné (HF format).",
    )
    parser.add_argument(
        "--hf-repo",
        default="Rafaltor/aitertainment-generator",
        help="Repo HF (message d'aide si poids manquants).",
    )
    args = parser.parse_args(argv)

    lora_dir = args.lora_dir if args.lora_dir.is_absolute() else _PROJECT_ROOT / args.lora_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else _PROJECT_ROOT / args.out_dir

    if not lora_dir.exists():
        print(f"Erreur : dossier LoRA introuvable : {lora_dir}", file=sys.stderr)
        return 1

    lora_dir = _resolve_lora_dir(lora_dir)
    if not _has_adapter_weights(lora_dir):
        _print_missing_weights_help(lora_dir, args.hf_repo)
        return 1

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(
            "Installe d'abord : pip install torch transformers peft accelerate",
            file=sys.stderr,
        )
        print(exc, file=sys.stderr)
        return 1

    print(f"Base   : {args.base_model}")
    print(f"LoRA   : {lora_dir}")
    print(f"Sortie : {out_dir}")
    print("Chargement base (CPU, fp16) — ~14 Go RAM, quelques minutes…")

    tokenizer = AutoTokenizer.from_pretrained(str(lora_dir), trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base,
        str(lora_dir),
        is_trainable=False,
        local_files_only=True,
    )
    model = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))

    print(f"✓ Modèle fusionné écrit dans {out_dir}")
    print(
        "\nÉtape suivante (GGUF) :\n"
        "  1. brew install cmake && git clone https://github.com/ggerganov/llama.cpp\n"
        "  2. cd llama.cpp && pip install -r requirements.txt\n"
        "  3. python convert_hf_to_gguf.py ../models/aitertainment-generator-merged \\\n"
        "       --outfile ../models/aitertainment-generator-f16.gguf --outtype f16\n"
        "  4. make -j quantize && ./quantize ../models/aitertainment-generator-f16.gguf \\\n"
        "       ../models/aitertainment-generator-q4km.gguf Q4_K_M\n"
        "  5. Édite deploy/Modelfile.generator (FROM → chemin .gguf)\n"
        "  6. ollama create aitertainment-generator -f deploy/Modelfile.generator"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
