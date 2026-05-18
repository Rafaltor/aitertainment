#!/usr/bin/env python3
"""Middleware autonome : Gemma (LM Studio) exécute, Claude (OpenRouter) supervise si blocage."""

from __future__ import annotations

import sys
from pathlib import Path

import requests

GEMMA_URL = "http://localhost:1234/v1/chat/completions"
GEMMA_MODEL = "google/gemma-4-e4b"
CLAUDE_URL = "https://openrouter.ai/api/v1/chat/completions"
CLAUDE_MODEL = "anthropic/claude-sonnet-4-6"
MAX_TURNS = 10
CONFIDENCE_KEYWORDS = [
    "je ne sais pas",
    "je ne peux pas",
    "je suis bloqué",
    "je ne trouve pas",
    "incertain",
    "unclear",
    "unsure",
    "I don't know",
    "I can't",
    "I'm not sure",
    "stuck",
    "help",
    "advice",
    "suggest",
    "recommend",
]

OPENROUTER_KEY = ""


def load_openrouter_key() -> str:
    """Charge OPENROUTER_API_KEY depuis ~/.hermes/.env."""
    env_path = Path.home() / ".hermes" / ".env"
    if not env_path.is_file():
        raise ValueError(f"Fichier introuvable : {env_path}")

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() != "OPENROUTER_API_KEY":
            continue
        val = value.strip().strip('"').strip("'")
        if val:
            return val

    raise ValueError("OPENROUTER_API_KEY absente ou vide dans ~/.hermes/.env")


def is_uncertain(response_text: str) -> bool:
    """True si le texte contient un indice d'incertitude (mots-clés, insensible à la casse)."""
    lower = response_text.lower()
    return any(kw.lower() in lower for kw in CONFIDENCE_KEYWORDS)


def ask_gemma(messages: list[dict]) -> str:
    try:
        r = requests.post(
            GEMMA_URL,
            json={"model": GEMMA_MODEL, "messages": messages},
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]["message"]["content"]
        return (choice or "").strip()
    except Exception:
        return ""


def ask_claude(context: str, gemma_response: str) -> str:
    try:
        global OPENROUTER_KEY
        r = requests.post(
            CLAUDE_URL,
            json={
                "model": CLAUDE_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Tu es un supervisor expert. Gemma est bloquée sur une tâche. "
                            "Donne-lui une guidance précise et actionnable en 3-5 lignes max. "
                            "Sois direct, technique, pas de blabla."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Contexte de la tâche:\n{context}\n\n"
                            f"Réponse de Gemma (bloquée):\n{gemma_response}\n\n"
                            f"Que doit-elle faire ?"
                        ),
                    },
                ],
            },
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]["message"]["content"]
        return (choice or "").strip()
    except Exception:
        return ""


def run_supervised_task(user_prompt: str) -> None:
    messages: list[dict] = [{"role": "user", "content": user_prompt}]
    consecutive_empty_gemma = 0

    for _ in range(MAX_TURNS):
        gemma_response = ask_gemma(messages)
        print(f"🤖 Gemma: {gemma_response}")

        if not gemma_response:
            consecutive_empty_gemma += 1
            if consecutive_empty_gemma >= 2:
                break
            continue

        consecutive_empty_gemma = 0

        if "?" not in gemma_response and not is_uncertain(gemma_response):
            messages.append({"role": "assistant", "content": gemma_response})
            break

        if is_uncertain(gemma_response):
            print("⚡ Gemma bloquée — consultation Claude...")
            claude_advice = ask_claude(user_prompt, gemma_response)
            print(f"🧠 Claude: {claude_advice}")
            messages.append({"role": "assistant", "content": gemma_response})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"[Conseil de Claude]: {claude_advice}\n"
                        f"Continue la tâche avec ce conseil."
                    ),
                }
            )
        else:
            messages.append({"role": "assistant", "content": gemma_response})

    print("✅ Tâche terminée")


def main() -> None:
    global OPENROUTER_KEY
    OPENROUTER_KEY = load_openrouter_key()

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:]).strip()
    else:
        prompt = input("Prompt: ").strip()

    if not prompt:
        print("Prompt vide.", file=sys.stderr)
        sys.exit(1)

    run_supervised_task(prompt)


if __name__ == "__main__":
    main()
