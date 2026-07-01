"""Appels texte Ollama (Mac mini local) — weave, fusion, etc."""

from __future__ import annotations

import logging
from typing import Any

import requests

import config

_LOG = logging.getLogger("aitertainment.ollama_text")


def call_ollama_text(
    prompt: str,
    *,
    model: str | None = None,
    num_predict: int = 128,
    temperature: float = 0.45,
    timeout: int | None = None,
) -> str:
    """Génération texte via ``OLLAMA_URL`` (défaut Mac mini)."""
    ollama_model = (model or config.OLLAMA_MODEL or "").strip()
    url = (config.OLLAMA_URL or "").strip()
    if not ollama_model or not url:
        _LOG.warning("Ollama indisponible (OLLAMA_URL / modèle absent).")
        return ""
    req_timeout = int(
        timeout
        if timeout is not None
        else getattr(config, "OLLAMA_REQUEST_TIMEOUT_S", 60)
    )
    body: dict[str, Any] = {
        "model": ollama_model,
        "prompt": str(prompt or ""),
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            "num_predict": num_predict,
            "stop": ["\n\n", "###", "\n###"],
        },
        "keep_alive": getattr(config, "OLLAMA_KEEP_ALIVE", "5m"),
    }
    try:
        resp = requests.post(url, json=body, timeout=req_timeout)
        resp.raise_for_status()
        return str(resp.json().get("response", "")).strip()
    except Exception as exc:
        _LOG.warning("Ollama texte échoué (modèle=%s) : %s", ollama_model, exc)
        return ""
