"""Utilitaires I/O pour ``data/training_comments_viral.json``."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAINING_PATH = _PROJECT_ROOT / "data" / "training_comments_viral.json"


class DatasetIOError(ValueError):
    """Erreur de lecture / écriture sur les fichiers du dataset."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise DatasetIOError(f"lecture impossible : {path} ({e})") from e
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise DatasetIOError(f"JSON invalide dans {path} : {e}") from e
    if not isinstance(data, dict):
        raise DatasetIOError(
            f"racine JSON doit être un objet ({type(data).__name__})"
        )
    return data


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        suffix=".tmp",
    ) as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        tmp_path = Path(fh.name)
    tmp_path.replace(path)


def _load_training(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_TRAINING_PATH
    if not p.exists():
        return {"entries": []}
    data = _read_json(p)
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise DatasetIOError(f'"entries" doit être une liste dans {p}')
    return {"entries": entries}


def _save_training(payload: dict[str, Any], *, path: Path | None = None) -> None:
    p = Path(path) if path else DEFAULT_TRAINING_PATH
    _atomic_write_json(p, {"entries": list(payload.get("entries") or [])})


__all__ = [
    "DEFAULT_TRAINING_PATH",
    "DatasetIOError",
    "_atomic_write_json",
    "_load_training",
    "_read_json",
    "_save_training",
]
