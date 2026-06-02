"""Écriture JSON atomique avec verrou fichier optionnel (daemons + scripts)."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover — Windows sans fcntl
    fcntl = None  # type: ignore[assignment]


class JsonLockTimeout(TimeoutError):
    """Impossible d'acquérir le verrou dans le délai imparti."""


@contextmanager
def json_lock(path: Path | str, *, timeout_s: float = 30.0) -> Iterator[None]:
    """Verrou exclusif sur ``path.lock`` (best-effort, Unix)."""
    path = Path(path)
    if fcntl is None:
        yield
        return

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    acquired = False
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise JsonLockTimeout(f"verrou occupé : {lock_path}") from None
                time.sleep(0.05)
        yield
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def atomic_write_json(
    path: Path | str,
    data: Any,
    *,
    indent: int = 2,
    use_lock: bool = True,
) -> None:
    """Écrit ``path`` via ``.tmp`` + ``replace``, optionnellement sous verrou."""
    path = Path(path)
    if use_lock:
        with json_lock(path):
            _write_unlocked(path, data, indent=indent)
    else:
        _write_unlocked(path, data, indent=indent)


def _write_unlocked(path: Path, data: Any, *, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
