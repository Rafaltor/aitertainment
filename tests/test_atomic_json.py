"""Tests modules.atomic_json — écriture atomique + verrou."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.atomic_json import JsonLockTimeout, atomic_write_json, json_lock


class AtomicJsonTest(unittest.TestCase):
    def test_atomic_write_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            atomic_write_json(path, {"a": 1}, use_lock=False)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data, {"a": 1})

    def test_lock_timeout_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "db.json"
            with patch("modules.atomic_json.fcntl") as mock_fcntl:
                mock_fcntl.LOCK_EX = 2
                mock_fcntl.LOCK_NB = 4
                mock_fcntl.LOCK_UN = 8
                mock_fcntl.flock.side_effect = BlockingIOError
                with self.assertRaises((JsonLockTimeout, TimeoutError)):
                    with json_lock(path, timeout_s=0.1):
                        pass


if __name__ == "__main__":
    unittest.main()
