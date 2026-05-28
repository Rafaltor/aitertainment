"""Tests des axes nommés (projection ancres sémantiques)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules import named_axes


class ProjectOnAxisTest(unittest.TestCase):
    def test_high_pole_closer_to_one(self) -> None:
        low = [1.0, 0.0, 0.0]
        high = [0.0, 1.0, 0.0]
        creator = [0.05, 0.95, 0.0]
        score = named_axes._project_on_axis(creator, low, high)
        self.assertGreater(score, 0.7)

    def test_low_pole_closer_to_zero(self) -> None:
        low = [1.0, 0.0, 0.0]
        high = [0.0, 1.0, 0.0]
        creator = [0.95, 0.05, 0.0]
        score = named_axes._project_on_axis(creator, low, high)
        self.assertLess(score, 0.3)


class ScoreNamedAxesTest(unittest.TestCase):
    def test_scores_in_zero_one_and_not_forced_span(self) -> None:
        anchors = {
            "energy_level": {
                "low": [1.0, 0.0, 0.0],
                "high": [0.0, 1.0, 0.0],
            },
            "scripted_vs_raw": {
                "low": [0.0, 1.0, 0.0],
                "high": [1.0, 0.0, 0.0],
            },
        }
        low_energy = named_axes.score_named_axes([0.9, 0.1, 0.0], anchors)
        high_energy = named_axes.score_named_axes([0.1, 0.9, 0.0], anchors)
        self.assertGreater(high_energy["energy_level"], low_energy["energy_level"])
        # Pas de min-max par compte : les deux profils n'ont pas tous un axe à 1.0
        self.assertFalse(all(v == 1.0 for v in high_energy.values()))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in high_energy.values()))


class AxisAnchorsPersistenceTest(unittest.TestCase):
    def test_build_and_load_roundtrip(self) -> None:
        def fake_embed(text: str) -> list[float]:
            base = float(len(text) % 7)
            return [base + i * 0.01 for i in range(8)]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "axis_anchors.json"
            with patch.object(named_axes, "AXIS_ANCHORS_PATH", path):
                built = named_axes.build_axis_anchors(
                    fake_embed, model="test-model", expected_dim=8, path=path
                )
            self.assertIsNotNone(built)
            loaded = named_axes.load_axis_anchors(
                model="test-model", expected_dim=8, path=path
            )
            self.assertIsNotNone(loaded)
            self.assertIn("energy_level", loaded)

    def test_wrong_model_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "axis_anchors.json"
            path.write_text(
                json.dumps({"model": "old", "dim": 4, "anchors": {}}) + "\n",
                encoding="utf-8",
            )
            self.assertIsNone(
                named_axes.load_axis_anchors(model="new", path=path)
            )


if __name__ == "__main__":
    unittest.main()
