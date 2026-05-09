"""Scénarios mock_data : détection + classification (Ollama simulé)."""

import unittest
from unittest.mock import patch

from modules.classifier import CommentClassifier, generate_comments
from modules.detector import SignalDetector
from tests.mock_data import (
    SCENARIOS,
    assert_signal_expectations,
    generate_mock_creator,
    ollama_post_double_stub,
)


class MockScenarioPipelineTest(unittest.TestCase):
    def test_all_scenarios_signal_and_classification(self) -> None:
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                bundle = generate_mock_creator(scenario)
                sig = SignalDetector(bundle.stats).detect(now=bundle.detect_now)
                assert_signal_expectations(bundle, sig)

                stub = ollama_post_double_stub(
                    bundle.simulated_classification,
                    bundle.simulated_generate,
                )
                with patch("modules.classifier._http_post", side_effect=stub):
                    clf = CommentClassifier()
                    out = clf.classify(bundle.comments, niches=bundle.niche)
                self.assertEqual(out["type"], bundle.expected_type)

                if bundle.expect_score_above_075 and out["type"] not in ("T1", "T3a"):
                    with patch("modules.classifier._http_post", side_effect=stub):
                        gen = generate_comments(
                            out, bundle.comments, niches=bundle.niche
                        )
                    self.assertEqual(len(gen), 3)


if __name__ == "__main__":
    unittest.main()
