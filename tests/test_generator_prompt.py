"""Tests pour ``modules.generator_prompt``."""

from __future__ import annotations

import unittest

from modules.generator_prompt import (
    build_alpaca_prompt,
    build_generator_input_block,
)


class GeneratorPromptTest(unittest.TestCase):
    def test_input_block_matches_training_format(self) -> None:
        block = build_generator_input_block(
            t_type_profile="T2b",
            niches=["humour", "sketch"],
            caption="test caption",
            hashtags=["f1", "monaco"],
            audio_id="AUD1",
            named_axes={"scripted_vs_raw": 0.11, "energy_level": 0.44},
        )
        self.assertIn("T-type commentateur: T2b", block)
        self.assertIn("Niches: humour, sketch", block)
        self.assertIn("Profil créateur:", block)
        self.assertIn("scripted_vs_raw=0.11", block)
        self.assertIn("Caption: test caption", block)

    def test_alpaca_prompt_ends_with_response_header(self) -> None:
        prompt = build_alpaca_prompt("T-type commentateur: T2\nNiches: humour")
        self.assertTrue(prompt.startswith("### Instruction:"))
        self.assertIn("### Input:\nT-type commentateur: T2", prompt)
        self.assertTrue(prompt.endswith("### Response:\n"))


if __name__ == "__main__":
    unittest.main()
