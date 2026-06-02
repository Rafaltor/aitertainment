"""Tests pour ``modules.generator_prompt``."""

from __future__ import annotations

import unittest

from modules.generator_prompt import (
    build_alpaca_prompt,
    build_generator_input_block,
    normalize_generator_output,
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

    def test_normalize_strips_emoji_and_mentions(self) -> None:
        out = normalize_generator_output("mdr trop vrai 🤣🤣 @someone extra words here")
        self.assertNotIn("@", out)
        self.assertNotIn("🤣", out)
        self.assertLessEqual(len(out.split()), 10)

    def test_transcript_in_input_block_and_alpaca_prompt(self) -> None:
        block = build_generator_input_block(
            t_type_profile="T2b",
            niches=["humour"],
            caption="reel test",
            transcript="il dit que le drop est vendredi",
        )
        self.assertIn("Transcript: il dit que le drop est vendredi", block)
        prompt = build_alpaca_prompt(block)
        self.assertIn("Transcript: il dit que le drop est vendredi", prompt)
        self.assertIn("Transcript est fourni", prompt)

    def test_transcript_truncated_at_500_chars(self) -> None:
        long_tr = "x" * 600
        block = build_generator_input_block(
            t_type_profile="T2",
            niches=["humour"],
            transcript=long_tr,
        )
        self.assertIn(f"Transcript: {'x' * 500}", block)
        self.assertNotIn("x" * 501, block)

    def test_creator_and_reel_in_input_block(self) -> None:
        block = build_generator_input_block(
            t_type_profile="T2b",
            niches=["humour"],
            creator_username="compte_a",
            reel_id="ABC123",
        )
        self.assertIn("Creator: @compte_a", block)
        self.assertIn("Reel: ABC123", block)

    def test_visual_description_in_input_block(self) -> None:
        block = build_generator_input_block(
            t_type_profile="T2b",
            niches=["humour"],
            visual_description="Un homme fait tomber un gâteau.",
        )
        self.assertIn("Visuel: Un homme fait tomber un gâteau.", block)
        prompt = build_alpaca_prompt(block)
        self.assertIn("Visuel est fourni", prompt)

    def test_visual_description_truncated_at_300_chars(self) -> None:
        long_vis = "y" * 400
        block = build_generator_input_block(
            t_type_profile="T2",
            niches=["humour"],
            visual_description=long_vis,
        )
        self.assertIn(f"Visuel: {'y' * 300}", block)
        self.assertNotIn("y" * 301, block)


if __name__ == "__main__":
    unittest.main()
