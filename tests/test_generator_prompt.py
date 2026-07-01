"""Tests pour ``modules.generator_prompt``."""

from __future__ import annotations

import unittest

from modules.generator_prompt import (
    build_alpaca_prompt,
    build_generator_input_block,
    build_generator_instruction,
    comment_length_bucket,
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
        )
        self.assertIn("T-type commentateur: T2b", block)
        self.assertIn("Niches: humour, sketch", block)
        self.assertIn("Caption: test caption", block)

    def test_alpaca_prompt_ends_with_response_header(self) -> None:
        prompt = build_alpaca_prompt("T-type commentateur: T2\nNiches: humour")
        self.assertTrue(prompt.startswith("### Instruction:"))
        self.assertIn("### Input:\nT-type commentateur: T2", prompt)
        self.assertTrue(prompt.endswith("### Response:\n"))

    def test_normalize_strips_emoji_and_mentions_short(self) -> None:
        out = normalize_generator_output(
            "mdr trop vrai 🤣🤣 @someone extra words here",
            length_bucket="short",
        )
        self.assertNotIn("@", out)
        self.assertNotIn("🤣", out)
        self.assertLessEqual(len(out.split()), 10)

    def test_normalize_keeps_long_comment(self) -> None:
        words = ["mot"] * 25
        raw = " ".join(words)
        out = normalize_generator_output(raw, length_bucket="long")
        self.assertEqual(len(out.split()), 25)

    def test_length_bucket_and_instruction(self) -> None:
        self.assertEqual(comment_length_bucket("mdr trop vrai"), "short")
        self.assertEqual(comment_length_bucket(" ".join(["mot"] * 12)), "long")
        self.assertIn("3 à 10 mots", build_generator_instruction("short"))
        self.assertIn("11 à 40 mots", build_generator_instruction("long"))
        block = build_generator_input_block(
            t_type_profile="T2",
            niches=["humour"],
            length_bucket="long",
        )
        self.assertIn("Longueur cible: développé", block)

    def test_video_context_in_input_block_and_alpaca_prompt(self) -> None:
        block = build_generator_input_block(
            t_type_profile="T2b",
            niches=["humour"],
            caption="reel test",
            video_context="Sketch où le drop est annoncé vendredi.",
        )
        self.assertIn(
            "Contexte vidéo: Sketch où le drop est annoncé vendredi.", block
        )
        prompt = build_alpaca_prompt(block)
        self.assertIn("Contexte vidéo: Sketch où le drop est annoncé vendredi.", prompt)
        self.assertIn("Transcript et/ou Visuel sont fournis", prompt)

    def test_video_context_truncated_at_600_chars(self) -> None:
        long_vc = "z" * 700
        block = build_generator_input_block(
            t_type_profile="T2",
            niches=["humour"],
            video_context=long_vc,
        )
        self.assertIn(f"Contexte vidéo: {'z' * 600}", block)
        self.assertNotIn("z" * 601, block)

    def test_creator_and_reel_in_input_block(self) -> None:
        block = build_generator_input_block(
            t_type_profile="T2b",
            niches=["humour"],
            creator_username="compte_a",
            reel_id="ABC123",
        )
        self.assertIn("Creator: @compte_a", block)
        self.assertIn("Reel: ABC123", block)



if __name__ == "__main__":
    unittest.main()
