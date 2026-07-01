"""Tests pour ``modules.ig1_spam_generator``."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules import ig1_spam_generator as gen


class Ig1SpamGeneratorTest(unittest.TestCase):
    def test_publishable_base_rejects_incomplete(self) -> None:
        self.assertFalse(
            gen.is_publishable_base_comment(
                "Pov: t'as juste pas préparé correctement ton oral mais tu"
            )
        )

    def test_publishable_base_accepts_short_comment(self) -> None:
        self.assertTrue(gen.is_publishable_base_comment("le prof est mort"))

    def test_substitutable_rejects_adverb(self) -> None:
        self.assertFalse(gen._is_substitutable_word("correctement"))
        self.assertTrue(gen._is_substitutable_word("prof"))

    def test_valid_substitution_requires_noun_or_verb_slot(self) -> None:
        self.assertTrue(
            gen._is_valid_substitution_weave(
                "le prof est mort", "le lowtaper67 est mort", "lowtaper67"
            )
        )
        self.assertFalse(
            gen._is_valid_substitution_weave(
                "t'as pas préparé correctement ton oral",
                "t'as pas préparé lowtaper67 ton oral",
                "lowtaper67",
            )
        )

    def test_fallback_picks_noun_not_adverb(self) -> None:
        out = gen._fallback_substitute_word(
            "t'as pas préparé correctement l'oral", "lowtaper67"
        )
        self.assertIn("lowtaper67", out.lower())
        self.assertNotIn("préparé", out.lower())
        self.assertIn("correctement", out)

    def test_weave_uses_ollama(self) -> None:
        with patch("modules.ig1_spam_generator.call_ollama_text") as mock:
            mock.return_value = "le lowtaper67 est mort"
            out = gen.weave_keyword_into_comment("le prof est mort", "lowtaper67")
        self.assertEqual(out, "le lowtaper67 est mort")

    @patch("modules.ig1_spam_generator._fuse_ig1_context", return_value="")
    @patch(
        "modules.ig1_spam_generator.weave_keyword_into_comment",
        return_value="le lowtaper67 est mort",
    )
    @patch(
        "modules.ig1_spam_generator._generate_base_comment",
        return_value="le prof est mort",
    )
    @patch("modules.ig1_spam_generator.config.OLLAMA_GENERATOR_MODEL", "gen-model")
    def test_single_base_single_weave(
        self,
        mock_base: unittest.mock.MagicMock,
        mock_weave: unittest.mock.MagicMock,
        mock_fuse: unittest.mock.MagicMock,
    ) -> None:
        pairs = gen.generate_ig1_spam_comments(caption="c", media_id="X")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0], ("le prof est mort", "le lowtaper67 est mort"))
        mock_weave.assert_called_once()

    @patch("modules.ig1_spam_generator.generate_ig1_spam_comments")
    @patch("modules.ig1_spam_generator.config.OLLAMA_GENERATOR_MODEL", "gen-model")
    def test_generate_returns_woven(self, mock_pairs: unittest.mock.MagicMock) -> None:
        mock_pairs.return_value = [("b", "le lowtaper67 est mort")]
        out = gen.generate_ig1_spam_comment(caption="c")
        self.assertEqual(out, "le lowtaper67 est mort")


if __name__ == "__main__":
    unittest.main()
