"""Tests des heuristiques comment_quality."""

import unittest

from modules.comment_quality import (
    REJECT_EMOJI_ONLY,
    REJECT_EMOJI_SPAM,
    REJECT_MENTION_ONLY,
    REJECT_PROMO_LINK,
    REJECT_TOO_LONG,
    REJECT_TOO_SHORT,
    assess_comment_quality,
    is_french_comment,
    is_incomplete_comment,
    is_repetitive_comment,
    strip_emojis,
)
from modules.generator_prompt import MAX_GENERATOR_OUTPUT_WORDS


class AssessCommentQualityTest(unittest.TestCase):
    def test_keeps_natural_short_comment(self) -> None:
        q = assess_comment_quality("mdr trop vrai 😂")
        self.assertTrue(q.ok)
        self.assertEqual(q.reasons, ())

    def test_rejects_emoji_only(self) -> None:
        q = assess_comment_quality("😂😂😂")
        self.assertFalse(q.ok)
        self.assertIn(REJECT_EMOJI_ONLY, q.reasons)

    def test_rejects_too_long(self) -> None:
        text = " ".join(["mot"] * (MAX_GENERATOR_OUTPUT_WORDS + 1))
        q = assess_comment_quality(text)
        self.assertFalse(q.ok)
        self.assertIn(REJECT_TOO_LONG, q.reasons)

    def test_rejects_emoji_spam(self) -> None:
        q = assess_comment_quality("JPP 😭😭😭😭😭")
        self.assertFalse(q.ok)
        self.assertIn(REJECT_EMOJI_SPAM, q.reasons)

    def test_rejects_promo(self) -> None:
        q = assess_comment_quality("Follow me on www.example.com")
        self.assertFalse(q.ok)
        self.assertIn(REJECT_PROMO_LINK, q.reasons)

    def test_rejects_mention_only(self) -> None:
        q = assess_comment_quality("@hortyunderscore")
        self.assertFalse(q.ok)
        self.assertIn(REJECT_MENTION_ONLY, q.reasons)

    def test_rejects_too_short(self) -> None:
        q = assess_comment_quality("ok")
        self.assertFalse(q.ok)
        self.assertIn(REJECT_TOO_SHORT, q.reasons)


class IsFrenchCommentTest(unittest.TestCase):
    def test_rejects_obvious_english(self) -> None:
        self.assertFalse(is_french_comment("bro is scientifically built to survive a car crash"))
        self.assertFalse(is_french_comment("how the hell did i reach french sonic"))

    def test_keeps_french(self) -> None:
        self.assertTrue(is_french_comment("mdr trop vrai"))
        self.assertTrue(is_french_comment("Pas mal hein ? C'est francais"))

    def test_keeps_emoji_reactions(self) -> None:
        self.assertTrue(is_french_comment("😂😂😂"))
        self.assertTrue(is_french_comment("💀💀"))


class IsIncompleteCommentTest(unittest.TestCase):
    def test_detects_cut_off_preposition(self) -> None:
        self.assertTrue(is_incomplete_comment("je suis un vieux gars depuis"))

    def test_detects_truncated_last_word(self) -> None:
        self.assertTrue(is_incomplete_comment("la balle perdue de fou j"))

    def test_keeps_complete_short_phrase(self) -> None:
        self.assertFalse(is_incomplete_comment("mdr trop vrai"))


class IsRepetitiveCommentTest(unittest.TestCase):
    def test_detects_bravo_loop(self) -> None:
        text = (
            "J'adore ! Bravo mec ! C'est trop fort !!!! Bravo de fou !!! "
            "Bravo gars !! Bravo toi !!!"
        )
        self.assertTrue(is_repetitive_comment(text))

    def test_keeps_ironic_single_bravo(self) -> None:
        text = "Bravo à toi je te déteste mdr très bon acting"
        self.assertFalse(is_repetitive_comment(text))


class StripEmojisTest(unittest.TestCase):
    def test_strips_trailing_emojis(self) -> None:
        self.assertEqual(strip_emojis("lets goo 🔥🔥"), "lets goo")

    def test_emoji_only_becomes_empty(self) -> None:
        self.assertEqual(strip_emojis("😂😂😂"), "")

    def test_keeps_accented_text(self) -> None:
        self.assertEqual(strip_emojis("mdr trop vrai 😂"), "mdr trop vrai")


if __name__ == "__main__":
    unittest.main()
