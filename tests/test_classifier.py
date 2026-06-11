"""Tests pour generate_comments (modèle Ollama fine-tuné, format Alpaca)."""

import json
import unittest
from unittest.mock import MagicMock, patch

from config import ORDERED_T_TYPES
from modules.classifier import (
    FORBIDDEN_GENERATOR_WORDS,
    ClassificationError,
    generate_comments,
    generate_comments_per_category,
)


def _ollama_json_response(payload_obj: object) -> MagicMock:
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json.return_value = {"response": json.dumps(payload_obj)}
    return r


class GenerateAlpacaFinetunedTest(unittest.TestCase):
    @patch("modules.classifier.config.OLLAMA_GENERATOR_MODEL", "aitertainment-generator")
    @patch("modules.classifier._http_post")
    def test_finetuned_path_uses_alpaca_prompt(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = [
            _ollama_json_response("mdr trop vrai"),
            _ollama_json_response("la ref est folle"),
            _ollama_json_response("j'ai dead"),
        ]
        out = generate_comments(
            {"type": "T2", "confidence": 0.9},
            [],
            niches=["humour"],
            t_type_profile="T2b",
            video_context={
                "caption": "test",
                "hashtags": ["humour"],
                "video_context": "",
            },
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], "mdr trop vrai")
        body = mock_post.call_args_list[0].kwargs["json_body"]
        self.assertEqual(body["model"], "aitertainment-generator")
        self.assertIn("### Instruction:", body["prompt"])
        self.assertIn("T-type commentateur: T2b", body["prompt"])
        self.assertIn("Longueur cible: court", body["prompt"])
        self.assertTrue(body["prompt"].endswith("### Response:\n"))
        opts = body["options"]
        self.assertEqual(opts["repeat_penalty"], 1.2)
        self.assertEqual(opts["num_predict"], 24)
        long_body = mock_post.call_args_list[1].kwargs["json_body"]
        self.assertIn("Longueur cible: développé", long_body["prompt"])
        self.assertEqual(long_body["options"]["num_predict"], 72)
        self.assertEqual(opts["temperature"], 0.65)
        self.assertEqual(long_body["options"]["temperature"], 0.5)

    @patch("modules.classifier.config.OLLAMA_GENERATOR_MODEL", "aitertainment-generator")
    @patch("modules.classifier._http_post")
    def test_finetuned_rejects_repetitive_bravo_loop(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = [
            _ollama_json_response("bravo bravo bravo bravo bravo bravo"),
            _ollama_json_response("mdr trop vrai"),
            _ollama_json_response("la ref est folle"),
            _ollama_json_response("j'ai dead"),
        ]
        out = generate_comments(
            {"type": "T2", "confidence": 0.9},
            [],
            niches=["humour"],
            t_type_profile="T2b",
            video_context={
                "caption": "test",
                "hashtags": ["humour"],
                "video_context": "",
            },
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], "mdr trop vrai")
        self.assertEqual(mock_post.call_count, 4)

    @patch("modules.classifier.config.OLLAMA_GENERATOR_MODEL", "aitertainment-generator")
    @patch("modules.classifier._http_post")
    def test_video_context_appears_in_alpaca_prompt(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = [
            _ollama_json_response("réf au vendredi mdr"),
            _ollama_json_response("trop vrai"),
            _ollama_json_response("j'ai dead"),
        ]
        generate_comments(
            {"type": "T2", "confidence": 0.9},
            [],
            niches=["humour"],
            t_type_profile="T2b",
            video_context={
                "caption": "drop",
                "hashtags": ["streetwear"],
                "video_context": "Le drop est annoncé vendredi à midi en cuisine.",
            },
        )
        prompt = mock_post.call_args_list[0].kwargs["json_body"]["prompt"]
        self.assertIn(
            "Contexte vidéo: Le drop est annoncé vendredi à midi en cuisine.",
            prompt,
        )

    @patch("modules.classifier.config.OLLAMA_GENERATOR_MODEL", "")
    def test_missing_finetuned_model_raises(self) -> None:
        with self.assertRaises(ClassificationError):
            generate_comments({"type": "T2"}, [], niches=["humour"])


class GeneratePerCategoryTest(unittest.TestCase):
    @patch("modules.classifier.config.OLLAMA_GENERATOR_MODEL", "aitertainment-generator")
    @patch("modules.classifier._http_post")
    def test_one_comment_per_t_type(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = [
            _ollama_json_response("admiration"),
            _ollama_json_response("vanne"),
            _ollama_json_response("punchline longue"),
            _ollama_json_response("haine"),
            _ollama_json_response("ironie"),
            _ollama_json_response("mème"),
            _ollama_json_response("ratio"),
        ]
        out = generate_comments_per_category(
            niches=["humour"],
            video_context={"caption": "test", "hashtags": ["humour"]},
        )
        self.assertEqual(set(out.keys()), set(ORDERED_T_TYPES))
        self.assertEqual(out["T1"], "admiration")
        self.assertEqual(mock_post.call_count, len(ORDERED_T_TYPES))
        prompts = [
            call.kwargs["json_body"]["prompt"] for call in mock_post.call_args_list
        ]
        for t_type, prompt in zip(ORDERED_T_TYPES, prompts):
            self.assertIn(f"T-type commentateur: {t_type}", prompt)


class ForbiddenWordsInfraTest(unittest.TestCase):
    """Notre infra ne doit pas introduire elle-même de mots interdits."""

    @patch("modules.classifier.config.OLLAMA_GENERATOR_MODEL", "aitertainment-generator")
    @patch("modules.classifier._http_post")
    def test_static_outputs_contain_no_forbidden_word(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = [
            _ollama_json_response("mdr trop vrai"),
            _ollama_json_response("j'en peux plus"),
            _ollama_json_response("le passage 0:08"),
        ]
        out = generate_comments(
            {"type": "T2", "confidence": 0.9}, [], niches=["humour"]
        )
        joined = " ".join(out).lower()
        for word in FORBIDDEN_GENERATOR_WORDS:
            self.assertNotIn(word.lower(), joined)


if __name__ == "__main__":
    unittest.main()
