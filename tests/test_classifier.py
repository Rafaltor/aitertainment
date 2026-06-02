"""Tests pour generate_comments (modèle Ollama fine-tuné, format Alpaca)."""

import json
import unittest
from unittest.mock import MagicMock, patch

from modules.classifier import (
    FORBIDDEN_GENERATOR_WORDS,
    ClassificationError,
    generate_comments,
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
                "transcript": "",
                "visual_description": "",
            },
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], "mdr trop vrai")
        body = mock_post.call_args_list[0].kwargs["json_body"]
        self.assertEqual(body["model"], "aitertainment-generator")
        self.assertIn("### Instruction:", body["prompt"])
        self.assertIn("T-type commentateur: T2b", body["prompt"])
        self.assertTrue(body["prompt"].endswith("### Response:\n"))
        opts = body["options"]
        self.assertEqual(opts["repeat_penalty"], 1.15)
        self.assertEqual(opts["num_predict"], 28)
        self.assertEqual(opts["temperature"], 0.7)

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
                "transcript": "",
                "visual_description": "",
            },
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], "mdr trop vrai")
        self.assertEqual(mock_post.call_count, 4)

    @patch("modules.classifier.config.OLLAMA_GENERATOR_MODEL", "aitertainment-generator")
    @patch("modules.classifier._http_post")
    def test_transcript_appears_in_alpaca_prompt(self, mock_post: MagicMock) -> None:
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
                "transcript": "le drop est vendredi à midi",
            },
        )
        prompt = mock_post.call_args_list[0].kwargs["json_body"]["prompt"]
        self.assertIn("Transcript: le drop est vendredi à midi", prompt)

    @patch("modules.classifier.config.OLLAMA_GENERATOR_MODEL", "aitertainment-generator")
    @patch("modules.classifier._http_post")
    def test_visual_description_appears_in_alpaca_prompt(
        self, mock_post: MagicMock
    ) -> None:
        mock_post.side_effect = [
            _ollama_json_response("la cuisine mdr"),
            _ollama_json_response("trop vrai"),
            _ollama_json_response("j'ai dead"),
        ]
        generate_comments(
            {"type": "T2", "confidence": 0.9},
            [],
            niches=["humour"],
            t_type_profile="T2b",
            video_context={
                "caption": "sketch",
                "hashtags": ["humour"],
                "transcript": "",
                "visual_description": "Deux potes dans une cuisine qui rigolent.",
            },
        )
        prompt = mock_post.call_args_list[0].kwargs["json_body"]["prompt"]
        self.assertIn(
            "Visuel: Deux potes dans une cuisine qui rigolent.", prompt
        )

    @patch("modules.classifier.config.OLLAMA_GENERATOR_MODEL", "")
    def test_missing_finetuned_model_raises(self) -> None:
        with self.assertRaises(ClassificationError):
            generate_comments({"type": "T2"}, [], niches=["humour"])


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
