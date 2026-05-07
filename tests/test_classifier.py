"""Tests pour CommentClassifier et generate_comments (Ollama)."""

import json
import unittest
from unittest.mock import MagicMock, patch

from modules.classifier import (
    ClassificationError,
    CommentClassifier,
    _parse_json_from_response,
    generate_comments,
)


class ParseJsonFromResponseTest(unittest.TestCase):
    def test_plain_object(self) -> None:
        raw = 'Voici le résultat {"type": "T1", "x": 1} fin'
        obj = _parse_json_from_response(raw)
        self.assertEqual(obj["type"], "T1")

    def test_fenced_json(self) -> None:
        raw = """```json
{"type": "T3b", "confidence": 0.9}
```"""
        obj = _parse_json_from_response(raw)
        self.assertEqual(obj["type"], "T3b")


def _ollama_json_response(payload_obj: dict) -> MagicMock:
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json.return_value = {"response": json.dumps(payload_obj)}
    return r


class CommentClassifierTest(unittest.TestCase):
    def test_classify_success(self) -> None:
        payload = {
            "type": "T2",
            "confidence": 0.82,
            "patterns": ["W", "X", "Y", "Z"],
            "tone": "Humour de niche.",
            "brand_risk": "low",
        }
        session = MagicMock()
        session.post.return_value = _ollama_json_response(payload)
        cc = CommentClassifier(client=session)
        out = cc.classify(["a", "b"], niche="streetwear")
        self.assertEqual(out["type"], "T2")
        self.assertAlmostEqual(out["confidence"], 0.82)
        self.assertEqual(out["patterns"], ["W", "X", "Y"])
        self.assertEqual(out["tone"], "Humour de niche.")
        self.assertEqual(out["brand_risk"], "low")
        session.post.assert_called_once()
        body = session.post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "qwen2.5:7b")
        self.assertEqual(body["stream"], False)
        self.assertIn("streetwear", body["prompt"])
        self.assertIn("T1", body["system"])

    def test_classify_invalid_json(self) -> None:
        session = MagicMock()
        session.post.return_value = _ollama_json_response_raw("pas du json")
        cc = CommentClassifier(client=session)
        with self.assertRaises(ClassificationError) as ctx:
            cc.classify(["x"], niche="")
        self.assertIsNotNone(ctx.exception.raw_text)

    def test_classify_invalid_type(self) -> None:
        session = MagicMock()
        session.post.return_value = _ollama_json_response(
            {
                "type": "TX",
                "confidence": 0.5,
                "patterns": [],
                "tone": "x",
                "brand_risk": "low",
            }
        )
        cc = CommentClassifier(client=session)
        with self.assertRaises(ClassificationError):
            cc.classify(["x"], niche="mode")

    def test_empty_comments(self) -> None:
        cc = CommentClassifier(client=MagicMock())
        with self.assertRaises(ValueError):
            cc.classify([], niche="x")


def _ollama_json_response_raw(text: str) -> MagicMock:
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json.return_value = {"response": text}
    return r


class GenerateCommentsTest(unittest.TestCase):
    @patch("modules.classifier._http_post")
    def test_generate_comments_success(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ollama_json_response(
            {"comments": ["un", "deux", "trois", "quatre"]}
        )
        cls = {"type": "T2", "confidence": 0.9, "patterns": [], "tone": "x", "brand_risk": "low"}
        out = generate_comments(cls, ["a", "b"], niche="streetwear")
        self.assertEqual(out, ["un", "deux", "trois"])
        mock_post.assert_called_once()
        body = mock_post.call_args.kwargs["json_body"]
        self.assertEqual(body["model"], "qwen2.5:7b")
        self.assertEqual(body["stream"], False)


if __name__ == "__main__":
    unittest.main()
