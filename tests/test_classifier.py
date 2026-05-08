"""Tests pour CommentClassifier et generate_comments (Ollama)."""

import json
import unittest
from unittest.mock import MagicMock, patch

from modules.classifier import (
    FORBIDDEN_GENERATOR_WORDS,
    GENERATE_COMMENTS_SYSTEM,
    GENERATE_FALLBACK_TTYPE,
    GENERATE_PROMPTS_BY_TTYPE,
    ClassificationError,
    CommentClassifier,
    _format_comments_sample,
    _normalize_video_context,
    _parse_json_from_response,
    _resolve_ttype_prompt,
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


class GenerateCommentsSystemPromptTest(unittest.TestCase):
    """Le system prompt global doit imposer les règles absolues du brief."""

    def test_system_prompt_contains_hard_rules(self) -> None:
        sp = GENERATE_COMMENTS_SYSTEM
        # Ton & posture
        self.assertIn("utilisateur lambda", sp)
        self.assertIn("RÈGLES ABSOLUES", sp)
        # Règles structurelles
        self.assertIn("Maximum 8 mots", sp)
        self.assertIn("Minuscule en début", sp)
        self.assertIn("0 ou 1 emoji", sp)
        self.assertIn("Pas de point final", sp)
        # Sortie JSON contraignante
        self.assertIn('"comments"', sp)

    def test_system_prompt_lists_all_forbidden_words(self) -> None:
        sp = GENERATE_COMMENTS_SYSTEM.lower()
        for word in FORBIDDEN_GENERATOR_WORDS:
            self.assertIn(word.lower(), sp, f"mot interdit absent du prompt : {word!r}")


class ResolveTtypePromptTest(unittest.TestCase):
    def test_known_ttypes_return_their_template(self) -> None:
        for t in ("T2", "T2b", "T3b", "T4", "T5"):
            resolved, tpl = _resolve_ttype_prompt(t)
            self.assertEqual(resolved, t)
            self.assertEqual(tpl, GENERATE_PROMPTS_BY_TTYPE[t])

    def test_unknown_ttype_falls_back_to_t2(self) -> None:
        for unknown in ("T1", "T3a", "TX", None, "", "  "):
            resolved, tpl = _resolve_ttype_prompt(unknown)
            self.assertEqual(resolved, "T2")
            self.assertEqual(resolved, GENERATE_FALLBACK_TTYPE)
            self.assertIn("inside joke", tpl)  # signature du prompt T2


class NormalizeVideoContextTest(unittest.TestCase):
    def test_full_context_normalizes_hashtag_list(self) -> None:
        out = _normalize_video_context(
            {"caption": "Top moment", "hashtags": ["F1", "monaco"]},
            niche="humour",
        )
        self.assertEqual(out["niche"], "humour")
        self.assertEqual(out["caption"], "Top moment")
        self.assertEqual(out["hashtags"], "#F1 #monaco")

    def test_none_inputs_yield_safe_defaults(self) -> None:
        out = _normalize_video_context(None, niche="")
        self.assertEqual(out["caption"], "(vide)")
        self.assertEqual(out["hashtags"], "(aucun)")
        self.assertEqual(out["niche"], "(non précisée)")

    def test_string_hashtags_passed_through(self) -> None:
        out = _normalize_video_context(
            {"caption": "x", "hashtags": "#manuel #déjà_formaté"}, niche="x"
        )
        self.assertEqual(out["hashtags"], "#manuel #déjà_formaté")


class FormatCommentsSampleTest(unittest.TestCase):
    def test_caps_at_twenty_lines(self) -> None:
        sample = [f"comment_{i}" for i in range(50)]
        text = _format_comments_sample(sample)
        self.assertEqual(text.count("\n"), 19)  # 20 lignes → 19 séparateurs
        self.assertIn("comment_0", text)
        self.assertIn("comment_19", text)
        self.assertNotIn("comment_20", text)

    def test_empty_yields_placeholder(self) -> None:
        self.assertEqual(_format_comments_sample([]), "(aucun commentaire disponible)")
        self.assertEqual(_format_comments_sample(["", "  ", None]), "(aucun commentaire disponible)")


class GenerateCommentsTest(unittest.TestCase):
    """Tests fonctionnels de bout en bout (Ollama mocké)."""

    @staticmethod
    def _capture_prompt(mock_post: MagicMock) -> tuple[str, str]:
        """Retourne ``(system, user_prompt)`` envoyés à Ollama."""
        body = mock_post.call_args.kwargs["json_body"]
        return body["system"], body["prompt"]

    @patch("modules.classifier._http_post")
    def test_generate_comments_success_returns_three(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ollama_json_response(
            {"comments": ["mdr", "ouais c'est ça", "trop vrai", "extra"]}
        )
        cls = {"type": "T2", "confidence": 0.9, "patterns": [], "tone": "x", "brand_risk": "low"}
        out = generate_comments(cls, ["a", "b"], niche="streetwear")
        self.assertEqual(out, ["mdr", "ouais c'est ça", "trop vrai"])
        mock_post.assert_called_once()
        body = mock_post.call_args.kwargs["json_body"]
        self.assertEqual(body["model"], "qwen2.5:7b")
        self.assertEqual(body["stream"], False)

    @patch("modules.classifier._http_post")
    def test_t2_uses_t2_template(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ollama_json_response(
            {"comments": ["x", "y", "z"]}
        )
        cls = {"type": "T2", "confidence": 0.9, "patterns": [], "tone": "x", "brand_risk": "low"}
        generate_comments(cls, [], niche="humour", video_context={
            "caption": "moment culte F1",
            "hashtags": ["F1", "monaco"],
        })
        system, prompt = self._capture_prompt(mock_post)
        # System global utilisé
        self.assertEqual(system, GENERATE_COMMENTS_SYSTEM.strip())
        # Template T2 (signature unique : "inside joke")
        self.assertIn("inside joke", prompt)
        self.assertIn("humour", prompt)
        self.assertIn("moment culte F1", prompt)
        self.assertIn("#F1 #monaco", prompt)

    @patch("modules.classifier._http_post")
    def test_t3b_uses_invisible_second_degree_template(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ollama_json_response(
            {"comments": ["bravo le débutant", "très pro pour 2026", "on sent l'expérience"]}
        )
        cls = {"type": "T3b", "confidence": 0.85, "patterns": [], "tone": "x", "brand_risk": "medium"}
        out = generate_comments(cls, ["test"], niche="cuisine", video_context={
            "caption": "ma première fois", "hashtags": []
        })
        _, prompt = self._capture_prompt(mock_post)
        # Le prompt T3b doit décrire le faux éloge / second degré.
        self.assertIn("second degré", prompt)
        self.assertIn("éloge", prompt)
        self.assertIn("liker", prompt)  # "doit pouvoir liker sans comprendre"
        # Les 3 commentaires retournés sont bien dans la forme attendue
        # (l'assertion structurelle ici = le mock délivre 3 strings,
        # peu importe leur contenu — la conformité au registre est la
        # responsabilité du modèle, pas du test).
        self.assertEqual(len(out), 3)
        self.assertTrue(all(isinstance(c, str) and c for c in out))

    @patch("modules.classifier._http_post")
    def test_unknown_ttype_falls_back_to_t2(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ollama_json_response(
            {"comments": ["a", "b", "c"]}
        )
        cls = {"type": "TX", "confidence": 0.5, "patterns": [], "tone": "?", "brand_risk": "low"}
        generate_comments(cls, [], niche="x")
        _, prompt = self._capture_prompt(mock_post)
        # On doit retomber sur T2 — signature : "inside joke".
        self.assertIn("inside joke", prompt)

    @patch("modules.classifier._http_post")
    def test_video_context_injected_into_user_prompt(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ollama_json_response(
            {"comments": ["a", "b", "c"]}
        )
        cls = {"type": "T4", "confidence": 0.9, "patterns": [], "tone": "x", "brand_risk": "low"}
        generate_comments(
            cls, [], niche="gaming",
            video_context={
                "caption": "GG la team",
                "hashtags": ["league", "esport"],
                "audio_id": "AUD123",  # ne doit pas fuiter dans le prompt user
            },
        )
        _, prompt = self._capture_prompt(mock_post)
        self.assertIn("gaming", prompt)
        self.assertIn("GG la team", prompt)
        self.assertIn("#league #esport", prompt)
        # ``audio_id`` n'a pas de slot dans les templates T-type — c'est
        # voulu : ce n'est pas un signal de registre, juste un identifiant
        # de classe. On vérifie qu'il n'apparaît PAS dans le prompt user.
        self.assertNotIn("AUD123", prompt)

    @patch("modules.classifier._http_post")
    def test_comments_sample_capped_at_twenty(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ollama_json_response(
            {"comments": ["a", "b", "c"]}
        )
        cls = {"type": "T2", "confidence": 0.9, "patterns": [], "tone": "x", "brand_risk": "low"}
        sample = [f"line_{i}" for i in range(50)]
        generate_comments(cls, sample, niche="x")
        _, prompt = self._capture_prompt(mock_post)
        self.assertIn("line_0", prompt)
        self.assertIn("line_19", prompt)
        self.assertNotIn("line_20", prompt)

    @patch("modules.classifier._http_post")
    def test_mock_response_respects_forbidden_words_check(self, mock_post: MagicMock) -> None:
        """Le mock simule un modèle bien dressé : 0 mot interdit dans la réponse.

        Ce test ne contraint pas le **vrai** modèle (responsabilité du LLM,
        pas du code Python), mais il vérifie que notre infra **n'introduit pas**
        elle-même de mots interdits (ex : message d'erreur, fallback statique).
        """
        clean_response = {"comments": ["mdr trop vrai", "j'en peux plus", "le passage 0:08"]}
        mock_post.return_value = _ollama_json_response(clean_response)
        cls = {"type": "T2", "confidence": 0.9, "patterns": [], "tone": "x", "brand_risk": "low"}
        out = generate_comments(cls, [], niche="humour")
        joined = " ".join(out).lower()
        for word in FORBIDDEN_GENERATOR_WORDS:
            self.assertNotIn(word.lower(), joined)


if __name__ == "__main__":
    unittest.main()
