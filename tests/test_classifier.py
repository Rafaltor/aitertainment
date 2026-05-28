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
    _format_niches,
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
    def test_classify_success_with_niches_list(self) -> None:
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
        # Schéma 2026-05 : ``niches`` (liste).
        out = cc.classify(["a", "b"], niches=["streetwear", "lifestyle"])
        self.assertEqual(out["type"], "T2")
        self.assertAlmostEqual(out["confidence"], 0.82)
        self.assertEqual(out["patterns"], ["W", "X", "Y"])
        self.assertEqual(out["tone"], "Humour de niche.")
        self.assertEqual(out["brand_risk"], "low")
        session.post.assert_called_once()
        body = session.post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "qwen2.5:7b")
        self.assertEqual(body["stream"], False)
        # Le prompt user contient ``Niches:`` (label nouveau schéma) + la
        # liste jointe par ", ".
        self.assertIn("Niches:", body["prompt"])
        self.assertIn("streetwear, lifestyle", body["prompt"])
        self.assertIn("T1", body["system"])

    def test_classify_accepts_niche_string_for_backward_compat(self) -> None:
        """Rétro-compat : ``niches`` peut être passé en string."""
        session = MagicMock()
        session.post.return_value = _ollama_json_response(
            {"type": "T2", "confidence": 0.5, "patterns": [], "tone": "x", "brand_risk": "low"}
        )
        cc = CommentClassifier(client=session)
        cc.classify(["a"], niches="streetwear")
        body = session.post.call_args.kwargs["json"]
        self.assertIn("Niches:", body["prompt"])
        self.assertIn("streetwear", body["prompt"])

    def test_classify_invalid_json(self) -> None:
        session = MagicMock()
        session.post.return_value = _ollama_json_response_raw("pas du json")
        cc = CommentClassifier(client=session)
        with self.assertRaises(ClassificationError) as ctx:
            cc.classify(["x"], niches="")
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
            cc.classify(["x"], niches=["mode"])

    def test_empty_comments(self) -> None:
        cc = CommentClassifier(client=MagicMock())
        with self.assertRaises(ValueError):
            cc.classify([], niches="x")


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


class FormatNichesTest(unittest.TestCase):
    """``_format_niches`` joint la liste / passe la string / défaut sûr."""

    def test_list_joined_with_commas(self) -> None:
        self.assertEqual(
            _format_niches(["humour", "sketch", "imitation"]),
            "humour, sketch, imitation",
        )

    def test_string_passed_through_after_strip(self) -> None:
        self.assertEqual(_format_niches("  humour, sketch  "), "humour, sketch")

    def test_none_falls_back(self) -> None:
        self.assertEqual(_format_niches(None), "(non précisée)")

    def test_empty_list_falls_back(self) -> None:
        self.assertEqual(_format_niches([]), "(non précisée)")
        self.assertEqual(_format_niches(["", "  "]), "(non précisée)")

    def test_filters_non_string_items(self) -> None:
        self.assertEqual(_format_niches(["humour", None, 42, "sketch"]), "humour, sketch")


class NormalizeVideoContextTest(unittest.TestCase):
    def test_full_context_normalizes_hashtag_list_and_niches_list(self) -> None:
        out = _normalize_video_context(
            {"caption": "Top moment", "hashtags": ["F1", "monaco"]},
            niches=["humour", "sketch"],
        )
        # Schéma 2026-05 : la clé est ``niches`` et contient la string formatée.
        self.assertEqual(out["niches"], "humour, sketch")
        self.assertEqual(out["caption"], "Top moment")
        self.assertEqual(out["hashtags"], "#F1 #monaco")

    def test_none_inputs_yield_safe_defaults(self) -> None:
        out = _normalize_video_context(None, niches=None)
        self.assertEqual(out["caption"], "(vide)")
        self.assertEqual(out["hashtags"], "(aucun)")
        self.assertEqual(out["niches"], "(non précisée)")

    def test_string_hashtags_and_niche_string_passed_through(self) -> None:
        out = _normalize_video_context(
            {"caption": "x", "hashtags": "#manuel #déjà_formaté"},
            niches="streetwear",
        )
        self.assertEqual(out["hashtags"], "#manuel #déjà_formaté")
        self.assertEqual(out["niches"], "streetwear")


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
            video_context={"caption": "test", "hashtags": ["humour"]},
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], "mdr trop vrai")
        body = mock_post.call_args_list[0].kwargs["json_body"]
        self.assertEqual(body["model"], "aitertainment-generator")
        self.assertIn("### Instruction:", body["prompt"])
        self.assertIn("T-type commentateur: T2b", body["prompt"])
        self.assertTrue(body["prompt"].endswith("### Response:\n"))


class GenerateWithNamedAxesTest(unittest.TestCase):
    @staticmethod
    def _capture_prompt(mock_post: MagicMock) -> str:
        return mock_post.call_args.kwargs["json_body"]["prompt"]

    @patch("modules.classifier._http_post")
    def test_non_empty_named_axes_adds_creator_profile_block(
        self, mock_post: MagicMock
    ) -> None:
        mock_post.return_value = _ollama_json_response(
            {"comments": ["a", "b", "c"]}
        )
        cls = {"type": "T2", "confidence": 0.9, "patterns": [], "tone": "x", "brand_risk": "low"}
        generate_comments(
            cls,
            [],
            niches=["humour"],
            t_type_profile="T2",
            named_axes={"scripted_vs_raw": 0.5, "solo_vs_collab": 0.6},
        )
        prompt = self._capture_prompt(mock_post)
        self.assertIn("Profil créateur:", prompt)
        self.assertIn("scripted_vs_raw=0.50", prompt)

    @patch("modules.classifier._http_post")
    def test_empty_named_axes_leaves_prompt_unchanged(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ollama_json_response(
            {"comments": ["a", "b", "c"]}
        )
        cls = {"type": "T2", "confidence": 0.9, "patterns": [], "tone": "x", "brand_risk": "low"}
        generate_comments(
            cls,
            [],
            niches=["humour"],
            t_type_profile="T2",
            video_context={"caption": "x", "hashtags": []},
        )
        baseline = self._capture_prompt(mock_post)
        mock_post.reset_mock()
        mock_post.return_value = _ollama_json_response(
            {"comments": ["a", "b", "c"]}
        )
        generate_comments(
            cls,
            [],
            niches=["humour"],
            t_type_profile="T2",
            video_context={"caption": "x", "hashtags": []},
            named_axes={},
        )
        self.assertEqual(self._capture_prompt(mock_post), baseline)
        self.assertNotIn("Profil créateur:", baseline)

    @patch("modules.classifier._http_post")
    def test_named_axes_values_formatted_with_two_decimals(
        self, mock_post: MagicMock
    ) -> None:
        mock_post.return_value = _ollama_json_response(
            {"comments": ["a", "b", "c"]}
        )
        cls = {"type": "T2", "confidence": 0.9, "patterns": [], "tone": "x", "brand_risk": "low"}
        generate_comments(
            cls,
            [],
            niches=["humour"],
            t_type_profile="T2",
            named_axes={"scripted_vs_raw": 0.123456, "solo_vs_collab": 1},
        )
        prompt = self._capture_prompt(mock_post)
        self.assertIn("scripted_vs_raw=0.12", prompt)
        self.assertIn("solo_vs_collab=1.00", prompt)


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
        out = generate_comments(cls, ["a", "b"], niches=["streetwear", "lifestyle"])
        self.assertEqual(out, ["mdr", "ouais c'est ça", "trop vrai"])
        mock_post.assert_called_once()
        body = mock_post.call_args.kwargs["json_body"]
        self.assertEqual(body["model"], "qwen2.5:7b")
        self.assertEqual(body["stream"], False)

    @patch("modules.classifier._http_post")
    def test_t2_uses_t2_template_with_niches_list(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ollama_json_response(
            {"comments": ["x", "y", "z"]}
        )
        cls = {"type": "T2", "confidence": 0.9, "patterns": [], "tone": "x", "brand_risk": "low"}
        generate_comments(
            cls, [],
            niches=["humour", "sketch"],
            video_context={
                "caption": "moment culte F1",
                "hashtags": ["F1", "monaco"],
            },
        )
        system, prompt = self._capture_prompt(mock_post)
        # System global utilisé
        self.assertEqual(system, GENERATE_COMMENTS_SYSTEM.strip())
        # Template T2 (signature unique : "inside joke")
        self.assertIn("inside joke", prompt)
        # Schéma 2026-05 : label ``Niches :`` (pluriel) avec liste jointe.
        self.assertIn("Niches : humour, sketch", prompt)
        self.assertIn("moment culte F1", prompt)
        self.assertIn("#F1 #monaco", prompt)

    @patch("modules.classifier._http_post")
    def test_t_type_profile_is_injected_in_prompt(self, mock_post: MagicMock) -> None:
        """``t_type_profile`` (persona du commentateur, distinct du T-type
        du contenu) doit apparaître sur sa propre ligne dans le prompt."""
        mock_post.return_value = _ollama_json_response(
            {"comments": ["x", "y", "z"]}
        )
        # Cas réaliste : le contenu est T2 (humour niche), notre persona
        # est T3b (second degré) — on commente du T2 en T3b.
        cls = {"type": "T2", "confidence": 0.9, "patterns": [], "tone": "x", "brand_risk": "low"}
        generate_comments(
            cls, [],
            niches=["humour"],
            t_type_profile="T3b",
            video_context={"caption": "x", "hashtags": []},
        )
        _, prompt = self._capture_prompt(mock_post)
        # Ligne dédiée avec le label exact du brief.
        self.assertIn("T-type commentateur : T3b", prompt)
        # Reste cohérent : c'est bien le template T2 (du contenu) qui est utilisé.
        self.assertIn("inside joke", prompt)

    @patch("modules.classifier._http_post")
    def test_t_type_profile_defaults_to_placeholder_when_missing(
        self, mock_post: MagicMock
    ) -> None:
        mock_post.return_value = _ollama_json_response(
            {"comments": ["a", "b", "c"]}
        )
        cls = {"type": "T2", "confidence": 0.9, "patterns": [], "tone": "x", "brand_risk": "low"}
        # Pas de t_type_profile → ``"(non précisé)"`` injecté.
        generate_comments(cls, [], niches=["humour"])
        _, prompt = self._capture_prompt(mock_post)
        self.assertIn("T-type commentateur : (non précisé)", prompt)

    @patch("modules.classifier._http_post")
    def test_t3b_uses_invisible_second_degree_template(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ollama_json_response(
            {"comments": ["bravo le débutant", "très pro pour 2026", "on sent l'expérience"]}
        )
        cls = {"type": "T3b", "confidence": 0.85, "patterns": [], "tone": "x", "brand_risk": "medium"}
        out = generate_comments(
            cls, ["test"],
            niches=["cuisine"],
            video_context={"caption": "ma première fois", "hashtags": []},
        )
        _, prompt = self._capture_prompt(mock_post)
        # Le prompt T3b doit décrire le faux éloge / second degré.
        self.assertIn("second degré", prompt)
        self.assertIn("éloge", prompt)
        self.assertIn("liker", prompt)  # "doit pouvoir liker sans comprendre"
        # Les 3 commentaires retournés sont bien dans la forme attendue.
        self.assertEqual(len(out), 3)
        self.assertTrue(all(isinstance(c, str) and c for c in out))

    @patch("modules.classifier._http_post")
    def test_unknown_ttype_falls_back_to_t2(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ollama_json_response(
            {"comments": ["a", "b", "c"]}
        )
        cls = {"type": "TX", "confidence": 0.5, "patterns": [], "tone": "?", "brand_risk": "low"}
        generate_comments(cls, [], niches="x")
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
            cls, [],
            niches=["gaming", "esport"],
            video_context={
                "caption": "GG la team",
                "hashtags": ["league", "esport"],
                "audio_id": "AUD123",  # ne doit pas fuiter dans le prompt user
            },
        )
        _, prompt = self._capture_prompt(mock_post)
        self.assertIn("Niches : gaming, esport", prompt)
        self.assertIn("GG la team", prompt)
        self.assertIn("#league #esport", prompt)
        # ``audio_id`` n'a pas de slot dans les templates T-type — c'est
        # voulu : ce n'est pas un signal de registre, juste un identifiant
        # de classe. On vérifie qu'il n'apparaît PAS dans le prompt user.
        self.assertNotIn("AUD123", prompt)

    @patch("modules.classifier._http_post")
    def test_niches_string_for_backward_compat(self, mock_post: MagicMock) -> None:
        """Rétro-compat : ``niches`` peut être une string."""
        mock_post.return_value = _ollama_json_response(
            {"comments": ["a", "b", "c"]}
        )
        cls = {"type": "T2", "confidence": 0.9, "patterns": [], "tone": "x", "brand_risk": "low"}
        generate_comments(cls, [], niches="streetwear")
        _, prompt = self._capture_prompt(mock_post)
        self.assertIn("Niches : streetwear", prompt)

    @patch("modules.classifier._http_post")
    def test_comments_sample_capped_at_twenty(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ollama_json_response(
            {"comments": ["a", "b", "c"]}
        )
        cls = {"type": "T2", "confidence": 0.9, "patterns": [], "tone": "x", "brand_risk": "low"}
        sample = [f"line_{i}" for i in range(50)]
        generate_comments(cls, sample, niches="x")
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
        out = generate_comments(cls, [], niches=["humour"])
        joined = " ".join(out).lower()
        for word in FORBIDDEN_GENERATOR_WORDS:
            self.assertNotIn(word.lower(), joined)


if __name__ == "__main__":
    unittest.main()
