"""Tests pour telegram_discovery_bot.py — handlers de validation Discovery."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import database
import telegram_discovery_bot as bot


SAMPLE_CANDIDATE = {
    "username": "promising_creator",
    "domain": "humour",
    "platform": "instagram",
    "followers": 12_000,
    "score": 742.0,
    "score_reels": 780.0,
    "score_posts": 600.0,
    "reel_weight": 0.7,
    "post_weight": 0.3,
    "reels_count": 7,
    "posts_count": 3,
    "reel_ratio_median": 1.85,
    "reel_engagement_median": 0.06,
    "reel_trend": "rising",
    "post_ratio_median": 0.05,
    "post_engagement_median": 0.04,
    "posting_rhythm": 0.8,
    "t_type_dominant": "T2",
    "t_type_distribution": {"T2": 0.7, "T3b": 0.3},
    "biography": "Je fais du contenu humour décalé pour la zoomer génération",
    "scored_at": "2026-05-07T12:00:00+00:00",
    "validated": False,
    "discovered_at": "2026-05-07T12:00:00+00:00",
}

LEGACY_CANDIDATE = {
    "username": "legacy_creator",
    "domain": "humour",
    "platform": "instagram",
    "followers": 8_000,
    "score": 600.0,
    "ratio_median": 1.4,
    "ratio_trend": "stable",
    "engagement_median": 0.05,
    "t_type_dominant": "T2",
    "t_type_distribution": {"T2": 1.0},
    "biography": "ancien candidat, schéma pré-segmentation",
    "scored_at": "2026-05-01T12:00:00+00:00",
    "validated": False,
    "discovered_at": "2026-05-01T12:00:00+00:00",
}


def _write_candidates(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps({"candidates": candidates}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_seeds(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "domains": [
                    {
                        "name": "humour",
                        "niche": "humour-zoomer",
                        "seeds": ["seed_one"],
                        "t_types_target": ["T2", "T3b"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


class BuildMessageTest(unittest.TestCase):
    def test_text_contains_key_fields(self) -> None:
        text = bot._build_candidate_text(SAMPLE_CANDIDATE)
        self.assertIn("Nouveau candidat", text)
        self.assertIn("@promising_creator", text)
        self.assertIn("12000", text)
        self.assertIn("742/1000", text)
        # Métriques par type segmentées
        self.assertIn("Reels 7", text)
        self.assertIn("1.85x", text)         # reel_ratio_median
        self.assertIn("rising", text)
        self.assertIn("📈", text)
        self.assertIn("Posts 3", text)
        self.assertIn("T2", text)
        self.assertIn("humour", text)

    def test_text_falls_back_to_legacy_schema(self) -> None:
        text = bot._build_candidate_text(LEGACY_CANDIDATE)
        self.assertIn("@legacy_creator", text)
        self.assertIn("600/1000", text)
        # Sans reels_count/posts_count : on retombe sur ratio_median + ratio_trend
        self.assertIn("1.40x", text)
        self.assertIn("stable", text)

    def test_bio_truncated_to_80(self) -> None:
        cand = {**SAMPLE_CANDIDATE, "biography": "x" * 200}
        text = bot._build_candidate_text(cand)
        # 80 chars max + ellipsis
        bio_line = next(l for l in text.splitlines() if l.startswith("📝 Bio"))
        self.assertLessEqual(len(bio_line), 100)
        self.assertTrue(bio_line.endswith("…"))

    def test_keyboard_has_five_buttons_with_correct_callbacks(self) -> None:
        kb = bot._build_candidate_keyboard("promising_creator")
        flat = [btn for row in kb["inline_keyboard"] for btn in row]
        # 4 actions par callback (v / r / m / ev) + 1 lien URL (Voir profil) = 5.
        self.assertEqual(len(flat), 5)
        callbacks = [b.get("callback_data") for b in flat if b.get("callback_data")]
        self.assertIn("v:promising_creator", callbacks)
        self.assertIn("r:promising_creator", callbacks)
        self.assertIn("m:promising_creator", callbacks)
        self.assertIn("ev:promising_creator", callbacks)
        urls = [b.get("url") for b in flat if b.get("url")]
        self.assertEqual(urls, ["https://www.instagram.com/promising_creator/"])

    def test_t_type_keyboard_has_seven_options(self) -> None:
        kb = bot._build_t_type_keyboard("u")
        flat = [btn for row in kb["inline_keyboard"] for btn in row]
        labels = [b["text"] for b in flat]
        self.assertEqual(set(labels), set(bot.T_TYPES_AVAILABLE))
        self.assertEqual(len(labels), 7)
        # tous en callback (pas de URL)
        for b in flat:
            self.assertTrue(b["callback_data"].startswith("s:"))


class ValidationsLogTest(unittest.TestCase):
    def test_append_validation_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "validations.json"
            bot.append_validation(
                {
                    "username": "u1",
                    "action": "validated",
                    "t_type_original": "T2",
                    "t_type_final": "T2",
                    "score": 600,
                    "domain": "humour",
                },
                path=p,
            )
            data = bot.load_validations(path=p)
            self.assertEqual(len(data["validations"]), 1)
            self.assertEqual(data["validations"][0]["username"], "u1")
            self.assertIn("validated_at", data["validations"][0])

    def test_append_validation_appends_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "validations.json"
            for i in range(3):
                bot.append_validation(
                    {
                        "username": f"u{i}",
                        "action": "validated",
                        "t_type_original": "T2",
                        "t_type_final": "T2",
                        "score": 600 + i,
                        "domain": "humour",
                    },
                    path=p,
                )
            data = bot.load_validations(path=p)
            self.assertEqual(len(data["validations"]), 3)


class HandleCallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.cand_path = base / "candidates.json"
        self.val_path = base / "validations.json"
        self.wl_path = base / "watchlist.json"
        self.seeds_path = base / "seeds.json"

        _write_candidates(self.cand_path, [SAMPLE_CANDIDATE])
        self.wl_path.write_text(json.dumps({"creators": []}), encoding="utf-8")
        _write_seeds(self.seeds_path)

        # On stub toute interaction réseau Telegram.
        self._post_patch = patch.object(bot, "_telegram_post", return_value={"ok": True})
        self.mock_post = self._post_patch.start()
        self.addCleanup(self._post_patch.stop)

    @staticmethod
    def _cq(data: str, *, chat_id: str = "42") -> dict[str, Any]:
        return {
            "id": "cb1",
            "data": data,
            "message": {"message_id": 100, "chat": {"id": chat_id}},
            "from": {"id": 999},
        }

    def test_validate_adds_to_watchlist_and_logs(self) -> None:
        result = bot.handle_callback(
            self._cq("v:promising_creator"),
            token="TKN",
            expected_chat_id="42",
            candidates_path=self.cand_path,
            validations_path=self.val_path,
            watchlist_path=self.wl_path,
            seeds_path=self.seeds_path,
        )
        self.assertIn("ajouté à la watchlist", result)

        wl = json.loads(self.wl_path.read_text(encoding="utf-8"))
        self.assertEqual(len(wl["creators"]), 1)
        creator = wl["creators"][0]
        self.assertEqual(creator["username"], "promising_creator")
        self.assertEqual(creator["t_type"], "T2")
        self.assertEqual(creator["niche"], "humour-zoomer")  # tiré de seeds.json
        self.assertIsNone(creator["last_post_id"])
        # engagement_baseline tiré de reel_engagement_median en priorité
        self.assertAlmostEqual(creator["engagement_baseline"], 0.06, places=4)

        val = bot.load_validations(path=self.val_path)
        self.assertEqual(len(val["validations"]), 1)
        rec = val["validations"][0]
        self.assertEqual(rec["action"], "validated")
        self.assertEqual(rec["t_type_original"], "T2")
        self.assertEqual(rec["t_type_final"], "T2")
        self.assertEqual(rec["score"], 742.0)

    def test_reject_logs_only(self) -> None:
        result = bot.handle_callback(
            self._cq("r:promising_creator"),
            token="TKN",
            expected_chat_id="42",
            candidates_path=self.cand_path,
            validations_path=self.val_path,
            watchlist_path=self.wl_path,
            seeds_path=self.seeds_path,
        )
        self.assertIn("ignoré", result)
        wl = json.loads(self.wl_path.read_text(encoding="utf-8"))
        self.assertEqual(wl["creators"], [])
        val = bot.load_validations(path=self.val_path)
        self.assertEqual(val["validations"][0]["action"], "rejected")

    def test_modify_opens_t_type_menu(self) -> None:
        result = bot.handle_callback(
            self._cq("m:promising_creator"),
            token="TKN",
            expected_chat_id="42",
            candidates_path=self.cand_path,
            validations_path=self.val_path,
            watchlist_path=self.wl_path,
            seeds_path=self.seeds_path,
        )
        self.assertIn("menu T-types", result)
        # Watchlist et validations restent vides : pas encore d'action finale.
        wl = json.loads(self.wl_path.read_text(encoding="utf-8"))
        self.assertEqual(wl["creators"], [])

        # Le bot a édité le message avec un keyboard de T-types
        edit_calls = [
            c for c in self.mock_post.call_args_list
            if c.args and c.args[0] == "editMessageText"
        ]
        self.assertEqual(len(edit_calls), 1)
        payload = edit_calls[0].args[1]
        self.assertIn("reply_markup", payload)
        labels = [
            b["text"]
            for row in payload["reply_markup"]["inline_keyboard"]
            for b in row
        ]
        self.assertEqual(set(labels), set(bot.T_TYPES_AVAILABLE))

    def test_set_ttype_records_correction_with_original_and_final(self) -> None:
        result = bot.handle_callback(
            self._cq("s:T4:promising_creator"),
            token="TKN",
            expected_chat_id="42",
            candidates_path=self.cand_path,
            validations_path=self.val_path,
            watchlist_path=self.wl_path,
            seeds_path=self.seeds_path,
        )
        self.assertIn("ajouté à la watchlist", result)
        self.assertIn("corrigé", result)

        wl = json.loads(self.wl_path.read_text(encoding="utf-8"))
        self.assertEqual(wl["creators"][0]["t_type"], "T4")

        val = bot.load_validations(path=self.val_path)
        rec = val["validations"][0]
        self.assertEqual(rec["action"], "corrected")
        self.assertEqual(rec["t_type_original"], "T2")
        self.assertEqual(rec["t_type_final"], "T4")

    def test_set_ttype_same_as_original_is_validated_not_corrected(self) -> None:
        bot.handle_callback(
            self._cq("s:T2:promising_creator"),
            token="TKN",
            expected_chat_id="42",
            candidates_path=self.cand_path,
            validations_path=self.val_path,
            watchlist_path=self.wl_path,
            seeds_path=self.seeds_path,
        )
        val = bot.load_validations(path=self.val_path)
        self.assertEqual(val["validations"][0]["action"], "validated")
        self.assertEqual(val["validations"][0]["t_type_final"], "T2")

    def test_set_ttype_unknown_is_rejected(self) -> None:
        result = bot.handle_callback(
            self._cq("s:T9:promising_creator"),
            token="TKN",
            expected_chat_id="42",
            candidates_path=self.cand_path,
            validations_path=self.val_path,
            watchlist_path=self.wl_path,
            seeds_path=self.seeds_path,
        )
        self.assertIn("inconnu", result)
        # Aucune validation ne doit être enregistrée.
        val = bot.load_validations(path=self.val_path)
        self.assertEqual(val["validations"], [])

    def test_callback_from_unauthorized_chat_is_ignored(self) -> None:
        result = bot.handle_callback(
            self._cq("v:promising_creator", chat_id="666"),
            token="TKN",
            expected_chat_id="42",
            candidates_path=self.cand_path,
            validations_path=self.val_path,
            watchlist_path=self.wl_path,
            seeds_path=self.seeds_path,
        )
        self.assertEqual(result, "ignored")
        # Aucune action n'a touché au disque.
        wl = json.loads(self.wl_path.read_text(encoding="utf-8"))
        self.assertEqual(wl["creators"], [])

    def test_validate_unknown_username_warns_without_crash(self) -> None:
        result = bot.handle_callback(
            self._cq("v:ghost"),
            token="TKN",
            expected_chat_id="42",
            candidates_path=self.cand_path,
            validations_path=self.val_path,
            watchlist_path=self.wl_path,
            seeds_path=self.seeds_path,
        )
        self.assertIn("introuvable", result)


class NotifyCandidateTest(unittest.TestCase):
    def test_mock_mode_does_not_call_telegram(self) -> None:
        with patch.object(bot, "_telegram_post") as mock_post:
            r = bot.notify_candidate(SAMPLE_CANDIDATE, mock=True)
            self.assertIsNone(r)
            mock_post.assert_not_called()

    def test_send_message_payload_structure(self) -> None:
        with patch.object(
            bot, "_telegram_post", return_value={"ok": True, "result": {"message_id": 1}}
        ) as mock_post:
            bot.notify_candidate(SAMPLE_CANDIDATE, token="TKN", chat_id="42")
            mock_post.assert_called_once()
            method, payload = mock_post.call_args.args
            self.assertEqual(method, "sendMessage")
            self.assertEqual(payload["chat_id"], "42")
            self.assertEqual(payload["parse_mode"], "HTML")
            self.assertIn("@promising_creator", payload["text"])
            self.assertIn("inline_keyboard", payload["reply_markup"])

    def test_missing_credentials_returns_none(self) -> None:
        with patch.object(bot.config, "TELEGRAM_DISCOVERY_TOKEN", ""), \
             patch.object(bot.config, "TELEGRAM_DISCOVERY_CHAT_ID", ""), \
             patch.object(bot, "_telegram_post") as mock_post:
            r = bot.notify_candidate(SAMPLE_CANDIDATE)
            self.assertIsNone(r)
            mock_post.assert_not_called()


class NotifyScoreEvolutionTest(unittest.TestCase):
    def test_format_rise(self) -> None:
        text = bot._format_score_evolution_text(
            "raikkonenaf", 448, 745, "B", "A"
        )
        self.assertIn("📈", text)
        self.assertIn("@raikkonenaf", text)
        self.assertIn("448", text)
        self.assertIn("745", text)
        self.assertIn("(+66%)", text)
        self.assertIn("Tier B→A", text)

    def test_format_drop(self) -> None:
        text = bot._format_score_evolution_text(
            "raikkonenaf", 745, 420, "A", "B"
        )
        self.assertIn("📉", text)
        self.assertIn("(-44%)", text)
        self.assertIn("Tier A→B", text)

    def test_format_stable_no_div_zero(self) -> None:
        # old_score = 0 ne doit pas faire planter le %.
        text = bot._format_score_evolution_text(
            "ghost", 0.0, 500.0, "?", "B"
        )
        self.assertIn("(+0%)", text)

    def test_format_handles_none_inputs(self) -> None:
        text = bot._format_score_evolution_text(
            "x", None, None, None, None  # type: ignore[arg-type]
        )
        self.assertIn("@x", text)
        self.assertIn("0 → 0", text)
        self.assertIn("Tier ?→?", text)

    def test_mock_mode_does_not_call_telegram(self) -> None:
        with patch.object(bot, "_telegram_post") as mock_post:
            r = bot.notify_score_evolution(
                "raikkonenaf", 448, 745, "B", "A", mock=True
            )
            self.assertIsNone(r)
            mock_post.assert_not_called()

    def test_send_message_payload_structure_no_keyboard(self) -> None:
        with patch.object(
            bot, "_telegram_post", return_value={"ok": True}
        ) as mock_post:
            bot.notify_score_evolution(
                "raikkonenaf", 448, 745, "B", "A", token="TKN", chat_id="42"
            )
            mock_post.assert_called_once()
            method, payload = mock_post.call_args.args
            self.assertEqual(method, "sendMessage")
            self.assertEqual(payload["chat_id"], "42")
            self.assertIn("@raikkonenaf", payload["text"])
            # PAS de boutons : c'est une notif info, pas une action.
            self.assertNotIn("reply_markup", payload)

    def test_missing_credentials_returns_none(self) -> None:
        with patch.object(bot.config, "TELEGRAM_DISCOVERY_TOKEN", ""), \
             patch.object(bot.config, "TELEGRAM_DISCOVERY_CHAT_ID", ""), \
             patch.object(bot, "_telegram_post") as mock_post:
            r = bot.notify_score_evolution("raikkonenaf", 100, 200, "C", "C")
            self.assertIsNone(r)
            mock_post.assert_not_called()


class RunBotTest(unittest.TestCase):
    def test_mock_mode_returns_immediately(self) -> None:
        with patch.object(bot, "_telegram_get") as mock_get:
            bot.run_bot(mock=True)
            mock_get.assert_not_called()


class HandleCallbackPersistsToDatabaseTest(unittest.TestCase):
    """Les handlers ✅ Valider et ✏️ Modifier T-type doivent aussi mettre à
    jour ``database.json`` (``validated=True``, ``t_type_final=...``)."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.cand_path = base / "candidates.json"
        self.val_path = base / "validations.json"
        self.wl_path = base / "watchlist.json"
        self.seeds_path = base / "seeds.json"
        self.db_path = base / "database.json"

        _write_candidates(self.cand_path, [SAMPLE_CANDIDATE])
        self.wl_path.write_text(json.dumps({"creators": []}), encoding="utf-8")
        _write_seeds(self.seeds_path)

        # Pré-upsert : on simule un profil déjà scoré une fois par discovery.
        db: dict[str, Any] = {"profiles": {}}
        database.upsert_profile(db, SAMPLE_CANDIDATE, added_via="discovery")
        database.save_db(db, path=self.db_path)

        self._post_patch = patch.object(bot, "_telegram_post", return_value={"ok": True})
        self.mock_post = self._post_patch.start()
        self.addCleanup(self._post_patch.stop)

    def _cq(self, data: str, *, chat_id: str = "42") -> dict[str, Any]:
        return {
            "id": "cb1",
            "data": data,
            "message": {"message_id": 100, "chat": {"id": chat_id}},
            "from": {"id": 999},
        }

    def test_validate_marks_profile_validated_in_db(self) -> None:
        bot.handle_callback(
            self._cq("v:promising_creator"),
            token="TKN",
            expected_chat_id="42",
            candidates_path=self.cand_path,
            validations_path=self.val_path,
            watchlist_path=self.wl_path,
            seeds_path=self.seeds_path,
            db_path=self.db_path,
        )
        db = database.load_db(path=self.db_path)
        profile = db["profiles"]["promising_creator"]
        self.assertTrue(profile["validated"])
        self.assertEqual(profile["t_type_final"], "T2")

    def test_set_ttype_writes_corrected_t_type_final_to_db(self) -> None:
        bot.handle_callback(
            self._cq("s:T4:promising_creator"),
            token="TKN",
            expected_chat_id="42",
            candidates_path=self.cand_path,
            validations_path=self.val_path,
            watchlist_path=self.wl_path,
            seeds_path=self.seeds_path,
            db_path=self.db_path,
        )
        db = database.load_db(path=self.db_path)
        profile = db["profiles"]["promising_creator"]
        self.assertTrue(profile["validated"])
        self.assertEqual(profile["t_type_final"], "T4")
        self.assertEqual(profile["t_type_original"], "T2")

    def test_validate_without_db_entry_does_not_crash(self) -> None:
        """Un candidat jamais scoré : la validation doit fonctionner (watchlist
        + validations.json) et juste skipper la maj DB."""
        empty_db_path = Path(self.tmp.name) / "empty_db.json"
        # Pas de pré-upsert : le profil n'existe pas dans la DB.
        result = bot.handle_callback(
            self._cq("v:promising_creator"),
            token="TKN",
            expected_chat_id="42",
            candidates_path=self.cand_path,
            validations_path=self.val_path,
            watchlist_path=self.wl_path,
            seeds_path=self.seeds_path,
            db_path=empty_db_path,
        )
        self.assertIn("ajouté à la watchlist", result)


class FormatEvolutionTest(unittest.TestCase):
    def test_single_score_returns_no_history(self) -> None:
        text = bot._format_evolution(
            "alice",
            [{"date": "2026-05-08T12:00:00", "score": 448}],
        )
        self.assertIn("Pas encore d'historique", text)

    def test_empty_history_returns_aucun(self) -> None:
        text = bot._format_evolution("alice", [])
        self.assertIn("aucun historique", text)

    def test_two_scores_show_evolution_with_pct_and_trend(self) -> None:
        # 448 → tier B, 750 → tier A (seuil >700) ; +67%.
        text = bot._format_evolution(
            "alice",
            [
                {"date": "2026-05-08T12:00:00", "score": 448},
                {"date": "2026-05-15T12:00:00", "score": 750},
            ],
        )
        self.assertIn("Évolution @alice", text)
        # J0 + J+7 attendus (08/05 → 15/05).
        self.assertIn("J0", text)
        self.assertIn("J+7", text)
        self.assertIn("08/05", text)
        self.assertIn("15/05", text)
        self.assertIn("448", text)
        self.assertIn("750", text)
        # Variation : (750-448)/448 ≈ +67%.
        self.assertIn("+67%", text)
        self.assertIn("Tier B", text)
        self.assertIn("Tier A", text)
        # Variation totale > +10% → rising.
        self.assertIn("rising", text)

    def test_declining_trend(self) -> None:
        text = bot._format_evolution(
            "alice",
            [
                {"date": "2026-05-01T12:00:00", "score": 800},
                {"date": "2026-05-15T12:00:00", "score": 600},
            ],
        )
        self.assertIn("declining", text)

    def test_stable_trend_within_10pct(self) -> None:
        text = bot._format_evolution(
            "alice",
            [
                {"date": "2026-05-01T12:00:00", "score": 500},
                {"date": "2026-05-15T12:00:00", "score": 525},
            ],
        )
        self.assertIn("stable", text)


class HandleEvolutionCallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "database.json"

        # 2 scores → message d'évolution complet.
        db: dict[str, Any] = {
            "profiles": {
                "alice": {
                    "platform": "instagram",
                    "tier": "A",
                    "scores_history": [
                        {"date": "2026-05-08T12:00:00", "score": 448.0},
                        {"date": "2026-05-15T12:00:00", "score": 612.0},
                    ],
                }
            }
        }
        database.save_db(db, path=self.db_path)

        self._post_patch = patch.object(bot, "_telegram_post", return_value={"ok": True})
        self.mock_post = self._post_patch.start()
        self.addCleanup(self._post_patch.stop)

    def _cq(self, data: str) -> dict[str, Any]:
        return {
            "id": "cb_ev",
            "data": data,
            "message": {"message_id": 200, "chat": {"id": "42"}},
        }

    def test_evolution_callback_pushes_new_message_with_history(self) -> None:
        result = bot.handle_callback(
            self._cq("ev:alice"),
            token="TKN",
            expected_chat_id="42",
            db_path=self.db_path,
        )
        self.assertIn("Évolution @alice", result)
        self.assertIn("J+7", result)

        send_calls = [
            c for c in self.mock_post.call_args_list
            if c.args and c.args[0] == "sendMessage"
        ]
        self.assertEqual(len(send_calls), 1)
        payload = send_calls[0].args[1]
        self.assertEqual(payload["chat_id"], "42")
        self.assertIn("Évolution @alice", payload["text"])

    def test_evolution_callback_for_unknown_user(self) -> None:
        result = bot.handle_callback(
            self._cq("ev:ghost"),
            token="TKN",
            expected_chat_id="42",
            db_path=self.db_path,
        )
        self.assertIn("pas encore en base", result)


if __name__ == "__main__":
    unittest.main()
