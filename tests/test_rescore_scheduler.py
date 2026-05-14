"""Tests pour ``rescore_scheduler.py`` (cycle, seuils de notif, sleep, CLI)."""

from __future__ import annotations

import io
import json
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

# Permet d'importer le module à la racine sans dépendre du PWD du test runner.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rescore_scheduler as scheduler  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso(offset_days: float = 0.0) -> str:
    dt = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=offset_days)
    return dt.isoformat(timespec="seconds")


def _mk_db(*profiles: dict) -> dict:
    return {"profiles": {p["username"]: {k: v for k, v in p.items() if k != "username"} for p in profiles}}


def _profile(
    username: str,
    *,
    last_score: float | None = None,
    next_rescore_offset_days: float = -1.0,
    archived: bool = False,
    tier: str | None = None,
    history_extra: list[dict] | None = None,
) -> dict:
    history: list[dict] = list(history_extra or [])
    if last_score is not None:
        history.append({"date": _now_iso(-30), "score": last_score})
    return {
        "username": username,
        "platform": "instagram",
        "followers": 50_000,
        "niche": "humour",
        "tier": tier or ("A" if (last_score or 0) > 700 else "B"),
        "validated": False,
        "t_type_original": "T2",
        "t_type_final": None,
        "added_via": "discovery",
        "added_at": _now_iso(-60),
        "last_scored_at": _now_iso(-30),
        "next_rescore_at": _now_iso(next_rescore_offset_days),
        "archived": archived,
        "scores_history": history,
    }


def _summary(score: float, tier: str = "A", *, username: str = "x") -> dict:
    return {
        "score_result": {"username": username, "score": score, "domain": "humour"},
        "profile": {},
        "tier": tier,
        "notified": False,
    }


def _write_db(tmp: Path, db: dict) -> Path:
    p = tmp / "database.json"
    p.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _pct_change / _last_score
# ---------------------------------------------------------------------------


class HelpersTest(unittest.TestCase):
    def test_pct_change_normal(self) -> None:
        self.assertAlmostEqual(scheduler._pct_change(100, 150), 0.5, places=6)

    def test_pct_change_zero_old_returns_zero(self) -> None:
        self.assertEqual(scheduler._pct_change(0, 500), 0.0)
        self.assertEqual(scheduler._pct_change(-10, 500), 0.0)
        self.assertEqual(scheduler._pct_change(None, 500), 0.0)

    def test_pct_change_invalid_old_does_not_crash(self) -> None:
        self.assertEqual(scheduler._pct_change("oops", 500), 0.0)  # type: ignore[arg-type]

    def test_last_score_reads_last_history_entry(self) -> None:
        prof = _profile("x", last_score=448, history_extra=[
            {"date": _now_iso(-60), "score": 100}
        ])
        score, tier = scheduler._last_score(prof)
        self.assertEqual(score, 448)
        self.assertEqual(tier, "B")  # 448 → tier B

    def test_last_score_empty_history_returns_none(self) -> None:
        prof = _profile("x", last_score=None)
        prof["scores_history"] = []
        self.assertEqual(scheduler._last_score(prof), (None, None))

    def test_last_score_invalid_value_returns_none(self) -> None:
        prof = _profile("x", last_score=None)
        prof["scores_history"] = [{"date": _now_iso(), "score": "n/a"}]
        self.assertEqual(scheduler._last_score(prof), (None, None))


class FormatDueSummaryTest(unittest.TestCase):
    def test_empty_returns_friendly_message(self) -> None:
        self.assertEqual(
            scheduler._format_due_summary([]),
            "Aucun profil dû pour rescore.",
        )

    def test_lists_each_due_profile(self) -> None:
        text = scheduler._format_due_summary(
            [
                {"username": "raikkonenaf", "tier": "B", "next_rescore_at": "2026-05-01T00:00:00"},
                {"username": "alonso", "tier": "A", "next_rescore_at": "2026-05-02T00:00:00"},
            ]
        )
        self.assertIn("2 profil(s)", text)
        self.assertIn("@raikkonenaf", text)
        self.assertIn("@alonso", text)
        self.assertIn("tier=A", text)


# ---------------------------------------------------------------------------
# run_rescore_cycle — cas standards
# ---------------------------------------------------------------------------


class RunRescoreCycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.addCleanup(self.tmpdir.cleanup)

    def _run(self, db: dict, *, score_fn=None, notify_fn=None, sleep_fn=None, mock: bool = True) -> tuple[dict, MagicMock, MagicMock, MagicMock]:
        db_path = _write_db(self.tmp, db)
        score_mock = score_fn or MagicMock(return_value=_summary(500, "B"))
        notify_mock = notify_fn or MagicMock()
        sleep_mock = sleep_fn or MagicMock()
        # rng stub déterministe pour les éventuels sleep
        rng = random.Random(0)
        stats = scheduler.run_rescore_cycle(
            mock=mock,
            db_path=db_path,
            sleep_fn=sleep_mock,
            rng=rng,
            score_fn=score_mock,
            notify_fn=notify_mock,
        )
        return stats, score_mock, notify_mock, sleep_mock

    def test_no_due_profiles_no_action(self) -> None:
        # next_rescore_at dans le futur → pas dû
        db = _mk_db(_profile("future_guy", last_score=500, next_rescore_offset_days=+5.0))
        stats, score, notify, sleep = self._run(db)
        self.assertEqual(stats["due_count"], 0)
        self.assertEqual(stats["processed"], 0)
        score.assert_not_called()
        notify.assert_not_called()
        sleep.assert_not_called()

    def test_calls_score_and_persist_for_each_due(self) -> None:
        db = _mk_db(
            _profile("p1", last_score=500, next_rescore_offset_days=-1),
            _profile("p2", last_score=600, next_rescore_offset_days=-2),
        )
        score_mock = MagicMock(side_effect=[
            _summary(520, "B", username="p1"),
            _summary(620, "B", username="p2"),
        ])
        stats, score, _notify, _sleep = self._run(db, score_fn=score_mock)
        self.assertEqual(stats["due_count"], 2)
        self.assertEqual(stats["processed"], 2)
        self.assertEqual(score.call_count, 2)
        # added_via="rescore" propagé
        for call in score.call_args_list:
            self.assertEqual(call.kwargs.get("added_via"), "rescore")

    def test_notifies_rise_above_15_percent(self) -> None:
        # 448 → 745 = +66%
        db = _mk_db(_profile("rise", last_score=448, tier="B"))
        score_mock = MagicMock(return_value=_summary(745, "A", username="rise"))
        stats, _score, notify, _sleep = self._run(db, score_fn=score_mock)
        self.assertEqual(stats["notified_rise"], 1)
        self.assertEqual(stats["notified_drop"], 0)
        notify.assert_called_once()
        kwargs = notify.call_args.kwargs
        self.assertEqual(kwargs["username"], "rise")
        self.assertEqual(kwargs["old_score"], 448)
        self.assertEqual(kwargs["new_score"], 745)
        self.assertEqual(kwargs["old_tier"], "B")
        self.assertEqual(kwargs["new_tier"], "A")

    def test_notifies_drop_below_minus_20_percent(self) -> None:
        # 745 → 420 = -43.6%
        db = _mk_db(_profile("drop", last_score=745, tier="A"))
        score_mock = MagicMock(return_value=_summary(420, "B", username="drop"))
        stats, _score, notify, _sleep = self._run(db, score_fn=score_mock)
        self.assertEqual(stats["notified_drop"], 1)
        self.assertEqual(stats["notified_rise"], 0)
        notify.assert_called_once()

    def test_no_notif_within_thresholds(self) -> None:
        # 500 → 510 = +2% → silence
        db = _mk_db(_profile("flat", last_score=500))
        score_mock = MagicMock(return_value=_summary(510, "B", username="flat"))
        stats, _score, notify, _sleep = self._run(db, score_fn=score_mock)
        self.assertEqual(stats["notified_rise"], 0)
        self.assertEqual(stats["notified_drop"], 0)
        notify.assert_not_called()

    def test_no_notif_when_no_history_baseline(self) -> None:
        # Profil dû mais sans baseline → on rescore quand même mais pas de notif
        db = _mk_db(_profile("fresh", last_score=None))
        score_mock = MagicMock(return_value=_summary(900, "A", username="fresh"))
        stats, _score, notify, _sleep = self._run(db, score_fn=score_mock)
        self.assertEqual(stats["processed"], 1)
        self.assertEqual(stats["notified_rise"], 0)
        self.assertEqual(stats["notified_drop"], 0)
        notify.assert_not_called()

    def test_sleep_between_profiles_not_after_last(self) -> None:
        db = _mk_db(
            _profile("p1", last_score=500),
            _profile("p2", last_score=500),
            _profile("p3", last_score=500),
        )
        score_mock = MagicMock(side_effect=[
            _summary(505, "B", username=u) for u in ("p1", "p2", "p3")
        ])
        # mock=False pour réellement déclencher les sleep, sleep_fn injecté.
        stats, _score, _notify, sleep = self._run(
            db, score_fn=score_mock, mock=False
        )
        self.assertEqual(stats["due_count"], 3)
        # 3 profils dûs → 2 sleep (entre p1-p2 et p2-p3, pas après p3).
        self.assertEqual(sleep.call_count, 2)
        # Wait dans la fenêtre 5–15 min.
        for call in sleep.call_args_list:
            wait = call.args[0]
            self.assertGreaterEqual(wait, scheduler.RESCORE_SLEEP_MIN_S)
            self.assertLessEqual(wait, scheduler.RESCORE_SLEEP_MAX_S)

    def test_mock_mode_skips_sleep_entirely(self) -> None:
        db = _mk_db(
            _profile("p1", last_score=500),
            _profile("p2", last_score=500),
        )
        score_mock = MagicMock(side_effect=[
            _summary(505, "B", username="p1"),
            _summary(505, "B", username="p2"),
        ])
        stats, _score, _notify, sleep = self._run(db, score_fn=score_mock, mock=True)
        self.assertEqual(stats["due_count"], 2)
        sleep.assert_not_called()

    def test_score_exception_increments_errors_continues_cycle(self) -> None:
        db = _mk_db(
            _profile("boom", last_score=500),
            _profile("ok", last_score=500),
        )
        score_mock = MagicMock(side_effect=[RuntimeError("network"), _summary(510, "B", username="ok")])
        stats, _score, notify, _sleep = self._run(db, score_fn=score_mock)
        self.assertEqual(stats["errors"], 1)
        self.assertEqual(stats["processed"], 1)
        notify.assert_not_called()  # +2% sur "ok" → silence

    def test_archived_profiles_are_skipped(self) -> None:
        db = _mk_db(
            _profile("alive", last_score=500),
            _profile("dead", last_score=500, archived=True),
        )
        score_mock = MagicMock(return_value=_summary(510, "B", username="alive"))
        stats, score, _notify, _sleep = self._run(db, score_fn=score_mock)
        self.assertEqual(stats["due_count"], 1)
        self.assertEqual(score.call_count, 1)

    def test_notify_failure_does_not_crash_cycle(self) -> None:
        db = _mk_db(_profile("rise", last_score=448, tier="B"))
        score_mock = MagicMock(return_value=_summary(745, "A", username="rise"))
        notify_mock = MagicMock(side_effect=RuntimeError("telegram down"))
        stats, _score, notify, _sleep = self._run(
            db, score_fn=score_mock, notify_fn=notify_mock
        )
        # Le compteur n'incrémente pas si la notif a levé.
        self.assertEqual(stats["notified_rise"], 0)
        notify.assert_called_once()


# ---------------------------------------------------------------------------
# load_db error / db absent
# ---------------------------------------------------------------------------


class RunRescoreCycleDbErrorsTest(unittest.TestCase):
    def test_missing_db_uses_empty(self) -> None:
        # Pas de fichier → load_db retourne {"profiles": {}} (cf. database.load_db)
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "ghost.json"
            stats = scheduler.run_rescore_cycle(
                mock=True,
                db_path=fake,
                score_fn=MagicMock(),
                notify_fn=MagicMock(),
                sleep_fn=MagicMock(),
            )
            self.assertEqual(stats["due_count"], 0)
            self.assertEqual(stats["errors"], 0)

    def test_corrupted_db_increments_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "database.json"
            p.write_text("{not json", encoding="utf-8")
            stats = scheduler.run_rescore_cycle(
                mock=True,
                db_path=p,
                score_fn=MagicMock(),
                notify_fn=MagicMock(),
                sleep_fn=MagicMock(),
            )
            self.assertEqual(stats["errors"], 1)
            self.assertEqual(stats["due_count"], 0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTest(unittest.TestCase):
    def test_cli_due_lists_profiles_without_running_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = _mk_db(_profile("dueguy", last_score=500))
            db_path = _write_db(Path(tmp), db)
            buf = io.StringIO()
            with patch.object(sys, "argv", [
                "rescore_scheduler.py", "--due", "--db-path", str(db_path)
            ]), patch.object(scheduler, "run_rescore_cycle") as cycle:
                with redirect_stdout(buf):
                    scheduler._main_cli()
            cycle.assert_not_called()
            out = buf.getvalue()
            self.assertIn("@dueguy", out)
            self.assertIn("1 profil", out)

    def test_cli_run_calls_cycle_and_prints_summary(self) -> None:
        buf = io.StringIO()
        fake_stats = {
            "due_count": 3,
            "processed": 3,
            "notified_rise": 1,
            "notified_drop": 1,
            "errors": 0,
        }
        with patch.object(sys, "argv", ["rescore_scheduler.py", "--mock"]), \
                patch.object(scheduler, "run_rescore_cycle", return_value=fake_stats) as cycle:
            with redirect_stdout(buf):
                scheduler._main_cli()
        cycle.assert_called_once()
        out = buf.getvalue()
        self.assertIn("dûs=3", out)
        self.assertIn("processed=3", out)
        self.assertIn("📈=1", out)
        self.assertIn("📉=1", out)


if __name__ == "__main__":
    unittest.main()
