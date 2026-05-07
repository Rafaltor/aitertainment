"""Données de démo réalistes pour valider le pipeline (détection + classification)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from modules.detector import CreatorStats

# Instant fixe pour des scores SignalDetector reproductibles
MOCK_DETECT_NOW = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)

ScenarioName = Literal["T2_serie", "T3b_hater", "T1_gros"]
SCENARIOS: tuple[ScenarioName, ...] = ("T2_serie", "T3b_hater", "T1_gros")


def _v(
    video_id: str,
    views: int,
    likes: int,
    comments: int,
    shares: int,
    saves: int,
    posted_at: datetime,
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "video_id": video_id,
        "views": views,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "saves": saves,
        "posted_at": posted_at,
    }
    row.update(extra)
    return row


def _comments_t2() -> list[str]:
    base = [
        "c'est nous les vrais day ones",
        "la niche qui sait",
        "mdrrr le cargo encore une fois",
        "Wsh la commu est trop saine",
        "personne parle du détail sur les coutures",
        "t'as capté ou pas",
        "on est pas sur du fast fashion la team",
        "ratio + L + respect",
        "le son du reel >>>",
        "jpp c'est trop notre humour",
        "street certified",
        "ça sent le grail",
        "les haters sont où",
        "on claim ce moment",
        "archive national street",
        "trop relatable pour la team",
        "c'est notre Avengers",
        "mdr le combo tech + baggy",
        "la légende raconte que...",
        "on est ensemble depuis le début",
        "personne fait mieux dans la niche",
        "c'est ça la France",
        "commu > algo",
        "on stream en silence",
        "trop iconique",
        "le lore est profond",
        "respect aux OG",
        "c'est notre MCU",
        "jamais vu mieux sur IG",
        "W la team",
    ]
    while len(base) < 30:
        base.append(f"phrase recurrente #{len(base)}")
    return base[:30]


def _comments_t3b() -> list[str]:
    base = [
        "super original encore",
        "wow trop naturel",
        "on adore tous sincèrement",
        "le talent saute aux yeux",
        "incroyable comme d'habitude",
        "ça change de tout ce qu'on voit... vraiment",
        "trop mérité ce succès",
        "personne ne remarque rien d'anormal",
        "quelle authenticité",
        "le glow up est flagrant (positif bien sûr)",
        "on est fans sans ironie",
        "tellement unique",
        "le niveau monte encore",
        "bravo pour cette masterclass",
        "c'est du jamais vu (dans le bon sens)",
        "tout le monde est d'accord",
        "quel génie",
        "la cohérence est parfaite",
        "on applaudit debout",
        "rien à redire comme toujours",
        "du pur hasard que ça ressemble à du déjà vu",
        "le hasard fait bien les choses",
        "trop nature comme approche",
        "les commentaires sont 100% sérieux",
        "aucune moquerie ici",
        "juste de l'admiration pure",
        "le second degré n'existe pas",
        "on est tous impressionnés pareil",
        "c'est du lourd (sans double sens)",
        "chef d'oeuvre reconnu universellement",
    ]
    return base[:30]


def _comments_t1() -> list[str]:
    base = [
        "merci pour ce que tu fais",
        "tu inspires beaucoup de gens",
        "continue comme ça",
        "trop beau message",
        "force à toi",
        "tu mérites tout le succès",
        "gratitude",
        "belle énergie",
        "ça fait du bien",
        "respect",
        "tu changes des vies",
        "on est fiers de toi",
        "belle évolution",
        "courage pour la suite",
        "tu assumes parfaitement",
        "incroyable mindset",
        "merci pour les conseils",
        "tu donnes envie d'aller mieux",
        "humble et fort",
        "belle personne",
        "tu restes authentique",
        "prends soin de toi aussi",
        "on suit depuis longtemps",
        "toujours pertinent",
        "merci pour l'inspi du jour",
        "ça motive",
        "belle journée à toi",
        "tu gères",
        "love",
        "proud of you",
    ]
    return base[:30]


@dataclass(frozen=True)
class MockCreatorBundle:
    """Jeu de données aligné sur un scénario de test."""

    scenario: str
    stats: CreatorStats
    comments: list[str]
    niche: str
    expected_type: str
    expect_score_above_075: bool
    detect_now: datetime
    simulated_classification: dict[str, Any]
    simulated_generate: dict[str, Any]


def assert_signal_expectations(bundle: MockCreatorBundle, signal_result: dict[str, Any]) -> None:
    """Vérifie le score viral attendu pour le scénario (lève AssertionError si incohérent)."""
    score = float(signal_result.get("score_viral", 0.0))
    if bundle.expect_score_above_075 and score <= 0.75:
        raise AssertionError(
            f"[{bundle.scenario}] score_viral={score:.3f} attendu > 0.75 (alerte virale)"
        )
    if not bundle.expect_score_above_075 and score > 0.75:
        raise AssertionError(
            f"[{bundle.scenario}] score_viral={score:.3f} attendu <= 0.75 (pas d'alerte)"
        )


def generate_mock_creator(scenario: str) -> MockCreatorBundle:
    """Construit stats + commentaires + réponses Ollama simulées pour un scénario."""
    t0 = MOCK_DETECT_NOW
    if scenario == "T2_serie":
        olds = [
            _v(f"sw_old_{i}", 200, 12, 4, 1, 0, t0 - timedelta(days=40 - i))
            for i in range(6)
        ]
        viral = [
            _v("sw_v2", 45_000, 4_000, 600, 120, 40, t0 - timedelta(hours=72), url="https://instagram.com/reel/sw2/"),
            _v("sw_v1", 62_000, 5_500, 900, 200, 60, t0 - timedelta(hours=48), url="https://instagram.com/reel/sw1/"),
            _v(
                "sw_v0",
                89_000,
                12_000,
                1_800,
                400,
                120,
                t0 - timedelta(hours=2),
                url="https://instagram.com/reel/sw0/",
                audio_id="tr_snd",
                audio_reels_count=800,
                audio_is_recent=True,
                duration_sec=90,
            ),
        ]
        stats = CreatorStats(
            creator_id="mock_t2",
            platform="instagram",
            username="atelier_street_mock",
            followers=8_000,
            recent_videos=olds + viral,
            follower_growth_7d_pct=12.0,
        )
        sim_class = {
            "type": "T2",
            "confidence": 0.91,
            "patterns": ["c'est nous les vrais day ones", "mdrrr le cargo encore une fois", "commu > algo"],
            "tone": "Humour tribal et insiders streetwear, registre 'on est la niche'.",
            "brand_risk": "low",
        }
        sim_gen = {
            "comments": [
                "le cargo est clean, la niche valide.",
                "on capte le détail couture, force.",
                "pas de pub, juste du respect pour le move.",
            ]
        }
        return MockCreatorBundle(
            scenario=scenario,
            stats=stats,
            comments=_comments_t2(),
            niche="streetwear",
            expected_type="T2",
            expect_score_above_075=True,
            detect_now=t0,
            simulated_classification=sim_class,
            simulated_generate=sim_gen,
        )

    if scenario == "T3b_hater":
        olds = [
            _v(f"ls_old_{i}", 120, 20, 8, 2, 1, t0 - timedelta(days=40 - i))
            for i in range(7)
        ]
        viral = [
            _v("ls_v2", 80_000, 14_000, 8_000, 900, 400, t0 - timedelta(hours=60)),
            _v("ls_v1", 100_000, 18_000, 11_000, 1_200, 500, t0 - timedelta(hours=36)),
            _v(
                "ls_head",
                120_000,
                55_000,
                28_000,
                6_000,
                2_500,
                t0 - timedelta(minutes=45),
                url="https://instagram.com/reel/ls1/",
                audio_id="ls_snd",
                audio_reels_count=200,
                audio_is_recent=True,
                duration_sec=75,
            ),
        ]
        stats = CreatorStats(
            creator_id="mock_t3b",
            platform="instagram",
            username="daily_lifestyle_mock",
            followers=15_000,
            recent_videos=olds + viral,
            follower_growth_7d_pct=18.0,
        )
        sim_class = {
            "type": "T3b",
            "confidence": 0.86,
            "patterns": [
                "super original encore",
                "wow trop naturel",
                "le glow up est flagrant (positif bien sûr)",
            ],
            "tone": "Second degré poli, compliments creux, moquerie implicite.",
            "brand_risk": "medium",
        }
        sim_gen = {
            "comments": [
                "oui oui très naturel, on adore.",
                "le niveau 'unique' se voit tout de suite.",
                "communauté unanime comme par hasard.",
            ]
        }
        return MockCreatorBundle(
            scenario=scenario,
            stats=stats,
            comments=_comments_t3b(),
            niche="lifestyle",
            expected_type="T3b",
            expect_score_above_075=True,
            detect_now=t0,
            simulated_classification=sim_class,
            simulated_generate=sim_gen,
        )

    if scenario == "T1_gros":
        flat = [
            _v(
                f"insp_{i}",
                30_000,
                2_200,
                180,
                40,
                25,
                t0 - timedelta(days=28 - i),
                url=f"https://instagram.com/reel/insp{i}/",
            )
            for i in range(10)
        ]
        stats = CreatorStats(
            creator_id="mock_t1",
            platform="instagram",
            username="mindset_daily_mock",
            followers=200_000,
            recent_videos=flat,
            follower_growth_7d_pct=1.0,
        )
        sim_class = {
            "type": "T1",
            "confidence": 0.84,
            "patterns": ["merci pour ce que tu fais", "tu inspires beaucoup de gens", "gratitude"],
            "tone": "Encouragements sincères, relation parasociale positive.",
            "brand_risk": "low",
        }
        sim_gen = {
            "comments": [
                "placeholder",
                "placeholder",
                "placeholder",
            ]
        }
        return MockCreatorBundle(
            scenario=scenario,
            stats=stats,
            comments=_comments_t1(),
            niche="inspiration / développement personnel",
            expected_type="T1",
            expect_score_above_075=False,
            detect_now=t0,
            simulated_classification=sim_class,
            simulated_generate=sim_gen,
        )

    raise ValueError(f"scenario inconnu: {scenario!r} (attendu un parmi {list(SCENARIOS)})")


def ollama_post_double_stub(
    classify_payload: dict[str, Any],
    generate_payload: dict[str, Any],
):
    """Fabrique une fonction compatible avec ``patch(..., side_effect=...)`` pour ``_http_post``."""

    def _post(url: str, *, json_body: dict[str, Any] | None = None, timeout: int = 120):
        _ = url, timeout
        system = (json_body or {}).get("system") or ""
        payload = (
            generate_payload
            if "conseiller culturel" in system.lower()
            else classify_payload
        )

        class _Resp:
            def raise_for_status(self) -> None:
                return None

            def json(self_inner) -> dict[str, Any]:
                return {"response": json.dumps(payload, ensure_ascii=False)}

        return _Resp()

    return _post
