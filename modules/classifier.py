"""Classification T1–T5 et génération de commentaires via Ollama (local)."""

from __future__ import annotations

import json
from typing import Any

import requests

import config

SYSTEM_PROMPT = """Tu es un expert en analyse culturelle des réseaux sociaux.
Classe la section de commentaires suivante selon ces types :
- T1 : Admiration sincère, encouragements, relation parasociale
- T2 : Humour tribal, communauté niche aimée, phrases récurrentes
- T3a : Haine directe non dissimulée
- T3b : Second degré invisible, moquerie que le créateur ne perçoit pas
- T4 : Identité communautaire forte, rituel de niche
- T5 : Créateur provocateur conscient qui exploite la haine

Retourne UNIQUEMENT un JSON avec :
{
  "type": "T1"|"T2"|"T3a"|"T3b"|"T4"|"T5",
  "confidence": <float entre 0 et 1>,
  "patterns": [<liste des 3 phrases/patterns les plus récurrents>],
  "tone": "<description en 1 phrase du registre dominant>",
  "brand_risk": "low"|"medium"|"high"
}"""

GENERATE_COMMENTS_SYSTEM = """Tu es conseiller culturel pour une marque sur les réseaux sociaux.
Tu proposes des réponses courtes, crédibles, dans le registre de la communauté (pas de ton publicitaire).
Retourne UNIQUEMENT un JSON valide, sans texte avant ou après :
{
  "comments": ["suggestion 1", "suggestion 2", "suggestion 3"]
}
Les trois chaînes doivent être des commentaires prêts à poster (une phrase chacune si possible)."""

VALID_TYPES = frozenset({"T1", "T2", "T3a", "T3b", "T4", "T5"})
VALID_RISK = frozenset({"low", "medium", "high"})


class ClassificationError(ValueError):
    """Réponse modèle illisible ou schéma invalide."""

    def __init__(self, message: str, *, raw_text: str | None = None) -> None:
        super().__init__(message)
        self.raw_text = raw_text


def _http_post(url: str, *, json_body: dict[str, Any], timeout: int) -> requests.Response:
    return requests.post(url, json=json_body, timeout=timeout)


def _ollama_response_text(resp: requests.Response) -> str:
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError as e:
        raise ValueError(f"corps HTTP non-JSON: {e}") from e
    return str(data.get("response", "")).strip()


def _parse_json_from_response(text: str) -> dict[str, Any]:
    """Extrait le premier objet JSON du texte (bloc ```json``` ou premier { … })."""
    s = text.strip()
    if not s:
        raise ValueError("réponse vide")

    fence = "```"
    if fence in s:
        start = s.find(fence)
        if start != -1:
            after = s[start + len(fence) :]
            if after.lower().startswith("json"):
                after = after[4:].lstrip()
            end = after.find(fence)
            if end != -1:
                inner = after[:end].strip()
                return json.loads(inner)

    brace = s.find("{")
    if brace == -1:
        raise ValueError("aucune accolade ouvrante")
    try:
        obj, _end = json.JSONDecoder().raw_decode(s[brace:])
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON mal formé: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("le JSON racine doit être un objet")
    return obj


def _normalize_classification(data: dict[str, Any]) -> dict[str, Any]:
    t = data.get("type")
    if not isinstance(t, str) or t not in VALID_TYPES:
        raise ClassificationError(
            f"type invalide: {t!r} (attendu un parmi {sorted(VALID_TYPES)})"
        )

    conf = data.get("confidence")
    try:
        c = float(conf)
    except (TypeError, ValueError) as e:
        raise ClassificationError(f"confidence invalide: {conf!r}") from e
    if c < 0.0 or c > 1.0:
        raise ClassificationError(f"confidence hors [0,1]: {c}")

    raw_patterns = data.get("patterns", [])
    if not isinstance(raw_patterns, list):
        raise ClassificationError(f"patterns doit être une liste, reçu: {type(raw_patterns)}")
    patterns = [str(p) for p in raw_patterns[:3]]

    tone = data.get("tone", "")
    if not isinstance(tone, str):
        tone = str(tone)

    risk = data.get("brand_risk", "")
    if not isinstance(risk, str):
        risk = str(risk)
    risk_l = risk.strip().lower()
    if risk_l not in VALID_RISK:
        raise ClassificationError(f"brand_risk invalide: {risk!r}")

    return {
        "type": t,
        "confidence": c,
        "patterns": patterns,
        "tone": tone.strip(),
        "brand_risk": risk_l,
    }


def _normalize_three_comments(data: dict[str, Any]) -> list[str]:
    raw = data.get("comments")
    if not isinstance(raw, list):
        raise ClassificationError(f'"comments" doit être une liste, reçu: {type(raw)!r}')
    out = [str(x).strip() for x in raw if str(x).strip()][:3]
    while len(out) < 3:
        out.append("—")
    return out[:3]


class CommentClassifier:
    """Classification T1–T5 via Ollama (endpoint /api/generate)."""

    MODEL = config.OLLAMA_MODEL

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: Any | None = None,
        ollama_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
    ) -> None:
        """api_key : conservé pour compatibilité (ignoré). client : objet type requests.Session avec .post()."""
        _ = api_key
        self.ollama_url = (ollama_url or config.OLLAMA_URL).strip()
        self.model = (model or config.OLLAMA_MODEL).strip()
        self.timeout = int(timeout)
        self._http = client if client is not None else None

    def _post(self, json_body: dict[str, Any]) -> requests.Response:
        if self._http is not None:
            return self._http.post(self.ollama_url, json=json_body, timeout=self.timeout)
        return _http_post(
            self.ollama_url, json_body=json_body, timeout=self.timeout
        )

    def _generate(self, *, system: str, prompt: str) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system.strip():
            body["system"] = system.strip()
        try:
            resp = self._post(body)
        except requests.RequestException as e:
            raise ClassificationError(f"Erreur HTTP Ollama: {e}") from e
        try:
            return _ollama_response_text(resp)
        except requests.RequestException as e:
            raise ClassificationError(f"Erreur HTTP Ollama (réponse): {e}") from e
        except ValueError as e:
            raise ClassificationError(f"Réponse Ollama invalide: {e}") from e

    def _build_user_message(self, niche: str, comments: list[str]) -> str:
        lines = [
            f"Niche / contexte marché : {niche.strip() or '(non précisé)'}",
            "",
            "Commentaires (un par ligne, ordre conservé) :",
        ]
        for i, c in enumerate(comments, start=1):
            lines.append(f"{i}. {c}")
        return "\n".join(lines)

    def classify(self, comments: list[str], niche: str) -> dict[str, Any]:
        """Envoie les commentaires à Ollama et retourne type, confidence, patterns, tone, brand_risk."""
        if not comments:
            raise ValueError("comments ne doit pas être vide")

        user_content = self._build_user_message(niche, comments)
        raw_text = self._generate(system=SYSTEM_PROMPT, prompt=user_content)

        try:
            payload = _parse_json_from_response(raw_text)
        except (json.JSONDecodeError, ValueError) as e:
            raise ClassificationError(
                f"Réponse JSON invalide ou introuvable: {e}",
                raw_text=raw_text,
            ) from e

        try:
            return _normalize_classification(payload)
        except ClassificationError as e:
            raise ClassificationError(str(e), raw_text=raw_text) from e


def generate_comments(
    classification: dict[str, Any],
    comments_sample: list[str],
    *,
    niche: str = "",
) -> list[str]:
    """Produit 3 suggestions via Ollama (même signature qu'avant)."""
    user_lines = [
        f"Niche / contexte : {niche.strip() or '(non précisé)'}",
        "",
        "Classification (JSON) :",
        json.dumps(classification, ensure_ascii=False),
        "",
        "Exemples de commentaires existants (extrait) :",
    ]
    for i, c in enumerate(comments_sample[:40], start=1):
        user_lines.append(f"{i}. {c}")
    prompt = "\n".join(user_lines)

    body: dict[str, Any] = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "system": GENERATE_COMMENTS_SYSTEM.strip(),
    }
    try:
        resp = _http_post(config.OLLAMA_URL, json_body=body, timeout=120)
    except requests.RequestException as e:
        raise ClassificationError(f"Erreur HTTP Ollama: {e}") from e

    try:
        raw_text = _ollama_response_text(resp)
    except requests.RequestException as e:
        raise ClassificationError(f"Erreur HTTP Ollama (réponse): {e}") from e
    except ValueError as e:
        raise ClassificationError(f"Réponse Ollama invalide: {e}") from e
    try:
        payload = _parse_json_from_response(raw_text)
    except (json.JSONDecodeError, ValueError) as e:
        raise ClassificationError(
            f"Réponse JSON invalide ou introuvable: {e}",
            raw_text=raw_text,
        ) from e

    try:
        return _normalize_three_comments(payload)
    except ClassificationError as e:
        raise ClassificationError(str(e), raw_text=raw_text) from e
