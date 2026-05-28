"""Classification T1–T5 et génération de commentaires via Ollama (local)."""

from __future__ import annotations

import json
from typing import Any

import requests

import config
from config import VALID_T_TYPES

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

GENERATE_COMMENTS_SYSTEM = """Tu es un utilisateur lambda qui commente des Reels Instagram.
Tu ne représentes aucune marque. Tu écris comme tu parlerais à un ami — vite, sans réfléchir, en 3 secondes.

RÈGLES ABSOLUES (violation = réponse rejetée) :
- Maximum 8 mots par commentaire
- Minuscule en début (sauf nom propre)
- 0 ou 1 emoji max, placé naturellement
- Jamais ces mots : incroyable, impressionnant, vraiment, tellement, félicitations, magnifique, superbe, excellent, contenu, créateur, vidéo, post
- Pas de point final
- Registre oral SMS, pas rédactionnel
- Imparfait orthographiquement si ça sonne plus naturel

Retourne UNIQUEMENT ce JSON, rien d'autre :
{"comments": ["commentaire 1", "commentaire 2", "commentaire 3"]}"""


# ---------------------------------------------------------------------------
# Prompts user spécialisés par T-type
# ---------------------------------------------------------------------------
#
# Chaque template est formaté avec ``str.format(**ctx)`` et reçoit :
# ``niche``, ``caption``, ``hashtags``, ``comments_sample``.
#
# T1 et T3a sont volontairement absents :
# - T1 (admiration sincère) → la marque ne génère pas dans ce registre
#   (cf. brief Watcher : T1 est exclu de ``_generate_for_post``).
# - T3a (haine directe) → toxique, pas de génération.
#
# Les T-types non listés ici retombent sur **T2** (registre le plus neutre
# parmi les générables ; cf. ``_resolve_ttype_prompt``).

GENERATE_PROMPTS_BY_TTYPE: dict[str, str] = {
    "T2": """Contexte : communauté humour niche soudée, inside jokes.
Tu fais partie du groupe. Tu réagis à l'inside joke sans l'expliquer.
Ton commentaire prouve que tu as compris.

T-type commentateur : {t_type_profile}
Niches : {niches}
Caption du Reel : {caption}
Hashtags : {hashtags}

Exemples de vrais commentaires de la communauté :
{comments_sample}""",
    "T2b": """Contexte : humour participatif, les gens continuent le sketch dans les commentaires.
Tu ajoutes une couche, tu continues la blague,
tu réponds dans le même registre que le créateur.

T-type commentateur : {t_type_profile}
Niches : {niches}
Caption du Reel : {caption}
Hashtags : {hashtags}

Exemples de vrais commentaires :
{comments_sample}""",
    "T3b": """Contexte : second degré invisible. Le créateur ne perçoit pas
qu'on se moque. Ton commentaire semble un éloge mais le sous-texte
est moqueur — subtil, pas agressif.
La cible doit pouvoir liker ton commentaire sans comprendre.

T-type commentateur : {t_type_profile}
Niches : {niches}
Caption du Reel : {caption}
Hashtags : {hashtags}

Exemples de vrais commentaires :
{comments_sample}""",
    "T4": """Contexte : identité communautaire ritualisée.
Tu poses un marqueur d'appartenance au groupe.
Phrase courte, rituelle, que seuls les membres comprennent.

T-type commentateur : {t_type_profile}
Niches : {niches}
Caption du Reel : {caption}
Hashtags : {hashtags}

Exemples de vrais commentaires :
{comments_sample}""",
    "T5": """Contexte : créateur provocateur conscient, troll assumé.
Tu joues le jeu — punchline directe, humour noir, tu assumes.
Pas d'agressivité gratuite, juste du piquant.

T-type commentateur : {t_type_profile}
Niches : {niches}
Caption du Reel : {caption}
Hashtags : {hashtags}

Exemples de vrais commentaires :
{comments_sample}""",
}

GENERATE_FALLBACK_TTYPE = "T2"

from modules.generator_prompt import (
    GENERATOR_INSTRUCTION,
    build_alpaca_prompt,
    build_generator_input_block,
)
from modules.named_axes import NAMED_AXES

# Mots interdits explicités côté system prompt — listés ici aussi pour
# permettre un check programmatique côté tests / debug (pas de filtrage
# automatique côté code de prod : on fait confiance au modèle, on ne fait
# pas de censure post-hoc qui dégraderait la cohérence des suggestions).
FORBIDDEN_GENERATOR_WORDS: frozenset[str] = frozenset({
    "incroyable",
    "impressionnant",
    "vraiment",
    "tellement",
    "félicitations",
    "magnifique",
    "superbe",
    "excellent",
    "contenu",
    "créateur",
    "vidéo",
    "post",
})

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
    if not isinstance(t, str) or t not in VALID_T_TYPES:
        raise ClassificationError(
            f"type invalide: {t!r} (attendu un parmi {sorted(VALID_T_TYPES)})"
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

    def _build_user_message(
        self, niches: list[str] | str, comments: list[str]
    ) -> str:
        niches_str = _format_niches(niches)
        lines = [
            f"Niches: {niches_str}",
            "",
            "Commentaires (un par ligne, ordre conservé) :",
        ]
        for i, c in enumerate(comments, start=1):
            lines.append(f"{i}. {c}")
        return "\n".join(lines)

    def classify(
        self,
        comments: list[str],
        niches: list[str] | str,
    ) -> dict[str, Any]:
        """Envoie les commentaires à Ollama et retourne type, confidence,
        patterns, tone, brand_risk.

        ``niches`` accepte indifféremment une liste ``list[str]`` (schéma
        2026-05) ou une string (rétro-compat). En interne on formate via
        ``_format_niches`` qui produit une chaîne ``"a, b, c"``.
        """
        if not comments:
            raise ValueError("comments ne doit pas être vide")

        user_content = self._build_user_message(niches, comments)
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


def _resolve_ttype_prompt(t_type: str | None) -> tuple[str, str]:
    """Retourne ``(t_type_effectif, template)``.

    Si ``t_type`` est inconnu / vide / pas dans ``GENERATE_PROMPTS_BY_TTYPE``,
    on retombe sur le prompt **T2** (registre le plus neutre parmi les
    générables — humour de niche, applicable à 80 % des cas). Le t_type
    effectif est aussi retourné pour que le caller puisse l'auditer.
    """
    key = str(t_type or "").strip()
    if key in GENERATE_PROMPTS_BY_TTYPE:
        return key, GENERATE_PROMPTS_BY_TTYPE[key]
    return GENERATE_FALLBACK_TTYPE, GENERATE_PROMPTS_BY_TTYPE[GENERATE_FALLBACK_TTYPE]


def _format_niches(niches: list[str] | str | None) -> str:
    """Joint les niches en une chaîne lisible pour le prompt.

    - ``list[str]`` → ``", ".join(niches)`` après strip et filtrage des
      chaînes vides. Si la liste résultante est vide → ``"(non précisée)"``.
    - ``str`` (rétro-compat) → strip puis fallback. La string n'est pas
      retravaillée — si un caller passe ``"humour, sketch"``, c'est
      conservé tel quel.
    - ``None`` → ``"(non précisée)"``.

    Garantit une string non vide en sortie (le placeholder ``{niches}``
    apparaît sinon vide dans le prompt et perturbe le LLM).
    """
    if niches is None:
        return "(non précisée)"
    if isinstance(niches, str):
        s = niches.strip()
        return s or "(non précisée)"
    if isinstance(niches, list):
        clean = [str(n).strip() for n in niches if isinstance(n, str) and str(n).strip()]
        return ", ".join(clean) if clean else "(non précisée)"
    return "(non précisée)"


def _normalize_video_context(
    video_context: dict[str, Any] | None,
    *,
    niches: list[str] | str | None,
) -> dict[str, str]:
    """Construit le contexte de format ``str.format(**ctx)`` du prompt T-type.

    On force des **strings** sur tous les champs : ``str.format`` doit pouvoir
    interpoler sans surprise même si l'appelant passe ``None`` ou des
    nombres. Les hashtags sont normalisés en chaîne ``#tag #tag2`` (ou
    ``(aucun)`` si vide) — c'est ce que le LLM voit le plus souvent dans le
    contenu réel d'Instagram.
    """
    ctx = video_context or {}
    raw_hashtags = ctx.get("hashtags") or []
    if isinstance(raw_hashtags, str):
        # Caller a passé une string pré-formatée : on la respecte.
        hashtags_str = raw_hashtags.strip() or "(aucun)"
    else:
        tags = [str(h).lstrip("#").strip() for h in raw_hashtags if str(h).strip()]
        hashtags_str = " ".join(f"#{t}" for t in tags) if tags else "(aucun)"

    return {
        "niches": _format_niches(niches),
        "caption": str(ctx.get("caption") or "").strip() or "(vide)",
        "hashtags": hashtags_str,
    }


def _coerce_named_axis(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_named_axes_block(named_axes: dict[str, Any]) -> str:
    values = [
        f"{axis}={_coerce_named_axis(named_axes.get(axis)):.2f}"
        for axis in NAMED_AXES
    ]
    return (
        "Profil créateur:\n"
        f"  {values[0]} {values[1]}\n"
        f"  {values[2]} {values[3]}\n"
        f"  {values[4]} {values[5]}\n"
        f"  {values[6]} {values[7]}\n"
        f"  {values[8]} {values[9]}"
    )


def _clean_generated_comment_line(text: str) -> str:
    """Première ligne utile après ``### Response:`` (sans JSON legacy)."""
    line = str(text or "").strip().split("\n", 1)[0].strip()
    if line.startswith("{"):
        return ""
    for prefix in ("### Response:", "### Input:", "### Instruction:"):
        if line.startswith(prefix):
            line = line[len(prefix) :].strip()
    return line.strip('"').strip("'")


def _generate_comments_alpaca(
    *,
    t_type_profile: str,
    niches: list[str] | str,
    video_context: dict[str, Any] | None,
    named_axes: dict[str, Any] | None,
    model: str,
    ollama_url: str,
    num_comments: int = 3,
) -> list[str]:
    """3 commentaires via le modèle fine-tuné (un appel Alpaca par commentaire)."""
    ctx = video_context or {}
    raw_hashtags = ctx.get("hashtags") or []
    if isinstance(raw_hashtags, str):
        hashtags: str | list[Any] = raw_hashtags
    else:
        tags = [str(h).strip() for h in raw_hashtags if str(h).strip()]
        hashtags = tags

    input_block = build_generator_input_block(
        t_type_profile=t_type_profile,
        niches=niches,
        caption=str(ctx.get("caption") or "").strip(),
        hashtags=hashtags,
        audio_id=str(ctx.get("audio_id") or ctx.get("audio") or "").strip(),
        named_axes=named_axes if isinstance(named_axes, dict) and named_axes else None,
    )
    prompt = build_alpaca_prompt(input_block, instruction=GENERATOR_INSTRUCTION)

    out: list[str] = []
    seen: set[str] = set()
    for _ in range(max(num_comments * 2, num_comments)):
        if len(out) >= num_comments:
            break
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.85, "top_p": 0.9, "num_predict": 40},
        }
        try:
            resp = _http_post(ollama_url, json_body=body, timeout=120)
            raw_text = _ollama_response_text(resp)
        except (requests.RequestException, ValueError) as e:
            raise ClassificationError(f"Erreur Ollama generator: {e}") from e

        comment = _clean_generated_comment_line(raw_text)
        if not comment:
            continue
        key = comment.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(comment)

    while len(out) < num_comments:
        out.append("—")
    return out[:num_comments]


def _format_comments_sample(
    comments_sample: list[str], *, max_lines: int = 20
) -> str:
    """Joint les vrais commentaires humains en bloc lisible pour le LLM.

    Le brief insiste : c'est le **signal le plus fort** pour calibrer le
    registre — on en passe jusqu'à 20 (vs 40 dans l'ancien prompt). Pas
    de numérotation : on veut imiter, pas analyser.
    """
    lines: list[str] = []
    for c in comments_sample[:max_lines]:
        s = str(c or "").strip()
        if s:
            lines.append(f"- {s}")
    return "\n".join(lines) if lines else "(aucun commentaire disponible)"


def generate_comments(
    classification: dict[str, Any],
    comments_sample: list[str],
    *,
    niches: list[str] | str = "",
    t_type_profile: str | None = None,
    video_context: dict[str, Any] | None = None,
    named_axes: dict[str, Any] | None = None,
) -> list[str]:
    """Produit 3 commentaires via Ollama, prompt **spécialisé par T-type**.

    Pipeline :

    1. Lit ``classification["type"]`` (T-type **du contenu** — sert à choisir
       le template) ; fallback **T2** si inconnu / absent / non couvert
       (cf. ``_resolve_ttype_prompt``).
    2. Construit le contexte vidéo : ``niches``, ``caption``, ``hashtags``
       depuis ``video_context`` (rétro-compat ``video_context=None``).
    3. Injecte ``t_type_profile`` (T-type **du commentateur**, lu dans
       ``watchlist.json`` côté caller) — c'est notre persona, distincte du
       T-type du contenu : on peut commenter en T3b un contenu T2.
    4. Injecte ``comments_sample[:20]`` comme exemples de registre humain.
    5. Appelle Ollama avec ``GENERATE_COMMENTS_SYSTEM`` (système global :
       règles absolues anti-marketing) + le template T-type comme user.

    En phase **Watcher**, ``comments_sample`` est typiquement vide (pas de
    scrape sur post frais) et tout repose sur ``video_context`` +
    ``t_type_profile``. En phase Discovery / generator de masse, on a au
    contraire ``comments_sample`` riche et ``video_context`` peut être
    ``None``.

    ``niches`` accepte ``list[str]`` (schéma 2026-05) ou ``str``
    (rétro-compat). ``t_type_profile=None`` produit ``"(non précisé)"``
    dans le prompt — utile pour les générations de masse hors watcher.

    Si ``OLLAMA_GENERATOR_MODEL`` est défini dans ``.env``, utilise le
    modèle fine-tuné (format Alpaca, une ligne par commentaire). Sinon,
    prompt T-type + JSON ``{"comments": [...]}`` via ``OLLAMA_MODEL``.
    """
    profile_tt = (t_type_profile or "").strip() or "(non précisé)"
    finetuned_model = getattr(config, "OLLAMA_GENERATOR_MODEL", "") or ""
    if finetuned_model:
        axes = named_axes if isinstance(named_axes, dict) and named_axes else None
        return _generate_comments_alpaca(
            t_type_profile=profile_tt,
            niches=niches,
            video_context=video_context,
            named_axes=axes,
            model=finetuned_model,
            ollama_url=config.OLLAMA_URL,
        )

    t_type_content, template = _resolve_ttype_prompt(
        (classification or {}).get("type")
    )
    ctx = _normalize_video_context(video_context, niches=niches)
    prompt = template.format(
        t_type_profile=profile_tt,
        niches=ctx["niches"],
        caption=ctx["caption"],
        hashtags=ctx["hashtags"],
        comments_sample=_format_comments_sample(comments_sample),
    )
    axes = named_axes if isinstance(named_axes, dict) and named_axes else None
    if axes:
        profile_line = f"T-type commentateur : {profile_tt}"
        prompt = prompt.replace(
            profile_line,
            f"{profile_line}\n{_format_named_axes_block(axes)}",
            1,
        )

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
        raise ClassificationError(
            f"{e} (t_type={t_type_content})", raw_text=raw_text
        ) from e
