"""Shared LLM labeling logic: personas, prompts, OpenRouter calls, caching, validation.

Used by scripts/api_eval.py (live loop) and scripts/replay_eval.py (offline replay)
so both always use byte-identical prompts.
"""

import hashlib
import json
import os
import time
from pathlib import Path

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TEMPERATURE = 0
CACHE_PATH = Path("runs/replay/llm_cache.json")
PROMPTS_DIR = Path("prompts")

RETRY_NOTE = (
    "\n\nYour previous answer was not valid JSON or did not name a team from the list exactly. "
    "Answer again with JSON only, using one team name verbatim from the list."
)


def load_env(path=".env"):
    path = Path(path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_persona(name, prompts_dir=None):
    folder = Path(prompts_dir) if prompts_dir else PROMPTS_DIR
    path = folder / f"{name}.json"
    if not path.exists():
        available = sorted(p.stem for p in folder.glob("*.json"))
        raise SystemExit(f"Unknown persona '{name}'. Available: {', '.join(available)}")
    return json.loads(path.read_text(encoding="utf-8")).get("description", "")


def system_prompt(teams, persona_text=None):
    text = (
        "You are an IT support agent triaging incoming tickets. "
        "Choose the team that should handle the ticket.\n\n"
        "The possible teams are:\n"
        + "\n".join(f"- {t}" for t in teams)
        + "\n\nRespond with JSON only:\n"
        '{"team": "<exact team name from the list>", '
        '"difficult": <true if the ticket is somewhat ambiguous but you can still make a reasoned choice>, '
        '"dont_know": <true if you do not know which team to assign and choosing would be a pure guess>, '
        '"confidence": <0.0-1.0>}'
    )
    if persona_text:
        text += "\n\n" + persona_text
    return text


def user_prompt(ticket, bundle, teams):
    lines = [
        "Label the following IT support ticket.",
        "",
        f"Subcategory: {ticket.get('Service subcategory->Name')}",
        f"Service: {ticket.get('Service->Name')}",
        f"Title: {ticket.get('Title_anon')}",
        "Description:",
        str(ticket.get("Description_anon") or ""),
    ]
    if bundle:
        lines += assistance_block(bundle, teams)
    return "\n".join(lines)


def assistance_block(bundle, teams):
    proba = bundle["infer_proba"]
    allowed = set(teams)
    ranked = [
        (c, p)
        for c, p in sorted(
            zip(proba["classes"], proba["probabilities"][0]),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if c in allowed
    ]
    lines = ["", "AI assistance (suggestions may be wrong; you decide):"]
    lines.append(
        f"Model prediction: {ranked[0][0]} ({ranked[0][1]:.0%}); "
        f"runner-up: {ranked[1][0]} ({ranked[1][1]:.0%})"
    )
    weights = bundle["explain_lime"][0]["word_weights"]
    support = [w for w, s in weights if s > 0]
    oppose = [w for w, s in weights if s < 0]
    lines.append(f"LIME words supporting the prediction: {', '.join(support) or '-'}")
    lines.append(f"LIME words opposing the prediction: {', '.join(oppose) or '-'}")
    nearest = bundle["nearest"][0]
    for heading, key in [
        (
            "Similar tickets from the predicted team "
            "(examples supporting the suggestion, not necessarily correct for this ticket):",
            "predicted_class_neighbors",
        ),
        (
            "Similar past tickets (from the historical pool, "
            "selected independently of the model's suggestion):",
            "historical_neighbors",
        ),
    ]:
        lines.append(heading)
        for nb in nearest[key]:
            lines.append(
                f"- [{nb['label']}] {str(nb['title']).strip()} - "
                f"{str(nb['best_sentence']).strip()} (similarity {nb['similarity']:.2f})"
            )
    return lines


def cache_key(model, system, user):
    return hashlib.sha1(f"{model}\n{system}\n{user}".encode("utf-8")).hexdigest()


def load_cache(path=None):
    path = Path(path) if path else CACHE_PATH
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache, path=None):
    path = Path(path) if path else CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def call_llm(client, model, system, user):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return {
        "raw": resp.choices[0].message.content,
        "usage": {"prompt": resp.usage.prompt_tokens, "completion": resp.usage.completion_tokens},
        "latency_ms": int((time.time() - t0) * 1000),
    }


def call_llm_cached(client, model, system, user, cache):
    if cache is None:
        return call_llm(client, model, system, user), False
    key = cache_key(model, system, user)
    if key in cache:
        return cache[key], True
    out = call_llm(client, model, system, user)
    cache[key] = out
    return out, False


def parse_decision(raw, teams):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    team = str(data.get("team", "")).strip()
    if team not in teams:
        return None
    return {
        "label": team,
        "difficult": bool(data.get("difficult")),
        "dont_know": bool(data.get("dont_know")),
        "confidence": data.get("confidence"),
    }


def ask_llm(client, model, teams, ticket, bundle, persona_text=None, cache=None):
    system = system_prompt(teams, persona_text)
    user = user_prompt(ticket, bundle, teams)
    out, cached = call_llm_cached(client, model, system, user, cache)
    decision = parse_decision(out["raw"], teams)
    if decision is None:
        out, cached = call_llm_cached(client, model, system, user + RETRY_NOTE, cache)
        decision = parse_decision(out["raw"], teams)
    return decision, {**out, "cached": cached}
