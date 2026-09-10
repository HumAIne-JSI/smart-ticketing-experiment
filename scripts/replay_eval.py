"""Offline replay: rebuild prompts from saved api_eval run logs, call an LLM via OpenRouter, score vs GT.

No backend involved: tickets, assistance bundles and modes come from runs/api_eval/*.json.
Ground truth comes from the Excel (Team->Name column).
Prompts, LLM calls and caching come from the shared scripts/llm_labeling.py module.

Usage (OPENROUTER_API_KEY env var required unless --dry-run):
    uv run python scripts/replay_eval.py --run "runs/api_eval/<file>.json" [--model ID] [--persona NAME] [--limit N] [--dry-run]
"""

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI

from llm_labeling import (
    OPENROUTER_BASE_URL,
    RETRY_NOTE,
    TEMPERATURE,
    cache_key,
    call_llm,
    call_llm_cached,
    load_cache,
    load_env,
    load_persona,
    parse_decision,
    save_cache,
    system_prompt,
    user_prompt,
)

LLM_MODEL = "openai/gpt-5.6-luna"
GT_XLSX = Path("data/User Request_last_team_ANON.xlsx")
RUN_OUTPUT_DIR = Path("runs/replay")

TEAMS = [
    "(GI-UX) Group",
    "(BF) Employee Platform (SAP SF)",
    "(GI-SaaS) Salesforce",
    "(GI-UX) Account Management",
    "(GI-IaaS) Network Cloud (Azure, Remote Access)",
    "(GI-UX) System Management & Anti Virus",
    "(GI-SM) Service Desk",
    "(GI-IaaS) Development Platform",
    "(LF) IT Office Access Italy",
    "(GI-UX) Mobile Device Management",
    "(GI-UX) Windows",
    "(GI-IaaS) Cloud Services",
    "(GI-IaaS) Admin - License & Asset Management",
    "(GI-IaaS) Network On-Prem (LAN,WLAN,WAN 2nd level)",
    "(GI-UX) Office365 & MS-Teams",
    "(GI-UX) Network Access",
    "(GI-UX) Unified Communication",
    "(GI-IaaS) Azure Data Center",
    "(GI-IaaS) Backend Application Srv. & Project Support",
    "(GI-UX) Application",
    "(GI-CyberSec) Security Operation Center",
    "(BF) Information Security Office",
    "(GI-UX) File & Print",
    "(GI-SaaS) SAP & Synertrade",
]


def load_gt():
    df = pd.read_excel(GT_XLSX).drop_duplicates(subset="Ref", keep="first")
    return {str(r["Ref"]): r["Team->Name"] for _, r in df.iterrows() if pd.notna(r["Team->Name"])}


def decide(client, model, teams, persona_text, user, cache):
    system = system_prompt(teams, persona_text)
    out, cached = call_llm_cached(client, model, system, user, cache)
    decision = parse_decision(out["raw"], teams)
    if decision is None:
        out, cached = call_llm_cached(client, model, system, user + RETRY_NOTE, cache)
        decision = parse_decision(out["raw"], teams)
    return decision, out, cached


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "N/A"


def replay_file(client, model, run_path, gt, cache, limit, dry_run, persona=None):
    data = json.loads(Path(run_path).read_text(encoding="utf-8"))
    mode = data["mode"]
    persona_name = persona or data.get("persona") or "base"
    persona_text = load_persona(persona_name)
    system = system_prompt(TEAMS, persona_text)
    records = data["records"][:limit] if limit else data["records"]
    print(f"[{mode}] replaying {len(records)} records from {run_path} (persona={persona_name})")

    results = []
    if dry_run:
        for r in records:
            print(f"--- {r['ref']} ---")
            print(user_prompt(r["ticket"], r["assistance_bundle"], TEAMS))
            print()
        return

    n_ok = n_fail = n_match = n_dont_know = n_difficult = 0
    base_random = base_model = base_n = 0
    tokens = {"prompt": 0, "completion": 0}
    cache_hits = 0
    for r in records:
        prompt = user_prompt(r["ticket"], r["assistance_bundle"], TEAMS)
        decision, out, cached = decide(client, model, TEAMS, persona_text, prompt, cache)
        cache_hits += int(cached)
        tokens["prompt"] += out["usage"]["prompt"]
        tokens["completion"] += out["usage"]["completion"]
        gt_label = gt.get(r["ref"])
        match = int(decision is not None and gt_label is not None and decision["label"] == gt_label)
        if decision is None:
            n_fail += 1
        else:
            n_ok += 1
            n_match += match
            n_dont_know += int(decision["dont_know"])
            n_difficult += int(decision["difficult"])
        if gt_label is not None:
            base_n += 1
            base_random += int(r["label"] == gt_label)
            if r["model_prediction"]:
                base_model += int(r["model_prediction"] == gt_label)
        print(
            f"  {r['ref']} -> {decision['label'] if decision else 'PARSE FAILURE'} "
            f"(GT: {gt_label}){' [cached]' if cached else ''}"
        )
        results.append(
            {
                "idx": r["idx"],
                "ref": r["ref"],
                "persona": persona_name,
                "prompt": prompt,
                "prompt_hash": cache_key(model, system, prompt),
                "llm_label": decision["label"] if decision else None,
                "difficult": decision["difficult"] if decision else None,
                "dont_know": decision["dont_know"] if decision else None,
                "confidence": decision["confidence"] if decision else None,
                "gt": gt_label,
                "match": bool(match) if decision else None,
                "usage": out["usage"],
                "latency_ms": out["latency_ms"],
                "cached": cached,
                "raw": out["raw"],
            }
        )

    model_gt_total = sum(1 for r in records if r["model_prediction"])
    print(
        f"[{mode}] records={len(records)} parse_failures={n_fail} | "
        f"llm/GT={pct(n_match, n_ok)} dont_know={pct(n_dont_know, n_ok)} "
        f"difficult={pct(n_difficult, n_ok)} | "
        f"baseline random/GT={pct(base_random, base_n)} model/GT={pct(base_model, model_gt_total)}"
    )
    print(f"[{mode}] tokens: prompt={tokens['prompt']} completion={tokens['completion']} cache_hits={cache_hits}")

    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = model.replace("/", "_")
    path = RUN_OUTPUT_DIR / f"{ts}_{mode}_{persona_name}_{slug}.json"
    payload = {
        "run_file": str(run_path),
        "mode": mode,
        "model": model,
        "persona": persona_name,
        "temperature": TEMPERATURE,
        "records": results,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{mode}] results written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Offline LLM replay of api_eval run logs.")
    parser.add_argument("--run", required=True, help="Run log path or glob (runs/api_eval/*.json).")
    parser.add_argument("--model", default=None, help="Override LLM_MODEL.")
    parser.add_argument("--persona", default=None, help="Override persona (default: from run log, else base).")
    parser.add_argument("--limit", type=int, default=None, help="Replay only the first N records.")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts, no API calls.")
    args = parser.parse_args()

    load_env()
    model = args.model or LLM_MODEL
    run_paths = sorted(glob.glob(args.run))
    if not run_paths:
        raise SystemExit(f"No files match --run {args.run}")
    gt = load_gt()

    client = None
    if not args.dry_run:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise SystemExit("Set the OPENROUTER_API_KEY env var.")
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])

    cache = {} if args.dry_run else load_cache()
    for run_path in run_paths:
        replay_file(client, model, run_path, gt, cache, args.limit, args.dry_run, args.persona)
    if not args.dry_run and cache:
        save_cache(cache)
        print(f"cache written to {len(cache)} entries")


if __name__ == "__main__":
    main()
