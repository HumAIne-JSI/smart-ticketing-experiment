"""Run the labeling eval against the HumAL API.

Two parts share one loop: control (no AI help) and treatment (with AI help).
Labels are chosen by the pluggable labeler (LABELER): random, model_accept or llm.

Usage:
    uv run python scripts/api_eval.py --mode control|treatment|both

After a successful run, export-all is extracted to database/llm_experiment/export_all_<ts>
and the instances created by the run are deleted (SAVE_EXPORT / DELETE_INSTANCE flags).
"""

import argparse
import io
import json
import os
import random
import time
import zipfile
from datetime import datetime
from pathlib import Path

import requests
from openai import OpenAI

from llm_labeling import (
    OPENROUTER_BASE_URL,
    ask_llm,
    load_cache,
    load_env,
    load_persona,
    save_cache,
)

BASE_URL = "http://localhost:8000"
USERNAME = "test"
PASSWORD = "test"
N_TICKETS = 10
MODEL_NAME = "logistic regression"
QS_STRATEGIES = ["Experiment 1", "Experiment 2"]
RUN_OUTPUT_DIR = Path("runs/api_eval")
EXPORT_DIR = Path("database/llm_experiment")
SAVE_EXPORT = True
DELETE_INSTANCE = True
LABELER = "random"
LLM_MODEL = "openai/gpt-5.6-luna"
PERSONA = "junior"


class Client:
    def __init__(self):
        self.base = BASE_URL.rstrip("/")
        self.username = USERNAME
        self.password = PASSWORD
        self.session = requests.Session()

    def login(self):
        r = self.session.post(
            f"{self.base}/users/login",
            json={"username": self.username, "password": self.password},
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Login failed -> {r.status_code}: {r.text[:200]}")
        self.session.headers["Authorization"] = f"Bearer {r.json()['access_token']}"

    def _request_with_retries(self, method, path, **kwargs):
        url = f"{self.base}{path}"
        last_exc = None
        for attempt in range(3):
            try:
                r = self.session.request(method, url, timeout=300, **kwargs)
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(2)
                continue
            if r.status_code == 401 and attempt < 2:
                self.login()
                continue
            if r.status_code >= 500 and attempt < 2:
                time.sleep(2)
                continue
            break
        else:
            raise RuntimeError(f"{method} {url} failed after retries: {last_exc}")
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {url} -> {r.status_code}: {r.text[:300]}")
        return r

    def call(self, method, path, **kwargs):
        return self._request_with_retries(method, path, **kwargs).json()

    def call_raw(self, method, path, **kwargs):
        return self._request_with_retries(method, path, **kwargs).content


def extract_instance_id(resp):
    if isinstance(resp, int):
        return resp
    if isinstance(resp, dict):
        for key in ("al_instance_id", "instance_id", "id"):
            if key in resp:
                return resp[key]
    raise RuntimeError(f"Cannot find instance id in response: {json.dumps(resp)[:300]}")


def rank_predictions(proba):
    return sorted(
        zip(proba["classes"], proba["probabilities"][0]),
        key=lambda pair: pair[1],
        reverse=True,
    )


class RandomLabeler:
    def label_ticket(self, ticket, assistance_bundle, teams):
        return {"label": random.choice(teams), "difficult": None, "dont_know": None, "raw": None}


class ModelAcceptLabeler:
    def label_ticket(self, ticket, assistance_bundle, teams):
        return {
            "label": rank_predictions(assistance_bundle["infer_proba"])[0][0],
            "difficult": None,
            "dont_know": None,
            "raw": None,
        }


class LLMLabeler:
    def __init__(self):
        load_env()
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("Set OPENROUTER_API_KEY (env var or .env).")
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
        self.model = LLM_MODEL
        self.persona_text = load_persona(PERSONA)
        self.cache = load_cache()
        self.cache_hits = 0

    def label_ticket(self, ticket, assistance_bundle, teams):
        decision, out = ask_llm(
            self.client, self.model, teams, ticket, assistance_bundle,
            persona_text=self.persona_text, cache=self.cache,
        )
        save_cache(self.cache)
        self.cache_hits += int(out["cached"])
        if decision is None:
            raise RuntimeError(f"LLM returned invalid team twice: {out['raw'][:200]}")
        return {
            **decision,
            "raw": out["raw"],
            "usage": out["usage"],
            "latency_ms": out["latency_ms"],
            "cached": out["cached"],
        }


LABELERS = {"random": RandomLabeler, "model_accept": ModelAcceptLabeler, "llm": LLMLabeler}


def run(client, mode, qs, n_tickets, teams, labeler):
    resp = client.call(
        "POST",
        "/activelearning/new",
        json={"model_name": MODEL_NAME, "qs_strategy": qs, "class_list": teams},
    )
    instance_id = extract_instance_id(resp)
    print(f"[{mode}] instance {instance_id} created (model={MODEL_NAME}, qs={qs})")

    records = []
    for i in range(1, n_tickets + 1):
        start = datetime.now()
        nxt = client.call("GET", f"/activelearning/{instance_id}/next")
        refs = nxt["query_idx"] if isinstance(nxt, dict) else nxt
        ref = refs[0]

        tickets = client.call("POST", "/data/tickets", json=[ref])["tickets"]
        ticket = tickets[0] if tickets else {}

        model_prediction = second_model_prediction = None
        bundle = None
        if mode == "treatment":
            proba = client.call(
                "POST", f"/activelearning/{instance_id}/infer_proba?query_idx={ref}"
            )
            ranked = rank_predictions(proba)
            model_prediction, second_model_prediction = ranked[0][0], ranked[1][0]
            bundle = {
                "infer_proba": proba,
                "explain_lime": client.call(
                    "POST", f"/xai/{instance_id}/explain_lime?query_idx={ref}"
                ),
                "nearest": client.call(
                    "POST", f"/xai/{instance_id}/nearest?query_idx={ref}"
                ),
            }

        decision = labeler.label_ticket(ticket, bundle, teams)
        label = decision["label"]
        end = datetime.now()
        info = {
            "ticket_id": ref,
            "label": label,
            "model_prediction": model_prediction,
            "second_model_prediction": second_model_prediction,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "most_helpful_feature": None,
            "is_tired": None,
            "is_difficult": decision["difficult"],
            "i_dont_know": decision["dont_know"],
        }
        client.call(
            "POST", f"/activelearning/{instance_id}/label-with-info", json=[info]
        )

        print(f"[{mode}] {i}/{n_tickets} {ref} -> {label} ({(end - start).total_seconds():.2f}s)")
        records.append(
            {
                "idx": i,
                "ref": ref,
                "ticket": ticket,
                "assistance_bundle": bundle,
                "label": label,
                "model_prediction": model_prediction,
                "second_model_prediction": second_model_prediction,
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "decision": decision,
            }
        )
    return records, instance_id


def export_and_cleanup(client, created_ids):
    if SAVE_EXPORT:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = EXPORT_DIR / f"export_all_{ts}"
        data = client.call_raw("GET", "/activelearning/export-all")
        zf = zipfile.ZipFile(io.BytesIO(data))
        target.mkdir(parents=True, exist_ok=True)
        zf.extractall(target)
        print(f"export extracted to {target} ({len(zf.namelist())} files)")
    if not DELETE_INSTANCE:
        return
    if not SAVE_EXPORT:
        raise SystemExit("DELETE_INSTANCE requires SAVE_EXPORT (deleting without export loses data).")
    for instance_id in created_ids:
        try:
            client.call("DELETE", f"/activelearning/{instance_id}")
            print(f"instance {instance_id} deleted")
        except RuntimeError as exc:
            print(f"instance {instance_id} delete failed: {exc}")


def save_run(mode, qs, records):
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = qs.strip().lower().replace(" ", "-")
    path = RUN_OUTPUT_DIR / f"{ts}_{mode}_{slug}.json"
    payload = {
        "mode": mode,
        "model_name": MODEL_NAME,
        "qs_strategy": qs,
        "labeler": LABELER,
        "persona": PERSONA,
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{mode}] run log written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run the labeling eval via the HumAL API.")
    parser.add_argument("--mode", choices=["control", "treatment", "both"], default="both")
    parser.add_argument("--n-tickets", type=int, default=None, help="Override N_TICKETS.")
    args = parser.parse_args()

    n_tickets = args.n_tickets if args.n_tickets is not None else N_TICKETS
    labeler = LABELERS[LABELER]()
    client = Client()
    client.login()
    teams = client.call("GET", "/data/teams")["teams"]

    modes = ["control", "treatment"] if args.mode == "both" else [args.mode]
    if LABELER == "model_accept" and "control" in modes:
        raise SystemExit(
            "model_accept labeler requires treatment mode (no model prediction in control)"
        )
    created_ids = []
    for mode in modes:
        idx = 0 if mode == "control" else min(1, len(QS_STRATEGIES) - 1)
        records, instance_id = run(client, mode, QS_STRATEGIES[idx], n_tickets, teams, labeler)
        save_run(mode, QS_STRATEGIES[idx], records)
        created_ids.append(instance_id)
    if getattr(labeler, "cache", None) is not None:
        print(f"LLM cache: {labeler.cache_hits} hits")
    export_and_cleanup(client, created_ids)


if __name__ == "__main__":
    main()
