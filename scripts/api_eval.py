"""Run the labeling eval against the HumAL API.

Two parts share one loop: control (no AI help) and treatment (with AI help).
Labels are chosen by the pluggable labeler (LABELER): random or model_accept; an LLM labeler comes later.

Usage:
    uv run python scripts/api_eval.py --mode control|treatment|both
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
USERNAME = "test"
PASSWORD = "test"
N_TICKETS = 10
MODEL_NAME = "logistic regression"
QS_STRATEGIES = ["Experiment 1", "Experiment 2"]
RUN_OUTPUT_DIR = Path("runs/api_eval")
LABELER = "random"


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

    def call(self, method, path, **kwargs):
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
        return r.json()


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


LABELERS = {"random": RandomLabeler, "model_accept": ModelAcceptLabeler}


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
            }
        )
    return records


def save_run(mode, qs, records):
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = qs.strip().lower().replace(" ", "-")
    path = RUN_OUTPUT_DIR / f"{ts}_{mode}_{slug}.json"
    payload = {
        "mode": mode,
        "model_name": MODEL_NAME,
        "qs_strategy": qs,
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
    for mode in modes:
        idx = 0 if mode == "control" else min(1, len(QS_STRATEGIES) - 1)
        records = run(client, mode, QS_STRATEGIES[idx], n_tickets, teams, labeler)
        save_run(mode, QS_STRATEGIES[idx], records)


if __name__ == "__main__":
    main()
