"""Audit labeling sessions in database/experiment/instances.

Read-only: prints a per-session and per-participant summary with flags.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cohort_config import add_cohort_args, resolve_cohort_from_args

ROOT = Path(__file__).resolve().parents[1] / "database" / "experiment" / "instances"


def load(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_ts(value):
    if value is None or (isinstance(value, float) and value != value):
        return None
    return datetime.fromisoformat(value)


def build_usernames(root=None):
    """Map user_id -> username, collected across all instances."""
    if root is None:
        root = ROOT
    mapping = {}
    for d in sorted(root.iterdir(), key=lambda p: int(p.name)):
        if not d.is_dir():
            continue
        db = d / "duckdb"
        for u in load(db / "users.json") or []:
            if u["username"] != "system":
                mapping[u["user_id"]] = u["username"]
        for e in load(db / "al_events.json") or []:
            if e.get("actor_type") == "human":
                mapping[e["user_id"]] = e["agent"]
    return mapping


def session_summary(instance_dir, usernames, min_minutes, max_minutes):
    db = instance_dir / "duckdb"
    al_instances = load(db / "al_instances.json") or []
    decisions = load(db / "label_decisions.json") or []
    events = load(db / "al_events.json") or []

    al_instance = al_instances[0] if al_instances else {}
    user_id = al_instance.get("user_id")
    username = usernames.get(user_id, "?")

    labeled = [d for d in decisions if d.get("labeled_at")]
    times = [parse_ts(d["labeled_at"]) for d in labeled]
    span = (max(times) - min(times)).total_seconds() / 60 if times else 0.0
    latencies = [d["latency_ms"] for d in decisions if d.get("latency_ms") is not None]
    n_idk = sum(1 for d in decisions if d.get("i_dont_know"))
    null_ts = len(decisions) - len(labeled)
    refs = [d["ref"] for d in decisions]
    dup_refs = len(refs) - len(set(refs))
    event_refs = {
        e["object_id"]
        for e in events
        if e.get("action") in ("confirm_label", "override_label")
    }
    valid_refs = {
        d["ref"]
        for d in decisions
        if not d.get("i_dont_know") and d.get("labeled_at")
    }

    flags = []
    if not decisions:
        flags.append("EMPTY")
    else:
        if not (min_minutes <= span <= max_minutes):
            flags.append(f"SPAN({span:.0f}m)")
        if null_ts:
            flags.append(f"NULL_TS({null_ts})")
        if dup_refs:
            flags.append(f"DUP_REFS({dup_refs})")
        if event_refs != valid_refs:
            extra = len(event_refs - valid_refs)
            missing = len(valid_refs - event_refs)
            flags.append(f"EVENT_MISMATCH(+{extra}/-{missing})")

    return {
        "instance": int(instance_dir.name),
        "username": username,
        "user_id": user_id,
        "created_at": parse_ts(al_instance.get("created_at")),
        "n_decisions": len(decisions),
        "n_idk": n_idk,
        "null_ts": null_ts,
        "dup_refs": dup_refs,
        "span_min": round(span, 1),
        "median_latency_s": round(median(latencies) / 1000, 1) if latencies else "-",
        "first": min(times) if times else None,
        "last": max(times) if times else None,
        "flags": flags,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-minutes", type=float, default=60)
    parser.add_argument("--max-minutes", type=float, default=120)
    add_cohort_args(parser)
    args = parser.parse_args()

    root = resolve_cohort_from_args(args).root
    sessions = []
    usernames = build_usernames(root)
    for d in sorted(root.iterdir(), key=lambda p: int(p.name)):
        if d.is_dir():
            sessions.append(
                session_summary(d, usernames, args.min_minutes, args.max_minutes)
            )

    print("=== Per-session summary ===")
    header = (
        f"{'inst':>4} {'user':<10} {'n':>4} {'idk':>4} {'null':>4} {'dup':>4} "
        f"{'span_m':>7} {'med_lat_s':>9} {'first':>16} {'flags'}"
    )
    print(header)
    for s in sessions:
        first = s["first"].strftime("%m-%d %H:%M") if s["first"] else "-"
        print(
            f"{s['instance']:>4} {s['username']:<10} {s['n_decisions']:>4} "
            f"{s['n_idk']:>4} {s['null_ts']:>4} {s['dup_refs']:>4} "
            f"{s['span_min']:>7} {s['median_latency_s']:>9} {first:>16} "
            f"{','.join(s['flags']) or 'OK'}"
        )

    by_user = {}
    for s in sessions:
        by_user.setdefault((s["username"], s["user_id"]), []).append(s)

    median_n = median(
        [s["n_decisions"] for s in sessions if s["n_decisions"] > 0] or [0]
    )

    print("\n=== Per-participant summary ===")
    print(f"{'user':<10} {'sessions':>8} {'n_labels':>18} {'gap_days':>9} {'flags'}")
    for (username, _), ss in sorted(by_user.items()):
        n_labels = "/".join(str(s["n_decisions"]) for s in ss)
        flags = []
        if len(ss) != 2:
            flags.append(f"SESSIONS({len(ss)})")
        if any(s["n_decisions"] < median_n / 2 for s in ss):
            flags.append("LOW_COUNT")
        gap = ""
        if len(ss) == 2 and all(s["first"] for s in ss):
            start1 = min(s["first"] for s in ss)
            start2 = max(s["first"] for s in ss)
            gap = f"{(start2 - start1).total_seconds() / 86400:.1f}"
            if start2 - start1 > __import__("datetime").timedelta(days=3):
                flags.append("BIG_GAP")
        print(
            f"{username:<10} {len(ss):>8} {n_labels:>18} {gap:>9} "
            f"{','.join(flags) or 'OK'}"
        )


if __name__ == "__main__":
    main()
