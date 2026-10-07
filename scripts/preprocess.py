"""Preprocess labeling decisions into analysis/decisions.csv and analysis/sessions.csv.

Session layout is defined in analysis/preprocess_config.json:
- "instances": list of instance indices merged into one session
- "trim_last_minutes": keep only decisions within the final N minutes
- "split_by_day": split merged decisions by labeled_at date (-d1, -d2, ...)
- "condition": "control"|"treatment", or "conditions_by_day" list for split sessions

Row cleaning: drop rows with null labeled_at (never labeled) and rows with
null label that are not "I don't know". idk rows are kept. Decision times
longer than LATENCY_CAP_MIN minutes are set to null (likely session breaks);
the decision itself is kept.

Session validity: a session must contain at least MIN_BLOCK_MIN minutes of
sustained labeling (longest same-day block; gaps up to BLOCK_GAP_MIN minutes
stay within a block). Shorter sessions are excluded entirely: printed during
the run, absent from the output CSVs. Participants left with only one
condition after this rule are dropped as well, so every condition-level
comparison is within-subject.

Also writes analysis/al_metrics.csv: one row per active-learning retrain
iteration (from <instance>/duckdb/metrics.json) with the session condition
assigned via the same day-split rule as the decisions.
"""

import json
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_sessions import ROOT, build_usernames, parse_ts

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"

LATENCY_CAP_MIN = 10
BLOCK_GAP_MIN = 30   # gaps up to this stay inside one sustained block
MIN_BLOCK_MIN = 45   # sessions whose longest block is shorter are excluded

COLUMNS = [
    "session_id",
    "condition",
    "al_instance_id",
    "ref",
    "label",
    "labeled_at",
    "model_prediction",
    "second_model_prediction",
    "latency_ms",
    "most_helpful_feature",
    "is_tired",
    "is_difficult",
    "i_dont_know",
    "skipped_for_training",
    "has_xai",
]


def load_instance_decisions(instance_id):
    path = ROOT / str(instance_id) / "duckdb" / "label_decisions.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


AL_METRIC_COLUMNS = ["al_instance_id", "iteration_id", "num_labeled", "f1_score", "created_at", "condition"]


def build_al_metrics(all_parts):
    parts = pd.concat(all_parts, ignore_index=True)
    parts["date"] = pd.to_datetime(parts["ts"]).dt.date
    cond_of = parts.groupby(["al_instance_id", "date"])["condition"].agg(lambda s: s.mode()[0])

    rows = []
    for inst in sorted(parts["al_instance_id"].unique()):
        path = ROOT / str(inst) / "duckdb" / "metrics.json"
        if not path.exists():
            continue
        for m in json.loads(path.read_text(encoding="utf-8")):
            day = parse_ts(m["created_at"]).date()
            per_day = cond_of.loc[inst]
            cond = per_day.get(day, per_day.iloc[0])
            rows.append({**{c: m[c] for c in AL_METRIC_COLUMNS[:-1]}, "condition": cond})
    return pd.DataFrame(rows)


def clean_rows(df, session_id):
    before = len(df)
    df = df[df["labeled_at"].notna()]
    null_ts = before - len(df)
    before = len(df)
    idk = df["i_dont_know"].fillna(False)
    df = df[df["label"].notna() | idk]
    bad_label = before - len(df)
    return df, null_ts, bad_label


def longest_block(part):
    """Minutes covered by the longest same-day run of decisions (gaps <= BLOCK_GAP_MIN)."""
    ts = part["ts"].sort_values()
    new_block = (ts.diff() > pd.Timedelta(minutes=BLOCK_GAP_MIN)) | \
                (ts.dt.date != ts.dt.date.shift())
    spans = ts.groupby(new_block.cumsum()).agg(["min", "max"])
    return (spans["max"] - spans["min"]).max().total_seconds() / 60


def resolve_conditions(spec, n_parts):
    if "condition" in spec and "conditions_by_day" in spec:
        raise ValueError(f"{spec['id']}: use either 'condition' or 'conditions_by_day', not both")
    if "conditions_by_day" in spec:
        if not spec.get("split_by_day"):
            raise ValueError(f"{spec['id']}: 'conditions_by_day' requires 'split_by_day'")
        if len(spec["conditions_by_day"]) != n_parts:
            raise ValueError(
                f"{spec['id']}: {n_parts} day-parts but {len(spec['conditions_by_day'])} conditions"
            )
        return list(spec["conditions_by_day"])
    if "condition" not in spec:
        raise ValueError(f"{spec['id']}: missing 'condition'")
    return [spec["condition"]] * n_parts


def build_session(spec, usernames):
    rows = []
    for inst in spec["instances"]:
        for d in load_instance_decisions(inst):
            rows.append({**d, "al_instance_id": inst})
    df = pd.DataFrame(rows)
    stats = {"raw": len(df)}

    if df.empty:
        return [], {"raw": 0, "dropped_null_ts": 0, "dropped_bad_label": 0}

    df["ts"] = df["labeled_at"].map(parse_ts)
    df = df.sort_values("ts")

    if "trim_last_minutes" in spec:
        cutoff = df["ts"].max() - timedelta(minutes=spec["trim_last_minutes"])
        stats["trimmed"] = int((df["ts"] < cutoff).sum())
        df = df[df["ts"] >= cutoff]

    cap_ms = LATENCY_CAP_MIN * 60_000
    stats["lat_capped"] = int((df["latency_ms"] > cap_ms).sum())
    df.loc[df["latency_ms"] > cap_ms, "latency_ms"] = float("nan")

    parts = []
    if spec.get("split_by_day"):
        for i, (_, day_df) in enumerate(df.groupby(df["ts"].dt.date, sort=True), 1):
            parts.append((f"{spec['id']}-d{i}", day_df))
    else:
        parts.append((spec["id"], df))

    out_rows = []
    conditions = resolve_conditions(spec, len(parts))
    for (session_id, part), condition in zip(parts, conditions):
        cleaned, null_ts, bad_label = clean_rows(part, session_id)
        stats["dropped_null_ts"] = stats.get("dropped_null_ts", 0) + null_ts
        stats["dropped_bad_label"] = stats.get("dropped_bad_label", 0) + bad_label
        block = longest_block(cleaned)
        if block < MIN_BLOCK_MIN:
            stats.setdefault("excluded", []).append((session_id, block))
            continue
        cleaned = cleaned.copy()
        cleaned["session_id"] = session_id
        cleaned["condition"] = condition
        cleaned["has_xai"] = cleaned["xai_result"].notna()
        out_rows.append(cleaned)

    return out_rows, stats


def main():
    config = json.loads((ANALYSIS / "preprocess_config.json").read_text(encoding="utf-8"))
    usernames = build_usernames()

    all_parts = []
    session_rows = []
    total_capped = 0
    for spec in config["sessions"]:
        parts, stats = build_session(spec, usernames)
        user = usernames.get(
            json.loads(
                (ROOT / str(spec["instances"][0]) / "duckdb" / "al_instances.json")
                .read_text(encoding="utf-8")
            )[0]["user_id"],
            "?",
        )
        for cleaned in parts:
            all_parts.append(cleaned)
            sid = cleaned["session_id"].iloc[0]
            ts = cleaned["ts"]
            span = (ts.max() - ts.min()).total_seconds() / 60 if len(ts) > 1 else 0
            dups = int(cleaned["ref"].duplicated().sum())
            session_rows.append(
                {
                    "session_id": sid,
                    "user": user,
                    "condition": cleaned["condition"].iloc[0],
                    "instances": ",".join(map(str, spec["instances"])),
                    "n_labels": len(cleaned),
                    "n_idk": int(cleaned["i_dont_know"].fillna(False).sum()),
                    "span_min": round(span, 1),
                    "dup_refs": dups,
                    "xai_share": round(cleaned["has_xai"].mean(), 2),
                    "n_latency_capped": stats.get("lat_capped", 0),
                    "longest_block_min": round(longest_block(cleaned), 1),
                }
            )
        dropped = stats.get("trimmed", 0) + stats.get("dropped_null_ts", 0) + stats.get("dropped_bad_label", 0)
        total_capped += stats.get("lat_capped", 0)
        excl = stats.get("excluded", [])
        note = "  EXCLUDED: " + ", ".join(f"{sid} ({b:.0f}m)" for sid, b in excl) if excl else ""
        print(
            f"{spec['id']}: raw={stats['raw']} kept={sum(len(p) for p in parts)} "
            f"dropped(trim={stats.get('trimmed', 0)}, null_ts={stats.get('dropped_null_ts', 0)}, "
            f"bad_label={stats.get('dropped_bad_label', 0)}) lat_capped={stats.get('lat_capped', 0)}{note}"
        )

    decisions = pd.concat(all_parts, ignore_index=True)[COLUMNS]
    sessions = pd.DataFrame(session_rows)

    complete = sessions.groupby("user")["condition"].nunique()
    keep_users = set(complete[complete == 2].index)
    partial = sessions[~sessions["user"].isin(keep_users)]
    if len(partial):
        for u, g in partial.groupby("user"):
            print(f"{u}: dropped entirely (only {'/'.join(sorted(g['condition'].unique()))} "
                  f"survived the {MIN_BLOCK_MIN}-min block rule)")
        keep_sids = set(sessions.loc[sessions["user"].isin(keep_users), "session_id"])
        all_parts = [p for p in all_parts if p["session_id"].iloc[0] in keep_sids]
        sessions = sessions[sessions["user"].isin(keep_users)]
        decisions = pd.concat(all_parts, ignore_index=True)[COLUMNS]

    cond_counts = decisions["condition"].value_counts()
    print("\nDecisions logged: " + ", ".join(f"{c}={n}" for c, n in cond_counts.items()))
    print(f"Latencies nulled (>{LATENCY_CAP_MIN} min): {total_capped}")

    decisions.to_csv(ANALYSIS / "decisions.csv", index=False)
    sessions.to_csv(ANALYSIS / "sessions.csv", index=False)

    al_metrics = build_al_metrics(all_parts)
    al_metrics.to_csv(ANALYSIS / "al_metrics.csv", index=False)
    per_cond = al_metrics.groupby("condition")["al_instance_id"].nunique()
    print(f"\nAL metrics: {len(al_metrics)} iterations, "
          + ", ".join(f"{cond}: {n} instances" for cond, n in per_cond.items()))

    print(f"\nTotal decisions: {len(decisions)} | sessions: {len(sessions)}")
    print("\n=== sessions.csv ===")
    print(sessions.to_string(index=False))
    print("\nAudit re-check on cleaned data:")
    bad = sessions[(sessions["dup_refs"] > 0) | (sessions["span_min"] > 120)]
    print(bad.to_string(index=False) if len(bad) else "no flags")

    print("\nCondition vs XAI sanity check (expected xai_share ~1.0 for treatment, ~0.0 for control):")
    for _, row in sessions.iterrows():
        expected = 1.0 if row["condition"] == "treatment" else 0.0
        if abs(row["xai_share"] - expected) > 0.5:
            print(
                f"  WARNING {row['session_id']} ({row['condition']}): "
                f"xai_share={row['xai_share']}"
            )


if __name__ == "__main__":
    main()
