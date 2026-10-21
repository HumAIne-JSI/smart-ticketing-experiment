"""Preprocess labeling decisions into analysis CSVs for three downstream views.

Session layout is defined in analysis/preprocess_config.json:
- "instances": list of instance indices merged into one session
- "condition": "control"|"treatment", or "conditions_by_day" list when the
  session mixes conditions
- "split_by_day": split merged decisions by labeled_at date (-d1, -d2, ...)
- "trim_last_minutes": fatigue view only - keep only decisions within the
  final N minutes of the raw instance (drops warm-up labels)

Artifacts (one run produces all three profiles over the same specs):

- fatigue -> decisions.csv, sessions.csv, al_metrics.csv, plus all_decisions.csv
  and all_sessions.csv (every configured day-part, including the ones excluded
  by the block rule; flag columns kept and excluded_reason).
  Time-based human analysis: trim applied, a session is excluded when its
  longest same-day labeling block is shorter than MIN_BLOCK_MIN.
- labels -> decisions_labels.csv, sessions_labels.csv.
  Label-count based paired human analysis: no trim and no block rule, so the
  full label sequences (from cold start) and short sessions are included.
- f1 -> decisions_f1.csv, sessions_f1.csv, al_metrics_f1.csv.
  Cold-start test-set F1 comparison: like labels, but for sessions whose two
  day-parts carry different conditions only the FIRST day-part is kept (the
  second part's model is warm-started by the first part's labels and cannot be
  attributed to its own condition). al_metrics_f1.csv keeps only the training
  iterations created up to the kept day-part's last decision date;
  num_labeled stays the raw training budget.

Row cleaning (all profiles): drop rows with null labeled_at (never labeled)
and rows with null label that are not "I don't know"; idk rows are kept.
Decision times longer than LATENCY_CAP_MIN minutes are set to null (likely
session breaks); the decision itself is kept.

Every user is kept in every artifact. Sessions tables carry a `paired` column
(True iff the user has both conditions alive in that artifact) so analyses can
filter per need instead of the former hard user-level exclusion.
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
MIN_BLOCK_MIN = 45   # fatigue profile: sessions whose longest block is shorter are excluded

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

PROFILES = {
    "fatigue": {"trim": True, "block_rule": True, "first_part_only": False},
    "labels": {"trim": False, "block_rule": False, "first_part_only": False},
    "f1": {"trim": False, "block_rule": False, "first_part_only": True},
}
SUFFIX = {"fatigue": "", "labels": "_labels", "f1": "_f1"}


def load_instance_decisions(instance_id):
    path = ROOT / str(instance_id) / "duckdb" / "label_decisions.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


AL_METRIC_COLUMNS = ["al_instance_id", "iteration_id", "num_labeled", "f1_score", "created_at", "condition"]


def build_al_metrics(parts, cut_dates=None):
    """One row per active-learning retrain iteration, condition assigned per day.

    cut_dates (f1 profile): {al_instance_id: last decision date of the kept
    day-part} - iterations created later are dropped (warm-start continuation).
    """
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["ts"]).dt.date
    cond_of = df.groupby(["al_instance_id", "date"])["condition"].agg(lambda s: s.mode()[0])

    rows = []
    for inst in sorted(df["al_instance_id"].unique()):
        path = ROOT / str(inst) / "duckdb" / "metrics.json"
        if not path.exists():
            continue
        cut = cut_dates.get(inst) if cut_dates else None
        per_day = cond_of.loc[inst]
        for m in json.loads(path.read_text(encoding="utf-8")):
            day = parse_ts(m["created_at"]).date()
            if cut is not None and day > cut:
                continue
            rows.append({**{c: m[c] for c in AL_METRIC_COLUMNS[:-1]},
                         "condition": per_day.get(day, per_day.iloc[0])})
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


def build_session(spec, profile):
    rows = []
    for inst in spec["instances"]:
        for d in load_instance_decisions(inst):
            rows.append({**d, "al_instance_id": inst})
    df = pd.DataFrame(rows)
    stats = {"raw": len(df)}

    if df.empty:
        return [], [], {"raw": 0, "dropped_null_ts": 0, "dropped_bad_label": 0}

    df["ts"] = df["labeled_at"].map(parse_ts)
    df = df.sort_values("ts")

    if profile["trim"] and "trim_last_minutes" in spec:
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

    conditions = resolve_conditions(spec, len(parts))
    if profile["first_part_only"] and len(parts) > 1:
        stats["first_only"] = [parts[0][0]]
        parts, conditions = parts[:1], conditions[:1]

    out_rows = []
    all_rows = []
    for (session_id, part), condition in zip(parts, conditions):
        cleaned, null_ts, bad_label = clean_rows(part, session_id)
        stats["dropped_null_ts"] = stats.get("dropped_null_ts", 0) + null_ts
        stats["dropped_bad_label"] = stats.get("dropped_bad_label", 0) + bad_label
        block = longest_block(cleaned)
        cleaned = cleaned.copy()
        cleaned["session_id"] = session_id
        cleaned["condition"] = condition
        cleaned["has_xai"] = cleaned["xai_result"].notna()
        all_rows.append(cleaned)
        if stats.get("first_only", [None])[0] == session_id:
            stats["cut"] = {inst: cleaned["ts"].max().date()
                            for inst in cleaned["al_instance_id"].unique()}
        if profile["block_rule"] and block < MIN_BLOCK_MIN:
            stats.setdefault("excluded", []).append((session_id, block))
            continue
        out_rows.append(cleaned)

    return out_rows, all_rows, stats


def session_summary_row(cleaned, user, spec, stats):
    ts = cleaned["ts"]
    span = (ts.max() - ts.min()).total_seconds() / 60 if len(ts) > 1 else 0
    return {
        "session_id": cleaned["session_id"].iloc[0],
        "user": user,
        "condition": cleaned["condition"].iloc[0],
        "instances": ",".join(map(str, spec["instances"])),
        "n_labels": len(cleaned),
        "n_idk": int(cleaned["i_dont_know"].fillna(False).sum()),
        "span_min": round(span, 1),
        "dup_refs": int(cleaned["ref"].duplicated().sum()),
        "xai_share": round(cleaned["has_xai"].mean(), 2),
        "n_latency_capped": stats.get("lat_capped", 0),
        "longest_block_min": round(longest_block(cleaned), 1),
    }


def add_paired_flag(sessions, label=""):
    n_cond = sessions.groupby("user")["condition"].nunique()
    paired_users = set(n_cond[n_cond == 2].index)
    sessions["paired"] = sessions["user"].isin(paired_users)
    for u in sorted(n_cond.index[~n_cond.eq(2)]):
        conds = sorted(sessions.loc[sessions["user"] == u, "condition"].unique())
        print(f"{u}: paired=False (only {'/'.join(conds)})")
    if not len(n_cond) - len(paired_users):
        print(f"{label}: all users are paired (both conditions present)" if label
              else "all users are paired (both conditions present)")


def run_profile(name, profile, config, usernames):
    suffix = SUFFIX[name]
    print(f"\n=== {name} artifact ===")
    kept_parts = []
    session_rows = []
    all_parts = []
    all_session_rows = []
    total_capped = 0
    cut_dates = {}

    for spec in config["sessions"]:
        parts, all_parts_of_spec, stats = build_session(spec, profile)
        user = usernames.get(
            json.loads(
                (ROOT / str(spec["instances"][0]) / "duckdb" / "al_instances.json")
                .read_text(encoding="utf-8")
            )[0]["user_id"],
            "?",
        )
        for cleaned in parts:
            kept_parts.append(cleaned)
            session_rows.append(session_summary_row(cleaned, user, spec, stats))
        if name == "fatigue":
            kept_sids = {p["session_id"].iloc[0] for p in parts}
            for cleaned in all_parts_of_spec:
                sid = cleaned["session_id"].iloc[0]
                row = session_summary_row(cleaned, user, spec, stats)
                row["kept"] = sid in kept_sids
                row["excluded_reason"] = "" if row["kept"] else f"short_block(<{MIN_BLOCK_MIN}m)"
                all_parts.append(cleaned)
                all_session_rows.append(row)
        total_capped += stats.get("lat_capped", 0)
        cut_dates.update(stats.get("cut", {}))
        excl = stats.get("excluded", [])
        note = "  EXCLUDED: " + ", ".join(f"{sid} ({b:.0f}m)" for sid, b in excl) if excl else ""
        if stats.get("first_only"):
            note += "  first-part only: " + ", ".join(stats["first_only"])
        print(
            f"{spec['id']}: raw={stats['raw']} kept={sum(len(p) for p in parts)} "
            f"dropped(trim={stats.get('trimmed', 0)}, null_ts={stats.get('dropped_null_ts', 0)}, "
            f"bad_label={stats.get('dropped_bad_label', 0)}) lat_capped={stats.get('lat_capped', 0)}{note}"
        )

    decisions = pd.concat(kept_parts, ignore_index=True)[COLUMNS]
    sessions = pd.DataFrame(session_rows)

    cond_counts = decisions["condition"].value_counts()
    print("\nDecisions logged: " + ", ".join(f"{c}={n}" for c, n in cond_counts.items()))
    print(f"Latencies nulled (>{LATENCY_CAP_MIN} min): {total_capped}")
    print(f"\nTotal decisions: {len(decisions)} | sessions: {len(sessions)}")
    print("\n=== sessions%s ===" % suffix)
    print(sessions.to_string(index=False))
    print("\npaired flags:")
    add_paired_flag(sessions, label=name)

    decisions.to_csv(ANALYSIS / f"decisions{suffix}.csv", index=False)
    sessions.to_csv(ANALYSIS / f"sessions{suffix}.csv", index=False)

    if name in ("fatigue", "f1"):
        al_metrics = build_al_metrics(kept_parts, cut_dates if name == "f1" else None)
        al_metrics.to_csv(ANALYSIS / f"al_metrics{suffix}.csv", index=False)
        per_cond = al_metrics.groupby("condition")["al_instance_id"].nunique()
        print(f"\nAL metrics: {len(al_metrics)} iterations, "
              + ", ".join(f"{cond}: {n} instances" for cond, n in per_cond.items()))

    if name == "fatigue":
        all_decisions = pd.concat(all_parts, ignore_index=True)[COLUMNS]
        all_sessions = pd.DataFrame(all_session_rows)
        paired_users = set(sessions.loc[sessions["paired"], "user"])
        all_sessions["paired"] = all_sessions["user"].isin(paired_users)
        all_decisions.to_csv(ANALYSIS / "all_decisions.csv", index=False)
        all_sessions.to_csv(ANALYSIS / "all_sessions.csv", index=False)

        print(f"\nAll-users tables: {len(all_sessions)} sessions "
              f"({int((~all_sessions['kept']).sum())} excluded), {len(all_decisions)} decisions")

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


def main():
    config = json.loads((ANALYSIS / "preprocess_config.json").read_text(encoding="utf-8"))
    usernames = build_usernames()
    for name, profile in PROFILES.items():
        run_profile(name, profile, config, usernames)


if __name__ == "__main__":
    main()
