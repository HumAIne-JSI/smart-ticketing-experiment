"""Analyze treatment effect from analysis/decisions.csv against ground truth.

Sections: a=accuracy & macro F1, b=complementarity (labels view = full
untrimmed data), c=learning curves, d=latency,
e=calibration, f=agreement, g=response rates, h=throughput, i=idk/difficult totals,
j=balanced analysis cutoffs, k=balanced-cohort learning curves,
l=total activity time, m=time-based retention cutoffs, n=time-based cohort curves,
o=area under the learning curve.
Run all (default) or one via --section a|b|c|d|e|f|g|h|i|j|k|l|m|n|o.
Use --human (default) or --llm to select the cohort; reads that cohort's
analysis dir and writes PNG charts to its figures/ dir and an HTML report
(reports/analyze_report.html or reports/analyze_report_llm.html).
"""

import argparse
import base64
import re
import sys
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cohort_config import add_cohort_args, resolve_cohort_from_args

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
FIGURES = ANALYSIS / "figures"
REPORT = Path(__file__).resolve().parents[1] / "reports" / "analyze_report.html"
COHORT_LABEL = "human"

# Section D truncated fatigue variant: per session keep only rows up to the
# Nth scorable decision (= preprocess max_scorable semantics). With the 70
# seed labels this corresponds to 70 + N total training labels (num_labeled).
TRIM_SCORABLE = 50

# Section K truncated paired-curves variant: keep retrain iterations with
# num_labeled <= this (70 seed labels + 50 human decisions = 120 total).
TRIM_NUM_LABELED = 120

# Section K truncated paired-curves plot: fixed y-range (model macro F1).
TRIM120_YLIM = (0.45, 0.75)

_sections: list[tuple[str, list]] = []
_blocks: list = []


def start_section(title):
    global _blocks
    _blocks = []
    _sections.append((title, _blocks))
    print(f"\n=== {title} ===")


def out(text=""):
    print(text)
    if text.strip():
        _blocks.append(("p", escape(text).replace("\n", "<br>")))


def out_table(df, title=None, index=False, ndigits=3):
    shown = df.round(ndigits) if ndigits is not None else df
    print(shown.to_string(index=index))
    if title:
        _blocks.append(("p", f"<b>{escape(title)}</b>"))
    _blocks.append(("table", shown.to_html(index=index, border=0)))


def save_fig(fig, name, caption=None):
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _blocks.append(("img", (path, caption)))


def render_report():
    def slug(title):
        return re.sub(r"\W+", "-", title.lower()).strip("-")

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Smart ticketing experiment — analysis report</title>",
        "<style>",
        "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;max-width:1100px;"
        "color:#222;line-height:1.45;padding:0 1rem}",
        "h1{border-bottom:2px solid #333;padding-bottom:.3rem}",
        "h2{margin-top:2.5rem;border-bottom:1px solid #ccc;padding-bottom:.2rem}",
        "nav{margin:1rem 0;padding:.6rem 1rem;background:#f5f5f5;border-radius:4px}",
        "nav a{margin-right:.9rem;white-space:nowrap}",
        "table{border-collapse:collapse;margin:.6rem 0;font-size:.9rem}",
        "th,td{border:1px solid #ccc;padding:3px 8px;text-align:right}",
        "th{background:#f0f0f0}",
        "th:first-child,td:first-child{text-align:left}",
        "img{max-width:100%;border:1px solid #eee;margin:.5rem 0}",
        ".caption{color:#666;font-size:.85rem;margin:.2rem 0}",
        "</style></head><body>",
        "<h1>Smart ticketing experiment — analysis report</h1>",
        f"<p>Generated {datetime.now():%Y-%m-%d %H:%M} from {ANALYSIS.name}/decisions.csv "
        f"(cohort: {COHORT_LABEL}; figures also saved as PNG in {ANALYSIS.name}/figures/).</p>",
        "<nav>",
        "".join(f"<a href='#{slug(t)}'>{escape(t)}</a>" for t, _ in _sections),
        "</nav>",
    ]
    for title, blocks in _sections:
        parts.append(f"<h2 id='{slug(title)}'>{escape(title)}</h2>")
        for kind, payload in blocks:
            if kind == "p":
                parts.append(f"<p>{payload}</p>")
            elif kind == "table":
                parts.append(payload)
            elif kind == "img":
                path, caption = payload
                data = base64.b64encode(path.read_bytes()).decode()
                alt = escape(caption or path.stem)
                parts.append(f"<img src='data:image/png;base64,{data}' alt='{alt}'>")
                if caption:
                    parts.append(f"<p class='caption'>{escape(caption)}</p>")
    parts.append("</body></html>")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(parts), encoding="utf-8")
    print(f"\nHTML report written to {REPORT}")


def ground_truth_map():
    excel = pd.read_excel(
        Path(__file__).resolve().parents[1] / "data" / "User Request_last_team_ANON.xlsx"
    )
    return dict(zip(excel["Ref"], excel["Team->Name"]))


def load_data():
    dec = pd.read_csv(ANALYSIS / "decisions.csv")
    sessions = pd.read_csv(ANALYSIS / "sessions.csv")

    # Keep paired users only (both conditions alive after preprocessing);
    # preprocessing now flags unpaired in sessions.csv instead of dropping them.
    paired = sessions["paired"].fillna(False).astype(bool)
    dropped_users = sorted(sessions.loc[~paired, "user"].unique())
    if dropped_users:
        print(f"Excluding unpaired users (not both conditions present): "
              f"{', '.join(dropped_users)}")
    sessions = sessions[paired]

    truth = ground_truth_map()
    dec["ground_truth"] = dec["ref"].map(truth)
    n_dec_all = len(dec)
    dec = dec.merge(sessions[["session_id", "user"]], on="session_id", how="inner")
    print(f"Decisions kept after unpaired-user filter: {len(dec)} / {n_dec_all}")
    dec["human_correct"] = dec["label"] == dec["ground_truth"]
    dec["model_correct"] = dec["model_prediction"] == dec["ground_truth"]
    dec["agrees_with_model"] = dec["label"] == dec["model_prediction"]
    dec["ts"] = pd.to_datetime(dec["labeled_at"], format="mixed")
    dec["position"] = dec.groupby("session_id")["ts"].rank(method="first")
    dec["session_size"] = dec.groupby("session_id")["ts"].transform("count")
    dec["pos_frac"] = dec["position"] / dec["session_size"]

    scorable = dec[~dec["i_dont_know"].fillna(False).astype(bool)]
    n_per_session = scorable.groupby("session_id").size()
    small = n_per_session[n_per_session < 20]
    print(f"Scorable decisions: {len(scorable)} / {len(dec)} "
          f"(idk excluded: {len(dec) - len(scorable)})")
    if len(small):
        print(f"WARNING sessions with < 20 scorable decisions: {list(small.index)}")
    print("Note: inst 4 (user1-s1, control) contains a 14% xai leak; kept as-is.")
    return dec, sessions, scorable


def slope_chart(pivot, ylabel, title, name, caption):
    users = pivot.dropna().index
    fig, ax = plt.subplots(figsize=(6, 5))
    for user in users:
        ax.plot([0, 1], [pivot.loc[user, "control"] * 100, pivot.loc[user, "treatment"] * 100],
                marker="o", color="gray", alpha=0.6)
    ax.plot([0, 1], [pivot.loc[users, "control"].mean() * 100,
                     pivot.loc[users, "treatment"].mean() * 100], marker="o",
            color="red", linewidth=4, markersize=10, label="mean")
    ax.set_xticks([0, 1], ["Unaided", "AI-assisted"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    save_fig(fig, name, caption)


def section_a(scorable, sessions):
    start_section("A. Accuracy & macro F1 (headline)")
    rows = []
    for (user, sid, cond), g in scorable.groupby(["user", "session_id", "condition"]):
        rows.append({
            "user": user,
            "session_id": sid,
            "condition": cond,
            "human_acc": g["human_correct"].mean(),
            "human_f1": f1_score(g["ground_truth"], g["label"], average="macro",
                                 zero_division=0),
        })
    scores = pd.DataFrame(rows)
    out_table(scores, ndigits=None)

    for metric, label in [("human_acc", "accuracy"), ("human_f1", "macro F1")]:
        pivot = scores.pivot(index="user", columns="condition", values=metric)
        users = pivot.dropna().index
        control = pivot.loc[users, "control"]
        treatment = pivot.loc[users, "treatment"]
        stat = stats.wilcoxon(treatment, control)
        out(f"\nMean {label}: control={control.mean():.3f}  "
            f"treatment={treatment.mean():.3f}  "
            f"difference={treatment.mean() - control.mean():+.3f}")
        out(f"Wilcoxon signed-rank (n={len(users)}): p={stat.pvalue:.4f}")
        slope_chart(
            pivot, f"labeling {label} (%)",
            f"Per-user {label}: Unaided vs. AI-assisted",
            "a_accuracy_slope.png" if metric == "human_acc" else "a_f1_slope.png",
            f"Per-user {label}, unaided (control) vs AI-assisted (treatment).")


def complementarity_rates(group):
    model_wrong = ~group["model_correct"]
    return {
        "n": len(group),
        "catch_rate": group.loc[model_wrong, "human_correct"].mean(),
        "overtrust_rate": 1 - group.loc[~model_wrong, "human_correct"].mean(),
        "model_acc": group["model_correct"].mean(),
        "human_acc": group["human_correct"].mean(),
    }


def complementarity_bars(charts_df, name, caption):
    """
    Two-panel complementarity bar chart: P(human correct | model wrong) and
    P(human wrong | model correct), one control/treatment bar pair each.
    `charts_df` needs columns condition, catch_rate, overtrust_rate.
    Each panel has its own fixed y-scale: left (catch rate) 0-0.35, right
    (overtrust rate) 0-0.55.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    x = range(len(charts_df))
    for ax, metric, title, ylim in [
        (axes[0], "catch_rate", "P(human correct | model wrong)", 0.35),
        (axes[1], "overtrust_rate", "P(human wrong | model correct)", 0.55),
    ]:
        ax.bar([c - 0.2 for c in x], charts_df[metric].where(charts_df.condition == "control"),
               width=0.4, label="control")
        ax.bar([c + 0.2 for c in x], charts_df[metric].where(charts_df.condition == "treatment"),
               width=0.4, label="treatment")
        ax.set_xticks(list(x), charts_df["condition"])
        ax.set_ylim(0, ylim)
        ax.set_title(title)
        ax.legend()
    save_fig(fig, name, caption)


def section_b():
    """Complementarity 2x2, built on the labels view (full untrimmed data).

    Like the other labels-based sections (L, M, O): no trim, no block rule,
    no user exclusions. Sessions that mix conditions are separated into
    day-parts (-d1/-d2, each with its own condition) already in
    decisions_labels.csv. IDK tickets stay excluded (not scorable against
    ground truth), matching the scoring rule of the trimmed sections.
    """
    start_section("B. Complementarity 2x2")
    dec = pd.read_csv(ANALYSIS / "decisions_labels.csv")
    ses = pd.read_csv(ANALYSIS / "sessions_labels.csv")
    user_of = dict(zip(ses["session_id"], ses["user"]))
    dec["user"] = dec["session_id"].map(user_of)
    missing = int(dec["user"].isna().sum())
    if missing:
        out(f"WARNING {missing} decisions without session user mapping; dropped")
        dec = dec[dec["user"].notna()]
    truth = ground_truth_map()
    dec["ground_truth"] = dec["ref"].map(truth)
    dec["human_correct"] = dec["label"] == dec["ground_truth"]
    dec["model_correct"] = dec["model_prediction"] == dec["ground_truth"]
    scorable = dec[~dec["i_dont_know"].fillna(False).astype(bool)]
    n_by_cond = {c: int((scorable["condition"] == c).sum())
                 for c in ("control", "treatment")}
    out(f"Labels-view data: {len(dec)} decisions ({len(dec) - len(scorable)} idk "
        f"excluded) -> scorable n={n_by_cond.get('control', 0)}/{n_by_cond.get('treatment', 0)} "
        f"(control/treatment); full untrimmed sequences, no block rule, no user "
        f"exclusions ({scorable['user'].nunique()} users).")

    rows = []
    for cond, group in scorable.groupby("condition"):
        rows.append({"condition": cond, **complementarity_rates(group)})
    table = pd.DataFrame(rows)
    out_table(table, title="Per-decision (all labeled decisions pooled)")

    complementarity_bars(
        table, "b_complementarity.png",
        "Complementarity: catch rate (human correct when model wrong) and "
        "overtrust rate (human wrong when model correct); "
        "labels-view (full untrimmed) data; left panel on the y-scale "
        "0-0.35, right panel 0-0.55.")

    per_user_rows = []
    for (user, cond), g in scorable.groupby(["user", "condition"]):
        per_user_rows.append({"user": user, "condition": cond,
                              **complementarity_rates(g)})
    per_user = pd.DataFrame(per_user_rows)
    summary = (
        per_user.groupby("condition")
        .agg(n_users=("user", "nunique"),
             mean_decisions=("n", "mean"),
             catch_rate=("catch_rate", "mean"),
             overtrust_rate=("overtrust_rate", "mean"),
             model_acc=("model_acc", "mean"),
             human_acc=("human_acc", "mean"))
        .reset_index()
    )
    out("\nPer-user rates (rate within each user x condition first, "
        "then mean over users):")
    out_table(summary, title="Per-user (rate within each user x condition, "
                             "then mean over users)")
    complementarity_bars(
        summary, "b_complementarity_per_user.png",
        "Complementarity per user: mean per-user catch rate (human correct "
        "when model wrong) and overtrust rate (human wrong when model "
        "correct), averaged over users; left panel on the y-scale "
        "0-0.35, right panel 0-0.55.")

    split_users = sorted(
        ses.loc[ses["session_id"].str.contains(r"-d\d+$"), "user"].unique()
    )
    if split_users:
        out(f"Note: split-by-day users ({', '.join(split_users)}): each "
            "condition's rows come only from that condition's day-part "
            "(e.g. user15-s1-d1 / user15-s1-d2).")

    for metric, label in [("catch_rate", "catch rate"),
                          ("overtrust_rate", "overtrust rate"),
                          ("model_acc", "model accuracy")]:
        pivot = per_user.pivot(index="user", columns="condition", values=metric)
        users = pivot.dropna().index
        control = pivot.loc[users, "control"]
        treatment = pivot.loc[users, "treatment"]
        stat = stats.wilcoxon(treatment, control)
        out(f"\nMean per-user {label}: control={control.mean():.3f}  "
            f"treatment={treatment.mean():.3f}  "
            f"difference={treatment.mean() - control.mean():+.3f}")
        out(f"Wilcoxon signed-rank (n={len(users)}): p={stat.pvalue:.4f}")


def section_c():
    start_section("C. Model learning curves")
    metrics = pd.read_csv(ANALYSIS / "al_metrics.csv")
    sessions = pd.read_csv(ANALYSIS / "sessions.csv")
    # Preprocessing no longer drops unpaired users; exclude their AL instances
    # here so the learning curves cover only paired (within-subject) users.
    paired = sessions["paired"].fillna(False).astype(bool)
    unpaired_inst = {
        int(i) for i in
        sessions.loc[~paired, "instances"].astype(str).str.split(",").explode()
        if i.strip()
    }
    if unpaired_inst:
        n_before = len(metrics)
        metrics = metrics[~metrics["al_instance_id"].isin(unpaired_inst)]
        out(f"Excluding AL instances of unpaired users {sorted(unpaired_inst)}: "
            f"{n_before - len(metrics)} of {n_before} iterations dropped")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"control": "tab:blue", "treatment": "tab:orange"}
    steps = {
        cond: group.groupby("num_labeled")["f1_score"].agg(["mean", "sem", "count"])
        for cond, group in metrics.groupby("condition")
    }
    cutoff = min(step.loc[step["count"] >= 5].index.max() for step in steps.values())
    if pd.isna(cutoff):
        out("No budget with >=5 instances in both conditions; showing individual runs only")
    else:
        out(f"Pooled mean shown up to num_labeled={cutoff} (>=5 instances in both conditions)")
    for cond, step in steps.items():
        color = colors[cond]
        group = metrics[metrics["condition"] == cond]
        for _, sub in group.groupby("al_instance_id"):
            sub = sub.sort_values("num_labeled")
            ax.plot(sub["num_labeled"], sub["f1_score"], color=color, alpha=0.25, linewidth=1)
        keep = (step["count"] >= 5) & (step.index <= cutoff)
        ax.plot(step.index[keep], step["mean"][keep], marker="o", markersize=3, color=color, label=cond)
        ax.fill_between(step.index[keep], (step["mean"] - step["sem"])[keep],
                        (step["mean"] + step["sem"])[keep], color=color, alpha=0.2)
    ax.set_xlabel("labels used for training")
    ax.set_ylabel("model macro F1 (test)")
    ax.set_title("Model learning curves\n(mean ± sem where ≥5 instances remain in both conditions)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="gray", alpha=0.4, linewidth=1))
    labels.append("individual instances")
    ax.legend(handles, labels)
    save_fig(fig, "c_learning_curves.png",
             "Model macro F1 learning curves in the live active learning loop; "
             "mean ± SEM where ≥5 instances remain in both conditions.")
    for cond, group in metrics.groupby("condition"):
        lo = group.nsmallest(10, "num_labeled")["f1_score"].mean()
        hi = group.nlargest(10, "num_labeled")["f1_score"].mean()
        out(f"{cond}: first-10 mean F1={lo:.3f}  last-10 mean F1={hi:.3f}  "
            f"({group['al_instance_id'].nunique()} instances, {len(group)} iterations)")

    out("\nPer-run first/last 10 retrain iterations (sorted by num_labeled):")
    summaries = {}
    for (cond, inst), g in metrics.groupby(["condition", "al_instance_id"]):
        g = g.sort_values("num_labeled")
        summaries.setdefault(cond, []).append(
            [inst, len(g), g.head(10)["f1_score"].mean(), g.tail(10)["f1_score"].mean()])
    def ci95(col):
        n = len(col)
        if n < 2:
            return None
        return stats.t.ppf(0.975, df=n - 1) * col.std() / n ** 0.5

    for cond, rows in summaries.items():
        table = pd.DataFrame(rows, columns=["al_instance_id", "n_iters", "first10_f1", "last10_f1"])
        out(f"{cond}:")
        out_table(table)
        first, last = table["first10_f1"], table["last10_f1"]
        h1, h2 = ci95(first), ci95(last)
        if h1 is None:
            out(f"  mean over runs: first10={first.mean():.3f}+/-{first.std():.3f}  "
                f"last10={last.mean():.3f}+/-{last.std():.3f}")
        else:
            out(f"  mean over runs (95% CI): first10={first.mean():.3f} "
                f"[{first.mean() - h1:.3f}, {first.mean() + h1:.3f}]  "
                f"last10={last.mean():.3f} [{last.mean() - h2:.3f}, {last.mean() + h2:.3f}]")


def fatigue_charts(sco, table_note, suffix="", title_note="", caption_note=""):
    """Fatigue aggregation + the two 'Fatigue:' figures for one decision set.

    `sco` needs columns session_id, condition, human_correct, latency_ms,
    ground_truth, label and pos_frac (fraction within its OWN session, so
    callers of truncated variants must recompute it on the truncated
    sequence). Figures get the name suffix; title/caption get the notes.
    """
    thirds = pd.cut(sco["pos_frac"], [0, 1 / 3, 2 / 3, 1],
                    labels=["first", "middle", "last"])
    per_session = (
        sco.groupby(["session_id", "condition", thirds], observed=True)
        .agg(acc=("human_correct", "mean"), lat_s=("latency_ms", "median"),
             n=("human_correct", "size"))
    )
    f1_third = (
        sco.groupby(["session_id", "condition", thirds], observed=True)
        .apply(lambda g: f1_score(g["ground_truth"], g["label"], average="macro",
                                  zero_division=0))
        .rename("f1")
    )
    per_session["lat_s"] = per_session["lat_s"] / 1000
    fatigue = (
        per_session.join(f1_third).reset_index()
        .groupby(["condition", "pos_frac"], observed=True)
        .agg(lat_s=("lat_s", "mean"), acc=("acc", "mean"), f1=("f1", "mean"),
             n_sessions=("session_id", "nunique"))
        .reset_index()
    )
    fatigue["acc_pct"] = fatigue["acc"] * 100
    fatigue["f1_pct"] = fatigue["f1"] * 100
    out(f"\nFatigue ({table_note}; latency = mean of per-session medians):")
    out_table(fatigue)
    fig2, ax_fat = plt.subplots(figsize=(6, 4.5))
    x_pos = [0, 1, 2]
    for cond, marker in [("control", "o-"), ("treatment", "^-")]:
        sub = fatigue[fatigue.condition == cond].sort_values("pos_frac")
        ax_fat.plot(x_pos, sub["acc_pct"], marker, label=f"{cond} accuracy (%)")
        ax_fat.plot(x_pos, sub["lat_s"], marker.replace("-", ""),
                    linestyle=":", label=f"{cond} latency (s)")
    ax_fat.set_xticks(x_pos, ["first", "middle", "last"])
    ax_fat.set_ylabel("accuracy (%) / median latency (s)")
    extra = f"{title_note}; " if title_note else ""
    ax_fat.set_title(f"Fatigue: accuracy & latency by session third\n"
                     f"({extra}latency: mean of per-session medians)")
    ax_fat.legend(fontsize=8)
    save_fig(fig2, f"d_fatigue{suffix}.png",
             f"Accuracy and latency across thirds of a session.{caption_note}")

    fig_f1, ax_f1 = plt.subplots(figsize=(6, 4.5))
    for cond, marker in [("control", "o-"), ("treatment", "^-")]:
        sub = fatigue[fatigue.condition == cond].sort_values("pos_frac")
        ax_f1.plot(x_pos, sub["f1_pct"], marker, label=f"{cond} macro F1 (%)")
        ax_f1.plot(x_pos, sub["lat_s"], marker.replace("-", ""),
                   linestyle=":", label=f"{cond} latency (s)")
    ax_f1.set_xticks(x_pos, ["first", "middle", "last"])
    ax_f1.set_ylabel("macro F1 (%) / median latency (s)")
    ax_f1.set_title(f"Fatigue: macro F1 & latency by session third\n"
                    f"({extra}latency: mean of per-session medians)")
    ax_f1.legend(fontsize=8)
    save_fig(fig_f1, f"d_fatigue_f1{suffix}.png",
             f"Macro F1 and latency across thirds of a session.{caption_note}")


def truncate_decisions(dec, n_scorable):
    """Per-session truncation to the first `n_scorable` scorable decisions.

    Same semantics as preprocess.py's max_scorable: keep every row (idk rows
    included) whose ts is at or before the ts of the Nth scorable ticket;
    sessions already at or below the cutoff stay whole. Returns (truncated
    copy, rows dropped, sessions shortened)."""
    before = len(dec)
    mask = ~dec["i_dont_know"].fillna(False).astype(bool)
    sc = dec.loc[mask, ["session_id", "ts"]].sort_values("ts")
    sc["nth"] = sc.groupby("session_id").cumcount() + 1
    cut = sc.loc[sc["nth"] == n_scorable]
    limit = dec["session_id"].map(dict(zip(cut["session_id"], cut["ts"])))
    trimmed = dec[limit.isna() | (dec["ts"] <= limit)].copy()
    dropped = before - len(trimmed)
    shortened = int((sc.groupby("session_id").size() > n_scorable).sum())
    return trimmed, dropped, shortened


def section_d(dec, scorable, sessions):
    start_section("D. Latency & fatigue")
    lat = (
        scorable.groupby(["user", "session_id", "condition"])["latency_ms"]
        .median()
        .reset_index()
    )
    lat["latency_s"] = lat["latency_ms"] / 1000
    out_table(lat, ndigits=None)

    per_dec = scorable.groupby("condition")["latency_ms"].agg(["median", "mean"]) / 1000
    per_user = (
        lat.groupby(["user", "condition"])["latency_s"].median()
        .groupby("condition").agg(["median", "mean"])
    )
    summary = pd.concat([per_dec, per_user], keys=["per-decision", "per-user"])
    summary.index.names = ["level", "condition"]
    out("\nPer-condition latency summary (s):")
    out_table(summary, index=True)

    pivot_lat = lat.pivot(index="user", columns="condition", values="latency_s")
    users = pivot_lat.dropna().index
    stat = stats.wilcoxon(pivot_lat.loc[users, "treatment"], pivot_lat.loc[users, "control"])
    out(f"Wilcoxon on per-user median latency (n={len(users)}): p={stat.pvalue:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    data = [scorable.loc[scorable.condition == c, "latency_ms"].dropna() / 1000
            for c in ("control", "treatment")]
    axes[0].boxplot(data, tick_labels=["control", "treatment"])
    axes[0].set_ylabel("latency (s)")
    axes[0].set_title("Per-decision latency")

    merged = sessions.merge(
        scorable.groupby("session_id")["human_correct"].mean().rename("acc"),
        left_on="session_id", right_index=True,
    ).merge(lat[["session_id", "latency_s"]], on="session_id")
    for cond, marker in [("control", "o"), ("treatment", "^")]:
        sub = merged[merged.condition == cond]
        axes[1].scatter(sub["latency_s"], sub["acc"], marker=marker, label=cond)
    axes[1].set_xlabel("median latency (s)")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy vs time per session")
    axes[1].legend()
    save_fig(fig, "d_latency_fatigue.png",
             "Left: per-decision latency by condition. Right: session accuracy vs median latency.")

    fatigue_charts(scorable, "thirds of session")

    # Same fatigue plots also at the "120 total labels" truncation: per
    # session keep only rows up to the 50th scorable decision (preprocess
    # max_scorable semantics: idk rows before the cutoff are kept; 70 seed
    # labels + 50 = num_labeled 120), pos_frac recomputed on the truncated
    # sequences like load_data does.
    dec_trim, dropped, shortened = truncate_decisions(dec, TRIM_SCORABLE)
    out(f"\nTruncated fatigue variant: per session only rows up to the "
        f"{TRIM_SCORABLE}th scorable decision are kept (training budget "
        f"= 70 seed labels + {TRIM_SCORABLE} = {70 + TRIM_SCORABLE} total "
        f"labels): {dropped} decisions cut, {shortened} of "
        f"{dec['session_id'].nunique()} sessions shortened.")
    dec_trim["position"] = dec_trim.groupby("session_id")["ts"].rank(method="first")
    dec_trim["session_size"] = dec_trim.groupby("session_id")["ts"].transform("count")
    dec_trim["pos_frac"] = dec_trim["position"] / dec_trim["session_size"]
    scorable_trim = dec_trim[~dec_trim["i_dont_know"].fillna(False).astype(bool)]
    fatigue_charts(
        scorable_trim,
        f"thirds of a truncated session: first {TRIM_SCORABLE} scorable "
        f"decisions per session (num_labeled = 70 seed + {TRIM_SCORABLE} = "
        f"{70 + TRIM_SCORABLE})",
        suffix="_trim120",
        title_note="first 50 scorable decisions",
        caption_note=f" Session truncated to the first {TRIM_SCORABLE} scorable "
        f"decisions per session (training budget {70 + TRIM_SCORABLE} incl. the "
        "70 seed labels); thirds recomputed on the truncated sequence.")

    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    bins = np.geomspace(0.1, 600, 40)
    colors = {"control": "tab:blue", "treatment": "tab:orange"}
    for cond, color in colors.items():
        lat_s = scorable.loc[scorable.condition == cond, "latency_ms"].dropna() / 1000
        ax3.hist(lat_s, bins=bins, color=color, alpha=0.6, label=cond)
        ax3.axvline(lat_s.median(), color=color, ls="--", lw=1)
        out(f"{cond} latency median={lat_s.median():.1f}s mean={lat_s.mean():.1f}s n={len(lat_s)}")
    ax3.set_xscale("log")
    ax3.set_xlabel("latency (s)")
    ax3.set_ylabel("decisions")
    ax3.set_title("Ticket latency distribution\n(log scale, dashed = median)")
    ax3.legend()
    save_fig(fig3, "d_latency_distribution.png",
             "Per-decision latency distribution (log scale; dashed = median).")


def section_e(dec, scorable):
    start_section("E. Calibration (IDK / difficulty)")
    dec["idk"] = dec["i_dont_know"].fillna(False).astype(bool)
    idk_rate = dec.groupby(["user", "condition"])["idk"].mean().reset_index()
    out("IDK rate per user:")
    out_table(idk_rate)

    out("\nP(model wrong | idk) vs P(model wrong | not idk):")
    for cond, group in dec.groupby("condition"):
        idk = group[group["idk"]]
        not_idk = group[~group["idk"]]
        p_idk = 1 - idk["model_correct"].mean() if len(idk) else float("nan")
        p_not = 1 - not_idk["model_correct"].mean()
        out(f"  {cond}: idk n={len(idk)} P(model wrong | idk)={p_idk:.3f}  "
            f"P(model wrong | not idk)={p_not:.3f}")

    dec["difficult"] = dec["is_difficult"].fillna(False).astype(bool)
    dec["tired"] = dec["is_tired"].fillna(False).astype(bool)
    selfrep = dec.groupby("condition").agg(
        n_difficult=("difficult", "sum"),
        n_answered_difficult=("is_difficult", "count"),
        n_tired=("tired", "sum"),
        n_answered_tired=("is_tired", "count"),
    )
    out("\nSelf-reports (with coverage):")
    out_table(selfrep, ndigits=None)


def agreement_table(df):
    agree = df["label"] == df["model_prediction"]
    in_top2 = agree | (df["label"] == df["second_model_prediction"])
    return {
        "n": len(df),
        "agree_top1": agree.mean(),
        "in_top2": in_top2.mean(),
        "acc_when_agree": df.loc[agree, "human_correct"].mean(),
        "acc_when_deviate": df.loc[~agree, "human_correct"].mean(),
    }


def emit_persona_tables():
    """Write analysis/persona_behavioral.csv from the labels view.

    One row per user,condition (control/treatment) pooled over all
    decisions_labels.csv rows (day-parts -d1/-d2 aggregated by their own
    condition; no trim/block). IDK included in n_total and medians;
    >10min capped null latencies excluded from the median; IDK counts as
    non-agree. pace_tertile is assigned per condition from cohort tertiles
    of median_latency_s (33rd/66th percentiles, linear): fastest third
    QUICK, middle STEADY, slowest SLOW; null median -> null.
    Cutoffs are cohort-specific (human vs llm runs separately).
    """
    dec = pd.read_csv(ANALYSIS / "decisions_labels.csv")
    ses = pd.read_csv(ANALYSIS / "sessions_labels.csv")
    user_of = dict(zip(ses["session_id"], ses["user"]))
    dec["user"] = dec["session_id"].map(user_of)
    missing = dec["user"].isna().sum()
    if missing:
        print(f"WARNING {missing} decisions without session user mapping; dropped")
        dec = dec[dec["user"].notna()]
    dec["latency_s"] = dec["latency_ms"] / 1000
    dec["is_difficult"] = dec["is_difficult"].fillna(False).astype(bool)
    dec["i_dont_know"] = dec["i_dont_know"].fillna(False).astype(bool)
    agree1 = dec["label"] == dec["model_prediction"]
    agree2 = agree1 | (dec["label"] == dec["second_model_prediction"])

    rows = []
    for (user, cond), g in dec.groupby(["user", "condition"]):
        idx = g.index
        rows.append({
            "user": user,
            "condition": cond,
            "n_total": int(len(g)),
            "median_latency_s": round(float(g["latency_s"].median(skipna=True)), 3)
            if g["latency_s"].notna().any() else float("nan"),
            "n_difficult": int(g["is_difficult"].sum()),
            "n_idk": int(g["i_dont_know"].sum()),
            "n_agree_top1": int(agree1.loc[idx].sum()),
            "n_agree_top2": int(agree2.loc[idx].sum()),
        })
    table = pd.DataFrame(rows).sort_values(["user", "condition"])
    cutoffs = {}
    table["pace_tertile"] = None
    for cond in ("control", "treatment"):
        meds = table.loc[table["condition"] == cond, "median_latency_s"].dropna()
        if not len(meds):
            continue
        q33 = float(meds.quantile(0.33))
        q66 = float(meds.quantile(0.66))
        cutoffs[cond] = (q33, q66)
        print(f"pace cutoffs {cond}: q33={q33:.3f}s q66={q66:.3f}s (n={len(meds)})")
        mask = table["condition"] == cond
        vals = table.loc[mask, "median_latency_s"]
        tert = pd.Series(None, index=table.loc[mask].index, dtype=object)
        tert[vals <= q33] = "QUICK"
        tert[(vals > q33) & (vals <= q66)] = "STEADY"
        tert[vals > q66] = "SLOW"
        table.loc[mask, "pace_tertile"] = tert.values
    out_path = ANALYSIS / "persona_behavioral.csv"
    table.to_csv(out_path, index=False)
    print(f"persona_behavioral: {len(table)} user×condition rows -> {out_path}")
    print(table.to_string(index=False))


def section_f(scorable):
    start_section("F. Agreement with model")
    rows = []
    for cond, group in scorable.groupby("condition"):
        row = {"condition": cond, **agreement_table(group)}
        rows.append(row)
    out_table(pd.DataFrame(rows))

    rows = []
    for (user, cond), group in scorable.groupby(["user", "condition"]):
        row = {"user": user, "condition": cond, **agreement_table(group)}
        rows.append(row)
    out("\nPer user:")
    out_table(pd.DataFrame(rows))


def section_g(dec):
    start_section("G. Per-condition response rates")
    top1 = dec["label"] == dec["model_prediction"]
    top2 = top1 | (dec["label"] == dec["second_model_prediction"])
    idk = dec["i_dont_know"].fillna(False).astype(bool)
    difficult = dec["is_difficult"].fillna(False).astype(bool)
    table = dec.assign(top1=top1, top2=top2, idk=idk, difficult=difficult).groupby("condition").agg(
        n=("top1", "size"), top1=("top1", "mean"), top2=("top2", "mean"),
        idk=("idk", "mean"), difficult=("difficult", "mean"),
    )
    out_table(table, index=True)


def section_h(scorable, dec, sessions):
    start_section("H. Throughput")
    n = scorable.groupby("session_id").size().rename("n")
    ts = dec.groupby("session_id")["ts"].agg(["min", "max"])
    active_h = ((ts["max"] - ts["min"]).dt.total_seconds() / 3600).rename("active_h")
    t = sessions.set_index("session_id").join(n).join(active_h)
    t["thr_span"] = t["n"] / (t["span_min"] / 60)
    t["thr_active"] = t["n"] / t["active_h"]
    out_table(t[["user", "condition", "n", "thr_span", "thr_active"]], index=True, ndigits=2)

    summary = t.groupby("condition")[["thr_span", "thr_active"]].agg(["mean", "median", "count"])
    out("\nPer-condition summary (tickets/hour):")
    out_table(summary, index=True, ndigits=2)

    for metric in ("thr_span", "thr_active"):
        pivot = t.pivot(index="user", columns="condition", values=metric)
        users = pivot.dropna().index
        stat = stats.wilcoxon(pivot.loc[users, "treatment"], pivot.loc[users, "control"])
        out(f"Wilcoxon {metric} (n={len(users)}): "
            f"control={pivot['control'].mean():.2f}  treatment={pivot['treatment'].mean():.2f}  "
            f"p={stat.pvalue:.4f}")


def section_i(dec):
    start_section("I. IDK & difficult tickets: totals and correct-team distributions")
    idk = dec["i_dont_know"].fillna(False).astype(bool)
    difficult = dec["is_difficult"].fillna(False).astype(bool)
    out(f"Total idk tickets: {idk.sum()}  Total difficult tickets: {difficult.sum()}  "
        f"(marked both: {(idk & difficult).sum()})")
    out("Note: idk tickets carry no human label (excluded from accuracy analysis); "
        "distributions are of the correct (ground-truth) team.")

    for name, mask in [("idk", idk), ("difficult", difficult)]:
        sub = dec[mask]
        dist = sub["ground_truth"].value_counts()
        out(f"\n{name} (n={len(sub)}) correct-team distribution (overall):")
        out_table(pd.DataFrame({"n": dist, "pct": (dist / len(sub) * 100).round(1)}),
                  index=True, ndigits=None)

        counts = (sub.groupby(["ground_truth", "condition"]).size()
                  .unstack("condition", fill_value=0)
                  .reindex(columns=["control", "treatment"], fill_value=0))
        pct = counts.div(counts.sum(axis=0), axis=1).mul(100).round(1)
        per_cond = counts.join(pct, lsuffix="_n", rsuffix="_pct").loc[dist.index]
        out(f"\n{name} (n={len(sub)}) per condition:")
        out_table(per_cond, index=True, ndigits=None)


def section_j():
    start_section("J. Balanced analysis cutoffs (retention by session length)")
    # Retention is evaluated on the f1 view (full label sequences, no trim,
    # first day-part only) so cutoffs reflect the input of the headline F1
    # analysis rather than the trimmed fatigue view.
    dec_f1 = pd.read_csv(ANALYSIS / "decisions_f1.csv")
    sessions_f1 = pd.read_csv(ANALYSIS / "sessions_f1.csv")
    sessions_f1["paired"] = sessions_f1["paired"].fillna(False).astype(bool)
    unpaired = sorted(sessions_f1.loc[~sessions_f1["paired"], "user"].unique())
    if unpaired:
        out(f"Unpaired users in the f1 view (one condition only): "
            f"{', '.join(unpaired)}. Their sessions stay in the session/ticket "
            "counts; users_kept counts paired users only.")
    scorable = dec_f1[~dec_f1["i_dont_know"].fillna(False).astype(bool)]
    n = scorable.groupby("session_id").size().rename("n_scorable")
    per_sid = sessions_f1.merge(n, on="session_id")

    cutoffs = sorted(per_sid["n_scorable"].unique())
    per_user = (per_sid[per_sid["paired"]].groupby("user")["n_scorable"].min())
    rows = []
    for t in cutoffs:
        kept = per_sid[per_sid["n_scorable"] >= t]
        rows.append({
            "cutoff": t,
            "control_sessions": (kept["condition"] == "control").sum(),
            "treatment_sessions": (kept["condition"] == "treatment").sum(),
            "users_kept": int((per_user >= t).sum()),
            "control_tickets": kept.loc[kept["condition"] == "control", "n_scorable"].sum(),
            "treatment_tickets": kept.loc[kept["condition"] == "treatment", "n_scorable"].sum(),
        })
    table = pd.DataFrame(rows)
    out("\nFor every cutoff on the f1 dataset (keep session if n_scorable >= "
        "cutoff; a user stays only if ALL of their sessions pass, keeping the "
        "within-subject pairing; session/ticket counts include unpaired users, "
        "users_kept is paired users only):")
    out_table(table, ndigits=None)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(table["cutoff"], table["users_kept"], "o-", label="users kept (paired)")
    ax.plot(table["cutoff"], table["control_sessions"], "^:", label="control sessions")
    ax.plot(table["cutoff"], table["treatment_sessions"], "s:", label="treatment sessions")
    ax.set_xlabel("cutoff: minimum scorable tickets per session")
    ax.set_ylabel("count kept")
    ax.set_title("Retention at each cutoff (f1 dataset)\n(user kept only if both conditions pass)")
    ax.legend(fontsize=8)
    save_fig(fig, "j_retention.png",
             "Sessions (f1 dataset, all users) and paired users retained as the "
             "minimum session length (scorable tickets) cutoff increases.")


def section_k():
    start_section("K. Model learning curves by retention cohort (f1 dataset)")
    # Like section C, but (a) sourced from the f1 artifacts and (b) restricted
    # to paired users (f1 view), split into the section-J retention cohorts.
    # Within every subplot the instance set is fixed by construction, so any
    # movement of the pooled mean is a learning effect rather than a change of
    # cohort composition; the whole-range panel keeps only users with BOTH
    # sessions present at each budget and marks every user loss explicitly.
    dec_f1 = pd.read_csv(ANALYSIS / "decisions_f1.csv")
    sessions_f1 = pd.read_csv(ANALYSIS / "sessions_f1.csv")
    sessions_f1["paired"] = sessions_f1["paired"].fillna(False).astype(bool)
    metrics = pd.read_csv(ANALYSIS / "al_metrics_f1.csv")
    unpaired = sorted(sessions_f1.loc[~sessions_f1["paired"], "user"].unique())
    out(f"Unpaired f1 users excluded: {', '.join(unpaired)}; "
        f"{sessions_f1['paired'].sum()} paired sessions remain.")

    scorable = dec_f1[~dec_f1["i_dont_know"].fillna(False).astype(bool)]
    n_scorable = scorable.groupby("session_id").size().rename("n_scorable")
    per_sid = sessions_f1.merge(n_scorable, on="session_id")
    per_user = per_sid[per_sid["paired"]].groupby("user")["n_scorable"].min().sort_values()
    inst_map = (
        sessions_f1[sessions_f1["user"].isin(per_user.index)]
        .assign(instance=lambda d: d["instances"].astype(str).str.split(","))
        .explode("instance")
    )
    inst_map = inst_map[inst_map["instance"].str.strip() != ""]
    inst_map["instance"] = inst_map["instance"].astype(int)
    cond_of = metrics.groupby("al_instance_id")["condition"].agg(lambda s: s.mode()[0])
    n_rows = inst_map.groupby("user").size()
    if (n_rows != 2).any():
        out(f"WARNING users with != 2 paired sessions: "
            f"{dict(n_rows[n_rows != 2])}")

    # Cohorts = section-J retention cutoffs, keeping cohorts with >= 5 paired
    # users (below that the >= 5 rule would leave no pooled mean at all).
    # Cutoffs that keep the same user set are collapsed into one cohort,
    # relabeled with the highest (tightest) cutoff of the run - the loose
    # lower label would understate the cohort's true common minimum.
    MIN_USERS = 5
    cohorts = []
    skipped_at = None
    for t in sorted(per_sid["n_scorable"].unique()):
        users = list(per_user.index[per_user >= t])
        if len(users) < MIN_USERS:
            if skipped_at is None:
                skipped_at = t
            continue
        if cohorts and len(users) == len(cohorts[-1][1]):
            cohorts[-1] = (t, users)
            continue
        cohorts.append((t, users))
    if skipped_at is not None:
        out(f"Cutoffs t>={skipped_at} keep fewer than {MIN_USERS} paired users; skipped.")

    rows = []
    for t, users in cohorts:
        inst = sorted(inst_map.loc[inst_map["user"].isin(users), "instance"].unique())
        met = metrics[metrics["al_instance_id"].isin(inst)]
        last = met.groupby("al_instance_id")["num_labeled"].max()
        conds = cond_of.reindex(inst)
        rows.append({
            "cutoff": t,
            "users": len(users),
            "control_inst": int((conds == "control").sum()),
            "treatment_inst": int((conds == "treatment").sum()),
            "x_max": int(last.min()) if len(met) else None,
        })
    alive = [i for i, r in enumerate(rows) if r["x_max"] is not None]
    if len(alive) != len(rows):
        out(f"Skipping cohorts without AL-metric rows: "
            f"{[rows[i]['cutoff'] for i in range(len(rows)) if i not in alive]}")
    cohorts = [cohorts[i] for i in alive]
    rows = [rows[i] for i in alive]
    if not cohorts:
        out(f"No retention cohorts with >={MIN_USERS} paired users; skipping cohort plots.")
        return
    out("\nRetention cohorts (each subplot keeps this exact instance set over "
        "its whole x-range; x stops at the cohort's lowest last training "
        "budget, not at the cutoff value, which is counted in human tickets):")
    out_table(pd.DataFrame(rows), ndigits=None)

    colors = {"control": "tab:blue", "treatment": "tab:orange"}
    xlim = (int(metrics["num_labeled"].min()),
            int(metrics[metrics["al_instance_id"].isin(inst_map["instance"].unique())]
                ["num_labeled"].max()))
    ylim = (metrics["f1_score"].min(), metrics["f1_score"].max())

    def draw(ax, met, x_max, title):
        for cond, color in colors.items():
            grp = met[met["condition"] == cond]
            for _, sub in grp.groupby("al_instance_id"):
                sub = sub.sort_values("num_labeled")
                ax.plot(sub["num_labeled"], sub["f1_score"], color=color,
                        alpha=0.25, linewidth=1)
            step = grp.groupby("num_labeled")["f1_score"].agg(["mean", "sem", "count"])
            keep = (step["count"] >= 5) & (step.index <= x_max)
            if keep.any():
                ax.plot(step.index[keep], step["mean"][keep], marker="o",
                        markersize=3, color=color, label=cond)
                ax.fill_between(step.index[keep], (step["mean"] - step["sem"])[keep],
                                (step["mean"] + step["sem"])[keep],
                                color=color, alpha=0.2)
        ax.set_xlim(xlim[0], x_max)
        ax.set_ylim(ylim)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)

    # Plot 1: option 1 - one panel per cohort, cohort fixed from x=0 on.
    n_rows_fig = (len(cohorts) + 1) // 2
    fig, axes = plt.subplots(n_rows_fig, 2, figsize=(12, 3.1 * n_rows_fig))
    axes = axes.ravel()
    for i, ((t, users), row) in enumerate(zip(cohorts, rows)):
        inst = sorted(inst_map.loc[inst_map["user"].isin(users), "instance"].unique())
        draw(axes[i], metrics[metrics["al_instance_id"].isin(inst)], row["x_max"],
             f"cutoff >= {t}: {len(users)} paired users")
        if i >= len(cohorts) - 2:
            axes[i].set_xlabel("labels used for training", fontsize=8)
        if i % 2 == 0:
            axes[i].set_ylabel("model macro F1 (test)", fontsize=8)
    for j in range(len(cohorts), len(axes)):
        axes[j].set_visible(False)
    handles = [Line2D([], [], color=colors["control"], marker="o", markersize=3,
                      label="control"),
               Line2D([], [], color=colors["treatment"], marker="o", markersize=3,
                      label="treatment"),
               Line2D([], [], color="gray", alpha=0.5, linewidth=1,
                      label="individual instances")]
    fig.suptitle("Model F1 by retention cohort (f1 dataset)\n"
                 "cohort fixed within each panel; mean ± SEM where >=5 instances",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.975))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    save_fig(fig, "k_cohort_curves.png",
             "Option 1: one panel per section-J retention cohort (>=5 paired "
             "users). The instance set of a panel is fixed from the start, so "
             "pooled-mean movement inside a panel is a learning effect; "
             "differences between panels are cohort composition. Each panel "
             "stops at the lowest last training budget among its instances.")

    # Plot 2: all cohorts merged into one panel, kept fully paired at every
    # point: a paired user contributes (to the pooled mean and their faint
    # lines) only while BOTH of their instances still have data at that
    # budget; the budget where a user's both-sessions data ends is marked as
    # a user loss.
    met_all = metrics[metrics["al_instance_id"].isin(inst_map["instance"].unique())]
    lo_x = met_all.groupby("al_instance_id")["num_labeled"].min()
    hi_x = met_all.groupby("al_instance_id")["num_labeled"].max()
    pair_of = {u: dict(zip(g["condition"], g["instance"]))
               for u, g in inst_map.groupby("user")}
    windows = {u: (max(lo_x[p["control"]], lo_x[p["treatment"]]),
                   min(hi_x[p["control"]], hi_x[p["treatment"]]))
               for u, p in pair_of.items()
               if set(p) == {"control", "treatment"}}
    parts = []
    for u, (f_u, l_u) in windows.items():
        for inst_id in pair_of[u].values():
            g = met_all[met_all["al_instance_id"] == inst_id]
            parts.append(g[(g["num_labeled"] >= f_u) & (g["num_labeled"] <= l_u)])
    kept = pd.concat(parts, ignore_index=True)

    lost_at = {}
    for u, (f_u, l_u) in windows.items():
        lost_at.setdefault(int(l_u), []).append(u)
    out("\nUser losses in plot 2 (rule: a user stays at a budget only while "
        "BOTH of their sessions still have data; each user's data ends at the "
        "num_labeled shown, remaining_after = users alive beyond it):")
    out_table(pd.DataFrame(
        [(l, ", ".join(sorted(lost_at[l])),
          sum(1 for _, l2 in windows.values() if l2 > l)) for l in sorted(lost_at)],
        columns=["num_labeled", "users_lost", "remaining_after"]), ndigits=None)

    fig2, ax = plt.subplots(figsize=(7.5, 4.8))
    draw(ax, kept, xlim[1], "Model F1, paired users (f1 dataset)\n"
         "mean over users with BOTH sessions present at each x")
    for l in lost_at:
        ax.axvline(l, color="gray", ls=":", lw=0.9, alpha=0.5)
    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1 + [Line2D([], [], color="gray", alpha=0.5, linewidth=1)],
              labels1 + ["individual instances"],
              fontsize=7, loc="lower right")
    fig2.tight_layout()
    save_fig(fig2, "k_paired_curves.png",
             "All paired users merged into one panel: every point of the pooled "
             "mean (and each faint line) uses only users whose control AND "
             "treatment session still have data at that budget. Vertical dotted "
             "lines mark the budgets where a user (both sessions together) "
             "drops out.")

    # Plot 2b: same construction truncated at the "120 total labels" point
    # (= 70 seed labels + 50 human decisions): instance curves and windows
    # are cut at num_labeled <= TRIM_NUM_LABELED, so users whose sessions
    # would run past the cap drop out there (or earlier, if their own data
    # already ended before it).
    met_trim = met_all[met_all["num_labeled"] <= TRIM_NUM_LABELED]
    if not len(met_trim):
        out(f"\nNo iterations with num_labeled <= {TRIM_NUM_LABELED}; "
            f"skipping the truncated paired-curves plot.")
    else:
        lo_x_t = met_trim.groupby("al_instance_id")["num_labeled"].min()
        hi_x_t = met_trim.groupby("al_instance_id")["num_labeled"].max()
        windows_t = {u: (max(lo_x_t[p["control"]], lo_x_t[p["treatment"]]),
                         min(hi_x_t[p["control"]], hi_x_t[p["treatment"]]))
                     for u, p in pair_of.items()
                     if set(p) == {"control", "treatment"}
                     and p["control"] in lo_x_t and p["treatment"] in lo_x_t}
        parts_t = []
        for u, (f_u, l_u) in windows_t.items():
            for inst_id in pair_of[u].values():
                g = met_trim[met_trim["al_instance_id"] == inst_id]
                parts_t.append(g[(g["num_labeled"] >= f_u) & (g["num_labeled"] <= l_u)])
        kept_t = pd.concat(parts_t, ignore_index=True)

        lost_at_t = {}
        for u, (f_u, l_u) in windows_t.items():
            lost_at_t.setdefault(int(l_u), []).append(u)
        out(f"\nUser losses in the truncated paired-curves plot (cut at "
            f"num_labeled <= {TRIM_NUM_LABELED}, same both-sessions-present "
            "rule as plot 2):")
        out_table(pd.DataFrame(
            [(l, ", ".join(sorted(lost_at_t[l])),
              sum(1 for _, l2 in windows_t.values() if l2 > l)) for l in sorted(lost_at_t)],
            columns=["num_labeled", "users_lost", "remaining_after"]), ndigits=None)

        fig2t, axt = plt.subplots(figsize=(7.5, 4.8))
        draw(axt, kept_t, TRIM_NUM_LABELED,
             f"Model F1, paired users (f1 dataset), truncated at "
             f"{TRIM_NUM_LABELED} total training labels\n"
             "(70 seed + 50 human decisions; mean over users with BOTH "
             "sessions present at each x)")
        axt.set_ylim(*TRIM120_YLIM)
        for l_t in lost_at_t:
            axt.axvline(l_t, color="gray", ls=":", lw=0.9, alpha=0.5)
        lines_t, labels_t = axt.get_legend_handles_labels()
        axt.legend(lines_t + [Line2D([], [], color="gray", alpha=0.5, linewidth=1)],
                   labels_t + ["individual instances"],
                   fontsize=7, loc="lower right")
        fig2t.tight_layout()
        save_fig(fig2t, "k_paired_curves_trim120.png",
                 f"Same paired-user panel as k_paired_curves.png, truncated at "
                 f"num_labeled <= {TRIM_NUM_LABELED} total training labels "
                 "(70 seed labels + 50 human decisions): every point beyond "
                 "that budget is ignored, and users whose sessions would run "
                 "past the cap drop out there; y-axis fixed to 0.45-0.75.")


def section_l():
    start_section("L. Total activity time per user & condition")
    # Per-user total "activity" time: summed decision latencies over ALL
    # decisions (idk tickets included), from the labels view (full untrimmed
    # sequences from cold start, no trim and no block rule).
    dec = pd.read_csv(ANALYSIS / "decisions_labels.csv")
    sessions = pd.read_csv(ANALYSIS / "sessions_labels.csv")
    user_of = dict(zip(sessions["session_id"], sessions["user"]))
    dec["user"] = dec["session_id"].map(user_of)
    missing = int(dec["user"].isna().sum())
    if missing:
        out(f"WARNING {missing} decisions without session user mapping; dropped")
        dec = dec[dec["user"].notna()]
    dec["latency_min"] = dec["latency_ms"] / 60000
    dec["n_null_latency"] = dec["latency_ms"].isna().astype(int)

    per_session = (
        dec.groupby(["user", "condition", "session_id"])
        .agg(n_decisions=("latency_min", "size"),
             n_null_latency=("n_null_latency", "sum"),
             total_min=("latency_min", "sum"))
        .reset_index()
        .sort_values(["user", "condition"])
    )
    out("\nPer-session total activity time (summed decision latencies, idk "
        "tickets included; n_null_latency = decisions with null latency, "
        "breaks >10 min, skipped by the sum):")
    out_table(per_session, ndigits=1)

    per_user = (
        per_session.groupby(["user", "condition"])["total_min"].sum().reset_index()
    )
    out("\nPer-user total activity time:")
    out_table(per_user, ndigits=1)
    split_users = sorted(
        sessions.loc[sessions["session_id"].str.contains(r"-d\d+$"), "user"].unique()
    )
    if split_users:
        out(f"Note: split-by-day users ({', '.join(split_users)}): each "
            "condition's total covers only that condition's day-part "
            "(e.g. user15-s1-d1 / user15-s1-d2).")

    pivot = (per_user.pivot(index="user", columns="condition", values="total_min")
             .reindex(columns=["control", "treatment"]))
    users = sorted(pivot.index)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(users))
    width = 0.4
    ax.bar(x - width / 2, pivot.loc[users, "control"], width,
           color="tab:blue", label="control")
    ax.bar(x + width / 2, pivot.loc[users, "treatment"], width,
           color="tab:orange", label="treatment")
    ax.set_xticks(x, users, rotation=45, ha="right")
    ax.set_ylabel("total activity time (min)")
    ax.set_title("Total activity time per user\n(summed decision latencies, idk tickets included)")
    ax.legend()
    save_fig(fig, "l_total_time_column.png",
             "Per-user total activity time (summed decision latencies, idk "
             "tickets included), unaided (control) vs AI-assisted (treatment); "
             "labels-view data (full untrimmed sessions).")


def section_m():
    start_section("M. Time-based retention cutoffs (f1 view)")
    # Section-J retention logic, but the cutoff metric is the session's total
    # activity time (summed decision latencies, idk tickets included) instead
    # of the number of scorable tickets. Built on the f1 view, so conditions
    # that do not start in a new session (warm-started second day-parts) are
    # already excluded; the cohorts are meant to later evaluate F1 curves.
    dec_f1 = pd.read_csv(ANALYSIS / "decisions_f1.csv")
    sessions_f1 = pd.read_csv(ANALYSIS / "sessions_f1.csv")
    sessions_f1["paired"] = sessions_f1["paired"].fillna(False).astype(bool)
    unpaired = sorted(sessions_f1.loc[~sessions_f1["paired"], "user"].unique())
    if unpaired:
        out(f"Unpaired users in the f1 view (one condition only): "
            f"{', '.join(unpaired)}. Their sessions stay in the session/ticket "
            "counts; users_kept counts paired users only.")
    dec_f1["latency_min"] = dec_f1["latency_ms"] / 60000
    total_min = (dec_f1.groupby("session_id")["latency_min"].sum(min_count=1)
                 .rename("total_min"))
    per_sid = sessions_f1.merge(total_min, on="session_id")

    cutoffs = sorted(per_sid["total_min"].dropna().unique())
    per_user = per_sid[per_sid["paired"]].groupby("user")["total_min"].min()
    rows = []
    for t in cutoffs:
        kept = per_sid[per_sid["total_min"] >= t]
        rows.append({
            "cutoff_min": round(t, 1),
            "control_sessions": int((kept["condition"] == "control").sum()),
            "treatment_sessions": int((kept["condition"] == "treatment").sum()),
            "users_kept": int((per_user >= t).sum()),
            "control_tickets": int(kept.loc[kept["condition"] == "control", "n_labels"].sum()),
            "treatment_tickets": int(kept.loc[kept["condition"] == "treatment", "n_labels"].sum()),
        })
    table = pd.DataFrame(rows)
    out("\nFor every cutoff on the f1 dataset (keep session if total activity "
        "time >= cutoff; a user stays only if ALL of their sessions pass, "
        "keeping the within-subject pairing; time = summed decision latencies, "
        "idk included; session/ticket counts include unpaired users, "
        "users_kept is paired users only):")
    out_table(table, ndigits=None)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(table["cutoff_min"], table["users_kept"], "o-", label="users kept (paired)")
    ax.plot(table["cutoff_min"], table["control_sessions"], "^:", label="control sessions")
    ax.plot(table["cutoff_min"], table["treatment_sessions"], "s:", label="treatment sessions")
    ax.set_xlabel("cutoff: minimum total activity time per session (min)")
    ax.set_ylabel("count kept")
    ax.set_title("Retention at each time cutoff (f1 dataset)\n(user kept only if both conditions pass)")
    ax.legend(fontsize=8)
    save_fig(fig, "m_retention_time.png",
             "Sessions (f1 dataset, all users) and paired users retained as the "
             "minimum session activity time (summed decision latencies, idk "
             "included) cutoff increases.")


def section_n():
    start_section("N. Model learning curves over activity time by retention cohort (f1 dataset)")
    # Like section K, but the x-axis is the elapsed active labeling time within
    # the session (cumulative summed decision latencies, idk included, breaks
    # >10min count 0) instead of num_labeled. Cohorts are the section-M
    # time-based retention cutoffs. Pooling uses a fixed GRID_MIN-minute grid
    # onto which every instance's curve is step-forward-filled: at minute t an
    # instance shows its last retrain with x <= t (the time analog of the
    # shared num_labeled budget grid).
    dec_f1 = pd.read_csv(ANALYSIS / "decisions_f1.csv")
    sessions_f1 = pd.read_csv(ANALYSIS / "sessions_f1.csv")
    sessions_f1["paired"] = sessions_f1["paired"].fillna(False).astype(bool)
    metrics = pd.read_csv(ANALYSIS / "al_metrics_f1.csv")
    unpaired = sorted(sessions_f1.loc[~sessions_f1["paired"], "user"].unique())
    out(f"Unpaired f1 users excluded: {', '.join(unpaired)}; "
        f"{sessions_f1['paired'].sum()} paired sessions remain.")
    GRID_MIN = 1
    MIN_USERS = 5

    # Per-decision elapsed active time within its session (idk included;
    # null latencies = breaks >10min, count 0 like the section-L/M sums).
    dec_f1["ts"] = pd.to_datetime(dec_f1["labeled_at"], format="mixed")
    dec_f1["latency_min"] = dec_f1["latency_ms"].fillna(0) / 60000
    dec_f1 = dec_f1.sort_values("ts")
    dec_f1["elapsed_min"] = dec_f1.groupby("session_id")["latency_min"].cumsum()

    # Per-retrain x: elapsed active time of the instance's session at created_at
    # (retrains fire right after a label submission; iterations created before
    # the session's first decision - the initial model - get x=0).
    session_of = dec_f1.groupby("al_instance_id")["session_id"].agg(lambda s: s.mode()[0])
    met = metrics.copy()
    met["created_at"] = pd.to_datetime(met["created_at"], format="mixed")
    unknown = sorted(set(met["al_instance_id"]) - set(session_of.index))
    if unknown:
        out(f"WARNING AL instances without f1 decisions (dropped from time curves): {unknown}")
        met = met[met["al_instance_id"].isin(session_of.index)]
    met["session_id"] = met["al_instance_id"].map(session_of)
    merged = pd.merge_asof(
        met.sort_values("created_at"),
        dec_f1[["session_id", "ts", "elapsed_min"]],
        left_on="created_at", right_on="ts", by="session_id", direction="backward",
    )
    merged["x_min"] = merged["elapsed_min"].fillna(0)

    # Retention cohorts from section M: per-user minimum session activity time.
    total = dec_f1.groupby("session_id")["elapsed_min"].max().rename("total_min")
    per_sid = sessions_f1.merge(total, on="session_id")
    per_user = per_sid[per_sid["paired"]].groupby("user")["total_min"].min().sort_values()
    inst_map = (
        sessions_f1[sessions_f1["user"].isin(per_user.index)]
        .assign(instance=lambda d: d["instances"].astype(str).str.split(","))
        .explode("instance")
    )
    inst_map = inst_map[inst_map["instance"].str.strip() != ""]
    inst_map["instance"] = inst_map["instance"].astype(int)
    n_rows = inst_map.groupby("user").size()
    if (n_rows != 2).any():
        out(f"WARNING users with != 2 paired sessions: {dict(n_rows[n_rows != 2])}")

    # Cutoffs that keep the same user set are collapsed into one cohort,
    # relabeled with the highest (tightest) cutoff of the run - the loose
    # lower label would understate the cohort's true common minimum.
    cohorts = []
    skipped_at = None
    for t in sorted(per_sid["total_min"].dropna().unique()):
        users = list(per_user.index[per_user >= t])
        if len(users) < MIN_USERS:
            if skipped_at is None:
                skipped_at = t
            continue
        if cohorts and len(users) == len(cohorts[-1][1]):
            cohorts[-1] = (t, users)
            continue
        cohorts.append((t, users))
    if skipped_at is not None:
        out(f"Cutoffs t>={skipped_at:.1f}min keep fewer than {MIN_USERS} paired users; skipped.")

    cond_of = metrics.groupby("al_instance_id")["condition"].agg(lambda s: s.mode()[0])
    rows = []
    for t, users in cohorts:
        inst = sorted(inst_map.loc[inst_map["user"].isin(users), "instance"].unique())
        met_c = merged[merged["al_instance_id"].isin(inst)]
        last = met_c.groupby("al_instance_id")["x_min"].max()
        x_max = float(last.min()) if len(last) else float("nan")
        conds = cond_of.reindex(inst)
        rows.append({
            "cutoff_min": round(t, 1),
            "users": len(users),
            "control_inst": int((conds == "control").sum()),
            "treatment_inst": int((conds == "treatment").sum()),
            "x_max": round(x_max, 1) if not np.isnan(x_max) else None,
        })
    alive = [i for i, r in enumerate(rows) if r["x_max"] is not None]
    if len(alive) != len(rows):
        out(f"Skipping cohorts without AL-metric rows: "
            f"{[rows[i]['cutoff_min'] for i in range(len(rows)) if i not in alive]}")
    cohorts = [cohorts[i] for i in alive]
    rows = [rows[i] for i in alive]
    if not cohorts:
        out(f"No retention cohorts with >={MIN_USERS} paired users; skipping cohort plots.")
        return
    out("\nTime retention cohorts (each subplot keeps this exact instance set "
        "over its whole x-range; x is elapsed active labeling time and stops "
        "at the cohort's lowest session total, not at the cutoff value, which "
        "is counted in human minutes):")
    out_table(pd.DataFrame(rows), ndigits=None)

    colors = {"control": "tab:blue", "treatment": "tab:orange"}
    inst_all = inst_map["instance"].unique()
    xlim = (0.0, float(merged.loc[merged["al_instance_id"].isin(inst_all), "x_min"].max()))
    ylim = (metrics["f1_score"].min(), metrics["f1_score"].max())

    def curve_at(sub, grid):
        """Step-forward-filled F1 at each grid minute (last retrain with x <= t)."""
        sub = sub.sort_values(["x_min", "created_at"])
        x = sub["x_min"].to_numpy()
        y = sub["f1_score"].to_numpy()
        idx = np.searchsorted(x, grid, side="right") - 1
        filled = np.full(grid.shape, np.nan)
        ok = idx >= 0
        filled[ok] = y[idx[ok]]
        filled[grid > x.max()] = np.nan
        return filled

    def draw(ax, inst_ids, x_max, title):
        sub_all = merged[merged["al_instance_id"].isin(inst_ids)]
        grid = np.arange(0, x_max + GRID_MIN, GRID_MIN)
        for cond, color in colors.items():
            grp = sub_all[sub_all["condition"] == cond]
            cols = {}
            for inst_id, sub in grp.groupby("al_instance_id"):
                sub = sub.sort_values(["x_min", "created_at"])
                ax.plot(sub["x_min"], sub["f1_score"], color=color, alpha=0.25,
                        linewidth=1, drawstyle="steps-post")
                cols[inst_id] = curve_at(sub, grid)
            mat = pd.DataFrame(cols)
            counts = mat.count(axis=1).to_numpy()
            keep = (counts >= 5) & (grid <= x_max)
            if keep.any():
                mean = mat.mean(axis=1).to_numpy()
                sem = mat.sem(axis=1).to_numpy()
                ax.plot(grid[keep], mean[keep], marker="o", markersize=3,
                        color=color, label=cond)
                ax.fill_between(grid[keep], (mean - sem)[keep], (mean + sem)[keep],
                                color=color, alpha=0.2)
        ax.set_xlim(xlim[0], x_max)
        ax.set_ylim(ylim)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)

    # Plot 1: one panel per cohort, cohort fixed from x=0 on.
    n_rows_fig = (len(cohorts) + 1) // 2
    fig, axes = plt.subplots(n_rows_fig, 2, figsize=(12, 3.1 * n_rows_fig))
    axes = axes.ravel()
    for i, ((t, users), row) in enumerate(zip(cohorts, rows)):
        inst = sorted(inst_map.loc[inst_map["user"].isin(users), "instance"].unique())
        draw(axes[i], inst, row["x_max"],
             f"cutoff >= {t:.1f} min: {len(users)} paired users")
        if i >= len(cohorts) - 2:
            axes[i].set_xlabel("active labeling time in session (min)", fontsize=8)
        if i % 2 == 0:
            axes[i].set_ylabel("model macro F1 (test)", fontsize=8)
    for j in range(len(cohorts), len(axes)):
        axes[j].set_visible(False)
    handles = [Line2D([], [], color=colors["control"], marker="o", markersize=3,
                      label="control"),
               Line2D([], [], color=colors["treatment"], marker="o", markersize=3,
                      label="treatment"),
               Line2D([], [], color="gray", alpha=0.5, linewidth=1,
                      label="individual instances")]
    fig.suptitle("Model F1 by time-based retention cohort (f1 dataset)\n"
                 "cohort fixed within each panel; mean ± SEM where >=5 instances",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.975))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    save_fig(fig, "n_cohort_curves_time.png",
             "One panel per section-M time-based retention cohort (>=5 paired "
             "users); x is elapsed active labeling time (summed decision "
             "latencies, idk included). The instance set of a panel is fixed "
             "from the start, so pooled-mean movement inside a panel is a "
             "learning effect; differences between panels are cohort "
             "composition. Each panel stops at the cohort's lowest session "
             "total time.")

    # Plot 2: all cohorts merged into one panel, kept fully paired at every
    # point: a user contributes only while BOTH of their sessions still have
    # data at that minute (i.e. while the section-M cutoff t keeps them); the
    # minute where a user's both-sessions data ends is marked as a user loss.
    met_all = merged[merged["al_instance_id"].isin(inst_all)]
    first_x = met_all.groupby("al_instance_id")["x_min"].min()
    last_x = met_all.groupby("al_instance_id")["x_min"].max()
    pair_of = {u: dict(zip(g["condition"], g["instance"]))
               for u, g in inst_map.groupby("user")}
    windows = {u: (max(first_x[p["control"]], first_x[p["treatment"]]),
                   min(last_x[p["control"]], last_x[p["treatment"]]))
               for u, p in pair_of.items()
               if set(p) == {"control", "treatment"}}
    grid_all = np.arange(0, xlim[1] + GRID_MIN, GRID_MIN)
    cond_curves = {"control": {}, "treatment": {}}
    for u, (f_u, l_u) in windows.items():
        for cond in ("control", "treatment"):
            c = curve_at(met_all[met_all["al_instance_id"] == pair_of[u][cond]], grid_all)
            c[(grid_all < f_u) | (grid_all > l_u)] = np.nan
            cond_curves[cond][u] = c

    lost_at = {}
    for u, (f_u, l_u) in windows.items():
        lost_at.setdefault(round(l_u, 1), []).append(u)
    out("\nUser losses in plot 2 (rule: a user stays at a time point only while "
        "BOTH of their sessions still have data; each user's data ends at the "
        "elapsed minutes shown, remaining_after = users alive beyond it):")
    out_table(pd.DataFrame(
        [(l, ", ".join(sorted(lost_at[l])),
          sum(1 for _, l2 in windows.values() if l2 > l)) for l in sorted(lost_at)],
        columns=["loss_min", "users_lost", "remaining_after"]), ndigits=None)

    fig2, ax = plt.subplots(figsize=(7.5, 4.8))
    for u, (f_u, l_u) in windows.items():
        for cond, color in colors.items():
            sub = met_all[met_all["al_instance_id"] == pair_of[u][cond]]
            sub = sub.sort_values(["x_min", "created_at"])
            sub = sub[(sub["x_min"] >= f_u) & (sub["x_min"] <= l_u)]
            ax.plot(sub["x_min"], sub["f1_score"], color=color, alpha=0.25,
                    linewidth=1, drawstyle="steps-post")
    for cond, color in colors.items():
        mat = pd.DataFrame(cond_curves[cond])
        counts = mat.count(axis=1).to_numpy()
        keep = counts >= 5
        if keep.any():
            mean = mat.mean(axis=1).to_numpy()
            sem = mat.sem(axis=1).to_numpy()
            ax.plot(grid_all[keep], mean[keep], marker="o", markersize=3,
                    color=color, label=cond)
            ax.fill_between(grid_all[keep], (mean - sem)[keep], (mean + sem)[keep],
                            color=color, alpha=0.2)
    ax.set_xlim(*xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("active labeling time in session (min)")
    ax.set_ylabel("model macro F1 (test)")
    ax.set_title("Model F1 over activity time, paired users (f1 dataset)\n"
                 "mean over users with BOTH sessions present at each minute")
    for l in lost_at:
        ax.axvline(l, color="gray", ls=":", lw=0.9, alpha=0.5)
    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1 + [Line2D([], [], color="gray", alpha=0.5, linewidth=1)],
              labels1 + ["individual instances"],
              fontsize=7, loc="lower right")
    fig2.tight_layout()
    save_fig(fig2, "n_paired_curves_time.png",
             "All paired users merged into one panel; x is elapsed active "
             "labeling time (summed decision latencies, idk included). Every "
             "point of the pooled mean (and each faint line) uses only users "
             "whose control AND treatment session still have data at that "
             "minute - the same alive rule as the section-M retention counts. "
             "Vertical dotted lines mark the minutes where a user (both "
             "sessions together) drops out.")


def section_o():
    start_section("O. Area under the learning curve (AULC)")
    # Area under the model F1 learning curve (f1 artifacts), computed over both
    # x-axes: labels used for training (budget, section-J axis) and elapsed
    # active labeling time (section-N construction, section-M axis). Within a
    # user, both conditions are integrated over the SHORTER session's range so
    # the paired areas are comparable; the Wilcoxon test runs on those per-user
    # areas. Integration is trapezoidal over the retrain points with an
    # interpolated endpoint at the integration limit; where a session's last
    # retrain precedes its limit (trailing idk tickets trigger no retrain),
    # the final model's F1 is held over the remaining minutes.
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    dec_f1 = pd.read_csv(ANALYSIS / "decisions_f1.csv")
    sessions_f1 = pd.read_csv(ANALYSIS / "sessions_f1.csv")
    sessions_f1["paired"] = sessions_f1["paired"].fillna(False).astype(bool)
    metrics = pd.read_csv(ANALYSIS / "al_metrics_f1.csv")
    unpaired = sorted(sessions_f1.loc[~sessions_f1["paired"], "user"].unique())
    out(f"Unpaired f1 users excluded: {', '.join(unpaired)}; "
        f"{sessions_f1['paired'].sum()} paired sessions remain.")

    # Per-decision elapsed active time within its session (idk included,
    # breaks >10min count 0), and per-retrain x on both axes.
    dec_f1["ts"] = pd.to_datetime(dec_f1["labeled_at"], format="mixed")
    dec_f1["latency_min"] = dec_f1["latency_ms"].fillna(0) / 60000
    dec_f1 = dec_f1.sort_values("ts")
    dec_f1["elapsed_min"] = dec_f1.groupby("session_id")["latency_min"].cumsum()

    session_of = dec_f1.groupby("al_instance_id")["session_id"].agg(lambda s: s.mode()[0])
    met = metrics.copy()
    met["created_at"] = pd.to_datetime(met["created_at"], format="mixed")
    unknown = sorted(set(met["al_instance_id"]) - set(session_of.index))
    if unknown:
        out(f"WARNING AL instances without f1 decisions (dropped): {unknown}")
        met = met[met["al_instance_id"].isin(session_of.index)]
    met["session_id"] = met["al_instance_id"].map(session_of)
    merged = pd.merge_asof(
        met.sort_values("created_at"),
        dec_f1[["session_id", "ts", "elapsed_min"]],
        left_on="created_at", right_on="ts", by="session_id", direction="backward",
    )
    merged["x_min"] = merged["elapsed_min"].fillna(0)

    def label_curve(inst_id):
        g = merged[merged["al_instance_id"] == inst_id].sort_values("num_labeled")
        return g["num_labeled"].to_numpy(float), g["f1_score"].to_numpy(float)

    def time_curve(inst_id):
        g = merged[merged["al_instance_id"] == inst_id].sort_values(
            ["x_min", "created_at"])
        x = g["x_min"].to_numpy(float)
        y = g["f1_score"].to_numpy(float)
        keep = np.append(x[1:] != x[:-1], True)  # keep the last retrain per x
        x, y = x[keep], y[keep]
        if x[0] > 0:  # the initial model exists from session start
            x, y = np.append(0.0, x), np.append(y[0], y)
        return x, y

    def trapz_to(x, y, limit, hold=False):
        """Trapezoid of the curve over [x[0], limit]; the endpoint is
        interpolated at the limit, or the last F1 is held when the curve ends
        before the limit."""
        if limit <= x[0]:
            return 0.0
        inside = x <= limit
        xs, ys = x[inside], y[inside]
        if xs[-1] < limit:
            xs = np.append(xs, limit)
            ys = np.append(ys, y[-1] if hold else np.interp(limit, x, y))
        return float(trapz(ys, xs))

    # Retention per-user minima (section J/M) and (user, condition) -> instance.
    scorable = dec_f1[~dec_f1["i_dont_know"].fillna(False).astype(bool)]
    n_scorable = scorable.groupby("session_id").size().rename("n_scorable")
    total_min = dec_f1.groupby("session_id")["latency_min"].sum().rename("total_min")
    per_sid = sessions_f1.merge(n_scorable, on="session_id").merge(
        total_min, on="session_id")
    per_user_labels = per_sid[per_sid["paired"]].groupby("user")["n_scorable"].min()
    per_user_time = per_sid[per_sid["paired"]].groupby("user")["total_min"].min()
    inst_map = (
        sessions_f1[sessions_f1["paired"]]
        .assign(instance=lambda d: d["instances"].astype(str).str.split(","))
        .explode("instance")
    )
    inst_map = inst_map[inst_map["instance"].str.strip() != ""]
    inst_map["instance"] = inst_map["instance"].astype(int)
    pairs = {}
    for user, g in inst_map.groupby("user"):
        d = dict(zip(g["condition"], g["instance"]))
        if set(d) == {"control", "treatment"}:
            pairs[user] = d
        else:
            out(f"WARNING user {user}: conditions {sorted(d)}; skipped")
    curves_l = {u: {c: label_curve(i) for c, i in d.items()} for u, d in pairs.items()}
    curves_t = {u: {c: time_curve(i) for c, i in d.items()} for u, d in pairs.items()}

    # 1. Per-user AULC, both conditions over the shorter session's range.
    rows = []
    for user, d in pairs.items():
        cut_l = int(per_user_labels[user])
        cut_t = float(per_user_time[user])
        row = {"user": user, "cutoff_labels": cut_l, "cutoff_time_min": round(cut_t, 1)}
        for cond in ("control", "treatment"):
            xl, yl = curves_l[user][cond]
            xt, yt = curves_t[user][cond]
            row[f"aulc_labels_{cond}"] = trapz_to(xl, yl, 70 + cut_l)
            row[f"aulc_time_{cond}"] = trapz_to(xt, yt, cut_t, hold=True)
        rows.append(row)
    aulc = pd.DataFrame(rows)
    out("\nPer-user AULC (both conditions integrated over the shorter session's "
        "range: labels from budget 70 to 70+cutoff_labels; time from session "
        "start to cutoff_time_min):")
    out_table(aulc, ndigits=2)
    n_hold = sum(
        1 for u in pairs
        if any(curves_t[u][c][0][-1] < per_user_time[u] - 1e-9 for c in pairs[u])
    )
    if n_hold:
        out(f"Note: for {n_hold} of {len(pairs)} users at least one session's last "
            f"retrain precedes the time cutoff (trailing idk decisions); the final "
            f"model's F1 is held over the remaining minutes.")

    # 2. Wilcoxon on the paired per-user AULCs.
    out("\nWilcoxon signed-rank on the paired per-user AULCs:")
    for crit, unit in (("labels", "F1 x labels"), ("time", "F1 x min")):
        c = aulc[f"aulc_{crit}_control"]
        t = aulc[f"aulc_{crit}_treatment"]
        stat = stats.wilcoxon(t, c)
        out(f"  {crit} ({unit}): control={c.mean():.2f}  treatment={t.mean():.2f}  "
            f"difference={t.mean() - c.mean():+.2f}  (n={len(aulc)})  "
            f"p={stat.pvalue:.4f}")

    # 3. Slope plots of the per-user AULCs.
    def aulc_slope(pivot, ylabel, title, name, caption):
        users = pivot.dropna().index
        fig, ax = plt.subplots(figsize=(6, 5))
        for user in users:
            ax.plot([0, 1], [pivot.loc[user, "control"], pivot.loc[user, "treatment"]],
                    marker="o", color="gray", alpha=0.6)
        ax.plot([0, 1], [pivot.loc[users, "control"].mean(),
                         pivot.loc[users, "treatment"].mean()],
                marker="o", color="red", linewidth=4, markersize=10, label="mean")
        ax.set_xticks([0, 1], ["Unaided", "AI-assisted"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        save_fig(fig, name, caption)

    piv_l = aulc.set_index("user").rename(columns={
        "aulc_labels_control": "control", "aulc_labels_treatment": "treatment"})
    aulc_slope(piv_l[["control", "treatment"]],
               "AULC (F1 × labels used for training)",
               "Per-user AULC over labels: Unaided vs. AI-assisted",
               "o_aulc_labels_slope.png",
               "Area under the model F1 learning curve over training budgets 70 "
               "to 70+cutoff (the shorter session's scorable count), per user.")
    piv_t = aulc.set_index("user").rename(columns={
        "aulc_time_control": "control", "aulc_time_treatment": "treatment"})
    aulc_slope(piv_t[["control", "treatment"]],
               "AULC (F1 × active labeling minutes)",
               "Per-user AULC over active time: Unaided vs. AI-assisted",
               "o_aulc_time_slope.png",
               "Area under the model F1 learning curve over the shorter "
               "session's active labeling time, per user.")

    # 4. Partial AULC per retention cutoff (mean over the paired users that
    # remain at each cutoff; both conditions integrated over the same range).
    rows = []
    for t in sorted(per_sid["n_scorable"].unique()):
        users = [u for u in pairs if per_user_labels[u] >= t]
        if not users:
            continue
        rows.append({
            "axis": "labels", "cutoff": int(t), "n_users": len(users),
            "control_mean": float(np.mean(
                [trapz_to(*curves_l[u]["control"], 70 + t) for u in users])),
            "treatment_mean": float(np.mean(
                [trapz_to(*curves_l[u]["treatment"], 70 + t) for u in users])),
        })
    for t in sorted(per_sid["total_min"].dropna().unique()):
        users = [u for u in pairs if per_user_time[u] >= t]
        if not users:
            continue
        rows.append({
            "axis": "time", "cutoff": round(float(t), 1), "n_users": len(users),
            "control_mean": float(np.mean(
                [trapz_to(*curves_t[u]["control"], t, hold=True) for u in users])),
            "treatment_mean": float(np.mean(
                [trapz_to(*curves_t[u]["treatment"], t, hold=True) for u in users])),
        })
    cut_table = pd.DataFrame(rows)
    out("\nPartial AULC per retention cutoff (mean over the paired users that "
        "remain at the cutoff; labels integrated over budgets 70..70+cutoff, "
        "time over 0..cutoff minutes with the final F1 held past the last "
        "retrain where needed):")
    out_table(cut_table, ndigits=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis_name, ax, xlabel, title in [
        ("labels", axes[0], "cutoff: minimum scorable tickets per session",
         "Partial AULC over labels\n(integrated from budget 70)"),
        ("time", axes[1], "cutoff: minimum session activity time (min)",
         "Partial AULC over active time\n(integrated from session start)"),
    ]:
        sub = cut_table[cut_table["axis"] == axis_name]
        ax.plot(sub["cutoff"], sub["control_mean"], "o-", color="tab:blue",
                label="control")
        ax.plot(sub["cutoff"], sub["treatment_mean"], "^-", color="tab:orange",
                label="treatment")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mean partial AULC")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Partial AULC by retention cutoff (f1 dataset)\n"
                 "mean over the paired users remaining at each cutoff",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "o_aulc_by_cutoff.png",
             "Mean partial AULC (trapezoidal; final F1 held past the last "
             "retrain where needed) among the paired users that remain at each "
             "section-J label cutoff (left) / section-M time cutoff (right). "
             "Within a cutoff both conditions are integrated over the same "
             "range; the widening window also raises the areas across cutoffs.")


def main():
    global ANALYSIS, FIGURES, REPORT, COHORT_LABEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"])
    parser.add_argument("--emit-persona-tables", action="store_true",
                        help="write analysis/persona_behavioral.csv from decisions_labels.csv and exit")
    add_cohort_args(parser)
    args = parser.parse_args()

    cohort = resolve_cohort_from_args(args)
    ANALYSIS = cohort.analysis
    FIGURES = cohort.figures
    REPORT = cohort.report
    COHORT_LABEL = cohort.label
    print(f"Cohort: {COHORT_LABEL} (input: {ANALYSIS}, report: {REPORT})")

    if args.emit_persona_tables:
        FIGURES.mkdir(parents=True, exist_ok=True)
        emit_persona_tables()
        return

    FIGURES.mkdir(parents=True, exist_ok=True)
    dec, sessions, scorable = load_data()

    if args.section in (None, "a"):
        section_a(scorable, sessions)
    if args.section in (None, "b"):
        section_b()
    if args.section in (None, "c"):
        section_c()
    if args.section in (None, "d"):
        section_d(dec, scorable, sessions)
    if args.section in (None, "e"):
        section_e(dec, scorable)
    if args.section in (None, "f"):
        section_f(scorable)
    if args.section in (None, "g"):
        section_g(dec)
    if args.section in (None, "h"):
        section_h(scorable, dec, sessions)
    if args.section in (None, "i"):
        section_i(dec)
    if args.section in (None, "j"):
        section_j()
    if args.section in (None, "k"):
        section_k()
    if args.section in (None, "l"):
        section_l()
    if args.section in (None, "m"):
        section_m()
    if args.section in (None, "n"):
        section_n()
    if args.section in (None, "o"):
        section_o()

    render_report()


if __name__ == "__main__":
    main()
