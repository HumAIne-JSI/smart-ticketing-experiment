"""Analyze treatment effect from analysis/decisions.csv against ground truth.

Sections: a=accuracy & macro F1, b=complementarity, c=learning curves, d=latency,
e=calibration, f=agreement, g=response rates, h=throughput, i=idk/difficult totals,
j=balanced analysis cutoffs, k=balanced-cohort learning curves.
Run all (default) or one via --section a|b|c|d|e|f|g|h|i|j|k.
Prints tables, writes PNG charts to analysis/figures/ and an HTML report
(reports/analyze_report.html) with the same content and embedded figures.
"""

import argparse
import base64
import re
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

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
FIGURES = ANALYSIS / "figures"
REPORT = Path(__file__).resolve().parents[1] / "reports" / "analyze_report.html"

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
        f"<p>Generated {datetime.now():%Y-%m-%d %H:%M} from analysis/decisions.csv "
        "(figures also saved as PNG in analysis/figures/).</p>",
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
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(parts), encoding="utf-8")
    print(f"\nHTML report written to {REPORT}")


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

    excel = pd.read_excel(
        Path(__file__).resolve().parents[1] / "data" / "User Request_last_team_ANON.xlsx"
    )
    truth = dict(zip(excel["Ref"], excel["Team->Name"]))
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


def section_b(scorable):
    start_section("B. Complementarity 2x2")
    rows = []
    for cond, group in scorable.groupby("condition"):
        model_wrong = ~group["model_correct"]
        rows.append({
            "condition": cond,
            "n": len(group),
            "catch_rate": group.loc[model_wrong, "human_correct"].mean(),
            "overtrust_rate": 1 - group.loc[~model_wrong, "human_correct"].mean(),
            "model_acc": group["model_correct"].mean(),
            "human_acc": group["human_correct"].mean(),
        })
    table = pd.DataFrame(rows)
    out_table(table)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    x = range(len(table))
    for ax, metric, title in [
        (axes[0], "catch_rate", "P(human correct | model wrong)"),
        (axes[1], "overtrust_rate", "P(human wrong | model correct)"),
    ]:
        ax.bar([c - 0.2 for c in x], table[metric].where(table.condition == "control"),
               width=0.4, label="control")
        ax.bar([c + 0.2 for c in x], table[metric].where(table.condition == "treatment"),
               width=0.4, label="treatment")
        ax.set_xticks(list(x), table["condition"])
        ax.set_title(title)
        ax.legend()
    save_fig(fig, "b_complementarity.png",
             "Complementarity: catch rate (human correct when model wrong) and "
             "overtrust rate (human wrong when model correct).")


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

    thirds = pd.cut(scorable["pos_frac"], [0, 1 / 3, 2 / 3, 1],
                    labels=["first", "middle", "last"])
    per_session = (
        scorable.groupby(["session_id", "condition", thirds], observed=True)
        .agg(acc=("human_correct", "mean"), lat_s=("latency_ms", "median"),
             n=("human_correct", "size"))
    )
    f1_third = (
        scorable.groupby(["session_id", "condition", thirds], observed=True)
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
    out("\nFatigue (thirds of session; latency = mean of per-session medians):")
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
    ax_fat.set_title("Fatigue: accuracy & latency by session third\n(latency: mean of per-session medians)")
    ax_fat.legend(fontsize=8)
    save_fig(fig2, "d_fatigue.png",
             "Accuracy and latency across thirds of a session.")

    fig_f1, ax_f1 = plt.subplots(figsize=(6, 4.5))
    for cond, marker in [("control", "o-"), ("treatment", "^-")]:
        sub = fatigue[fatigue.condition == cond].sort_values("pos_frac")
        ax_f1.plot(x_pos, sub["f1_pct"], marker, label=f"{cond} macro F1 (%)")
        ax_f1.plot(x_pos, sub["lat_s"], marker.replace("-", ""),
                   linestyle=":", label=f"{cond} latency (s)")
    ax_f1.set_xticks(x_pos, ["first", "middle", "last"])
    ax_f1.set_ylabel("macro F1 (%) / median latency (s)")
    ax_f1.set_title("Fatigue: macro F1 & latency by session third\n(latency: mean of per-session medians)")
    ax_f1.legend(fontsize=8)
    save_fig(fig_f1, "d_fatigue_f1.png",
             "Macro F1 and latency across thirds of a session.")

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

    # Cohorts = section-J retention cutoffs that decrease the paired-user
    # count, keeping cohorts with >= 5 paired users (below that the >= 5
    # rule would leave no pooled mean at all).
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"])
    args = parser.parse_args()

    FIGURES.mkdir(exist_ok=True)
    dec, sessions, scorable = load_data()

    if args.section in (None, "a"):
        section_a(scorable, sessions)
    if args.section in (None, "b"):
        section_b(scorable)
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

    render_report()


if __name__ == "__main__":
    main()
