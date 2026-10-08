"""Analyze treatment effect from analysis/decisions.csv against ground truth.

Sections: a=accuracy, b=complementarity, c=learning curves, d=latency, e=calibration,
f=agreement, g=response rates, h=throughput, i=idk/difficult totals & team distributions.
Run all (default) or one via --section a|b|c|d|e|f|g|h|i.
Prints tables and writes PNG charts to analysis/figures/.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
FIGURES = ANALYSIS / "figures"


def load_data():
    dec = pd.read_csv(ANALYSIS / "decisions.csv")
    sessions = pd.read_csv(ANALYSIS / "sessions.csv")
    excel = pd.read_excel(
        Path(__file__).resolve().parents[1] / "data" / "User Request_last_team_ANON.xlsx"
    )
    truth = dict(zip(excel["Ref"], excel["Team->Name"]))
    dec["ground_truth"] = dec["ref"].map(truth)
    dec = dec.merge(sessions[["session_id", "user"]], on="session_id", how="left")
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


def section_a(scorable, sessions):
    print("\n=== A. Accuracy (headline) ===")
    acc = (
        scorable.groupby(["user", "session_id", "condition"])["human_correct"]
        .mean()
        .reset_index()
    )
    print(acc.to_string(index=False))

    pivot = acc.pivot(index="user", columns="condition", values="human_correct")
    users = pivot.dropna().index
    control = pivot.loc[users, "control"]
    treatment = pivot.loc[users, "treatment"]
    stat = stats.wilcoxon(treatment, control)
    print(f"\nMean accuracy: control={control.mean():.3f}  treatment={treatment.mean():.3f}  "
          f"difference={treatment.mean() - control.mean():+.3f}")
    print(f"Wilcoxon signed-rank (n={len(users)}): p={stat.pvalue:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    for user in users:
        ax.plot([0, 1], [pivot.loc[user, "control"] * 100, pivot.loc[user, "treatment"] * 100],
                marker="o", color="gray", alpha=0.6)
    ax.plot([0, 1], [control.mean() * 100, treatment.mean() * 100], marker="o",
            color="red", linewidth=4, markersize=10, label="mean")
    ax.set_xticks([0, 1], ["Unaided", "AI-assisted"])
    ax.set_ylabel("labeling accuracy (%)")
    ax.set_title("Per-user accuracy: Unaided vs. AI-assisted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "a_accuracy_slope.png", dpi=150)

    return acc


def section_b(scorable):
    print("\n=== B. Complementarity 2x2 ===")
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
    print(table.round(3).to_string(index=False))

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
    fig.tight_layout()
    fig.savefig(FIGURES / "b_complementarity.png", dpi=150)


def section_c():
    print("\n=== C. Model learning curves ===")
    metrics = pd.read_csv(ANALYSIS / "al_metrics.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"control": "tab:blue", "treatment": "tab:orange"}
    steps = {
        cond: group.groupby("num_labeled")["f1_score"].agg(["mean", "sem", "count"])
        for cond, group in metrics.groupby("condition")
    }
    cutoff = min(step.loc[step["count"] >= 5].index.max() for step in steps.values())
    print(f"Pooled mean shown up to num_labeled={cutoff} (>=5 instances in both conditions)")
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
    fig.tight_layout()
    fig.savefig(FIGURES / "c_learning_curves.png", dpi=150)
    for cond, group in metrics.groupby("condition"):
        lo = group.nsmallest(10, "num_labeled")["f1_score"].mean()
        hi = group.nlargest(10, "num_labeled")["f1_score"].mean()
        print(f"{cond}: first-10 mean F1={lo:.3f}  last-10 mean F1={hi:.3f}  "
              f"({group['al_instance_id'].nunique()} instances, {len(group)} iterations)")

    print("\nPer-run first/last 10 retrain iterations (sorted by num_labeled):")
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
        print(f"\n{cond}:")
        print(table.round(3).to_string(index=False))
        first, last = table["first10_f1"], table["last10_f1"]
        h1, h2 = ci95(first), ci95(last)
        if h1 is None:
            print(f"  mean over runs: first10={first.mean():.3f}+/-{first.std():.3f}  "
                  f"last10={last.mean():.3f}+/-{last.std():.3f}")
        else:
            print(f"  mean over runs (95% CI): first10={first.mean():.3f} "
                  f"[{first.mean() - h1:.3f}, {first.mean() + h1:.3f}]  "
                  f"last10={last.mean():.3f} [{last.mean() - h2:.3f}, {last.mean() + h2:.3f}]")


def section_d(dec, scorable, sessions):
    print("\n=== D. Latency & fatigue ===")
    lat = (
        scorable.groupby(["user", "session_id", "condition"])["latency_ms"]
        .median()
        .reset_index()
    )
    lat["latency_s"] = lat["latency_ms"] / 1000
    print(lat.to_string(index=False))

    per_dec = scorable.groupby("condition")["latency_ms"].agg(["median", "mean"]) / 1000
    per_user = (
        lat.groupby(["user", "condition"])["latency_s"].median()
        .groupby("condition").agg(["median", "mean"])
    )
    summary = pd.concat([per_dec, per_user], keys=["per-decision", "per-user"])
    summary.index.names = ["level", "condition"]
    print("\nPer-condition latency summary (s):")
    print(summary.round(3).to_string())

    pivot_lat = lat.pivot(index="user", columns="condition", values="latency_s")
    users = pivot_lat.dropna().index
    stat = stats.wilcoxon(pivot_lat.loc[users, "treatment"], pivot_lat.loc[users, "control"])
    print(f"Wilcoxon on per-user median latency (n={len(users)}): p={stat.pvalue:.4f}")

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

    thirds = pd.cut(scorable["pos_frac"], [0, 1 / 3, 2 / 3, 1],
                    labels=["first", "middle", "last"])
    per_session = (
        scorable.groupby(["session_id", "condition", thirds], observed=True)
        .agg(acc=("human_correct", "mean"), lat_s=("latency_ms", "median"),
             n=("human_correct", "size"))
    )
    per_session["lat_s"] = per_session["lat_s"] / 1000
    fatigue = (
        per_session.reset_index()
        .groupby(["condition", "pos_frac"], observed=True)
        .agg(lat_s=("lat_s", "mean"), acc=("acc", "mean"), n_sessions=("session_id", "nunique"))
        .reset_index()
    )
    fatigue["acc_pct"] = fatigue["acc"] * 100
    print("\nFatigue (thirds of session; latency = mean of per-session medians):")
    print(fatigue.round(3).to_string(index=False))
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
    fig2.tight_layout()
    fig2.savefig(FIGURES / "d_fatigue.png", dpi=150)
    fig.tight_layout()
    fig.savefig(FIGURES / "d_latency_fatigue.png", dpi=150)

    colors = {"control": "tab:blue", "treatment": "tab:orange"}
    lat_by_cond = {c: scorable.loc[scorable.condition == c, "latency_ms"].dropna() / 1000
                   for c in ("control", "treatment")}
    for cond, lat_s in lat_by_cond.items():
        print(f"{cond} latency median={lat_s.median():.1f}s mean={lat_s.mean():.1f}s n={len(lat_s)}")

    fig3, (ax_geo, ax_lin) = plt.subplots(1, 2, figsize=(12, 4.5))
    panels = [(ax_geo, np.geomspace(0.1, 600, 40), True, "log scale"),
              (ax_lin, np.linspace(0.1, 600, 40), False, "linear scale")]
    for ax, bins, is_log, name in panels:
        for cond, lat_s in lat_by_cond.items():
            ax.hist(lat_s, bins=bins, color=colors[cond], alpha=0.6, label=cond)
            ax.axvline(lat_s.median(), color=colors[cond], ls="--", lw=1)
        if is_log:
            ax.set_xscale("log")
        ax.set_xlabel("latency (s)")
        ax.set_ylabel("decisions")
        ax.set_title(f"Ticket latency distribution ({name})\ndashed = median", fontsize=10)
        ax.legend()
    fig3.tight_layout()
    fig3.savefig(FIGURES / "d_latency_distribution.png", dpi=150)


def section_e(dec, scorable):
    print("\n=== E. Calibration (IDK / difficulty) ===")
    dec["idk"] = dec["i_dont_know"].fillna(False).astype(bool)
    idk_rate = dec.groupby(["user", "condition"])["idk"].mean().reset_index()
    print("IDK rate per user:")
    print(idk_rate.round(3).to_string(index=False))

    print("\nP(model wrong | idk) vs P(model wrong | not idk):")
    for cond, group in dec.groupby("condition"):
        idk = group[group["idk"]]
        not_idk = group[~group["idk"]]
        p_idk = 1 - idk["model_correct"].mean() if len(idk) else float("nan")
        p_not = 1 - not_idk["model_correct"].mean()
        print(f"  {cond}: idk n={len(idk)} P(model wrong | idk)={p_idk:.3f}  "
              f"P(model wrong | not idk)={p_not:.3f}")

    dec["difficult"] = dec["is_difficult"].fillna(False).astype(bool)
    dec["tired"] = dec["is_tired"].fillna(False).astype(bool)
    selfrep = dec.groupby("condition").agg(
        n_difficult=("difficult", "sum"),
        n_answered_difficult=("is_difficult", "count"),
        n_tired=("tired", "sum"),
        n_answered_tired=("is_tired", "count"),
    )
    print("\nSelf-reports (with coverage):")
    print(selfrep.to_string(index=False))


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
    print("\n=== F. Agreement with model ===")
    rows = []
    for cond, group in scorable.groupby("condition"):
        row = {"condition": cond, **agreement_table(group)}
        rows.append(row)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    rows = []
    for (user, cond), group in scorable.groupby(["user", "condition"]):
        row = {"user": user, "condition": cond, **agreement_table(group)}
        rows.append(row)
    print("\nPer user:")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def section_g(dec):
    print("\n=== G. Per-condition response rates ===")
    top1 = dec["label"] == dec["model_prediction"]
    top2 = top1 | (dec["label"] == dec["second_model_prediction"])
    idk = dec["i_dont_know"].fillna(False).astype(bool)
    difficult = dec["is_difficult"].fillna(False).astype(bool)
    table = dec.assign(top1=top1, top2=top2, idk=idk, difficult=difficult).groupby("condition").agg(
        n=("top1", "size"), top1=("top1", "mean"), top2=("top2", "mean"),
        idk=("idk", "mean"), difficult=("difficult", "mean"),
    )
    print(table.round(3).to_string())


def section_h(scorable, dec, sessions):
    print("\n=== H. Throughput ===")
    n = scorable.groupby("session_id").size().rename("n")
    ts = dec.groupby("session_id")["ts"].agg(["min", "max"])
    active_h = ((ts["max"] - ts["min"]).dt.total_seconds() / 3600).rename("active_h")
    t = sessions.set_index("session_id").join(n).join(active_h)
    t["thr_span"] = t["n"] / (t["span_min"] / 60)
    t["thr_active"] = t["n"] / t["active_h"]
    print(t[["user", "condition", "n", "thr_span", "thr_active"]].round(2).to_string())

    summary = t.groupby("condition")[["thr_span", "thr_active"]].agg(["mean", "median", "count"])
    print("\nPer-condition summary (tickets/hour):")
    print(summary.round(2).to_string())

    for metric in ("thr_span", "thr_active"):
        pivot = t.pivot(index="user", columns="condition", values=metric)
        users = pivot.dropna().index
        stat = stats.wilcoxon(pivot.loc[users, "treatment"], pivot.loc[users, "control"])
        print(f"Wilcoxon {metric} (n={len(users)}): "
              f"control={pivot['control'].mean():.2f}  treatment={pivot['treatment'].mean():.2f}  "
              f"p={stat.pvalue:.4f}")


def section_i(dec):
    print("\n=== I. IDK & difficult tickets: totals and correct-team distributions ===")
    idk = dec["i_dont_know"].fillna(False).astype(bool)
    difficult = dec["is_difficult"].fillna(False).astype(bool)
    print(f"Total idk tickets: {idk.sum()}  Total difficult tickets: {difficult.sum()}  "
          f"(marked both: {(idk & difficult).sum()})")
    print("Note: idk tickets carry no human label (excluded from accuracy analysis); "
          "distributions are of the correct (ground-truth) team.")

    for name, mask in [("idk", idk), ("difficult", difficult)]:
        sub = dec[mask]
        dist = sub["ground_truth"].value_counts()
        print(f"\n{name} (n={len(sub)}) correct-team distribution (overall):")
        print(pd.DataFrame({"n": dist, "pct": (dist / len(sub) * 100).round(1)}).to_string())

        counts = (sub.groupby(["ground_truth", "condition"]).size()
                  .unstack("condition", fill_value=0)
                  .reindex(columns=["control", "treatment"], fill_value=0))
        pct = counts.div(counts.sum(axis=0), axis=1).mul(100).round(1)
        per_cond = counts.join(pct, lsuffix="_n", rsuffix="_pct").loc[dist.index]
        print(f"\n{name} (n={len(sub)}) per condition:")
        print(per_cond.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=["a", "b", "c", "d", "e", "f", "g", "h", "i"])
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


if __name__ == "__main__":
    main()
