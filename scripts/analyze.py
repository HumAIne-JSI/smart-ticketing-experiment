"""Analyze treatment effect from analysis/decisions.csv against ground truth.

Sections: a=accuracy, b=complementarity, c=learning curves, d=latency, e=calibration.
Run all (default) or one via --section a|b|c|d|e.
Prints tables and writes PNG charts to analysis/figures/.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_sessions import ROOT, parse_ts

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
    cond_of = {4: "control", 5: "treatment", 6: "treatment", 7: "control",
               8: "treatment", 9: "control", 15: "treatment", 16: "control",
               35: "treatment", 36: "control"}
    curves = {c: [] for c in ("control", "treatment")}
    for inst, cond in cond_of.items():
        path = ROOT / str(inst) / "duckdb" / "metrics.json"
        if not path.exists():
            continue
        for m in json.loads(path.read_text(encoding="utf-8")):
            curves[cond].append((m["num_labeled"], m["f1_score"]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond, pts in curves.items():
        df = pd.DataFrame(pts, columns=["n", "f1"]).sort_values("n")
        binned = df.groupby(pd.cut(df["n"], bins=12))["f1"].agg(["mean", "sem"])
        centers = [iv.mid for iv in binned.index]
        ax.plot(centers, binned["mean"], marker="o", label=cond)
        ax.fill_between(centers, binned["mean"] - binned["sem"],
                        binned["mean"] + binned["sem"], alpha=0.2)
    ax.set_xlabel("labels used for training")
    ax.set_ylabel("model macro F1 (test)")
    ax.set_title("Model learning curves (mean ± sem)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "c_learning_curves.png", dpi=150)
    for cond, pts in curves.items():
        df = pd.DataFrame(pts, columns=["n", "f1"])
        print(f"{cond}: first-10 mean F1={df.nsmallest(10, 'n')['f1'].mean():.3f}  "
              f"last-10 mean F1={df.nlargest(10, 'n')['f1'].mean():.3f}  "
              f"({len(df)} iterations)")


def section_d(dec, scorable, sessions):
    print("\n=== D. Latency & fatigue ===")
    lat = (
        scorable.groupby(["user", "session_id", "condition"])["latency_ms"]
        .median()
        .reset_index()
    )
    lat["latency_s"] = lat["latency_ms"] / 1000
    print(lat.to_string(index=False))
    stat = stats.wilcoxon(
        lat[lat.condition == "treatment"].set_index("user")["latency_s"],
        lat[lat.condition == "control"].set_index("user")["latency_s"],
    )
    print(f"Wilcoxon on per-user median latency: p={stat.pvalue:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    data = [scorable.loc[scorable.condition == c, "latency_ms"] / 1000
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
    fatigue = scorable.groupby(["condition", thirds], observed=True).agg(
        acc=("human_correct", "mean"), lat_s=("latency_ms", lambda x: x.mean() / 1000)
    ).reset_index()
    fatigue["acc_pct"] = fatigue["acc"] * 100
    print("\nFatigue (thirds of session):")
    print(fatigue.round(3).to_string(index=False))
    fig2, ax_fat = plt.subplots(figsize=(6, 4.5))
    x_pos = [0, 1, 2]
    for cond, marker in [("control", "o-"), ("treatment", "^-")]:
        sub = fatigue[fatigue.condition == cond].sort_values("pos_frac")
        ax_fat.plot(x_pos, sub["acc_pct"], marker, label=f"{cond} accuracy (%)")
        ax_fat.plot(x_pos, sub["lat_s"], marker.replace("-", ""),
                    linestyle=":", label=f"{cond} latency (s)")
    ax_fat.set_xticks(x_pos, ["first", "middle", "last"])
    ax_fat.set_ylabel("accuracy (%) / latency (s)")
    ax_fat.set_title("Fatigue: accuracy & latency by session third")
    ax_fat.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(FIGURES / "d_fatigue.png", dpi=150)
    fig.tight_layout()
    fig.savefig(FIGURES / "d_latency_fatigue.png", dpi=150)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=["a", "b", "c", "d", "e", "f"])
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


if __name__ == "__main__":
    main()
