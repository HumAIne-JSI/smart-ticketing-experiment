"""Paired analysis of the smart-ticketing experiment on a unified preprocessing.

Unlike scripts/analyze.py (which used a different data view per section), this
script applies ONE preprocessing to every section and to both cohorts:

- the source for decisions is the labels view (decisions_labels.csv /
  sessions_labels.csv: no trim, no block rule, idk rows kept, latencies
  >10 min already nulled by preprocess.py);
- a user is kept only if their control AND treatment session both hold at
  least --min-decisions scorable (non-idk) decisions; users with any shorter
  session are dropped entirely (the within-subject pairing is never broken);
- every kept session is then truncated to its first --min-decisions scorable
  decisions (idk rows at or before the cutoff stay in, but do not count);
- split-by-day sessions (...-d1/-d2) are dropped entirely unless
  --include-split is given;
- the LLM cohort additionally keeps only the user ids that survived the human
  preprocessing (LLM-only ids and ids whose human run was dropped are
  excluded; the intersection is applied on top of the LLM side's own keep
  rule).

Sections: a accuracy & macro F1, b complementarity (catch / overtrust),
c model learning curves cut at the training state after the Nth scorable
decision, d latency & fatigue, o area under the learning curve.
Each section is shown for the human cohort first, then for the LLM cohort.
Charts go to figures/paired/human|llm/; both cohorts share the HTML report
reports/analyze_report_paired.html (chart PNGs are embedded and copied
there as standalone files).
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
from cohort_config import resolve_cohort

REPO = Path(__file__).resolve().parents[1]
FIG_ROOT = REPO / "figures" / "paired"
REPORT = REPO / "reports" / "analyze_report_paired.html"

SEED_LABELS = 70  # labels the initial model is trained on (both cohorts)
CONDITIONS = ("control", "treatment")
CONDITION_COLORS = {"control": "tab:blue", "treatment": "tab:orange"}

_sections: list[tuple[str, list]] = []
_blocks: list = []
_fig_dir: Path = FIG_ROOT / "human"
_args = None


# ---------------------------------------------------------------- report I/O

def start_section(title):
    global _blocks
    _blocks = []
    _sections.append((title, _blocks))
    print(f"\n=== {title} ===")


def start_cohort(data):
    """Open a cohort subsection inside the current section; retarget figures."""
    global _fig_dir
    label = data["label"]
    _fig_dir = FIG_ROOT / label
    _fig_dir.mkdir(parents=True, exist_ok=True)
    if label == "human":
        header = "Human cohort"
    else:
        header = (f"LLM cohort ({len(data['kept_users'])} users - the same "
                  f"user ids that survived the human preprocessing)")
    print(f"\n--- {header} ---")
    _blocks.append(("h3", header))


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
    path = _fig_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _blocks.append(("img", (path, caption)))


def render_report():
    def slug(title):
        return re.sub(r"\W+", "-", title.lower()).strip("-")

    split_note = ("split-by-day sessions kept (--include-split)"
                  if _args.include_split else "split-by-day sessions dropped")
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Smart ticketing experiment - paired analysis report</title>",
        "<style>",
        "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;max-width:1100px;"
        "color:#222;line-height:1.45;padding:0 1rem}",
        "h1{border-bottom:2px solid #333;padding-bottom:.3rem}",
        "h2{margin-top:2.5rem;border-bottom:1px solid #ccc;padding-bottom:.2rem}",
        "h3{margin-top:1.4rem;color:#444;font-size:1.05rem}",
        "nav{margin:1rem 0;padding:.6rem 1rem;background:#f5f5f5;border-radius:4px}",
        "nav a{margin-right:.9rem;white-space:nowrap}",
        "table{border-collapse:collapse;margin:.6rem 0;font-size:.9rem}",
        "th,td{border:1px solid #ccc;padding:3px 8px;text-align:right}",
        "th{background:#f0f0f0}",
        "th:first-child,td:first-child{text-align:left}",
        "img{max-width:100%;border:1px solid #eee;margin:.5rem 0}",
        ".caption{color:#666;font-size:.85rem;margin:.2rem 0}",
        "</style></head><body>",
        "<h1>Smart ticketing experiment - paired analysis report</h1>",
        f"<p>Generated {datetime.now():%Y-%m-%d %H:%M}. Unified preprocessing: "
        f"keep paired users whose BOTH sessions hold &ge;{_args.min_decisions} "
        f"scorable decisions, truncate every kept session to its first "
        f"{_args.min_decisions} scorable decisions; {split_note}; human cohort "
        f"first, LLM cohort (same user ids) second. Charts also saved as PNG "
        f"under {FIG_ROOT.relative_to(REPO)}/&lt;cohort&gt;/.</p>",
        "<nav>",
        "".join(f"<a href='#{slug(t)}'>{escape(t)}</a>" for t, _ in _sections),
        "</nav>",
    ]
    for title, blocks in _sections:
        parts.append(f"<h2 id='{slug(title)}'>{escape(title)}</h2>")
        for kind, payload in blocks:
            if kind == "p":
                parts.append(f"<p>{payload}</p>")
            elif kind == "h3":
                parts.append(f"<h3>{payload}</h3>")
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


# ------------------------------------------------------------ preprocessing

def ground_truth_map():
    excel = pd.read_excel(REPO / "data" / "User Request_last_team_ANON.xlsx")
    return dict(zip(excel["Ref"], excel["Team->Name"]))


def user_sort_key(user):
    digits = re.findall(r"\d+", user)
    return (int(digits[-1]) if digits else 10**6, user)


def order_by_user(df, *secondary):
    """Sort a per-user table numerically (user2 before user13), then by the
    given secondary columns."""
    df = df.copy()
    df["_u"] = df["user"].map(user_sort_key)
    cols = ["_u"] + [c for c in secondary if c in df.columns]
    return (df.sort_values(cols).drop(columns="_u").reset_index(drop=True))


def preprocess(cohort, n_min, include_split, truth, allowed_users=None):
    """The ONE preprocessing every section uses (applied per cohort).

    Returns a dict with the truncated decisions, kept sessions/users, the
    budget-cut AL metrics for sections C/O and the bookkeeping shown in
    section 0. `allowed_users` (LLM side only): user ids that survived the
    human preprocessing; everything else is excluded after the cohort's own
    keep rule.
    """
    tag = cohort.label
    llm_counts_idk = tag == "llm"

    dec = pd.read_csv(cohort.analysis / "decisions_labels.csv")
    ses = pd.read_csv(cohort.analysis / "sessions_labels.csv")
    dec["i_dont_know"] = dec["i_dont_know"].fillna(False).astype(bool)
    ses["paired"] = ses["paired"].fillna(False).astype(bool)
    dec["ts"] = pd.to_datetime(dec["labeled_at"], format="mixed")

    n_sc_full = dec.loc[~dec["i_dont_know"]].groupby("session_id").size()
    ses["n_scorable"] = ses["session_id"].map(n_sc_full).fillna(0).astype(int)

    split_mask = ses["session_id"].str.contains(r"-d\d+$", regex=True)
    split_sids = set(ses.loc[split_mask, "session_id"])
    if not include_split:
        ses = ses.loc[~split_mask].copy()

    dropped: dict[str, str] = {}
    kept: list[str] = []
    status_rows = []
    for user, g in ses.groupby("user"):
        if not include_split and all(sid in split_sids for sid in g["session_id"]):
            status_rows.append({
                "user": user, "sessions": len(g), "conditions": "-",
                "min_scorable": None, "status": "dropped",
                "reason": "split-by-day sessions only (kept only with --include-split)",
            })
            dropped[user] = "split-by-day sessions only (kept only with --include-split)"
            continue
        conds = set(g["condition"])
        mins = int(g["n_scorable"].min())
        if not {"control", "treatment"} <= conds:
            reason = f"unpaired (conditions present: {sorted(conds)})"
        elif mins < n_min:
            reason = f"a session has only {mins} scorable decisions < {n_min}"
        else:
            reason = "kept"
        if allowed_users is not None and reason == "kept" \
                and user not in allowed_users:
            reason = "user id not kept in the human cohort"
        if reason == "kept":
            kept.append(user)
        else:
            dropped[user] = reason
        status_rows.append({
            "user": user, "sessions": len(g),
            "conditions": "/".join(sorted(conds)), "min_scorable": mins,
            "status": "kept" if reason == "kept" else "dropped", "reason": reason,
        })
    kept.sort(key=user_sort_key)
    status = pd.DataFrame(status_rows).sort_values(
        "user", key=lambda s: s.map(user_sort_key)).reset_index(drop=True)

    sessions = ses[ses["user"].isin(kept)].sort_values(
        "user", key=lambda s: s.map(user_sort_key)).reset_index(drop=True)
    keep_sids = set(sessions["session_id"])
    user_of = dict(zip(sessions["session_id"], sessions["user"]))
    dec = dec[dec["session_id"].isin(keep_sids)].copy()
    n_dec_before_trunc = len(dec)

    # Unified truncation: keep every row (idk included) whose ts is at or
    # before the ts of the n_min-th scorable decision of its session.
    kept_parts, trunc_info = [], []
    for sid, g in dec.sort_values("ts").groupby("session_id", sort=True):
        sc_ts = g.loc[~g["i_dont_know"], "ts"]
        if len(sc_ts) > n_min:
            cutoff = sc_ts.iloc[n_min - 1]
            tg = g[g["ts"] <= cutoff]
            trunc_info.append({"session_id": sid, "user": user_of[sid],
                               "condition": g["condition"].iloc[0],
                               "n_scorable_before": len(sc_ts),
                               "rows_dropped": len(g) - len(tg)})
        else:
            tg = g
        kept_parts.append(tg)
    dec = pd.concat(kept_parts, ignore_index=True).sort_values(
        ["session_id", "ts"]).reset_index(drop=True)

    dec["ground_truth"] = dec["ref"].map(truth)
    dec["human_correct"] = dec["label"] == dec["ground_truth"]
    dec["model_correct"] = dec["model_prediction"] == dec["ground_truth"]
    dec["user"] = dec["session_id"].map(user_of)
    dec["latency_min"] = dec["latency_ms"].fillna(0) / 60000
    dec["position"] = dec.groupby("session_id")["ts"].rank(method="first")
    dec["session_size"] = dec.groupby("session_id")["ts"].transform("count")
    dec["pos_frac"] = dec["position"] / dec["session_size"]
    dec["elapsed_min"] = dec.groupby("session_id")["latency_min"].cumsum()
    scorable = dec.loc[~dec["i_dont_know"]].copy()

    # --- AL metrics for sections C/O -------------------------------------
    metrics = pd.read_csv(cohort.analysis / "al_metrics_f1.csv")
    inst_of_sid = (dec.groupby("session_id")["al_instance_id"]
                   .agg(lambda s: s.mode()[0]).astype(int).to_dict())
    sids_of_inst: dict[int, list[str]] = {}
    for sid, inst in inst_of_sid.items():
        sids_of_inst.setdefault(inst, []).append(sid)

    # Warm-start attribution check: one AL instance used by >=2 kept sessions
    # of the same user (split-by-day users kept via --include-split) cannot
    # have its model attributed per condition; the user leaves C/O only.
    warm_users: dict[str, int] = {}
    for inst, sids in sids_of_inst.items():
        users = sorted({user_of[sid] for sid in sids})
        if len(users) > 1:
            for u in users:
                warm_users[u] = inst
        elif len(sids) > 1:
            warm_users[users[0]] = inst
    eligible_insts = sorted(i for i, sids in sids_of_inst.items()
                            if all(user_of[sid] not in warm_users for sid in sids))

    # Budget cut per instance: keep iterations created up to the training
    # state right after the n_min-th scorable decision. num_labeled counts
    # seed 70 + submitted labels (idk too on the LLM side, scorable only in
    # the human loop), hence the different caps.
    n_sc_kept = scorable.groupby("session_id").size()
    n_idk_kept = dec.groupby("session_id")["i_dont_know"].sum()
    cond_of_inst, cap_of_inst, sid_of_inst = {}, {}, {}
    for inst in eligible_insts:
        sid = sids_of_inst[inst][0]
        sid_of_inst[inst] = sid
        cond_of_inst[inst] = sessions.loc[sessions["session_id"] == sid,
                                          "condition"].iloc[0]
        cap_of_inst[inst] = (SEED_LABELS + int(n_sc_kept[sid])
                             + (int(n_idk_kept[sid]) if llm_counts_idk else 0))
    met = metrics[metrics["al_instance_id"].isin(eligible_insts)].copy()
    met["budget_cap"] = met["al_instance_id"].map(cap_of_inst)
    n_met_before_cut = len(met)
    metrics = met[met["num_labeled"] <= met["budget_cap"]].drop(columns="budget_cap")

    user_of_inst = {inst: user_of[sid_of_inst[inst]] for inst in eligible_insts}
    pairs = {}
    for user in kept:
        d = {cond_of_inst[inst]: inst for inst, u in user_of_inst.items() if u == user}
        if set(d) == {"control", "treatment"}:
            pairs[user] = d

    return {
        "label": tag, "analysis": cohort.analysis,
        "n_min": n_min, "include_split": include_split,
        "llm_counts_idk": llm_counts_idk,
        "dec": dec, "sessions": sessions, "scorable": scorable,
        "kept_users": kept, "dropped": dropped, "status": status,
        "n_dec_before_trunc": n_dec_before_trunc,
        "trunc_info": pd.DataFrame(trunc_info),
        "metrics": metrics, "n_met_before_cut": n_met_before_cut,
        "warm_users": warm_users, "pairs": pairs,
        "cond_of_inst": cond_of_inst, "sid_of_inst": sid_of_inst,
    }


# ----------------------------------------------------------------- section 0

def section_zero(datasets):
    start_section("0. Preprocessing (unified)")
    out(f"Keep rule (both cohorts): paired users whose control AND treatment "
        f"session both hold >= {_args.min_decisions} scorable (non-idk) "
        f"decisions; users with any shorter session are dropped entirely.")
    out(f"Truncation: every kept session is cut to its first "
        f"{_args.min_decisions} scorable decisions; idk tickets at or before "
        f"the cutoff are kept but do not count. Every section below uses "
        f"exactly this one preprocessed dataset.")
    out(f"Split-by-day sessions (...-d1/-d2): "
        f"{'kept (--include-split)' if _args.include_split else 'dropped (default)'}.")
    out("LLM cohort: only the user ids that survived the human preprocessing "
        "are analysed (intersection applied after the LLM side's own keep "
        "rule).")
    for data in datasets:
        start_cohort(data)
        out(f"\nSource: {data['analysis'].name}/decisions_labels.csv + "
            f"sessions_labels.csv (+ al_metrics_f1.csv for sections C/O)")
        out_table(data["status"], title="User status (keep rule)", ndigits=None)
        n_ses = len(data["sessions"])
        n_dec = len(data["dec"])
        n_sc = len(data["scorable"])
        idk = int(data["dec"]["i_dont_know"].sum())
        per_cond = (data["scorable"].groupby("condition").size()
                    .reindex(list(CONDITIONS)).astype(int))
        out(f"Kept users: {len(data['kept_users'])}  sessions: {n_ses}  "
            f"decisions: {n_dec} ({idk} idk kept before the cutoff, {n_sc} "
            f"scorable)  scorable per condition: "
            f"{per_cond['control']}/{per_cond['treatment']} "
            f"(control/treatment)")
        if len(data["trunc_info"]):
            ti = data["trunc_info"]
            out(f"Sessions truncated (more than {_args.min_decisions} scorable "
                f"before): {len(ti)} of {n_ses}; "
                f"{int(ti['rows_dropped'].sum())} decisions cut "
                f"({data['n_dec_before_trunc']} -> {n_dec}).")
        else:
            out("No session needed truncation at this cutoff.")
        if data["warm_users"]:
            out(f"Warm-start users excluded from sections C/O only (same AL "
                f"instance in both of their kept sessions): "
                f"{', '.join(sorted(data['warm_users']))}.")
        missing = sorted(data["dec"].loc[
            data["dec"]["ground_truth"].isna() & ~data["dec"]["i_dont_know"],
            "ref"].unique())
        if missing:
            out(f"WARNING refs without ground truth: {missing}")


# ------------------------------------------------------------------ section A

def slope_chart(pivot, ylabel, title, name, caption):
    users = pivot.dropna().index.tolist()
    fig, ax = plt.subplots(figsize=(6, 5))
    for user in users:
        ax.plot([0, 1], [pivot.loc[user, "control"] * 100,
                         pivot.loc[user, "treatment"] * 100],
                marker="o", color="gray", alpha=0.6)
    ax.plot([0, 1], [pivot.loc[users, "control"].mean() * 100,
                     pivot.loc[users, "treatment"].mean() * 100],
            marker="o", color="red", linewidth=4, markersize=10, label="mean")
    ax.set_xticks([0, 1], ["Unaided", "AI-assisted"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    save_fig(fig, name, caption)


def section_a(data):
    start_cohort(data)
    sc = data["scorable"]
    rows = []
    for (user, sid, cond), g in sc.groupby(["user", "session_id", "condition"]):
        rows.append({
            "user": user, "session_id": sid, "condition": cond, "n": len(g),
            "human_acc": g["human_correct"].mean(),
            "human_f1": f1_score(g["ground_truth"], g["label"], average="macro",
                                 zero_division=0),
        })
    scores = pd.DataFrame(rows)
    scores = order_by_user(scores, "session_id")
    out(f"Per user x session on the truncated scorable data "
        f"(the first {data['n_min']} scorable decisions per session):")
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
            f"Per-user {label} on the first {data['n_min']} scorable decisions "
            f"per session, unaided (control) vs AI-assisted (treatment).")


# ------------------------------------------------------------------ section B

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
    """Two-panel bar chart; left panel y-range 0-0.35, right 0-0.55."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    x = range(len(charts_df))
    for ax, metric, title, ylim in [
        (axes[0], "catch_rate", "P(human correct | model wrong)", 0.35),
        (axes[1], "overtrust_rate", "P(human wrong | model correct)", 0.55),
    ]:
        ax.bar([c - 0.2 for c in x],
               charts_df[metric].where(charts_df.condition == "control"),
               width=0.4, label="control")
        ax.bar([c + 0.2 for c in x],
               charts_df[metric].where(charts_df.condition == "treatment"),
               width=0.4, label="treatment")
        ax.set_xticks(list(x), charts_df["condition"])
        ax.set_ylim(0, ylim)
        ax.set_title(title)
        ax.legend()
    save_fig(fig, name, caption)


def section_b(data):
    start_cohort(data)
    sc = data["scorable"]
    per_cond = (sc.groupby("condition").size()
                .reindex(list(CONDITIONS)).astype(int))
    out(f"On the unified truncated data (idk excluded; n={per_cond['control']}"
        f"/{per_cond['treatment']} scorable decisions, control/treatment):")

    rows = []
    for cond, group in sc.groupby("condition"):
        rows.append({"condition": cond, **complementarity_rates(group)})
    table = pd.DataFrame(rows)
    out("\nPer-decision (all truncated scorable decisions pooled):")
    out_table(table, title="Pooled per condition")
    complementarity_bars(
        table, "b_complementarity.png",
        f"Complementarity on the truncated data: catch rate (human correct "
        f"when model wrong) and overtrust rate (human wrong when model "
        f"correct); left panel on the y-scale 0-0.35, right panel 0-0.55.")

    per_user_rows = []
    for (user, cond), g in sc.groupby(["user", "condition"]):
        per_user_rows.append({"user": user, "condition": cond,
                              **complementarity_rates(g)})
    per_user = pd.DataFrame(per_user_rows)
    out("\nPer-user rates (rate within each user x condition first, "
        "then mean over users):")
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
    out_table(summary, title="Mean over users")
    complementarity_bars(
        summary, "b_complementarity_per_user.png",
        "Complementarity per user (mean over users) on the truncated data; "
        "left panel on the y-scale 0-0.35, right panel 0-0.55.")

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


# ------------------------------------------------------------------ section C

def section_c(data):
    start_cohort(data)
    met = data["metrics"]
    n_inst = met.groupby("condition")["al_instance_id"].nunique()
    idk_note = (" (LLM budget counter includes idk labels)"
                if data["llm_counts_idk"] else "")
    out(f"Every kept instance is cut at the training state right after the "
        f"{data['n_min']}th scorable decision (num_labeled <= "
        f"{SEED_LABELS}+{data['n_min']}{idk_note}); instances per condition: "
        f"{n_inst.to_dict()}. The instance set is fixed over the whole "
        f"window, so pooled-mean movement is a learning effect.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond in CONDITIONS:
        color = CONDITION_COLORS[cond]
        group = met[met["condition"] == cond]
        for _, sub in group.groupby("al_instance_id"):
            sub = sub.sort_values("num_labeled")
            ax.plot(sub["num_labeled"], sub["f1_score"], color=color,
                    alpha=0.25, linewidth=1)
        step = group.groupby("num_labeled")["f1_score"].agg(
            ["mean", "sem", "count"])
        keep = step["count"] >= 5
        ax.plot(step.index[keep], step["mean"][keep], marker="o", markersize=3,
                color=color, label=cond)
        ax.fill_between(step.index[keep], (step["mean"] - step["sem"])[keep],
                        (step["mean"] + step["sem"])[keep], color=color, alpha=0.2)
    ax.set_xlabel("labels used for training")
    ax.set_ylabel("model macro F1 (test)")
    ax.set_title(f"Model learning curves (first {data['n_min']} scorable "
                 f"decisions per session;\nmean ± SEM where ≥5 instances)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="gray", alpha=0.4, linewidth=1))
    labels.append("individual instances")
    ax.legend(handles, labels)
    save_fig(fig, "c_learning_curves.png",
             f"Model macro F1 over the shared training window ({SEED_LABELS} "
             f"seed labels + the first {data['n_min']} scorable decisions per "
             f"session); individual instances faint, pooled mean ± SEM where "
             f"≥5 instances.")

    summary = met.groupby("condition").agg(
        n_instances=("al_instance_id", "nunique"),
        f1_initial=("f1_score", "first"), f1_final=("f1_score", "last"),
        budget_min=("num_labeled", "min"), budget_max=("num_labeled", "max"))
    out("\nSummary per condition (initial = num_labeled 70 model, final = "
        "last kept iteration):")
    out_table(summary, index=True)


# ------------------------------------------------------------------ section D

def fatigue_charts(sco, table_note, title_note="", n_min=None):
    """Fatigue aggregation + the two 'Fatigue:' figures (already-truncated data)."""
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
    ax_fat.set_title(f"Fatigue: accuracy & latency by session third\n"
                     f"({title_note}; latency: mean of per-session medians)")
    ax_fat.legend(fontsize=8)
    trunc_note = (f"first {n_min} scorable decisions per session"
                  if n_min is not None else "all scorable decisions")
    save_fig(fig2, "d_fatigue.png",
             f"Accuracy and latency across thirds of the truncated session "
             f"({trunc_note}).")

    fig_f1, ax_f1 = plt.subplots(figsize=(6, 4.5))
    for cond, marker in [("control", "o-"), ("treatment", "^-")]:
        sub = fatigue[fatigue.condition == cond].sort_values("pos_frac")
        ax_f1.plot(x_pos, sub["f1_pct"], marker, label=f"{cond} macro F1 (%)")
        ax_f1.plot(x_pos, sub["lat_s"], marker.replace("-", ""),
                   linestyle=":", label=f"{cond} latency (s)")
    ax_f1.set_xticks(x_pos, ["first", "middle", "last"])
    ax_f1.set_ylabel("macro F1 (%) / median latency (s)")
    ax_f1.set_title(f"Fatigue: macro F1 & latency by session third\n"
                    f"({title_note}; latency: mean of per-session medians)")
    ax_f1.legend(fontsize=8)
    save_fig(fig_f1, "d_fatigue_f1.png",
             f"Macro F1 and latency across thirds of the truncated session "
             f"({trunc_note}).")


def section_d(data):
    start_cohort(data)
    sc = data["scorable"]
    n_min = data["n_min"]

    lat = (sc.groupby(["user", "session_id", "condition"])["latency_ms"]
           .median().reset_index())
    lat["latency_s"] = lat["latency_ms"] / 1000
    lat = order_by_user(lat, "session_id")
    out(f"Per-user median latency on the truncated sessions "
        f"(the first {n_min} scorable decisions per session):")
    out_table(lat, ndigits=None)

    per_dec = sc.groupby("condition")["latency_ms"].agg(["median", "mean"]) / 1000
    per_user = (lat.groupby(["user", "condition"])["latency_s"].median()
                .groupby("condition").agg(["median", "mean"]))
    summary = pd.concat([per_dec, per_user], keys=["per-decision", "per-user"])
    summary.index.names = ["level", "condition"]
    out("\nPer-condition latency summary (s):")
    out_table(summary, index=True)

    pivot_lat = lat.pivot(index="user", columns="condition", values="latency_s")
    users = pivot_lat.dropna().index
    stat = stats.wilcoxon(pivot_lat.loc[users, "treatment"],
                          pivot_lat.loc[users, "control"])
    out(f"Wilcoxon on per-user median latency (n={len(users)}): "
        f"p={stat.pvalue:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    data_s = [sc.loc[sc.condition == c, "latency_ms"].dropna() / 1000
              for c in CONDITIONS]
    axes[0].boxplot(data_s, tick_labels=list(CONDITIONS))
    axes[0].set_ylabel("latency (s)")
    axes[0].set_title("Per-decision latency")

    merged = data["sessions"].merge(
        sc.groupby("session_id")["human_correct"].mean().rename("acc"),
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
             f"Left: per-decision latency by condition (truncated sessions). "
             f"Right: session accuracy vs median latency.")

    fatigue_charts(
        sc,
        f"thirds of the truncated session: first {n_min} scorable decisions "
        f"each (idk before the cutoff included)",
        title_note=f"first {n_min} scorable decisions per session",
        n_min=n_min)

    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    bins = np.geomspace(0.1, 600, 40)
    for cond, color in CONDITION_COLORS.items():
        lat_s = sc.loc[sc.condition == cond, "latency_ms"].dropna() / 1000
        ax3.hist(lat_s, bins=bins, color=color, alpha=0.6, label=cond)
        ax3.axvline(lat_s.median(), color=color, ls="--", lw=1)
        out(f"{cond} latency median={lat_s.median():.1f}s "
            f"mean={lat_s.mean():.1f}s n={len(lat_s)}")
    ax3.set_xscale("log")
    ax3.set_xlabel("latency (s)")
    ax3.set_ylabel("decisions")
    ax3.set_title("Ticket latency distribution\n(log scale, dashed = median)")
    ax3.legend()
    save_fig(fig3, "d_latency_distribution.png",
             "Per-decision latency distribution on the truncated sessions "
             "(log scale; dashed = median).")


# ------------------------------------------------------------------ section O

def section_o(data):
    start_cohort(data)
    dec, met, pairs = data["dec"], data["metrics"], data["pairs"]
    skipped = [u for u in data["kept_users"] if u not in pairs]
    if skipped:
        out(f"Users skipped (no model attributable to one of their "
            f"conditions): {', '.join(skipped)}")
    out(f"AULC over both x-axes: labels used for training (budget "
        f"{SEED_LABELS}..{SEED_LABELS + data['n_min']}, the same window for "
        f"both conditions of a user) and elapsed active labeling time "
        f"(0..min of the two session totals, final F1 held past the last "
        f"retrain). Wilcoxon on the per-user paired areas.")

    sid_of_inst = data["sid_of_inst"]
    inst_ids = [inst for d in pairs.values() for inst in d.values()]
    sids_o = sorted({sid_of_inst[i] for i in inst_ids})
    met_o = met[met["al_instance_id"].isin(inst_ids)].copy()
    met_o["created_at"] = pd.to_datetime(met_o["created_at"], format="mixed")
    met_o["session_id"] = met_o["al_instance_id"].map(sid_of_inst)
    dec_o = dec[dec["session_id"].isin(sids_o)][["session_id", "ts", "elapsed_min"]]
    merged = pd.merge_asof(
        met_o.sort_values("created_at"),
        dec_o.sort_values("ts"),
        left_on="created_at", right_on="ts", by="session_id",
        direction="backward",
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

    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    def trapz_to(x, y, limit, hold=False):
        """Trapezoid over [x[0], limit]; the endpoint is interpolated at the
        limit, or the last F1 is held when the curve ends before it."""
        if limit <= x[0]:
            return 0.0
        inside = x <= limit
        xs, ys = x[inside], y[inside]
        if xs[-1] < limit:
            xs = np.append(xs, limit)
            ys = np.append(ys, y[-1] if hold else np.interp(limit, x, y))
        return float(trapz(ys, xs))

    total_min = dec.groupby("session_id")["latency_min"].sum().reindex(sids_o)
    rows, held_users = [], []
    for user, d in pairs.items():
        sid_c, sid_t = sid_of_inst[d["control"]], sid_of_inst[d["treatment"]]
        cut_t = float(min(total_min[sid_c], total_min[sid_t]))
        held = False
        row = {"user": user, "cutoff_labels": data["n_min"],
               "cutoff_time_min": round(cut_t, 1)}
        for cond in CONDITIONS:
            xl, yl = label_curve(d[cond])
            xt, yt = time_curve(d[cond])
            row[f"aulc_labels_{cond}"] = trapz_to(xl, yl,
                                                  SEED_LABELS + data["n_min"])
            row[f"aulc_time_{cond}"] = trapz_to(xt, yt, cut_t, hold=True)
            if xt[-1] < cut_t - 1e-9:
                held = True
        if held:
            held_users.append(user)
        rows.append(row)
    aulc = pd.DataFrame(rows)
    aulc = order_by_user(aulc)
    out("\nPer-user AULC (both conditions integrated over the same window):")
    out_table(aulc, ndigits=2)

    out("\nWilcoxon signed-rank on the paired per-user AULCs:")
    for crit, unit in (("labels", "F1 x labels"), ("time", "F1 x min")):
        c = aulc[f"aulc_{crit}_control"]
        t = aulc[f"aulc_{crit}_treatment"]
        stat = stats.wilcoxon(t, c)
        out(f"  {crit} ({unit}): control={c.mean():.2f}  "
            f"treatment={t.mean():.2f}  "
            f"difference={t.mean() - c.mean():+.2f}  (n={len(aulc)})  "
            f"p={stat.pvalue:.4f}")

    if held_users:
        out(f"Note: for {len(held_users)} of {len(pairs)} users at least one "
            f"session's last retrain precedes the time cutoff ({', '.join(held_users)}; "
            f"trailing idk decisions trigger no retrain); the final model's "
            f"F1 is held over the remaining minutes.")

    def aulc_slope(pivot, ylabel, title, name, caption):
        users = pivot.dropna().index.tolist()
        fig, ax = plt.subplots(figsize=(6, 5))
        for user in users:
            ax.plot([0, 1], [pivot.loc[user, "control"],
                             pivot.loc[user, "treatment"]],
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
               f"Area under the model F1 learning curve over training budgets "
               f"{SEED_LABELS} to {SEED_LABELS + data['n_min']} (truncated "
               f"session), per user.")
    piv_t = aulc.set_index("user").rename(columns={
        "aulc_time_control": "control", "aulc_time_treatment": "treatment"})
    aulc_slope(piv_t[["control", "treatment"]],
               "AULC (F1 × active labeling minutes)",
               "Per-user AULC over active time: Unaided vs. AI-assisted",
               "o_aulc_time_slope.png",
               "Area under the model F1 learning curve over the shorter "
               "session's active labeling time (truncated sessions), per user.")


# ---------------------------------------------------------------------- main

SECTIONS = [
    ("a", "A. Accuracy & macro F1", section_a),
    ("b", "B. Complementarity (catch / overtrust)", section_b),
    ("c", "C. Model learning curves (cut at the Nth scorable decision)", section_c),
    ("d", "D. Latency & fatigue (truncated sessions)", section_d),
    ("o", "O. Area under the learning curve (AULC)", section_o),
]


def main():
    global _args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--min-decisions", type=int, default=40,
        help="paired users are kept when BOTH of their sessions hold >= N "
             "scorable (non-idk) decisions; every kept session is truncated "
             "to its first N scorable decisions (default 40)",
    )
    parser.add_argument(
        "--include-split", action="store_true",
        help="also include split-by-day sessions (...-d1/-d2, dropped by "
             "default); their users stay out of sections C/O (warm-start)",
    )
    parser.add_argument(
        "--section", choices=[k for k, _, _ in SECTIONS],
        help="run a single section instead of all",
    )
    _args = parser.parse_args()
    if _args.min_decisions < 1:
        parser.error("--min-decisions must be >= 1")

    truth = ground_truth_map()
    human = preprocess(resolve_cohort(False), _args.min_decisions,
                       _args.include_split, truth)
    llm = preprocess(resolve_cohort(True), _args.min_decisions,
                     _args.include_split, truth,
                     allowed_users=set(human["kept_users"]))
    datasets = [human, llm]

    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Unified paired analysis (min_decisions={_args.min_decisions}, "
          f"include_split={_args.include_split})")

    section_zero(datasets)
    for key, title, fn in SECTIONS:
        if _args.section is not None and key != _args.section:
            continue
        start_section(title)
        for data in datasets:
            fn(data)

    render_report()


if __name__ == "__main__":
    main()
