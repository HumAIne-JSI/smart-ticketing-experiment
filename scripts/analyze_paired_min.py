"""Paired min-truncation analysis of the human cohort (final paired view).

Ingests the preprocessed labels view (analysis/decisions_labels.csv +
analysis/sessions_labels.csv) and applies ONE preprocessing:

- a user is kept only if they have exactly one control and one treatment
  session and BOTH sessions hold more than --min-scorable scorable (non-idk)
  decisions; split-by-day day-parts (...-d1/-d2) count as one session each
  (each holds exactly one condition; there is no learning-curve / AULC
  section here, so no warm-start exclusion is needed);
- each kept session is then truncated to its user's N = min(scorable count of
  the two sessions) by an index walk in labeled_at order: rows are kept until
  the Nth scorable decision is counted; idk rows interleaved before that
  point stay in the window (they belong to the idk/difficult denominators),
  rows after it are dropped;
- latency metrics use only scorable decisions with a latency < 10 min (the
  >10-min latencies are already nulled by preprocess.py); those decisions
  still count in every other metric's denominator;
- everything is computed per user x condition first; means, differences and
  Wilcoxon signed-rank tests are over users (decisions are never pooled
  across users).

Metric notes:
- idk rate = idk count / ALL decisions in the window (blanks in the source
  column mean a normal labeled decision, so fillna(False) is lossless);
- difficult rate = marked-difficult count / ALL decisions in the window,
  with unanswered (blank) treated as not difficult - users who never used
  the flag are kept with rate 0 (coverage reported alongside).

Sections: 0 cohort & truncation, a accuracy & macro F1, b correction rates
(model correction / false correction, tables only), c latency, d idk &
difficult rates, e thirds of the truncated session (accuracy/F1 & latency).
Run all (default) or one via --section 0|a|b|c|d|e.
Figures go to figures/paired_min/, the report to
reports/analyze_report_paired_min.html.
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

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "analysis"
FIGURES = REPO / "figures" / "paired_min"
REPORT = REPO / "reports" / "analyze_report_paired_min.html"

LATENCY_CAP_MS = 10 * 60 * 1000  # 10 minutes, same cap preprocess.py applies
CONDITIONS = ("control", "treatment")
CONDITION_COLORS = {"control": "tab:blue", "treatment": "tab:orange"}

_sections: list[tuple[str, list]] = []
_blocks: list = []
_args = None


# ---------------------------------------------------------------- report I/O

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
        "<title>Smart ticketing experiment - paired min-truncation analysis</title>",
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
        "<h1>Smart ticketing experiment - paired min-truncation analysis (human cohort)</h1>",
        f"<p>Generated {datetime.now():%Y-%m-%d %H:%M} from {ANALYSIS.name}/decisions_labels.csv. "
        f"Unified preprocessing: keep users with one control and one treatment session, both holding "
        f"more than {_args.min_scorable} scorable (non-idk) decisions; truncate every kept session to its "
        f"user's N = min(scorable counts of the two sessions) by an index walk in labeled_at order (idk "
        f"rows interleaved before the Nth scorable decision stay in the window and count toward the "
        f"idk/difficult denominators); latency metrics use only scorable decisions with latency &lt; 10 min. "
        f"Every metric is computed per user x condition first; means, differences and Wilcoxon signed-rank "
        f"tests are over users. Charts also saved as PNG in {FIGURES.relative_to(REPO)}/.</p>",
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


# ------------------------------------------------------------------- helpers

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


def paired_test(per_user, metric, label):
    """Paired per-user statistics for one metric: means per condition, mean
    paired difference and Wilcoxon signed-rank over users. Users with a NaN
    on either side are excluded (noted); when scipy cannot compute (e.g. all
    paired differences zero) the p-value cell says so. Returns a row dict
    for the section's statistics table."""
    pivot = per_user.pivot(index="user", columns="condition", values=metric)
    users = pivot.dropna().index
    c = pivot.loc[users, "control"]
    t = pivot.loc[users, "treatment"]
    n_nonzero = int(((t - c) != 0).sum())
    note = (f"{len(pivot) - len(users)} users excluded: value undefined in "
            f"one condition" if len(pivot) > len(users) else "")
    try:
        stat = stats.wilcoxon(t, c)
        if np.isnan(stat.pvalue):
            raise ValueError("p-value is nan")
        p = f"{stat.pvalue:.4f}"
    except (ValueError, TypeError):
        p = "not computable (all paired differences zero)"
    return {
        "metric": label,
        "mean control": c.mean(),
        "mean treatment": t.mean(),
        "mean difference (treatment - control)": t.mean() - c.mean(),
        "n users": len(users),
        "non-zero paired diffs": n_nonzero,
        "Wilcoxon p": p,
        "note": note,
    }


def paired_table(rows, title):
    """Emit one section's paired-statistics table (rows from paired_test)."""
    out("")
    out_table(pd.DataFrame(rows), title=title)


def slope_chart(pivot, ylabel, title, name, caption, percent=True):
    users = pivot.dropna().index.tolist()
    fig, ax = plt.subplots(figsize=(6, 5))
    scale = 100 if percent else 1
    for user in users:
        ax.plot([0, 1], [pivot.loc[user, "control"] * scale,
                         pivot.loc[user, "treatment"] * scale],
                marker="o", color="gray", alpha=0.6)
    ax.plot([0, 1], [pivot.loc[users, "control"].mean() * scale,
                     pivot.loc[users, "treatment"].mean() * scale],
            marker="o", color="red", linewidth=4, markersize=10, label="mean")
    ax.set_xticks([0, 1], ["Unaided", "AI-assisted"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    save_fig(fig, name, caption)


# ------------------------------------------------------------- preprocessing

def prepare(min_scorable, truth):
    """The ONE preprocessing every section uses.

    Returns a dict with the truncated decisions, kept sessions/users, the
    per-user truncation targets N and the bookkeeping shown in section 0.
    """
    dec = pd.read_csv(ANALYSIS / "decisions_labels.csv")
    ses = pd.read_csv(ANALYSIS / "sessions_labels.csv")

    # Keep the raw answered-ness of the optional difficult flag for coverage
    # reporting BEFORE filling blanks with False.
    dec["difficult_answered"] = dec["is_difficult"].notna()
    dec["i_dont_know"] = dec["i_dont_know"].fillna(False).astype(bool)
    dec["is_difficult"] = dec["is_difficult"].fillna(False).astype(bool)
    dec["ts"] = pd.to_datetime(dec["labeled_at"], format="mixed")

    dec["ground_truth"] = dec["ref"].map(truth)
    dec["human_correct"] = dec["label"] == dec["ground_truth"]
    dec["model_correct"] = dec["model_prediction"] == dec["ground_truth"]
    dec["latency_s"] = dec["latency_ms"] / 1000
    dec["latency_ok"] = dec["latency_ms"].notna() & (dec["latency_ms"] < LATENCY_CAP_MS)
    dec["latency_s_ok"] = np.where(dec["latency_ok"], dec["latency_s"], np.nan)

    n_sc_full = dec.loc[~dec["i_dont_know"]].groupby("session_id").size()
    ses["n_scorable"] = ses["session_id"].map(n_sc_full).fillna(0).astype(int)

    # --- keep rule ---------------------------------------------------------
    kept, dropped, status_rows = [], {}, []
    for user, g in ses.groupby("user"):
        conds = sorted(g["condition"])
        mins = int(g["n_scorable"].min())
        if conds != list(CONDITIONS):
            reason = (f"does not have exactly one control and one treatment "
                      f"session (conditions present: {conds})")
        elif mins <= min_scorable:
            reason = (f"a session holds only {mins} scorable decisions "
                      f"(<= {min_scorable})")
        else:
            reason = "kept"
        if reason == "kept":
            kept.append(user)
        else:
            dropped[user] = reason
        status_rows.append({
            "user": user,
            "sessions": len(g),
            "conditions": "/".join(conds),
            "min_scorable": mins,
            "N_used": mins if reason == "kept" else None,
            "status": "kept" if reason == "kept" else "dropped",
            "reason": reason if reason != "kept" else "",
        })
    kept.sort(key=user_sort_key)
    status = pd.DataFrame(status_rows).sort_values(
        "user", key=lambda s: s.map(user_sort_key)).reset_index(drop=True)

    sessions = ses[ses["user"].isin(kept)].sort_values(
        "user", key=lambda s: s.map(user_sort_key)).reset_index(drop=True)
    keep_sids = set(sessions["session_id"])
    user_of = dict(zip(ses["session_id"], ses["user"]))

    # Per-user truncation target: min scorable count across the two sessions.
    n_of_user = (sessions.groupby("user")["n_scorable"].min()
                 .to_dict())

    # --- truncation (index walk, no timestamp logic) -----------------------
    dec = dec[dec["session_id"].isin(keep_sids)].copy()
    n_dec_before_trunc = len(dec)
    dec = dec.sort_values(["session_id", "ts"], kind="stable").reset_index(drop=True)
    sc_int = (~dec["i_dont_know"]).astype(int)
    # count of scorable decisions strictly BEFORE this row within its session
    cum_before = sc_int.groupby(dec["session_id"]).cumsum() - sc_int
    n_cut = dec["session_id"].map({sid: n_of_user[user_of[sid]] for sid in keep_sids})
    dec = dec.loc[cum_before < n_cut].copy()

    n_idk_kept = int(dec["i_dont_know"].sum())
    dec["user"] = dec["session_id"].map(user_of)
    dec["position"] = dec.groupby("session_id")["ts"].rank(method="first")
    dec["session_size"] = dec.groupby("session_id")["ts"].transform("count")
    dec["pos_frac"] = dec["position"] / dec["session_size"]
    scorable = dec.loc[~dec["i_dont_know"]].copy()

    missing = sorted(scorable.loc[scorable["ground_truth"].isna(), "ref"].unique())
    if missing:
        print(f"WARNING scorable refs without ground truth: {missing}")

    return {
        "min_scorable": min_scorable,
        "dec": dec, "scorable": scorable, "sessions": sessions,
        "kept": kept, "dropped": dropped, "status": status,
        "n_of_user": n_of_user, "user_of": user_of,
        "n_dec_before_trunc": n_dec_before_trunc,
        "n_idk_kept": n_idk_kept,
    }


# ----------------------------------------------------------------- section 0

def section_zero(data):
    start_section("0. Cohort & truncation")
    out(f"Keep rule: users with exactly one control and one treatment session, "
        f"both holding more than {data['min_scorable']} scorable (non-idk) "
        f"decisions; every other user is dropped entirely (the within-subject "
        f"pairing is never broken). Split-by-day day-parts (...-d1/-d2) count "
        f"as one session each; each holds exactly one condition.")
    out(f"Truncation: every kept session is walked in labeled_at order and cut "
        f"at its user's N = min(scorable count of the two sessions); idk rows "
        f"interleaved before the Nth scorable decision stay in the window and "
        f"count toward the idk/difficult denominators; rows after it are dropped.")
    out(f"Latency: only scorable decisions with a latency < 10 min enter the "
        f"latency metrics (the >10-min latencies are already nulled by "
        f"preprocess.py); those decisions still count in every other metric.")
    out("Aggregation: every metric is computed per user x condition first; "
        "means, differences and Wilcoxon signed-rank tests are over users.")
    out_table(data["status"], title="User status (keep rule)", ndigits=None)

    kept_list = ", ".join(f"{u} (N={data['n_of_user'][u]})"
                          for u in data["kept"])
    out(f"\nKept users ({len(data['kept'])}): {kept_list}")
    if data["dropped"]:
        dropped_list = "; ".join(f"{u} ({r})" for u, r in data["dropped"].items())
        out(f"Dropped users ({len(data['dropped'])}): {dropped_list}")

    n_ses = len(data["sessions"])
    dec, sc = data["dec"], data["scorable"]
    per_cond = (sc.groupby("condition").size().reindex(list(CONDITIONS)).astype(int))
    out(f"\nKept sessions: {n_ses}  window decisions: {len(dec)} "
        f"({data['n_dec_before_trunc']} before truncation; {data['n_idk_kept']} "
        f"idk kept inside the windows)  scorable: {len(sc)} "
        f"(control/treatment: {per_cond['control']}/{per_cond['treatment']})")
    n_used = pd.Series(data["n_of_user"])
    out(f"Truncation targets N over users: min={n_used.min()}  "
        f"median={int(n_used.median())}  max={n_used.max()}")


# ------------------------------------------------------------------ metrics

def user_condition_metrics(g):
    """All metrics for one user x condition group of window rows."""
    sc = g.loc[~g["i_dont_know"]]
    model_wrong = ~sc["model_correct"]
    model_right = ~model_wrong
    lat = sc.loc[sc["latency_ok"], "latency_s"]
    return {
        "n_window": len(g),
        "n_scorable": len(sc),
        "n_idk": int(g["i_dont_know"].sum()),
        "idk_rate": g["i_dont_know"].mean(),
        "n_difficult": int(g["is_difficult"].sum()),
        "n_difficult_answered": int(g["difficult_answered"].sum()),
        "difficult_rate": g["is_difficult"].mean(),
        "accuracy": sc["human_correct"].mean(),
        "macro_f1": f1_score(sc["ground_truth"], sc["label"], average="macro",
                             zero_division=0),
        "n_model_wrong": int(model_wrong.sum()),
        "n_model_correct": int(model_right.sum()),
        "model_correction_rate": (sc.loc[model_wrong, "human_correct"].mean()
                                  if model_wrong.any() else np.nan),
        "false_correction_rate": (1 - sc.loc[model_right, "human_correct"].mean()
                                  if model_right.any() else np.nan),
        "n_lat_ok": int(sc["latency_ok"].sum()),
        "median_latency_s": (lat.median() if lat.notna().any() else np.nan),
    }


def compute_per_user(dec):
    rows = []
    for (user, cond), g in dec.groupby(["user", "condition"]):
        rows.append({"user": user, "condition": cond,
                     **user_condition_metrics(g)})
    return order_by_user(pd.DataFrame(rows))


def compute_per_session(dec):
    rows = []
    for (user, sid, cond), g in dec.groupby(["user", "session_id", "condition"]):
        rows.append({"user": user, "session_id": sid, "condition": cond,
                     **user_condition_metrics(g)})
    return order_by_user(pd.DataFrame(rows), "session_id")


# ------------------------------------------------------------------ section A

def section_a(data):
    per_user, per_session = data["per_user"], data["per_session"]
    show_cols = ["user", "session_id", "condition", "n_scorable",
                 "accuracy", "macro_f1"]
    out(f"Per user x session on the truncated windows (first N scorable "
        f"decisions per session, N = min over the two sessions, "
        f"N > {data['min_scorable']}):")
    out_table(per_session[show_cols], ndigits=None)

    paired_table(
        [paired_test(per_user, "accuracy", "accuracy"),
         paired_test(per_user, "macro_f1", "macro F1")],
        "Paired per-user statistics (Wilcoxon signed-rank): accuracy & macro F1")

    for metric, ylabel, name in [
        ("accuracy", "Labeling accuracy (%)", "a_accuracy_slope.png"),
        ("macro_f1", "Labeling macro F1", "a_f1_slope.png"),
    ]:
        pivot = per_user.pivot(index="user", columns="condition", values=metric)
        label = "accuracy" if metric == "accuracy" else "macro F1"
        slope_chart(
            pivot, ylabel,
            f"Per-User {label.title()}: Unaided vs. AI-Assisted",
            name,
            f"Per-user {label} on the truncated windows (first N scorable "
            f"decisions per session, N = min over the two sessions), unaided "
            f"(control) vs AI-assisted (treatment).")


# ------------------------------------------------------------------ section B

def section_b(data):
    per_user = data["per_user"]
    cols = ["user", "condition", "n_scorable", "n_model_wrong",
            "n_model_correct", "model_correction_rate", "false_correction_rate"]
    out("Model correction rate = P(human correct | model wrong); false "
        "correction rate = P(human wrong | model correct). Computed per user "
        "x condition on the truncated windows; undefined (NaN) when the "
        "conditioning set is empty for that user x condition.")
    out_table(order_by_user(per_user[cols]), title="Per user x condition",
              ndigits=None)

    summary = (
        per_user.groupby("condition")
        .agg(n_users=("user", "size"),
             mean_model_wrong=("n_model_wrong", "mean"),
             mean_model_correct=("n_model_correct", "mean"),
             model_correction_rate=("model_correction_rate", "mean"),
             false_correction_rate=("false_correction_rate", "mean"))
        .reindex(list(CONDITIONS))
        .reset_index()
    )
    out("\nMean over users (users with an undefined rate in a condition are "
        "excluded from that rate's mean only):")
    out_table(summary, title="Mean over users")

    paired_table(
        [paired_test(per_user, "model_correction_rate", "model correction rate"),
         paired_test(per_user, "false_correction_rate", "false correction rate")],
        "Paired per-user statistics (Wilcoxon signed-rank): correction rates")


# ------------------------------------------------------------------ section C

def section_c(data):
    per_user, per_session = data["per_user"], data["per_session"]
    cols = ["user", "session_id", "condition", "n_scorable", "n_lat_ok",
            "median_latency_s"]
    out("Per-user median decision latency on the truncated windows, over "
        "scorable decisions with a latency < 10 min only:")
    out_table(per_session[cols], title="Per user x session", ndigits=None)

    summary = (
        per_session.groupby("condition")
        .agg(n_sessions=("session_id", "size"),
             mean_of_medians=("median_latency_s", "mean"),
             median_of_medians=("median_latency_s", "median"))
        .reindex(list(CONDITIONS))
        .reset_index()
    )
    out("\nSummary per condition (mean/median of the per-session medians):")
    out_table(summary)

    pivot = per_user.pivot(index="user", columns="condition",
                           values="median_latency_s")
    paired_table(
        [paired_test(per_user, "median_latency_s", "median latency (s)")],
        "Paired per-user statistics (Wilcoxon signed-rank): median latency")
    slope_chart(
        pivot, "median latency (s)",
        "Per-User Median Latency: Unaided vs. AI-Assisted",
        "c_latency_slope.png",
        "Per-user median decision latency (s) on the truncated windows, "
        "unaided (control) vs AI-assisted (treatment); latencies >= 10 min "
        "excluded.", percent=False)


# ------------------------------------------------------------------ section D

def section_d(data):
    per_user, per_session = data["per_user"], data["per_session"]
    cols = ["user", "condition", "n_window", "n_idk", "idk_rate",
            "n_difficult", "n_difficult_answered", "difficult_rate"]
    out("IDK rate = idk count / ALL decisions in the window; difficult rate = "
        "marked-difficult count / ALL decisions in the window, with "
        "unanswered (blank) treated as not difficult - users who never used "
        "the flag are kept with rate 0 (n_difficult_answered shows the "
        "coverage). All rates are per user x condition on the truncated "
        "windows.")
    out_table(order_by_user(per_user[cols]), title="Per user x condition",
              ndigits=None)

    ses_cols = ["session_id", "user", "condition", "n_window", "n_idk",
                "idk_rate", "n_difficult", "n_difficult_answered",
                "difficult_rate"]
    out("\nPer session:")
    out_table(per_session[ses_cols], title="Per session", ndigits=None)

    summary = (
        per_user.groupby("condition")
        .agg(n_users=("user", "size"),
             mean_idk_rate=("idk_rate", "mean"),
             mean_difficult_rate=("difficult_rate", "mean"),
             mean_difficult_answered=("n_difficult_answered", "mean"))
        .reindex(list(CONDITIONS))
        .reset_index()
    )
    out("\nMean over users:")
    out_table(summary, title="Mean over users")

    paired_table(
        [paired_test(per_user, "idk_rate", "IDK rate"),
         paired_test(per_user, "difficult_rate", "difficult rate")],
        "Paired per-user statistics (Wilcoxon signed-rank): IDK & difficult rates")


# ------------------------------------------------------------------ section E

def thirds_charts(sc):
    """Fatigue-style table + the two thirds figures on the truncated windows."""
    thirds = pd.cut(sc["pos_frac"], [0, 1 / 3, 2 / 3, 1],
                    labels=["first", "middle", "last"])
    per_session = (
        sc.groupby(["session_id", "condition", thirds], observed=True)
        .agg(acc=("human_correct", "mean"),
             lat_s=("latency_s_ok", "median"),
             n=("human_correct", "size"))
    )
    f1_third = (
        sc.groupby(["session_id", "condition", thirds], observed=True)
        .apply(lambda g: f1_score(g["ground_truth"], g["label"], average="macro",
                                  zero_division=0))
        .rename("f1")
    )
    fatigue = (
        per_session.join(f1_third).reset_index()
        .groupby(["condition", "pos_frac"], observed=True)
        .agg(lat_s=("lat_s", "mean"), acc=("acc", "mean"), f1=("f1", "mean"),
             n_sessions=("session_id", "nunique"))
        .reset_index()
    )
    fatigue["acc_pct"] = fatigue["acc"] * 100
    fatigue["f1_pct"] = fatigue["f1"] * 100
    out("Per-session thirds (by position within the truncated window), then "
        "averaged over sessions per condition; latency = mean of the "
        "per-session-third medians (< 10 min only):")
    out_table(fatigue)

    x_pos = [0, 1, 2]
    cond_label = {"control": "Unaided", "treatment": "AI-assisted"}
    for y_metric, y_pct, ylabel, title, name, caption in [
        ("acc", "acc_pct", "Accuracy (%) / median latency (s)",
         "Accuracy & Latency Over Session Thirds",
         "e_thirds_acc_latency.png",
         "Accuracy and latency across thirds of the truncated session "
         "(first N scorable decisions per session, N = min over the two "
         "sessions); latency: mean of per-session-third medians (< 10 min only)."),
        ("f1", "f1_pct", "Macro F1 / median latency (s)",
         "Macro F1 & Latency Over Session Thirds",
         "e_thirds_f1_latency.png",
         "Macro F1 and latency across thirds of the truncated session "
         "(first N scorable decisions per session, N = min over the two "
         "sessions); latency: mean of per-session-third medians (< 10 min only)."),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        metric_name = "accuracy (%)" if y_metric == "acc" else "macro F1"
        for cond in CONDITIONS:
            sub = fatigue[fatigue.condition == cond].sort_values("pos_frac")
            color = CONDITION_COLORS[cond]
            ax.plot(x_pos, sub[y_pct], "o-", color=color,
                    label=f"{cond_label[cond]} {metric_name}")
            ax.plot(x_pos, sub["lat_s"], "o:", color=color,
                    label=f"{cond_label[cond]} latency (s)")
        ax.set_xticks(x_pos, ["First", "Middle", "Last"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        save_fig(fig, name, caption)


def section_e(data):
    thirds_charts(data["scorable"])


# ---------------------------------------------------------------------- main

SECTIONS = [
    ("a", "A. Accuracy & macro F1", section_a),
    ("b", "B. Correction rates (model correction / false correction)", section_b),
    ("c", "C. Latency", section_c),
    ("d", "D. IDK & difficult rates", section_d),
    ("e", "E. Thirds of the truncated session", section_e),
]


def main():
    global _args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--min-scorable", type=int, default=20,
        help="users are kept when BOTH of their sessions hold strictly more "
             "than N scorable (non-idk) decisions (default 20)",
    )
    parser.add_argument(
        "--section", choices=["0"] + [k for k, _, _ in SECTIONS],
        help="run a single section instead of all",
    )
    _args = parser.parse_args()
    if _args.min_scorable < 1:
        parser.error("--min-scorable must be >= 1")

    truth = ground_truth_map()
    data = prepare(_args.min_scorable, truth)
    data["per_user"] = compute_per_user(data["dec"])
    data["per_session"] = compute_per_session(data["dec"])

    FIGURES.mkdir(parents=True, exist_ok=True)
    print(f"Paired min-truncation analysis of the human cohort "
        f"(min_scorable={_args.min_scorable}; a user is kept when BOTH "
        f"sessions hold strictly more than {_args.min_scorable} scorable decisions)")
    print(f"Kept users ({len(data['kept'])}): "
          + ", ".join(f"{u} (N={data['n_of_user'][u]})" for u in data["kept"]))
    for user, reason in data["dropped"].items():
        print(f"Dropped {user}: {reason}")

    section_zero(data)
    for key, title, fn in SECTIONS:
        if _args.section is not None and _args.section != key:
            continue
        start_section(title)
        fn(data)

    render_report()


if __name__ == "__main__":
    main()
