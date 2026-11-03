"""Compare LLM personas with name-matched human users on condition deltas.

For every user name that exists and is paired (both conditions) in BOTH the
human (analysis/) and the LLM persona (analysis_llm/) cohort f1 view:

- per (user x condition x cohort):
    * labeling accuracy and macro F1 over ALL scorable (non-idk) decisions
      (f1 view: full label sequences, first day-part only, no truncation),
    * AULC of the model macro-F1 learning curve over the labels axis:
      trapezoidal integral of al_metrics_f1's F1 vs num_labeled from budget
      70 to 70 + N, where N = per-user minimum scorable count across the
      user's two sessions (section O rule); computed separately per cohort.
- per (user x cohort): delta = treatment - control for all three metrics.
- paired by user name (real user1 vs LLM persona user1):
    * alignment rate = share of pairs where sign(human delta) ==
      sign(LLM delta) (strict: a zero delta counts as aligned only against
      another zero),
    * Pearson correlation between the human and LLM deltas,
    * MAE of the human vs LLM deltas (magnitude agreement),
    * 3-panel scatter: x = human delta, y = LLM persona delta.

Two batches are reported per run:

- "full" (sections 1-2): everything above.
- "first-<window>" (sections 3-4, --window, default 40): per user x condition
  only the first N scorable labels in chronological order are kept and users
  with fewer than N scorable labels in either condition are dropped; AULC is
  integrated over the fixed labels window 70..70+N for BOTH cohorts, so human
  and LLM deltas share identical windows (the LLM cohort is naturally capped
  at 50 scorable by preprocessing).
- "matched-min" (sections 5-6): for every name-matched pair, N = the minimum
  scorable-label count over that user's FOUR sessions (human control/
  treatment AND LLM control/treatment, e.g. human c-70/t-60 and llm c-50/
  t-50 -> N = 50); each of the four views keeps its first N labels and AULC
  integrates over 70..70+N within the pair. Windows are identical across
  cohorts within a pair (all 17 pairs kept), but they differ across pairs.

Inputs:  analysis*/decisions_f1.csv, sessions_f1.csv, al_metrics_f1.csv.
Outputs: reports/persona_comparison_report.html,
         analysis/persona_comparison/persona_metrics.csv  (+label_window col),
         analysis/persona_comparison/persona_deltas.csv   (+label_window col),
         analysis/persona_comparison/figures/persona_delta_scatter.png,
         analysis/persona_comparison/figures/persona_delta_scatter_first<N>.png,
         analysis/persona_comparison/figures/persona_delta_scatter_matched_min.png.

Run both cohorts' preprocessing first (python scripts/preprocess.py and
python scripts/preprocess.py --llm), then: python scripts/compare_personas.py
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
import analyze
from analyze import ground_truth_map, out, out_table, save_fig, start_section

REPO = Path(__file__).resolve().parents[1]
COHORTS = {"human": REPO / "analysis", "llm": REPO / "analysis_llm"}
OUT_DIR = REPO / "analysis" / "persona_comparison"
FIGURE_DIR = OUT_DIR / "figures"
REPORT = REPO / "reports" / "persona_comparison_report.html"

# Cold-start seed labels: the model's first recorded test F1 sits at this
# training budget before any session label is added (section O's axis start).
START_BUDGET = 70

METRICS = [
    ("acc", "Accuracy", "proportion (pct. pts in plot)"),
    ("f1", "Macro F1", "proportion (pct. pts in plot)"),
    ("aulc", "AULC (F1 × labels)", "F1 × labels"),
]
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def require_inputs():
    needed = [d / n for d in COHORTS.values()
              for n in ("decisions_f1.csv", "sessions_f1.csv", "al_metrics_f1.csv")]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        print("Missing inputs:\n  " + "\n  ".join(missing), file=sys.stderr)
        print("Run both cohorts' preprocessing first:\n"
              "  python scripts/preprocess.py\n"
              "  python scripts/preprocess.py --llm", file=sys.stderr)
        raise SystemExit(1)


def _utf8_console():
    """Windows consoles default to cp1252, which chokes on some chars used here."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass


def user_sort_key(u):
    return (0, int(u[4:])) if u.startswith("user") and u[4:].isdigit() else (1, u)


def trapz_to(x, y, limit):
    """Trapezoid of the curve over [x[0], limit]; the endpoint is interpolated
    at the limit (labels axis: retrains stop at the last scorable ticket)."""
    if limit <= x[0]:
        return 0.0
    inside = x <= limit
    xs, ys = x[inside], y[inside]
    if xs[-1] < limit:
        xs = np.append(xs, limit)
        ys = np.append(ys, np.interp(limit, x, y))
    return float(trapz(ys, xs))


def load_cohort(analysis_dir, label, announce=True):
    """Scorable (non-idk) f1-view decisions of paired sessions, with user and
    correctness mapped. Returns (dec, sessions).`announce=False` suppresses
    notes that only depend on the raw data (already printed in section 1)."""
    dec = pd.read_csv(analysis_dir / "decisions_f1.csv")
    ses = pd.read_csv(analysis_dir / "sessions_f1.csv")
    paired = ses["paired"].fillna(False).astype(bool)
    unpaired = sorted(ses.loc[~paired, "user"].unique())
    if unpaired and announce:
        out(f"{label}: unpaired f1-view users excluded: {', '.join(unpaired)}")
    ses = ses[paired]
    user_of = dict(zip(ses["session_id"], ses["user"]))

    dec = dec[dec["session_id"].isin(user_of)].copy()
    dec["user"] = dec["session_id"].map(user_of)
    dec["ground_truth"] = dec["ref"].map(ground_truth_map())
    missing = int(dec["ground_truth"].isna().sum())
    if missing:
        out(f"WARNING {label}: {missing} decisions without ground truth; dropped")
        dec = dec[dec["ground_truth"].notna()]
    dec["correct"] = dec["label"] == dec["ground_truth"]
    dec = dec[~dec["i_dont_know"].fillna(False).astype(bool)]

    multi = ses.groupby(["user", "condition"])["session_id"].count()
    multi = multi[multi > 1]
    if len(multi) and announce:
        out(f"NOTE {label}: multiple sessions per user×condition exist; "
            f"they are pooled for accuracy/F1: {multi.to_dict()}")
    return dec, ses


def truncate_window(dec, window=None, cut_by_user=None, label=""):
    """First N scorable labels per (user, condition) in chronological order,
    dropped users included in the note. Two modes:
    - window=int: uniform N; a user with fewer than N scorable labels in
      either condition is dropped entirely (both conditions) - section 3/4.
    - cut_by_user=Series (user -> N): per-user N from the name-matched
      minimum; users absent from the map (or with a missing condition) are
      dropped - sections 5/6."""
    n_all = dec.groupby(["user", "condition"]).size().unstack().reindex(
        columns=["control", "treatment"])
    if cut_by_user is not None:
        limit_of = cut_by_user.reindex(n_all.index).astype(float)
        bad = limit_of.isna() | n_all.isna().any(axis=1) | \
            (n_all.sub(limit_of, axis="index") < 0).any(axis=1)
        header = (f"{label}: dropped users not name-matched in both cohorts")
    else:
        limit_of = pd.Series(float(window), index=n_all.index)
        bad = (n_all.fillna(0) < window).any(axis=1)
        header = (f"{label}: dropped users with fewer than {window} scorable "
                  f"labels in a condition")
    if bad.any():
        parts = []
        for u, r in n_all[bad].iterrows():
            parts.append(f"{u} (control={int(r['control']) if pd.notna(r['control']) else 0}"
                         f", treatment={int(r['treatment']) if pd.notna(r['treatment']) else 0})")
        out(f"{header}: {', '.join(parts)}")
    keep = set(n_all.index[~bad])
    dec = dec[dec["user"].isin(keep)].copy()
    dec["ts"] = pd.to_datetime(dec["labeled_at"], format="mixed")
    dec = dec.sort_values(["user", "condition", "ts"])
    kept = [g.head(int(limit_of[u]))
            for (u, _), g in dec.groupby(["user", "condition"], sort=False)]
    if not kept:
        return dec.iloc[0:0]
    return pd.concat(kept)


def aulc_frame(dec, ses, analysis_dir, label, window=None, cut_by_user=None,
               announce=True):
    """Per-user AULC over the labels axis, for each condition.

    - window=None & cut_by_user=None (section O rule): both conditions of a
      user are integrated over 70..70+cut, cut = the user's minimum
      per-session scorable count.
    - window=N (fixed-window batch): only users present in `dec` (the
      truncated view) are integrated, over 70..70+N, i.e. the same window
      for both cohorts.
    - cut_by_user=Series user -> N (matched-min batch): each user integrates
      over 70..70+N with that pair's own N."""

    def note(msg):
        if announce:
            out(msg)

    al = pd.read_csv(analysis_dir / "al_metrics_f1.csv")
    curves = {}
    for inst in al["al_instance_id"].unique():
        g = al[al["al_instance_id"] == inst].sort_values("num_labeled")
        curves[inst] = (g["num_labeled"].to_numpy(float), g["f1_score"].to_numpy(float))

    if cut_by_user is not None:
        cuts = cut_by_user
        keep_users = set(dec["user"].unique())
    elif window is None:
        n_sid = dec.groupby("session_id").size().rename("n_scorable")
        cuts = ses.merge(n_sid, on="session_id", how="left")
        cuts = cuts.groupby("user")["n_scorable"].min()
        keep_users = set(cuts.index)
    else:
        cuts = None
        keep_users = set(dec["user"].unique())

    inst_map = ses.assign(instance=ses["instances"].astype(str).str.split(","))
    inst_map = inst_map.explode("instance")
    inst_map = inst_map[inst_map["instance"].str.strip() != ""]
    inst_map["instance"] = inst_map["instance"].astype(int)

    rows = []
    for user, grp in inst_map.groupby("user"):
        if user not in keep_users:
            continue
        pair = dict(zip(grp["condition"], grp["instance"]))
        if set(pair) != {"control", "treatment"}:
            note(f"WARNING {label} user {user}: conditions {sorted(pair)}; "
                 f"skipped for AULC")
            continue
        if cut_by_user is not None:
            cut = int(cut_by_user[user])
        else:
            cut = int(cuts.get(user, np.nan)) if window is None else int(window)
        if pd.isna(cut):
            note(f"WARNING {label} user {user}: no scorable decisions; skipped for AULC")
            continue
        limit = START_BUDGET + cut
        row = {"user": user, "cut_n": cut}
        for cond in ("control", "treatment"):
            x, y = curves.get(pair[cond], (np.array([]), np.array([])))
            if len(x) == 0:
                note(f"WARNING {label} user {user} {cond}: AL instance "
                     f"{pair[cond]} has no metric rows; AULC set to NaN")
                row[f"aulc_{cond}"] = np.nan
            else:
                row[f"aulc_{cond}"] = trapz_to(x, y, limit)
        rows.append(row)
    return pd.DataFrame(rows).set_index("user")


def cohort_metric_frame(dec, ses, analysis_dir, label, window=None,
                        cut_by_user=None, announce=True):
    """Wide per-user frame: {n, acc, f1, aulc}_{control, treatment} + cut_n."""
    rows = []
    for (user, cond), g in dec.groupby(["user", "condition"]):
        rows.append({
            "user": user, "condition": cond, "n": len(g),
            "acc": g["correct"].mean(),
            "f1": f1_score(g["ground_truth"], g["label"], average="macro",
                           zero_division=0),
        })
    wide = pd.DataFrame(rows).set_index(["user", "condition"]).unstack("condition")
    # a condition with zero scorable decisions would surface here as missing
    # (metric, condition) columns; fill them in so the delta pairing warns.
    want = pd.MultiIndex.from_product([("n", "acc", "f1"),
                                       ("control", "treatment")])
    for mi in want.difference(wide.columns):
        if announce:
            out(f"WARNING {label}: no scorable decisions for {mi[1]} ({mi[0]}); "
                f"filled with NaN")
        wide[mi] = np.nan
    wide.columns = [f"{m}_{c}" for m, c in wide.columns]
    wide = wide.join(aulc_frame(dec, ses, analysis_dir, label, window=window,
                                cut_by_user=cut_by_user, announce=announce),
                     how="left")
    wide = wide.sort_index()
    if announce:
        both = wide["acc_control"].notna() & wide["acc_treatment"].notna()
        # one NaN accuracy would break the delta pairing later; surface it once
        if not both[wide.index].all():
            out(f"WARNING {label}: users without a full acc/f1 metric pair: "
                f"{list(wide.index[~both])}")
    return wide


def matched_cut_map(raw):
    """Per name-matched user's label budget N: the minimum scorable-label
    count over that user's FOUR sessions (human control/treatment AND LLM
    control/treatment). Users not present with all four views are dropped
    from the map. Prints the count table; returns (n_map, count_table)."""
    counts = {name: dec.groupby(["user", "condition"]).size().unstack().reindex(
                  columns=["control", "treatment"])
              for name, (dec, ses) in raw.items()}
    # a pair is matchable only when BOTH cohorts hold BOTH conditions for it
    complete = counts["human"].index.intersection(counts["llm"].index)
    four = pd.concat([counts["human"].loc[complete],
                      counts["llm"].loc[complete]])
    per_cond = four.groupby(level="user").min()
    ok = per_cond.notna().all(axis=1) & (per_cond >= 1).all(axis=1)
    complete_list = list(per_cond.index[ok])
    n_map = per_cond.min(axis=1)[complete_list].astype(int)
    n_map = n_map.reindex(sorted(n_map.index, key=user_sort_key))
    tbl = per_cond.loc[complete_list].astype(int).reindex(n_map.index)
    tbl["N_min"] = n_map
    out_table(tbl,
              title="Matched label budget per pair: scorable counts per "
                    "user (human ctrl/treat, LLM ctrl/treat) and "
                    "N = minimum over all four sessions")
    return n_map, tbl


def metric_tables(frames):
    """Per metric: table of human/LLM control/treatment values and deltas,
    paired on the user name (inner join), NaN pairs dropped."""
    tables = {}
    for m, mlabel, _ in METRICS:
        h, l = frames["human"], frames["llm"]
        idx = h.index.intersection(l.index)
        cols = {}
        for cname, w in (("human", h), ("llm", l)):
            for cond in ("control", "treatment"):
                cols[f"{cname}_{cond}"] = w.loc[idx, f"{m}_{cond}"]
        t = pd.DataFrame(cols, index=idx)
        missing = t[["human_control", "human_treatment",
                     "llm_control", "llm_treatment"]].isna().any(axis=1)
        if missing.any():
            out(f"WARNING {mlabel}: dropped name-matched pairs "
                f"(metric missing in one cohort): {list(t.index[missing])}")
            t = t[~missing]
        t["human_delta"] = t["human_treatment"] - t["human_control"]
        t["llm_delta"] = t["llm_treatment"] - t["llm_control"]
        t["aligned"] = np.sign(t["human_delta"]) == np.sign(t["llm_delta"])
        tables[m] = t
    return tables


def summary_table(tables):
    """Alignment rate, Pearson correlation and MAE per metric."""
    rows = []
    for m, mlabel, unit in METRICS:
        t = tables[m]
        dh, dl = t["human_delta"], t["llm_delta"]
        n = len(t)
        aligned = int(t["aligned"].sum())
        rate = aligned / n if n else np.nan
        if n < 3 or dh.std(ddof=1) == 0 or dl.std(ddof=1) == 0:
            r = p = np.nan
        else:
            r, p = stats.pearsonr(dh, dl)
        mae = float((dh - dl).abs().mean()) if n else np.nan
        rows.append({
            "metric": mlabel,
            "delta_unit": unit,
            "n_pairs": n,
            "aligned": f"{aligned}/{n}",
            "alignment_rate": rate,
            "pearson_r": r,
            "pearson_p": p,
            "mae": mae,
        })
    return pd.DataFrame(rows)


def delta_scatter(tables, fig_name, caption):
    """3-panel scatter of paired deltas: x = human delta, y = LLM delta."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for ax, (m, mlabel, unit) in zip(axes, METRICS):
        t = tables[m]
        factor = 100.0 if m in ("acc", "f1") else 1.0
        x, y = t["human_delta"] * factor, t["llm_delta"] * factor
        finite = np.concatenate([x.dropna().to_numpy(), y.dropna().to_numpy()])
        lim = max(np.max(np.abs(finite)) * 1.18, 1e-9) if len(finite) else 1.0
        ax.plot([-lim, lim], [-lim, lim], ls="--", color="gray", lw=1, zorder=1)
        ax.axvline(0, color="k", lw=0.8, alpha=0.5, zorder=1)
        ax.axhline(0, color="k", lw=0.8, alpha=0.5, zorder=1)
        colors = ["tab:green" if a else "tab:red" for a in t["aligned"]]
        ax.scatter(x, y, c=colors, s=32, zorder=3)
        # cycle four offset directions so neighbouring labels do not collide
        offsets = [(5, 5, "left"), (-5, 5, "right"), (5, -12, "left"),
                   (-5, -12, "right")]
        for i, u in enumerate(t.index):
            dx, dy, ha = offsets[i % len(offsets)]
            ax.annotate(u, (x[u], y[u]), fontsize=6, xytext=(dx, dy),
                        textcoords="offset points", ha=ha, color="0.35")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(mlabel)
        shown_unit = "pct. points" if m in ("acc", "f1") else unit
        ax.set_xlabel(f"Δ human ({shown_unit})")
        if ax is axes[0]:
            ax.set_ylabel(f"Δ LLM persona ({shown_unit})")
    handles = [
        Line2D([], [], color="tab:green", marker="o", ls="",
               label="aligned (same sign)"),
        Line2D([], [], color="tab:red", marker="o", ls="",
               label="opposite sign"),
        Line2D([], [], color="gray", ls="--", lw=1, label="y = x"),
    ]
    axes[0].legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, fig_name, caption)


def metrics_csv_rows(frames, label_window):
    """Long rows (one per user x cohort x condition) for persona_metrics.csv."""
    rows = []
    for label, w in frames.items():
        for user, r in w.iterrows():
            for cond in ("control", "treatment"):
                rows.append({
                    "label_window": label_window,
                    "user": user, "cohort": label, "condition": cond,
                    "n_scorable": int(r[f"n_{cond}"])
                    if pd.notna(r[f"n_{cond}"]) else "",
                    "acc": r[f"acc_{cond}"], "f1": r[f"f1_{cond}"],
                    "aulc": r[f"aulc_{cond}"],
                    "aulc_limit_n": int(r["cut_n"])
                    if pd.notna(r["cut_n"]) else "",
                })
    return rows


def deltas_csv_rows(tables, label_window):
    """Long rows (one per user x metric) for persona_deltas.csv."""
    rows = []
    for m, mlabel, _ in METRICS:
        for user, r in tables[m].iterrows():
            rows.append({
                "label_window": label_window,
                "user": user, "metric": mlabel,
                "human_delta": r["human_delta"], "llm_delta": r["llm_delta"],
                "aligned": bool(r["aligned"]),
            })
    return rows


def run_batch(window, titles, intro, scatter_name, scatter_caption,
              announce=True, matched=False):
    """One analysis pass: per-user metrics, name-matched deltas, stats tables
    and the delta scatter. Returns (frames, tables, summary).

    matched=True: per-pair label budgets N = min over the user's four sessions
    (human control/treatment, LLM control/treatment) - sections 5/6."""
    start_section(titles[0])
    out(intro)
    if matched:
        wdesc = "each pair's name-matched minimum label budget"
        raw = {label: load_cohort(adir, label, announce=announce)
               for label, adir in COHORTS.items()}
        n_map, _ = matched_cut_map(raw)
    elif window is not None:
        wdesc = f"the first {window} scorable labels per condition"
    else:
        wdesc = "all scorable decisions"
    frames = {}
    for label, adir in COHORTS.items():
        dec, ses = raw[label] if matched else load_cohort(adir, label,
                                                         announce=announce)
        if matched:
            dec = truncate_window(dec, cut_by_user=n_map, label=label)
        elif window is not None:
            dec = truncate_window(dec, window=window, label=label)
        frames[label] = cohort_metric_frame(
            dec, ses, adir, label, window=window,
            cut_by_user=n_map if matched else None, announce=announce)
        wide = frames[label]
        n_dec = int(np.nansum(wide[["n_control", "n_treatment"]].to_numpy()))
        out(f"\n{label}: {wide.index.nunique()} paired users, "
            f"{n_dec} scorable decisions across both conditions ({wdesc}).")
        out_table(wide.round(3), title=f"{label} per-user metrics ({wdesc})",
                  index=True)

    common = frames["human"].index.intersection(frames["llm"].index)
    if len(common) == 0:
        out("\nNo name-matched pairs across cohorts; nothing to compare.")
        return frames, None, None
    out(f"\nName-matched pairs available in both cohorts: {len(common)} "
        f"({', '.join(sorted(common, key=user_sort_key))}).")

    tables = metric_tables(frames)
    summary = summary_table(tables)

    start_section(titles[1])
    out("Delta = treatment - control within the same cohort (real user1 "
        "compared to LLM persona user1). Alignment uses strict sign equality: "
        "positive treatment effect in both cohorts (or negative in both) "
        "counts as aligned; a zero delta aligns only against another zero. "
        "Pearson r/p correlate the human and LLM deltas across users; MAE is "
        "the mean absolute human-vs-LLM delta difference (deltas for "
        "acc/F1 in proportions, AULC in F1 × labels).")
    for i, (m, mlabel, unit) in enumerate(METRICS):
        t = tables[m]
        row = summary.iloc[i]
        out_table(t.round(3), title=f"{mlabel} — deltas paired by user name",
                  index=True)
        dh, dl = t["human_delta"], t["llm_delta"]
        out(f"\n{mlabel} ({unit}): mean delta human={dh.mean():+.4f}  "
            f"mean delta LLM={dl.mean():+.4f}")
        r_s = f"{row['pearson_r']:+.3f}" if pd.notna(row["pearson_r"]) else "n/a"
        p_s = f"p={row['pearson_p']:.4f}" if pd.notna(row["pearson_p"]) else ""
        out(f"  aligned {row['aligned']} ({100 * row['alignment_rate']:.1f}%)  "
            f"Pearson r={r_s} {p_s}  MAE={row['mae']:.4f}")
    out_table(summary, title="Summary (strict-sign alignment, Pearson "
                             "correlation of deltas, MAE of deltas)")

    delta_scatter(tables, scatter_name, scatter_caption)
    return frames, tables, summary


def render_report():
    def slug(title):
        return re.sub(r"\W+", "-", title.lower()).strip("-")

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Smart ticketing — human vs LLM persona deltas</title>",
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
        "<h1>Human vs. LLM persona — condition-delta comparison</h1>",
        f"<p>Generated {datetime.now():%Y-%m-%d %H:%M} from analysis/ (human) and "
        f"analysis_llm/ (LLM persona) f1-view artifacts; delta = treatment - control "
        f"within each cohort, pairs matched on the user name. Figures also saved "
        f"as PNG in analysis/persona_comparison/figures/.</p>",
        "<nav>",
        "".join(f"<a href='#{slug(t)}'>{escape(t)}</a>" for t, _ in analyze._sections),
        "</nav>",
    ]
    for title, blocks in analyze._sections:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=40,
                        help="first-N scorable-label window used by the second "
                        "batch of sections (0 disables that batch)")
    args = parser.parse_args()

    _utf8_console()
    require_inputs()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    analyze.FIGURES = FIGURE_DIR  # reuse analyze.save_fig's output location

    all_metrics_rows, all_delta_rows = [], []
    scatter_caption = (
        "Per name-matched pair: delta (treatment - control) of the LLM "
        "persona vs the corresponding real user. Green = aligned (same "
        "sign), red = opposite sign; dashed y = x marks perfect delta "
        "agreement. Accuracy/F1 deltas in percentage points, AULC in "
        "F1 × labels.")

    # ---- batch 1: full data (original analysis) ----
    frames, tables, summary = run_batch(
        window=None,
        titles=("1. Per-user metrics by condition (human vs LLM personas)",
                "2. Deltas paired by name: alignment, Pearson correlation, MAE"),
        intro="Data basis: f1 view (decisions_f1.csv — full label sequences, first "
              "day-part only), paired sessions only. Accuracy and macro F1 use ALL "
              "scorable (non-idk) decisions per user × condition (idk excluded, no "
              "truncation). AULC integrates the model F1 learning curve (labels "
              f"axis) from budget {START_BUDGET} to {START_BUDGET}+N, N = per-user "
              "minimum scorable count across the user's two sessions (section O "
              "rule), computed separately for each cohort. n = scorable decisions "
              "per cell; cut_n = the AULC integration limit's N.",
        scatter_name="persona_delta_scatter.png",
        scatter_caption=scatter_caption)
    if tables is not None:
        all_metrics_rows += metrics_csv_rows(frames, "full")
        all_delta_rows += deltas_csv_rows(tables, "full")

    # ---- batch 2: first-N scorable-label windows ----
    win = args.window
    if win is not None and win > 0:
        win = int(win)
        frames_w, tables_w, summary_w = run_batch(
            window=win,
            titles=(
                f"3. Per-user metrics on the first {win} scorable labels "
                f"(human vs LLM personas)",
                f"4. Deltas on first-{win}-label windows: alignment, Pearson "
                f"correlation, MAE"),
            intro=(f"Data basis: f1 view (decisions_f1.csv), paired sessions only; "
                   f"per user × condition only the FIRST {win} scorable (non-idk) "
                   "labels in chronological order are kept (preprocessing-style "
                   f"max_scorable={win}), and users with fewer than {win} scorable "
                   "labels in either condition are dropped. AULC integrates the "
                   f"model F1 learning curve over the fixed labels window "
                   f"{START_BUDGET}..{START_BUDGET + win} for BOTH cohorts "
                   "(num_labeled counts only scorable labels), so human and LLM "
                   "deltas share identical windows for all three metrics. This "
                   "batch removes the unequal-support asymmetry of sections 1-2 "
                   "(LLM sessions are naturally capped at 50 scorable)."),
            scatter_name=f"persona_delta_scatter_first{win}.png",
            scatter_caption=(
                "Same construction as the full-data scatter, but each user × "
                f"condition is truncated to the first {win} scorable labels "
                f"(AULC window {START_BUDGET}..{START_BUDGET + win} for both "
                "cohorts), removing the unequal-support asymmetry of the "
                "full-data scatter."),
            announce=False)
        if tables_w is not None:
            all_metrics_rows += metrics_csv_rows(frames_w, f"first-{win}")
            all_delta_rows += deltas_csv_rows(tables_w, f"first-{win}")

    # ---- batch 3: per-pair name-matched minimum label budget ----
    frames_m, tables_m, summary_m = run_batch(
        window=None, matched=True,
        titles=(
            "5. Per-user metrics matched to each persona's minimum label "
            "budget (human vs LLM personas)",
            "6. Deltas on name-matched minimum windows: alignment, Pearson "
            "correlation, MAE"),
        intro=(f"Data basis: f1 view (decisions_f1.csv), paired sessions only. "
               "The label budget is MATCHED per persona: for every name-matched "
               "pair N = the minimum scorable-label count over that user's four "
               "sessions (human control/treatment AND LLM control/treatment; "
               "e.g. human c-70/t-60 and llm c-50/t-50 -> N = 50). Each of the "
               "four views keeps its first N scorable labels in chronological "
               "order, and AULC integrates the model F1 learning curve over "
               f"{START_BUDGET}..{START_BUDGET}+N within the pair, so every "
               "metric is window-identical across the cohorts within a pair. "
               "All matched pairs are kept (user15/user20 stay excluded: "
               "unpaired in the human f1 view); budgets still differ ACROSS "
               "pairs (unlike the fixed first-40 windows of sections 3-4), so "
               "cross-user comparisons mix deltas estimated with different "
               "precision."),
        scatter_name="persona_delta_scatter_matched_min.png",
        scatter_caption=(
            "Deltas (treatment − control) per name-matched pair where BOTH "
            "cohorts are truncated to that pair's minimum label budget N over "
            f"all four sessions (AULC window {START_BUDGET}..{START_BUDGET}+N). "
            "Exact window matching within the pair; note N varies across pairs "
            "per persona, so cross-user comparisons have heterogeneous "
            "precision."),
        announce=False)
    if tables_m is not None:
        all_metrics_rows += metrics_csv_rows(frames_m, "matched-min")
        all_delta_rows += deltas_csv_rows(tables_m, "matched-min")

    metrics_path = OUT_DIR / "persona_metrics.csv"
    pd.DataFrame(all_metrics_rows).to_csv(metrics_path, index=False)
    deltas_path = OUT_DIR / "persona_deltas.csv"
    pd.DataFrame(all_delta_rows).to_csv(deltas_path, index=False)
    out(f"\nWrote {metrics_path.relative_to(REPO)}")
    out(f"Wrote {deltas_path.relative_to(REPO)}")

    render_report()


if __name__ == "__main__":
    main()
