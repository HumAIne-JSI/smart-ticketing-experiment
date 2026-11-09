"""Consistency analysis of the user3 replicate pilot (database/consistency_experiment).

Eight replicate runs (user3con1..8) of persona user3, executed by the LLM
agent (deepseek/deepseek-v4.1-flash), each running the same two-session
protocol copied from user3's experiment_design.yaml cells: control +
Experiment 1 -> treatment + Experiment 2. This script quantifies how
consistent that LLM persona behaves across its repeated runs:

- per run: labeling accuracy and macro F1 of the actual labels over all
  scorable (non-idk) decisions vs the Excel ground truth, plus the AULC of
  the model macro-F1 learning curve (labels axis) integrated from the
  cold-start budget 70 to 70 + N, N = per-run minimum scorable count across
  the run's two sessions (section O rule);
- plots 1-2: the original per-condition distributions of all three metrics
  (paired across conditions) and the bootstrap distribution of their means
  with percentile 95% CIs (alpha = 5%, --ci to change the level);
- plot 3: the persona-comparison scatter restricted to user3 — the real
  (human) user3's delta (treatment - control, from the original main-cohort
  sessions tabulated in analysis/persona_comparison/persona_deltas.csv,
  default first-40 window) on x, and each LLM persona-user3 replicate run
  (user3conN) with its own paired delta on y; the y = x line marks perfect
  agreement with the original user3, and the shaded horizontal band spans
  the replicate runs' min-max delta range.

Inputs:  database/consistency_experiment/users_config.yaml (active runs),
         database/consistency_experiment/instances/<id>/duckdb/*.json,
         data/User Request_last_team_ANON.xlsx (ground truth),
         analysis/persona_comparison/persona_deltas.csv (user3's
         original-session delta).
Outputs: analysis/consistency/consistency_session_metrics.csv,
         analysis/consistency/consistency_bootstrap_summary.csv,
         analysis/consistency/consistency_user3_deltas.csv,
         analysis/consistency/consistency_replicates_vs_user3.csv,
         analysis/consistency/figures/*.png,
         reports/consistency_report.html.

Run: python scripts/analyze_consistency.py [--b 10000] [--seed 0] [--ci 95]
                                          [--window first-40]
"""

import argparse
import base64
import json
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
import yaml
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze
from analyze import ground_truth_map, out, out_table, save_fig, start_section

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "database" / "consistency_experiment"
CONFIG_PATH = DATA_ROOT / "users_config.yaml"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
OUT_DIR = REPO / "analysis" / "consistency"
FIGURE_DIR = OUT_DIR / "figures"
REPORT = REPO / "reports" / "consistency_report.html"
PERSONA_DELTAS = REPO / "analysis" / "persona_comparison" / "persona_deltas.csv"

# Cold-start seed labels: the first recorded test F1 sits at this training
# budget before any session label is added (section O's labels-axis start).
START_BUDGET = 70

CONDITION_ORDER = ("control", "treatment")
STRATEGY_TO_CONDITION = {"Experiment 1": "control", "Experiment 2": "treatment"}
CONDITION_COLORS = {"control": "tab:blue", "treatment": "tab:orange"}

METRICS = [
    ("acc", "Accuracy", "proportion (pct. pts in plot)"),
    ("f1", "Macro F1", "proportion (pct. pts in plot)"),
    ("aulc", "AULC (F1 × labels)", "F1 × labels"),
]
PCT = {"acc", "f1"}  # metrics plotted/stored as percentages in figures
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _utf8_console():
    """Windows consoles default to cp1252, which chokes on some chars used here."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass


def norm_metric(label):
    """Normalization key so '×'/'x'/mojibake variants of the AULC label match."""
    return re.sub(r"[×x\uFFFD]", "x", str(label)).strip().lower()


def require_inputs():
    needed = [CONFIG_PATH, DATA_ROOT / "instances", PERSONA_DELTAS]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        print("Missing inputs:\n  " + "\n  ".join(missing), file=sys.stderr)
        raise SystemExit(1)


def load_config(path):
    """Active (uncommented) users from users_config.yaml, conditions resolved
    from the strategy (Experiment 1 -> control, Experiment 2 -> treatment)."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    users = []
    for u in cfg.get("users") or []:
        sessions = {}
        for s in u.get("sessions") or []:
            cond = STRATEGY_TO_CONDITION[s["strategy"]]
            if s.get("condition") and s["condition"] != cond:
                out(f"WARNING config user {u['username']}: condition "
                    f"'{s['condition']}' does not match strategy "
                    f"'{s['strategy']}' (-> {cond}); using {cond}")
            if cond in sessions:
                raise SystemExit(f"config user {u['username']}: duplicate {cond} session")
            sessions[cond] = int(s["instance_id"])
        missing = [c for c in CONDITION_ORDER if c not in sessions]
        if missing:
            out(f"WARNING config user {u['username']}: no session for "
                f"{', '.join(missing)}; excluded")
            continue
        users.append({"username": u["username"], "sessions": sessions})
    if not users:
        raise SystemExit("no active users in users_config.yaml")
    return users


def load_curve(instance_id):
    """Model macro-F1 learning curve of one AL instance: (num_labeled, f1)."""
    path = DATA_ROOT / "instances" / str(instance_id) / "duckdb" / "metrics.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda m: m["num_labeled"])
    x = np.array([m["num_labeled"] for m in rows], float)
    y = np.array([m["f1_score"] for m in rows], float)
    keep = np.concatenate([[True], np.diff(x) > 0])  # dedupe repeated budgets
    return x[keep], y[keep]


def trapz_to(x, y, limit):
    """Trapezoid of the curve over [x[0], limit]; the endpoint is interpolated
    at the limit (labels axis: retrains stop at the last scorable ticket)."""
    if len(x) == 0 or limit <= x[0]:
        return 0.0
    inside = x <= limit
    xs, ys = x[inside], y[inside]
    if len(xs) == 0:
        return 0.0
    if xs[-1] < limit:
        xs = np.append(xs, limit)
        ys = np.append(ys, np.interp(limit, x, y))
        if x[-1] < limit:
            out(f"WARNING learning curve ends at num_labeled={x[-1]:.0f} < "
                f"integration limit {limit:.0f}; the tail is extrapolated flat")
    return float(trapz(ys, xs))


def load_session(instance_id, gt):
    """Scorable (non-idk) decisions of one session with correctness mapped."""
    path = DATA_ROOT / "instances" / str(instance_id) / "duckdb" / "label_decisions.json"
    dec = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(dec)
    n_raw = len(df)
    if df.empty:
        return df, n_raw, 0
    idk = df["i_dont_know"].fillna(False).astype(bool)
    df = df[df["labeled_at"].notna() & (df["label"].notna() | idk)].copy()
    df["ground_truth"] = df["ref"].map(gt)
    n_missing_gt = int(df["ground_truth"].isna().sum())
    if n_missing_gt:
        out(f"WARNING instance {instance_id}: {n_missing_gt} decisions without "
            f"ground truth; dropped")
        df = df[df["ground_truth"].notna()]
    df["correct"] = df["label"] == df["ground_truth"]
    df = df[~df["i_dont_know"].fillna(False).astype(bool)]
    return df, n_raw, int(n_raw - len(df) - n_missing_gt)


def build_session_frame(users, gt):
    """Long per-session frame: one row per user x condition with acc/f1/aulc."""
    rows = []
    for u in users:
        sessions = {}
        for cond in CONDITION_ORDER:
            inst = u["sessions"][cond]
            dec, n_raw, n_dropped = load_session(inst, gt)
            n_scorable = len(dec)
            cell = {"username": u["username"], "condition": cond,
                    "instance_id": inst, "n_decisions_raw": n_raw,
                    "n_scorable": n_scorable, "n_idk_or_dropped": n_dropped,
                    "acc": np.nan, "f1": np.nan, "aulc": np.nan}
            if n_scorable:
                cell["acc"] = float(dec["correct"].mean())
                cell["f1"] = float(f1_score(dec["ground_truth"], dec["label"],
                                            average="macro", zero_division=0))
            sessions[cond] = cell
        # section O rule: both conditions integrate over the same labels window
        cut = int(min(c["n_scorable"] for c in sessions.values()))
        for cond, cell in sessions.items():
            x, y = load_curve(cell["instance_id"])
            cell["cut_n"] = cut
            cell["aulc"] = (trapz_to(x, y, START_BUDGET + cut) if cut > 0
                            else np.nan)
            if cut == 0:
                out(f"WARNING user {cell['username']} {cond}: no scorable "
                    f"decisions; acc/f1/aulc set to NaN")
            rows.append(cell)
    return pd.DataFrame(rows)


def session_wide(long_df):
    """Wide per-user frame with {n_scorable, acc, f1, aulc}_{control, treatment}."""
    stats = ["n_scorable", "acc", "f1", "aulc"]
    wide = long_df.set_index(["username", "condition"])[stats].unstack("condition")
    want = pd.MultiIndex.from_product([stats, CONDITION_ORDER])
    wide = wide.reindex(columns=want)
    wide.columns = [f"{s}_{c}" for s, c in wide.columns]
    wide["cut_n"] = long_df.groupby("username")["cut_n"].first()
    return wide.sort_index()


def bootstrap_means(values, b, rng):
    """Bootstrap distribution of the mean: b resamples of the session values."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    idx = rng.integers(0, len(v), size=(b, len(v)))
    return v[idx].mean(axis=1)


def percentile_ci(boot, ci_level):
    alpha = (100.0 - ci_level) / 2.0
    return np.percentile(boot, [alpha, 100.0 - alpha])


def plot_distributions(wide, fig_name, caption):
    """Original per-condition distributions of the three metrics, replicate
    runs paired across conditions (points AND paired lines share the same
    per-run x offset, so every line touches exactly its two points)."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))
    jitter = {u: o for u, o in zip(wide.index, np.linspace(-0.13, 0.13, len(wide)))}
    for ax, (m, mlabel, unit) in zip(axes, METRICS):
        factor = 100.0 if m in PCT else 1.0
        for u, row in wide.iterrows():
            pair = [row[f"{m}_{c}"] * factor for c in CONDITION_ORDER]
            if not any(pd.isna(pair)):
                off = jitter[u]
                ax.plot([0 + off, 1 + off], pair, color="0.78", lw=1.1,
                        zorder=1)
        for xi, cond in enumerate(CONDITION_ORDER):
            sub = wide[f"{m}_{cond}"].dropna()
            v = sub.to_numpy(float) * factor
            ax.scatter(xi + np.array([jitter[u] for u in sub.index]), v,
                       s=36, zorder=3, color=CONDITION_COLORS[cond],
                       edgecolor="white", lw=0.6)
        ax.set_xticks([0, 1], CONDITION_ORDER)
        ax.set_xlim(-0.5, 1.5)
        if m in PCT:
            ax.set_ylim(0, 100)
        ax.set_title(mlabel)
        if ax is axes[0]:
            ax.set_ylabel("pct. points" if m in PCT else unit)
        else:
            ax.set_ylabel("")
    handles = [
        Line2D([], [], color=CONDITION_COLORS["control"], marker="o", ls="",
               label="replicate run (control)"),
        Line2D([], [], color=CONDITION_COLORS["treatment"], marker="o", ls="",
               label="replicate run (treatment)"),
        Line2D([], [], color="0.78", lw=1.1,
               label="same replicate run across conditions"),
    ]
    axes[0].legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, fig_name, caption)


def plot_bootstrap_means(boot, summary, fig_name, caption, ci_level):
    """Bootstrap distributions of the per-condition means with percentile CIs."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))
    for ax, (m, mlabel, unit) in zip(axes, METRICS):
        factor = 100.0 if m in PCT else 1.0
        lo_all, hi_all = np.inf, -np.inf
        for cond in CONDITION_ORDER:
            b = boot[(m, cond)] * factor
            s = summary[(m, cond)]
            lo, hi = s["ci_low"] * factor, s["ci_high"] * factor
            lo_all, hi_all = min(lo_all, b.min()), max(hi_all, b.max())
            ax.hist(b, bins=40, density=True, histtype="stepfilled",
                    alpha=0.18, color=CONDITION_COLORS[cond])
            ax.hist(b, bins=40, density=True, histtype="step", lw=1.6,
                    color=CONDITION_COLORS[cond],
                    label=f"{cond}: CI [{lo:.1f}, {hi:.1f}]")
            ax.axvline(s["observed_mean"] * factor, color=CONDITION_COLORS[cond],
                       lw=1.8, zorder=2)
            ax.axvline(lo, color=CONDITION_COLORS[cond], ls="--", lw=1.4,
                       zorder=3)
            ax.axvline(hi, color=CONDITION_COLORS[cond], ls="--", lw=1.4,
                       zorder=3)
        ax.set_xlim(lo_all, hi_all)
        ax.set_title(mlabel)
        if ax is axes[0]:
            ax.set_ylabel("bootstrap density")
        ax.set_xlabel("pct. points" if m in PCT else unit)
    handles = [
        Line2D([], [], color=CONDITION_COLORS["control"], lw=1.8,
               label="observed mean (control)"),
        Line2D([], [], color=CONDITION_COLORS["treatment"], lw=1.8,
               label="observed mean (treatment)"),
        Line2D([], [], color="black", lw=1.4, ls="--",
               label=f"percentile CI bounds ({ci_level:.0f}%)"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.9)
    axes[1].legend(loc="upper left", fontsize=7, framealpha=0.9, title="condition: 95% CI")
    fig.tight_layout()
    save_fig(fig, fig_name, caption)


def load_user3_original_deltas(window):
    """Real (human) user3's per-metric delta (treatment - control) from the
    original main-cohort sessions, as tabulated in persona_deltas.csv."""
    df = pd.read_csv(PERSONA_DELTAS)
    df = df[(df["label_window"] == window) & (df["user"] == "user3")]
    if df.empty:
        raise SystemExit(f"no user3 rows for label_window '{window}' in "
                         f"{PERSONA_DELTAS}")
    deltas = {}
    for m, mlabel, _ in METRICS:
        want = norm_metric(mlabel)
        sub = df[df["metric"].map(norm_metric) == want]
        deltas[m] = float(sub["human_delta"].iloc[0])
    return deltas


def plot_replicate_scatter(wide, user3_delta, fig_name, caption):
    """Persona-comparison scatter restricted to user3: x = the real (human)
    user3's delta from the original main-cohort sessions, y = each LLM
    persona-user3 replicate run's own delta; y = x marks perfect agreement,
    the horizontal band spans the replicate runs' min-max delta range."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    offsets = [(5, 5, "left"), (-5, 5, "right"), (5, -12, "left"),
               (-5, -12, "right")]
    for ax, (m, mlabel, unit) in zip(axes, METRICS):
        factor = 100.0 if m in PCT else 1.0
        x_val = user3_delta[m] * factor
        deltas = (wide[f"{m}_treatment"] - wide[f"{m}_control"]).dropna() * factor
        y_min, y_max = deltas.min(), deltas.max()
        ax.axhspan(y_min, y_max, color=CONDITION_COLORS["control"], alpha=0.13,
                   zorder=0)
        ax.axvline(x_val, color="0.25", lw=1.0, ls=":", zorder=1)
        aligned = np.sign(deltas) == np.sign(x_val)
        all_vals = np.concatenate([deltas.to_numpy(), [x_val]])
        lim = max(np.max(np.abs(all_vals)) * 1.18, 1e-9)
        ax.plot([-lim, lim], [-lim, lim], ls="--", color="gray", lw=1, zorder=1)
        ax.axvline(0, color="k", lw=0.8, alpha=0.5, zorder=1)
        ax.axhline(0, color="k", lw=0.8, alpha=0.5, zorder=1)
        ax.scatter(np.full(len(deltas), x_val), deltas,
                   c=["tab:green" if a else "tab:red" for a in aligned],
                   s=32, zorder=3)
        # the original user3's own delta as a reference marker on the y = x line
        ax.scatter([x_val], [x_val], marker="D", s=42, facecolor="white",
                   edgecolor="black", lw=1.2, zorder=4)
        for i, u in enumerate(deltas.index):
            dx, dy, ha = offsets[i % len(offsets)]
            ax.annotate(u, (x_val, deltas[u]), fontsize=6, xytext=(dx, dy),
                        textcoords="offset points", ha=ha, color="0.35")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(mlabel)
        shown_unit = "pct. points" if m in PCT else unit
        ax.set_xlabel(f"Δ user3, original sessions ({shown_unit})")
        if ax is axes[0]:
            ax.set_ylabel(f"Δ user3con runs — LLM persona user3 ({shown_unit})")
    handles = [
        Line2D([], [], color="tab:green", marker="o", ls="",
               label="aligned (same sign as user3)"),
        Line2D([], [], color="tab:red", marker="o", ls="", label="opposite sign"),
        Line2D([], [], marker="D", ls="", ms=7, markerfacecolor="white",
               markeredgecolor="black", color="none",
               label="user3 (original sessions)"),
        Line2D([], [], color=CONDITION_COLORS["control"], lw=8, alpha=0.25,
               label="replicate-run delta range (min–max)"),
        Line2D([], [], color="gray", ls="--", lw=1, label="y = x"),
    ]
    axes[0].legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, fig_name, caption)


def render_report(intro_text):
    def slug(title):
        return re.sub(r"\W+", "-", title.lower()).strip("-")

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Smart ticketing — user3 consistency pilot</title>",
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
        "<h1>user3 consistency pilot — replicate-run agreement (LLM persona user3)</h1>",
        f"<p>Generated {datetime.now():%Y-%m-%d %H:%M} from "
        f"database/consistency_experiment/ (persona user3 replicated by the "
        f"LLM agent; each user3conN run does control = Experiment 1 and "
        f"treatment = Experiment 2). "
        f"{escape(intro_text)} Figures also saved as PNG in "
        f"analysis/consistency/figures/.</p>",
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
    parser.add_argument("--b", type=int, default=10000,
                        help="number of bootstrap resamples per panel")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the bootstrap")
    parser.add_argument("--ci", type=float, default=95.0,
                        help="percentile CI level in %% (default 95, i.e. the "
                             "5%%-significance interval requested for the pilot)")
    parser.add_argument("--window", default="first-40",
                        choices=["full", "first-40", "matched-min"],
                        help="label_window of the real user3's delta used as "
                             "the scatter's x reference (default first-40: "
                             "identical 40-label support as the replicate "
                             "runs' sessions)")
    args = parser.parse_args()

    _utf8_console()
    require_inputs()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    analyze.FIGURES = FIGURE_DIR  # route analyze.save_fig to the pilot's dir

    users = load_config(CONFIG_PATH)
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        run_users = {u["username"] for u in manifest.get("users", [])}
        active = {u["username"] for u in users}
        excluded = sorted(run_users - active)
        fields = [("design", manifest.get("design")),
                  ("agent", manifest.get("agent_model")),
                  ("model", manifest.get("model")),
                  ("n_tickets", manifest.get("n_tickets")),
                  ("status", manifest.get("status")),
                  ("exported_at", manifest.get("exported_at"))]
        out("Run: " + ", ".join(f"{k} {v}" for k, v in fields if v is not None))
        out(f"Config users: {len(active)} active "
            f"({', '.join(sorted(active))}); excluded via config (commented "
            f"out): {', '.join(excluded) if excluded else 'none'}")
    else:
        out(f"Config users: {len(users)} active "
            f"({', '.join(u['username'] for u in users)})")

    # ---- section 1: per-replicate session metrics ---------------------------
    start_section("1. Per-replicate session metrics by condition")
    out("Data basis: scorable (non-idk) decisions per session "
        "(label_decisions.json; rows without a submitted label are artifact "
        "rows of the 40-ticket quota and are dropped), ground truth from "
        "data/User Request_last_team_ANON.xlsx. Accuracy/macro F1 describe "
        "the actual labeling; AULC integrates the model macro-F1 learning "
        f"curve over labels {START_BUDGET}..{START_BUDGET}+N, N = per-user "
        "minimum scorable count across the user's two sessions (section O "
        "rule). control = Experiment 1, treatment = Experiment 2.")
    gt = ground_truth_map()
    long_df = build_session_frame(users, gt)
    wide = session_wide(long_df)
    out_table(wide[["n_scorable_control", "n_scorable_treatment",
                    "acc_control", "acc_treatment", "f1_control",
                    "f1_treatment", "aulc_control", "aulc_treatment",
                    "cut_n"]].round(3),
              title="Per-replicate-run metrics (n = scorable decisions; "
                    "acc/f1/aulc per condition; cut_n = AULC window N)",
              index=True)
    plot_distributions(
        wide, "consistency_distributions.png",
        "Original per-condition distributions across the LLM persona-user3 "
        "replicate runs: dots = replicate runs (blue control, orange "
        "treatment), gray lines pair the same replicate run across "
        "conditions. Accuracy/macro F1 in pct. points, AULC in F1 × labels "
        f"(labels axis {START_BUDGET}..{START_BUDGET}+N with N = cut_n).")
    session_csv = OUT_DIR / "consistency_session_metrics.csv"
    long_df.to_csv(session_csv, index=False)
    out(f"Wrote {session_csv.relative_to(REPO)}")

    # ---- section 2: bootstrap of the condition means ------------------------
    start_section("2. Bootstrap distribution of the condition means "
                  f"({args.ci:.0f}% percentile CI)")
    out(f"Bootstrap: B = {args.b} resamples (seed {args.seed}) of the "
        f"per-condition values across the {len(wide)} replicate runs; the CI "
        f"is the percentile interval at the {args.ci:.0f}% level "
        f"(alpha = {100 - args.ci:.0f}%).")
    rng = np.random.default_rng(args.seed)
    boot, summary, summary_rows = {}, {}, []
    for m, mlabel, unit in METRICS:
        for cond in CONDITION_ORDER:
            values = wide[f"{m}_{cond}"].dropna().to_numpy(float)
            b = bootstrap_means(values, args.b, rng)
            lo, hi = percentile_ci(b, args.ci)
            boot[(m, cond)] = b
            summary[(m, cond)] = {"observed_mean": float(values.mean()),
                                  "observed_sd": float(values.std(ddof=1)),
                                  "ci_low": float(lo), "ci_high": float(hi)}
            summary_rows.append({
                "metric": mlabel, "condition": cond, "n": len(values),
                "observed_mean": values.mean(), "observed_sd": values.std(ddof=1),
                "boot_mean": b.mean(), "ci_low": lo, "ci_high": hi})
    summary_df = pd.DataFrame(summary_rows).set_index(["metric", "condition"])
    out_table(summary_df.round(4), title="Observed mean ± SD and bootstrap "
                                         "mean with percentile CI", index=True)
    plot_bootstrap_means(
        boot, summary, "consistency_bootstrap_means.png",
        f"Bootstrap distributions of the mean per metric and condition "
        f"(B = {args.b} resamples of the {len(wide)} replicate-run session "
        f"values, seed {args.seed}): filled step histograms per condition, "
        f"solid lines = observed means, dashed lines = percentile CI bounds "
        f"({args.ci:.0f}% level, values in the legend). Accuracy/macro F1 in "
        f"pct. points, AULC in F1 × labels.",
        ci_level=args.ci)
    boot_csv = OUT_DIR / "consistency_bootstrap_summary.csv"
    summary_df.reset_index().to_csv(boot_csv, index=False)
    out(f"Wrote {boot_csv.relative_to(REPO)}")

    # ---- section 3: real user3 vs the LLM persona-user3 replicate runs -----
    start_section("3. Real user3 (original sessions) vs LLM persona-user3 "
                  f"replicate runs (user3conN; window '{args.window}')")
    out("Delta = treatment - control per replicate run (each run does both "
        "conditions, so deltas are paired). The scatter mirrors the layout "
        "of analysis/persona_comparison/figures/persona_delta_scatter.png "
        "but restricted to user3: x = the real (human) user3's delta from "
        "its original main-cohort sessions (persona_deltas.csv "
        f"human_delta, label_window '{args.window}'), y = each LLM "
        "persona-user3 replicate run's own delta; y = x marks perfect "
        "agreement with the original user3, and the shaded horizontal band "
        "spans the replicate runs' min-max delta range (run-to-run "
        "variability of the persona).")
    delta_rows = []
    for m, mlabel, _ in METRICS:
        d = (wide[f"{m}_treatment"] - wide[f"{m}_control"]).dropna()
        for username, v in d.items():
            delta_rows.append({"metric": mlabel, "username": username,
                               "delta": v})
    deltas_long = pd.DataFrame(delta_rows)
    deltas_wide = deltas_long.pivot(index="username", columns="metric",
                                    values="delta")
    deltas_wide.columns = [norm_metric(c) for c in deltas_wide.columns]
    delta_csv = OUT_DIR / "consistency_user3_deltas.csv"
    deltas_long.to_csv(delta_csv, index=False)
    out(f"Wrote {delta_csv.relative_to(REPO)}")

    boot_delta_rows = []
    for m, mlabel, unit in METRICS:
        key = norm_metric(mlabel)
        d = deltas_wide[key].dropna().to_numpy(float)
        b = bootstrap_means(d, args.b, rng)
        lo, hi = percentile_ci(b, args.ci)
        boot_delta_rows.append({
            "metric": mlabel, "condition": "delta (treatment - control)",
            "n": len(d), "observed_mean": d.mean(),
            "observed_sd": d.std(ddof=1), "boot_mean": b.mean(),
            "ci_low": lo, "ci_high": hi})
        rngs = ", ".join(f"{v:+.3f}" for v in d)
        out(f"\n{mlabel} per-replicate deltas: {rngs}")
        out(f"  mean {d.mean():+.4f}  SD {d.std(ddof=1):+.4f}  "
            f"min {d.min():+.4f}  max {d.max():+.4f}  "
            f"bootstrap {args.ci:.0f}% CI of the mean "
            f"[{lo:+.4f}, {hi:+.4f}]")
    all_summary = pd.concat([summary_df.reset_index(),
                             pd.DataFrame(boot_delta_rows)])
    boot_csv_all = OUT_DIR / "consistency_bootstrap_summary.csv"
    all_summary.to_csv(boot_csv_all, index=False)
    out(f"\nUpdated {boot_csv_all.relative_to(REPO)} with the paired-delta rows")

    user3_delta = load_user3_original_deltas(args.window)
    rows = []
    for m, mlabel, _ in METRICS:
        key = norm_metric(mlabel)
        d = deltas_wide[key].dropna()
        for u, v in d.items():
            rows.append({
                "metric": mlabel, "replicate_run": u, "replicate_delta": v,
                "user3_original_delta": user3_delta[m],
                "aligned": bool(np.sign(v) == np.sign(user3_delta[m]))})
    rep_df = pd.DataFrame(rows)
    rep_csv = OUT_DIR / "consistency_replicates_vs_user3.csv"
    rep_df.to_csv(rep_csv, index=False)
    out(f"Wrote {rep_csv.relative_to(REPO)}")

    align = rep_df.groupby("metric")["aligned"].agg(["sum", "count"])
    key_of = {norm_metric(mlabel): m for m, mlabel, _ in METRICS}
    align_tbl = pd.DataFrame({
        "aligned": align["sum"].astype(int).astype(str) + "/" + align["count"].astype(int).astype(str),
        "alignment_rate": align["sum"] / align["count"],
        "user3_original_delta": [user3_delta[key_of[norm_metric(ml)]]
                                 for ml in align.index],
        "replicate_delta_mean": [deltas_wide[norm_metric(ml)].dropna().mean()
                                 for ml in align.index],
        "replicate_delta_min": [deltas_wide[norm_metric(ml)].dropna().min()
                                for ml in align.index],
        "replicate_delta_max": [deltas_wide[norm_metric(ml)].dropna().max()
                                for ml in align.index],
    }).reindex(columns=["aligned", "alignment_rate", "user3_original_delta",
                        "replicate_delta_mean", "replicate_delta_min",
                        "replicate_delta_max"])
    out_table(align_tbl.round(3),
              title="Replicate runs vs the original user3: strict-sign "
                    "alignment with user3's original-session delta and the "
                    "replicate deltas' spread", index=True)

    plot_replicate_scatter(
        wide, user3_delta, "consistency_persona_scatter_user3.png",
        f"Persona comparison restricted to user3: x = the real (human) "
        f"user3's delta from its original main-cohort sessions "
        f"(persona_deltas.csv, label_window '{args.window}'), y = each LLM "
        f"persona-user3 replicate run's own delta (user3conN; accuracy/F1 "
        f"deltas in pct. points, AULC in F1 × labels). Shaded horizontal "
        f"band = replicate runs' min-max delta range; white diamond = "
        f"user3's original delta sitting on the y = x line (perfect "
        f"agreement reference); green = replicate delta aligned in sign "
        f"with user3's, red = opposite sign.")

    render_report(
        f"LLM persona-user3 replicate-run agreement across {len(users)} runs "
        f"({', '.join(u['username'] for u in users)}), agent "
        f"deepseek/deepseek-v4.1-flash; bootstrap B = {args.b}, seed "
        f"{args.seed}, {args.ci:.0f}% percentile CIs; user3 reference "
        f"window '{args.window}'.")


if __name__ == "__main__":
    main()
