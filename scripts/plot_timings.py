"""Visualise label timing for every session: cumulative labels vs elapsed time.

One panel per session. The slope is labeling pace, plateaus are breaks,
and dashed red lines mark gaps > 30 min (participant likely left the desk).
Dots are colored by that decision's latency (capped at 10 min; gray = nulled).

Writes <cohort>/figures/timings_per_session.png from decisions.csv +
sessions.csv (only sessions that survived preprocessing) and
<cohort>/figures/timings_per_session_all.png from all_decisions.csv +
all_sessions.csv (every configured session, titles of excluded ones prefixed
with an X; x-axes share the longest kept session's span, longer excluded
sessions draw on their own extended axis).
Use --human (default) or --llm to select the cohort.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cohort_config import add_cohort_args, resolve_cohort_from_args

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
FIGURES = ANALYSIS / "figures"

GAP_BREAK_MIN = 30   # gaps larger than this are marked as breaks
LAT_CAP_S = 600      # latency color scale cap (10 min)
NCOLS = 6


def load_sessions(prefix=""):
    """prefix "" -> cleaned tables; "all_" -> tables incl. excluded sessions."""
    dec = pd.read_csv(ANALYSIS / f"{prefix}decisions.csv")
    ses = pd.read_csv(ANALYSIS / f"{prefix}sessions.csv")
    dec["ts"] = pd.to_datetime(dec["labeled_at"], format="mixed")
    return dec, ses.set_index("session_id")


def draw_session(ax, g, x, span, title, cmap, norm):
    y = np.arange(1, len(g) + 1)
    lat = (g["latency_ms"] / 1000).clip(upper=LAT_CAP_S)
    for xi in x[x.diff() > GAP_BREAK_MIN]:
        ax.axvline(xi, color="crimson", ls="--", lw=0.9, alpha=0.8)
    ax.plot(x, y, drawstyle="steps-post", color="0.85", lw=1, zorder=1)
    ax.scatter(x, y, c=lat, cmap=cmap, norm=norm, s=14, zorder=2)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)


def plot(dec, sessions, out_path, suptitle, shared_xlim=False):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.75")
    norm = Normalize(vmin=0, vmax=LAT_CAP_S)

    groups = []
    def _sort_key(s):
        user = str(sessions.loc[s, "user"])
        # human: user12 -> 12; llm: LLM_User1 -> 1; fallback to name sort
        digits = "".join(ch for ch in user if ch.isdigit())
        try:
            return (int(digits), s) if digits else (10**9, user, s)
        except ValueError:
            return (10**9, user, s)
    order = sorted(sessions.index, key=_sort_key)
    for sid in order:
        g = dec[dec.session_id == sid].sort_values("ts")
        if g.empty:
            continue
        x = (g["ts"] - g["ts"].iloc[0]).dt.total_seconds() / 60
        groups.append((sid, g, x))
    if shared_xlim:
        base = max(x.iloc[-1] for sid, g, x in groups
                   if "kept" not in sessions or bool(sessions.loc[sid, "kept"]))

    nrows = int(np.ceil(len(groups) / NCOLS))
    fig, axes = plt.subplots(nrows, NCOLS,
                             figsize=(17, 3.2 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (sid, g, x) in zip(axes, groups):
        kept = bool(sessions.loc[sid, "kept"]) if "kept" in sessions else True
        marker = "" if kept else "\u2717 "
        span = sessions.loc[sid, "span_min"]
        title = f"{marker}{sid} ({g['condition'].iloc[0]}, n={len(g)}, span={span:.0f}m)"
        draw_session(ax, g, x, span, title, cmap, norm)
        if shared_xlim:
            ax.set_xlim(0, max(base, x.iloc[-1]))
    for ax in axes[len(groups):]:
        ax.set_visible(False)

    for ax in axes[::NCOLS]:
        ax.set_ylabel("labels", fontsize=8)
    for ax in axes[-NCOLS:]:
        ax.set_xlabel("minutes since first label", fontsize=8)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                        shrink=0.6, pad=0.01)
    cbar.set_label("decision latency (s), cap 600 s — gray = nulled", fontsize=8)
    fig.suptitle(suptitle, fontsize=11)

    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path} ({len(groups)} sessions, {len(dec)} decisions)")


def main():
    global ANALYSIS, FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    add_cohort_args(parser)
    args = parser.parse_args()
    cohort = resolve_cohort_from_args(args)
    ANALYSIS = cohort.analysis
    FIGURES = cohort.figures
    FIGURES.mkdir(parents=True, exist_ok=True)
    print(f"Cohort: {cohort.label} (input: {ANALYSIS})")
    dec, ses = load_sessions()
    plot(dec, ses, FIGURES / "timings_per_session.png",
         "Label timing per session (dashed red = gap > 30 min)")

    dec, ses = load_sessions("all_")
    plot(dec, ses, FIGURES / "timings_per_session_all.png",
         "\u2717 = session excluded by preprocessing (x-axis = longest kept "
         "session; longer \u2717 sessions extend their own); dashed red = gap > 30 min",
         shared_xlim=True)


if __name__ == "__main__":
    main()
