"""Visualise label timing for every session: cumulative labels vs elapsed time.

One panel per session. The slope is labeling pace, plateaus are breaks,
and dashed red lines mark gaps > 30 min (participant likely left the desk).
Dots are colored by that decision's latency (capped at 10 min; gray = nulled).

Reads analysis/decisions.csv + analysis/sessions.csv, writes
analysis/figures/timings_per_session.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
FIGURES = ANALYSIS / "figures"

GAP_BREAK_MIN = 30   # gaps larger than this are marked as breaks
LAT_CAP_S = 600      # latency color scale cap (10 min)


def main():
    dec = pd.read_csv(ANALYSIS / "decisions.csv")
    ses = pd.read_csv(ANALYSIS / "sessions.csv")
    dec = dec.merge(ses[["session_id", "user"]], on="session_id", how="left")
    dec["ts"] = pd.to_datetime(dec["labeled_at"], format="mixed")

    sessions = ses.set_index("session_id")
    order = sorted(sessions.index,
                   key=lambda s: (int(sessions.loc[s, "user"][4:]), s))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.75")
    norm = Normalize(vmin=0, vmax=LAT_CAP_S)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    ncols = 6
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 3.2 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, sid in zip(axes, order):
        g = dec[dec.session_id == sid].sort_values("ts")
        x = (g["ts"] - g["ts"].iloc[0]).dt.total_seconds() / 60
        y = np.arange(1, len(g) + 1)
        lat = (g["latency_ms"] / 1000).clip(upper=LAT_CAP_S).fillna(np.nan)
        span = sessions.loc[sid, "span_min"]

        for xi in x[x.diff() > GAP_BREAK_MIN]:
            ax.axvline(xi, color="crimson", ls="--", lw=0.9, alpha=0.8)

        ax.plot(x, y, drawstyle="steps-post", color="0.85", lw=1, zorder=1)
        ax.scatter(x, y, c=lat, cmap=cmap, norm=norm, s=14, zorder=2)
        ax.set_title(f"{sid} ({g['condition'].iloc[0]}, n={len(g)}, span={span:.0f}m)",
                     fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(order):]:
        ax.set_visible(False)

    for ax in axes[::ncols]:
        ax.set_ylabel("labels", fontsize=8)
    for ax in axes[-ncols:]:
        ax.set_xlabel("minutes since first label", fontsize=8)

    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.01)
    cbar.set_label("decision latency (s), cap 600 s — gray = nulled", fontsize=8)
    fig.suptitle("Label timing per session (dashed red = gap > 30 min)", fontsize=11)

    out = FIGURES / "timings_per_session.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out} ({len(order)} sessions, {len(dec)} decisions)")


if __name__ == "__main__":
    main()
