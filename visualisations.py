import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "experiment"
OUTPUT_PATH = RESULTS_DIR / "condition_comparison.png"

CONTROL_SESSION_IDS = [4, 7, 9]
TREATMENT_SESSION_IDS = [5, 6, 8]

CONTROL_LABEL = "No AI assistance"
TREATMENT_LABEL = "With AI assistance"
CONDITIONS = [(CONTROL_LABEL, CONTROL_SESSION_IDS), (TREATMENT_LABEL, TREATMENT_SESSION_IDS)]
COLORS = {CONTROL_LABEL: "#4C72B0", TREATMENT_LABEL: "#DD8452"}

METRIC_COLUMNS = [
    ("latency_min", "latency (min)", 2),
    ("labeler_gt", "lab-GT (%)", 2),
    ("difficult", "difficult (%)", 2),
    ("dont_know", "i-dont-know (%)", 2),
    ("model_labeler", "mod-lab (%)", 2),
    ("model_gt", "mod-GT (%)", 2),
    ("final_f1", "final F1", 4),
]

SINGLE_PLOTS = [
    ("latency_min", "Avg latency per ticket (min)"),
    ("labeler_gt", "Labeler-GT agreement (%)"),
    ("final_f1", "Final macro F1"),
]

GROUPED_PLOTS = [
    (["difficult", "dont_know"], ["Difficult", "I don't know"], "Self-reported rates"),
    (["model_labeler", "model_gt"], ["Model-labeler", "Model-GT"], "Model agreement (%)"),
]


def load_sessions(session_ids):
    sessions = []
    for sid in session_ids:
        path = RESULTS_DIR / f"{sid}.json"
        with open(path, encoding="utf-8") as f:
            sessions.append(json.load(f))
    return sessions


def extract_metrics(session):
    computed = session["computed_metrics"]
    buttons = session["self_reported_buttons"]
    f1_history = session.get("macro_f1_by_iteration") or []
    latency_ms = computed.get("average_latency_ms")
    return {
        "latency_min": None if latency_ms is None else latency_ms / 60000,
        "labeler_gt": computed.get("labeler_gt_agreement_pct"),
        "difficult": buttons.get("difficult_pct"),
        "dont_know": buttons.get("i_dont_know_pct"),
        "model_labeler": computed.get("model_labeler_agreement_pct"),
        "model_gt": computed.get("model_gt_agreement_pct"),
        "final_f1": f1_history[-1]["f1_score"] if f1_history else None,
    }


def fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def mean_of(values):
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def print_report(metrics_by_condition):
    header = "  ".join(f"{label:>14}" for _, label, _ in METRIC_COLUMNS)
    print("Per-session metrics")
    print(f"  {'session':>14}  {header}")
    for cond_label, rows in metrics_by_condition.items():
        print(f"{cond_label}:")
        for sid, metrics in rows:
            cells = "  ".join(f"{fmt(metrics[key], digits):>14}" for key, _, digits in METRIC_COLUMNS)
            print(f"  {sid:>14}  {cells}")

    print("\nCondition means (mean over sessions)")
    for cond_label, rows in metrics_by_condition.items():
        cells = "  ".join(
            f"{fmt(mean_of([m[key] for _, m in rows]), digits):>14}"
            for key, _, digits in METRIC_COLUMNS
        )
        print(f"  {cond_label:<24} {cells}")


def bar_subplot(ax, keys, key_labels, title, metrics_by_condition):
    group_width = 0.8
    bar_width = group_width / len(CONDITIONS)
    for ci, (cond_label, rows) in enumerate(metrics_by_condition.items()):
        for ki, key in enumerate(keys):
            values = [m[key] for _, m in rows if m[key] is not None]
            x = ki - group_width / 2 + bar_width / 2 + ci * bar_width
            mean = mean_of(values)
            ax.bar(
                x,
                0 if mean is None else mean,
                width=bar_width * 0.9,
                color=COLORS[cond_label],
                label=cond_label if ki == 0 else None,
            )
    ax.set_title(title)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(key_labels)
    ax.set_ylim(bottom=0)


def make_figure(metrics_by_condition):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title) in zip(axes.flat[:3], SINGLE_PLOTS):
        bar_subplot(ax, [key], [""], title, metrics_by_condition)
    for ax, (keys, key_labels, title) in zip(axes.flat[3:5], GROUPED_PLOTS):
        bar_subplot(ax, keys, key_labels, title, metrics_by_condition)
    axes.flat[5].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=2)
    fig.suptitle("AI assisted vs. unaided labeling")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_PATH, dpi=150)


def main():
    metrics_by_condition = {}
    for cond_label, session_ids in CONDITIONS:
        metrics_by_condition[cond_label] = [
            (sid, extract_metrics(session))
            for sid, session in zip(session_ids, load_sessions(session_ids))
        ]
    print_report(metrics_by_condition)
    make_figure(metrics_by_condition)
    print(f"\nFigure saved to: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
