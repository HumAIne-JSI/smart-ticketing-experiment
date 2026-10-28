"""Build per-user persona JSONs from behavioral tables + questionnaires.

Ingests (no recomputation of latencies):
- analysis/persona_behavioral.csv (from analyze.py --emit-persona-tables,
  itself computed from analysis/decisions_labels.csv)
- questionnaires/responses/*.xlsx Data sheets (numeric codes)

Writes:
- figures/persona_latency_medians.png (grouped bars manual vs assisted
  + mean-of-medians lines)
- personas/persona_inputs/<user>.json per behavioral user

Usage:
    python scripts/persona_preprocessing.py [--human|--llm]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cohort_config import add_cohort_args, resolve_cohort_from_args

REPO = Path(__file__).resolve().parents[1]

PRE_MAP = {
    "gender": {1: "Male", 2: "Female", 3: "Non-binary", 4: "Prefer not to say"},
    "age": {1: "Under 25", 2: "25-34", 3: "35-44", 4: "45-54",
            5: "55-64", 6: "65 or older", 7: "Prefer not to say"},
    "edu": {1: "Secondary", 2: "Vocational", 3: "Bachelor",
            4: "Master", 5: "Doctoral", 6: "Prefer not to say"},
    "lead": {1: "No", 2: "Yes, currently", 3: "Yes, previously"},
    "years": {1: "None", 2: "<1 year", 3: "1-3 years",
              4: "3-5 years", 5: ">5 years"},
    "tools": {1: "No", 2: "Yes"},
    "freq": {1: "Never", 2: "Rarely", 3: "Sometimes", 4: "Often",
             5: "Daily or almost daily"},
}
AI_Q6 = {1: "Easy / I was already certain about",
         2: "Challenging / I was uncertain about",
         3: "Both equally"}
AI_Q7 = {1: "Model prediction", 2: "Confidence scores",
         3: "LIME explanations", 4: "Closest predicted class tickets",
         5: "Closest past tickets"}
AI_Q5 = {1: "Model prediction", 2: "Confidence scores",
         3: "LIME explanations", 4: "Closest predicted class tickets",
         5: "Closest past tickets"}
AI_Q11 = {1: "Automatically", 2: "On request"}
OVERALL_Q2 = {1: "With AI", 2: "Without AI", 3: "No difference"}
TERTILE_COLORS = {"QUICK": "#2ca02c", "STEADY": "#ff7f0e", "SLOW": "#d62728"}


def to_int_or_neg1(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        if isinstance(v, str) and v.strip() in ("-1", "'-1' unanswered question,"):
            return -1
        f = float(str(v).strip())
        if f == -1:
            return -1
        return int(f)
    except (ValueError, TypeError):
        return None


def free_text(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s in ("-1", "'-1' unanswered question,"):
        return -1
    return s


def read_data_sheet(path, user_col):
    df = pd.read_excel(path, sheet_name="Data", header=None)
    codes = [str(c).strip() for c in df.iloc[0].tolist()]
    out = {}
    for r in range(2, len(df)):
        user = str(df.iloc[r, user_col]).strip()
        if not user or user.lower() == "nan":
            continue
        if user in out:
            print(f"WARNING duplicate {user} in {path.name}; keeping first")
            continue
        row = {}
        for i, code in enumerate(codes):
            row[code] = df.iloc[r, i]
        out[user] = row
    return out


def build_json(user, beh, pre, man, ai, ov):
    def ii(d, code):
        if d is None:
            return None
        return to_int_or_neg1(d.get(code))

    def mapped(d, code, book):
        if d is None:
            return None
        v = to_int_or_neg1(d.get(code))
        if v is None:
            return None
        if v == -1:
            return -1
        return book.get(v)

    pre_sec = None
    if pre is not None:
        pre_sec = {
            "gender": mapped(pre, "Q3", PRE_MAP["gender"]),
            "age_group": mapped(pre, "Q4", PRE_MAP["age"]),
            "education": mapped(pre, "Q5", PRE_MAP["edu"]),
            "current_role": free_text(pre.get("Q6")),
            "lead_experience": mapped(pre, "Q7", PRE_MAP["lead"]),
            "years_ticket_experience": mapped(pre, "Q8", PRE_MAP["years"]),
            "ml_familiarity_1_5": ii(pre, "Q9"),
            "prior_model_assisted_use": mapped(pre, "Q10", PRE_MAP["tools"]),
            "labeling_frequency": mapped(pre, "Q11", PRE_MAP["freq"]),
        }

    man_sec = None
    if man is not None:
        man_sec = {
            "difficulty_1_5": ii(man, "Q2"),
            "mental_effort_agree_1_5": ii(man, "Q3c"),
            "time_consuming_agree_1_5": ii(man, "Q3e"),
            "satisfaction_agree_1_5": ii(man, "Q3f"),
        }

    ai_sec = None
    if ai is not None:
        def cue_rank(code):
            v = to_int_or_neg1(ai.get(code))
            if v is None or v == -1:
                return None
            return AI_Q5.get(v)
        ranking = [cue_rank("Q5a"), cue_rank("Q5b"), cue_rank("Q5c")]
        if len(set(r for r in ranking if r is not None)) != len([r for r in ranking if r is not None]):
            print(f"WARNING {user}: duplicate cue_ranking entries {ranking}")
        ai_sec = {
            "difficulty_1_5": ii(ai, "Q2"),
            "overall_experience_1_5": {
                "assistance_useful": ii(ai, "Q3k"),
                "trusted_predictions_stated": ii(ai, "Q3n"),
                "felt_pressured_to_agree": ii(ai, "Q3o"),
            },
            "xai_usefulness_1_5": {
                "prediction": ii(ai, "Q4a"),
                "confidence_scores": ii(ai, "Q4b"),
                "lime_highlights": ii(ai, "Q4c"),
                "closest_predicted_class_tickets": ii(ai, "Q4d"),
                "closest_past_tickets": ii(ai, "Q4e"),
            },
            "cue_ranking_1st_2nd_3rd": ranking,
            "most_helpful_for": mapped(ai, "Q6", AI_Q6),
            "least_helpful_feature": mapped(ai, "Q7", AI_Q7),
            "clarity_confusion_1_5": {
                "explanations_clear": ii(ai, "Q8a"),
                "understand_how_ai_decides": ii(ai, "Q12c"),
            },
            "wrong_label_frequency_1_5": ii(ai, "Q9"),
            "prefer_auto_vs_on_request": mapped(ai, "Q11", AI_Q11),
            "trust_dynamics_1_5": {
                "trust_system_in_my_interest": ii(ai, "Q12f"),
                "reduced_mental_effort": ii(ai, "Q12g"),
                "trusted_more_over_time": ii(ai, "Q12h"),
                "less_useful_at_beginning": ii(ai, "Q12i"),
            },
        }

    ov_sec = None
    if ov is not None:
        ov_sec = {
            "preferred_condition": mapped(ov, "Q2", OVERALL_Q2),
            "more_accurate_with_ai_1_5": ii(ov, "Q3a"),
            "faster_with_ai_1_5": ii(ov, "Q3b"),
            "less_effort_with_ai_1_5": ii(ov, "Q3c"),
            "changed_criteria_1_5": ii(ov, "Q3e"),
            "prefer_ai_for_real_work_1_5": ii(ov, "Q3f"),
            "biggest_difference_text": free_text(ov.get("Q4")),
        }

    def side(cond):
        r = beh.get(cond)
        if r is None:
            return None
        pace = r.get("pace_tertile")
        if pace is None or (isinstance(pace, float) and pd.isna(pace)):
            pace = None
        elif str(pace).strip() not in ("QUICK", "STEADY", "SLOW"):
            pace = None
        else:
            pace = str(pace).strip()
        med = r["median_latency_s"]
        med = float(med) if pd.notna(med) else float("nan")
        base = {
            "n_total": int(r["n_total"]),
            "median_resolution_time_s": med,
            "pace_tertile": pace,
            "n_difficult": int(r["n_difficult"]),
            "n_idk": int(r["n_idk"]),
        }
        if cond == "treatment":
            base["n_agree_top1"] = int(r["n_agree_top1"])
            base["n_agree_top2"] = int(r["n_agree_top2"])
        return base

    return {
        "participant_id": user,
        "pre_experiment": pre_sec,
        "manual_session": man_sec,
        "ai_assisted_session": ai_sec,
        "overall_comparison": ov_sec,
        "behavioral_logs": {
            "manual": side("control"),
            "assisted": side("treatment"),
        },
    }


def plot_latencies(beh_wide, out_path):
    import matplotlib.patches as mpatches
    users = sorted(beh_wide.keys())
    manual = [beh_wide[u].get("control", {}).get("median_latency_s", float("nan"))
              for u in users]
    assisted = [beh_wide[u].get("treatment", {}).get("median_latency_s", float("nan"))
                for u in users]
    manual_pace = [beh_wide[u].get("control", {}).get("pace_tertile") for u in users]
    assisted_pace = [beh_wide[u].get("treatment", {}).get("pace_tertile") for u in users]
    # Display cutoffs recomputed from ingested medians (same 33/66 linear rule);
    # assignment itself comes from analyze.py and is only visualized here.
    m_series = pd.Series(manual, dtype=float).dropna()
    a_series = pd.Series(assisted, dtype=float).dropna()
    m_cuts = (float(m_series.quantile(0.33)), float(m_series.quantile(0.66))) if len(m_series) else (float("nan"), float("nan"))
    a_cuts = (float(a_series.quantile(0.33)), float(a_series.quantile(0.66))) if len(a_series) else (float("nan"), float("nan"))
    x = np.arange(len(users))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(users)), 5))
    m_colors = [TERTILE_COLORS.get(p, "0.7") if p is not None and not (isinstance(p, float) and pd.isna(p)) else "0.7"
                for p in manual_pace]
    a_colors = [TERTILE_COLORS.get(p, "0.7") if p is not None and not (isinstance(p, float) and pd.isna(p)) else "0.7"
                for p in assisted_pace]
    ax.bar(x - w / 2, manual, w, color=m_colors, edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, assisted, w, color=a_colors, edgecolor="black", linewidth=0.5,
           hatch="//")
    mean_m = float(np.nanmean(manual)) if len(manual) else float("nan")
    mean_a = float(np.nanmean(assisted)) if len(assisted) else float("nan")
    ax.axhline(mean_m, color="blue", linestyle="--", linewidth=1)
    ax.axhline(mean_a, color="orange", linestyle=":", linewidth=1)
    for cut, ls, col in [(m_cuts[0], (0, (2, 1)), "blue"), (m_cuts[1], (0, (2, 1)), "blue"),
                        (a_cuts[0], (0, (1, 1)), "orange"), (a_cuts[1], (0, (1, 1)), "orange")]:
        if pd.notna(cut):
            ax.axhline(cut, color=col, linestyle=ls, linewidth=1, alpha=0.7)
    ax.text(0.01, 0.97, f"mean manual={mean_m:.1f}s cut {m_cuts[0]:.1f}/{m_cuts[1]:.1f}s",
            transform=ax.transAxes, va="top", fontsize=8, color="blue")
    ax.text(0.01, 0.90, f"mean assisted={mean_a:.1f}s cut {a_cuts[0]:.1f}/{a_cuts[1]:.1f}s",
            transform=ax.transAxes, va="top", fontsize=8, color="orange")
    ax.set_xticks(x, users, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("median per-ticket latency (s)")
    ax.set_title("Per-user median ticket latency by condition (labels view)\ncolor=pace tertile, hatch=assisted; dotted=tertile cuts")
    handles = [mpatches.Patch(color=c, label=k) for k, c in TERTILE_COLORS.items()]
    handles += [mpatches.Patch(facecolor="white", edgecolor="black", label="left=manual"),
                mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="right=assisted")]
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path} (mean manual={mean_m:.1f}s assisted={mean_a:.1f}s)")
    print(f"display cuts manual q33/q66={m_cuts[0]:.3f}/{m_cuts[1]:.3f} assisted q33/q66={a_cuts[0]:.3f}/{a_cuts[1]:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_cohort_args(parser)
    args = parser.parse_args()
    cohort = resolve_cohort_from_args(args)
    analysis = cohort.analysis
    figures = cohort.figures
    figures.mkdir(parents=True, exist_ok=True)
    qdir = REPO / "questionnaires" / "responses"
    out_dir = REPO / "personas" / (
        "persona_inputs_llm" if cohort.label == "llm" else "persona_inputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    beh_path = analysis / "persona_behavioral.csv"
    if not beh_path.exists():
        raise SystemExit(f"missing {beh_path}; run analyze.py --emit-persona-tables first")
    beh_df = pd.read_csv(beh_path)
    beh_wide = {}
    for _, r in beh_df.iterrows():
        beh_wide.setdefault(str(r["user"]), {})[str(r["condition"])] = r.to_dict()

    plot_latencies(beh_wide, figures / "persona_latency_medians.png")

    pre = read_data_sheet(qdir / "pre_experiment.xlsx", user_col=1)  # Q2 username
    man = read_data_sheet(qdir / "manual_session.xlsx", user_col=0)  # Q1
    ai = read_data_sheet(qdir / "ai_assisted.xlsx", user_col=0)
    ov = read_data_sheet(qdir / "overall_comparison.xlsx", user_col=0)

    for user in sorted(beh_wide.keys()):
        data = build_json(user, beh_wide[user],
                          pre.get(user), man.get(user), ai.get(user), ov.get(user))
        for sec in ("pre_experiment", "manual_session",
                    "ai_assisted_session", "overall_comparison"):
            if data[sec] is None:
                print(f"{user}: missing {sec} questionnaire -> null")
        for side in ("manual", "assisted"):
            if data["behavioral_logs"][side] is None:
                print(f"{user}: missing {side} condition -> null")
        (out_dir / f"{user}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(beh_wide)} persona JSONs -> {out_dir}")


if __name__ == "__main__":
    main()
