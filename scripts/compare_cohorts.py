"""Compare LLM vs human figures in one HTML report.

Reads fixed inputs:
  human: analysis/figures/
  llm:   analysis_llm/figures/
Writes: reports/compare_report.html (self-contained, PNGs embedded).

Run analyze --human / --llm first; missing inputs fail loudly.
"""

import base64
import sys
from datetime import datetime
from html import escape
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HUMAN_FIG = REPO / "analysis" / "figures"
LLM_FIG = REPO / "analysis_llm" / "figures"
OUT = REPO / "reports" / "compare_report.html"

SECTIONS = [
    {
        "title": "1. Per-user macro F1",
        "desc": "Labeling macro F1, unaided (control) vs AI-assisted (treatment); gray lines = users, red = mean.",
        "images": [
            ("human", HUMAN_FIG / "a_f1_slope.png",
             "Human paired users (n=15)."),
            ("llm", LLM_FIG / "a_f1_slope.png",
             "LLM paired users (n=2)."),
        ],
    },
    {
        "title": "2. Complementarity",
        "desc": "Catch rate P(human correct | model wrong) and overtrust rate P(human wrong | model correct) per condition.",
        "images": [
            ("human", HUMAN_FIG / "b_complementarity.png",
             "Human complementarity."),
            ("llm", LLM_FIG / "b_complementarity.png",
             "LLM complementarity."),
        ],
    },
    {
        "title": "3. Fatigue macro F1",
        "desc": "Macro F1 and latency across session thirds (equal-count label thirds in chronological order).",
        "images": [
            ("human", HUMAN_FIG / "d_fatigue_f1.png",
             "Human F1 and latency by session third."),
            ("llm", LLM_FIG / "d_fatigue_f1.png",
             "LLM F1 and latency by session third."),
        ],
    },
    {
        "title": "4. LLM learning curves (section C)",
        "desc": "Model test-F1 vs training labels; faint lines = instances, no pooled mean (<5 instances).",
        "images": [
            ("llm", LLM_FIG / "c_learning_curves.png",
             "LLM active-learning loop only."),
        ],
    },
    {
        "title": "5. Human retention curve (section K)",
        "desc": "Paired-users merged F1 panel; pooled mean uses only users with both sessions present at each budget, dotted lines = dropout.",
        "images": [
            ("human", HUMAN_FIG / "k_paired_curves.png",
             "Human paired-users panel only."),
        ],
    },
]


def main():
    missing = [str(p) for s in SECTIONS for _, p, _ in s["images"] if not p.exists()]
    if missing:
        print("Missing figures:\n  " + "\n  ".join(missing), file=sys.stderr)
        print("Run: python scripts/analyze.py --human  and  python scripts/analyze.py --llm",
              file=sys.stderr)
        raise SystemExit(1)

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>LLM vs human — figure comparison</title>",
        "<style>",
        "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;max-width:1100px;"
        "color:#222;line-height:1.45;padding:0 1rem}",
        "h1{border-bottom:2px solid #333;padding-bottom:.3rem}",
        "h2{margin-top:2.5rem;border-bottom:1px solid #ccc;padding-bottom:.2rem}",
        ".tag{display:inline-block;font-size:.75rem;font-weight:bold;padding:1px 8px;"
        "border-radius:4px;margin-right:.4rem}",
        ".human{background:#dceaff;color:#1a3a8f}",
        ".llm{background:#ffe4cc;color:#8a3c00}",
        "img{max-width:100%;border:1px solid #eee;margin:.5rem 0}",
        ".caption{color:#666;font-size:.85rem;margin:.2rem 0}",
        "</style></head><body>",
        "<h1>LLM vs human — figure comparison</h1>",
        f"<p>Generated {datetime.now():%Y-%m-%d %H:%M} from "
        f"analysis/figures/ [human] and analysis_llm/figures/ [llm].</p>",
    ]
    for s in SECTIONS:
        parts.append(f"<h2>{escape(s['title'])}</h2><p>{escape(s['desc'])}</p>")
        for cohort, path, caption in s["images"]:
            data = base64.b64encode(path.read_bytes()).decode()
            parts.append(
                f"<p><span class='tag {cohort}'>[{cohort}]</span>"
                f"<span class='caption'>{escape(path.name)} — {escape(caption)}</span></p>"
                f"<img src='data:image/png;base64,{data}' alt='[{cohort}] {escape(path.name)}'>"
            )
    parts.append("</body></html>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Compare report written to {OUT}")


if __name__ == "__main__":
    main()
