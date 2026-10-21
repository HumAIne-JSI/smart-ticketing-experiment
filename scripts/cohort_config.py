"""Cohort path resolution for --human / --llm flags.

--human (default) uses the human experiment database and the existing
analysis/ + reports/analyze_report.html outputs.
--llm uses the LLM experiment database and isolated analysis_llm/ +
reports/analyze_report_llm.html outputs so the two cohorts never
overwrite each other.
"""

from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CohortPaths:
    label: str
    root: Path
    config: Path
    analysis: Path
    figures: Path
    report: Path


def resolve_cohort(is_llm: bool) -> CohortPaths:
    if is_llm:
        analysis = REPO / "analysis_llm"
        return CohortPaths(
            label="llm",
            root=REPO / "database" / "llm_experiment" / "instances",
            config=REPO / "analysis" / "preprocess_config_llm.json",
            analysis=analysis,
            figures=analysis / "figures",
            report=REPO / "reports" / "analyze_report_llm.html",
        )
    analysis = REPO / "analysis"
    return CohortPaths(
        label="human",
        root=REPO / "database" / "experiment" / "instances",
        config=REPO / "analysis" / "preprocess_config.json",
        analysis=analysis,
        figures=analysis / "figures",
        report=REPO / "reports" / "analyze_report.html",
    )


def add_cohort_args(parser):
    """Add mutually-exclusive --human / --llm flags (default: human)."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--human", action="store_true",
                       help="use human experiment data (default)")
    group.add_argument("--llm", action="store_true",
                       help="use LLM experiment data")
    return parser


def resolve_cohort_from_args(args) -> CohortPaths:
    return resolve_cohort(bool(getattr(args, "llm", False)))
