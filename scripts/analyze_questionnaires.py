"""Analyze questionnaire responses per question and write an HTML report.

Reads the 1ka survey exports from questionnaires/responses/*.xlsx (sheets
"Data" with the raw answer codes and "Response labels" with the question
texts, per-respondent labels and the scale definitions), classifies every
question and reports it individually:

  * number - Likert / 1-5 scale answers: average, median, distribution of
             the answers per scale value, number of users answering and
             number leaving it blank,
  * choice - nominal single-choice questions (e.g. gender, "Automatically
             vs on request", ranked features): frequency table per answer
             option, number of users answering and leaving it blank,
  * open   - free-text answers: every answer listed verbatim with the
             participant ID, plus the number of users answering and the
             number who left the field blank.

Answer code -1 (and empty cells) means the participant left the question
blank; it is treated as blank everywhere.  The participant-ID column
("user<NN>") is not analyzed as a question; it identifies respondents.

Participant selection: set INCLUDED_USERS below to an iterable of user IDs
(e.g. {"user1", "user2"}) to restrict the whole analysis to those users,
or leave it as None to include everybody.  It can be overridden at run
time with --users user1,user2,user3.  Listed IDs that match no respondent
produce a console warning.

Output: self-contained HTML report (reports/questionnaire_analysis.html).
"""

import argparse
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path

from openpyxl import load_workbook

REPO = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO / "questionnaires" / "responses"
REPORT_PATH = REPO / "reports" / "questionnaire_analysis.html"

# ---------------------------------------------------------------------------
# Participant selection: which users to include in the analysis.
#   None or empty  -> include every participant (default).
#   Otherwise      -> only rows of these participant IDs are analyzed,
#                     e.g. INCLUDED_USERS = {"user1", "user2", "user8"}.
# ---------------------------------------------------------------------------
# All that participated in the study
# INCLUDED_USERS: set[str] | list[str] | tuple[str, ...] | None = ["user1", "user2", "user3", "user6", "user7", "user8", "user9", "user10", "user11", "user13", "user14", "user15", "user16", "user17", "user20", "user21", "user22", "user23", "user24", "user25"]
# All that labeled at least 40 tickets
#INCLUDED_USERS: set[str] | list[str] | tuple[str, ...] | None = ["user2", "user3", "user9", "user13", "user14", "user16", "user21", "user22", "user23", "user24", "user25"]
# all that labeled at least 20 tickets

INCLUDED_USERS: set[str] | list[str] | tuple[str, ...] | None = ["user1", "user2", "user3", "user6", "user7", "user9", "user11", "user13", "user14", "user15", "user16", "user17", "user20", "user21", "user22", "user23", "user24", "user25"]


ID_RE = re.compile(r"^user\d+$", re.IGNORECASE)
BLANK_CODE = -1

# Display titles / preferred section order (any other file is appended).
TITLE_BY_STEM = {
    "pre_experiment": "Pre-Experiment Questionnaire",
    "manual_session": "Manual Session Questionnaire",
    "ai_assisted": "AI-Assisted Session Questionnaire",
    "overall_comparison": "Overall Comparison Questionnaire",
}
SECTION_ORDER = ["pre_experiment", "manual_session", "ai_assisted", "overall_comparison"]

# Canonical 1-5 scale vocabularies.  If the value->label mapping of a
# numeric question fits into one of these, the question is treated as a
# numerical scale; otherwise it is a nominal choice question.
SCALE_VOCABULARIES = [
    frozenset({"strongly disagree", "disagree", "neutral", "agree", "strongly agree"}),
    frozenset({"strongly disagree", "disagree", "neither agree nor disagree", "agree", "strongly agree"}),
    frozenset({"not useful", "slightly useful", "moderately useful", "very useful", "extremely useful"}),
    frozenset({"never", "rarely", "sometimes", "often", "very often"}),
    frozenset({"never", "rarely", "sometimes", "often", "daily or almost daily"}),
    frozenset({"very easy", "easy", "moderately easy", "moderately difficult", "difficult", "very difficult"}),
    frozenset({"much less", "less", "about the same", "more", "much more"}),
]

KIND_LABELS = {"number": "Numerical (scale)", "choice": "Nominal choice", "open": "Open text"}


# ---------------------------------------------------------------------------
# Loading and classification
# ---------------------------------------------------------------------------
def clean_text(text) -> str:
    """Collapse all whitespace (incl. tabs / nbsp) to single spaces."""
    return " ".join(str(text).split()) if text is not None else ""


def is_blank(value) -> bool:
    """-1 (numeric or textual), None or '' means the question was left blank."""
    if value is None:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == BLANK_CODE
    return str(value).strip() in ("", "-1")


def classify_numeric(q: dict) -> None:
    """Decide between 'number' (scale) and 'choice' (nominal) for numeric codes.

    Builds a code -> most common label mapping from the Data / Response-labels
    pairing.  A question is a numerical scale when its labels are just the
    numbers as text (or absent) or when they fit a canonical scale vocabulary.
    """
    code_labels: dict[int, Counter] = {}
    for _, value, label in q["answers"]:
        if is_blank(value):
            continue
        code = int(value)
        word = clean_text(label) if not is_blank(label) else ""
        code_labels.setdefault(code, Counter())[word] += 1

    vmap: dict[int, str] = {}
    words: set[str] = set()
    labels_are_numbers = True
    for code, counter in code_labels.items():
        best = counter.most_common(1)[0][0]
        vmap[code] = best
        if best:
            words.add(best.lower())
            if best != str(code):
                labels_are_numbers = False

    if not words or labels_are_numbers or any(words <= vocab for vocab in SCALE_VOCABULARIES):
        q["kind"] = "number"
    else:
        q["kind"] = "choice"
    q["vmap"] = {c: vmap[c] for c in sorted(vmap)}


def load_questionnaire(path: Path) -> dict:
    """Load one xlsx export: participant list + per-question answer lists."""
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        ws_data, ws_labels = wb["Data"], wb["Response labels"]
    except KeyError:  # fall back to the first two sheets
        names = wb.sheetnames
        ws_data, ws_labels = wb[names[0]], wb[names[1]]
    drows = [tuple(r) for r in ws_data.iter_rows(values_only=True)]
    lrows = [tuple(r) for r in ws_labels.iter_rows(values_only=True)]
    wb.close()

    qids = [clean_text(c) for c in drows[0]]
    qtexts = [clean_text(c) for c in (lrows[1] if len(lrows) > 1 else drows[1])]
    n_rows = min(len(drows), len(lrows))

    # Rows 3+ hold one respondent per row; row 0/1 of the labels sheet carry
    # the ID list and the full question texts.
    values: list[list] = [[drows[row][col] for row in range(2, n_rows)] for col in range(len(qids))]
    labels: list[list] = [[lrows[row][col] for row in range(2, n_rows)] for col in range(len(qids))]

    # The participant-ID column: every non-blank value matches user<NN>.
    id_col = None
    for col in range(len(qids)):
        vals = [v for v in values[col] if not is_blank(v)]
        if vals and all(isinstance(v, str) and ID_RE.match(v.strip()) for v in vals):
            id_col = col
            break

    participant_ids = [clean_text(v).strip() for v in values[id_col]] if id_col is not None else []
    questions = []
    for col, qid in enumerate(qids):
        if col == id_col:
            continue
        answers = list(zip(participant_ids, values[col], labels[col]))
        q = {"id": qid, "text": qtexts[col] if col < len(qtexts) else "", "answers": answers}
        if any(isinstance(v, str) and not is_blank(v) for v in values[col]):
            q["kind"] = "open"
        else:
            classify_numeric(q)
        questions.append(q)

    return {
        "stem": path.stem,
        "file": path.name,
        "title": TITLE_BY_STEM.get(path.stem, path.stem),
        "participants": participant_ids,
        "questions": questions,
    }


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
def esc_text(text) -> str:
    return escape(str(text)).replace("\n", "<br>")


def fmt_num(x) -> str:
    if isinstance(x, int):
        return str(x)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}"


def pct(count: int, total: int) -> str:
    return f"{100 * count / total:.1f}%" if total else "n/a"


def stats_table(cells: str) -> str:
    return f'<table class="summary"><tr>{cells}</tr></table>'


def answered_blank_cells(answered: int, blank: int) -> str:
    return f"<th>Answered</th><td>{answered}</td><th>Left blank</th><td>{blank}</td>"


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def render_question(q: dict) -> str:
    blank = sum(1 for _, v, _ in q["answers"] if is_blank(v))
    answered = len(q["answers"]) - blank
    parts = [f'<div class="qid">{esc_text(q["id"])} · {esc_text(KIND_LABELS.get(q["kind"], q["kind"]))}</div>']
    if q["text"]:
        parts.append(f'<div class="qtext">{esc_text(q["text"])}</div>')

    if q["kind"] == "number":
        values = [int(v) for _, v, _ in q["answers"] if not is_blank(v)]
        mean_cell = f"{statistics.fmean(values):.2f}" if values else "n/a"
        med_cell = fmt_num(statistics.median(values)) if values else "n/a"
        parts.append(stats_table(
            f"<th>Average</th><td>{mean_cell}</td><th>Median</th><td>{med_cell}</td>"
            + answered_blank_cells(answered, blank)))
        if values:
            counts = Counter(values)
            rows = []
            for code in range(min(values), max(values) + 1):
                count = counts.get(code, 0)
                word = q["vmap"].get(code, "")
                label = "" if not word or word == str(code) else esc_text(word)
                rows.append([str(code), label, str(count), pct(count, answered)])
            parts.append(table(["Value", "Scale label", "Count", "% of answered"], rows))
    elif q["kind"] == "choice":
        parts.append(stats_table(answered_blank_cells(answered, blank)))
        counts = Counter(int(v) for _, v, _ in q["answers"] if not is_blank(v))
        rows = []
        for code, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            word = q["vmap"].get(code, "")
            label = esc_text(word) if word else f"(code {code})"
            rows.append([label, str(code), str(count), pct(count, answered)])
        if rows:
            parts.append(table(["Answer option", "Code", "Count", "% of answered"], rows))
    elif q["kind"] == "open":
        parts.append(stats_table(answered_blank_cells(answered, blank)))
        for pid, value, _ in q["answers"]:
            if is_blank(value):
                continue
            # Undo the 1ka CSV-style escaping of quotes inside answers.
            text = str(value).replace('\\"', '"').replace("\\'", "'")
            parts.append(f'<div class="answer"><span class="uid">{esc_text(pid)}</span>{esc_text(text)}</div>')
    return '<div class="q">' + "".join(parts) + "</div>"


def build_report(sections: list[dict], selection_note: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    toc = (
        '<p class="meta"><strong>Contents:</strong> '
        + " · ".join(f"{esc_text(s['title'])} ({esc_text(s['respondents'])})" for s in sections)
        + "</p>"
    )
    body = "".join(
        f'<section><h2>{esc_text(s["title"])}</h2><p class="meta">{esc_text(s["meta"])}</p>'
        + "".join(render_question(q) for q in s["questions"]) + "</section>"
        for s in sections
    )
    css = """
body{font-family:'Segoe UI',Arial,sans-serif;color:#1c2733;background:#f5f7fa;margin:0;padding:24px}
.container{max-width:1100px;margin:0 auto}
h1{margin:.1em 0 .1em}
.meta{color:#5a6b7b;font-size:.92em;margin:.15em 0 .6em}
section{background:#fff;border:1px solid #dde5ec;border-radius:8px;padding:14px 22px 8px;margin:18px 0}
section h2{margin:.2em 0 .2em}
.q{border:1px solid #e4eaf0;border-radius:6px;padding:10px 14px 12px;margin:12px 0;background:#fdfefe}
.qid{font-weight:600;color:#2b4a63;letter-spacing:.02em}
.qtext{margin:2px 0 8px;color:#33424f}
table{border-collapse:collapse;margin:4px 0 8px}
th,td{border:1px solid #dfe6ec;padding:4px 12px;font-size:.9em;text-align:left}
th{background:#eef3f7;font-weight:600}
.answer{background:#f4f8fb;border:1px solid #e3ecf3;border-radius:4px;padding:6px 10px;margin:4px 0;font-size:.92em}
.uid{font-weight:600;color:#2b6cb0;margin-right:10px}
"""
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        "<title>Questionnaire Response Analysis</title><style>" + css + "</style></head><body>"
        '<div class="container">'
        "<h1>Questionnaire Response Analysis</h1>"
        f'<p class="meta">Generated {esc_text(now)} · An answer code of -1 means the participant '
        f"left the question blank · {esc_text(selection_note)}</p>"
        + toc + body + "</div></body></html>"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def resolve_selection(cli_users: str | None) -> set[str] | None:
    if cli_users:
        users = {u.strip() for u in cli_users.split(",") if u.strip()}
        return users or None
    if INCLUDED_USERS is None:
        return None
    return {u.strip() for u in INCLUDED_USERS} or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-question analysis of questionnaire responses, written to an HTML report.")
    parser.add_argument("--input", type=Path, default=INPUT_DIR, help="folder with the xlsx response exports")
    parser.add_argument("--output", type=Path, default=REPORT_PATH, help="HTML report destination")
    parser.add_argument("--users", type=str, default=None,
                        help="comma-separated participant IDs to include (overrides INCLUDED_USERS)")
    args = parser.parse_args(argv)

    files = sorted(args.input.glob("*.xlsx"))
    if not files:
        print(f"No .xlsx files found in {args.input}", file=sys.stderr)
        return 1
    files.sort(key=lambda p: SECTION_ORDER.index(p.stem) if p.stem in SECTION_ORDER else len(SECTION_ORDER))

    workbooks = [load_questionnaire(path) for path in files]
    selection = resolve_selection(args.users)
    known_ids = {pid for wb in workbooks for pid in wb["participants"]}
    if selection is not None:
        missing = selection - known_ids
        if missing:
            print(f"Warning: selection contains IDs with no respondent data: {', '.join(sorted(missing))}",
                  file=sys.stderr)

    selection_note = (
        "Participant selection: all participants found in the exports."
        if selection is None else
        f"Participant selection: {', '.join(sorted(selection))}.")

    sections = []
    print(f"Participant selection: {'ALL' if selection is None else ', '.join(sorted(selection))}")
    for wb in workbooks:
        if selection is not None:
            keep = selection
            questions = [{**q, "answers": [row for row in q["answers"] if row[0] in keep]}
                         for q in wb["questions"]]
            respondents = [pid for pid in wb["participants"] if pid in keep]
        else:
            questions = wb["questions"]
            respondents = list(wb["participants"])

        kinds = Counter(q["kind"] for q in questions)
        meta = (f"{len(respondents)} respondents"
                + ("" if selection is None else f" of {len(selection)} selected")
                + f" · {len(questions)} questions: {kinds.get('number', 0)} numerical, "
                f"{kinds.get('choice', 0)} choice, {kinds.get('open', 0)} open")
        sections.append({"title": wb["title"], "meta": meta, "questions": questions,
                         "respondents": f"{len(respondents)} respondents"})

        print(f"\n== {wb['title']} ({wb['file']}) - {len(respondents)} respondents ==")
        for q in questions:
            blank = sum(1 for _, v, _ in q["answers"] if is_blank(v))
            answered = len(q["answers"]) - blank
            extra = ""
            if q["kind"] == "number" and answered:
                extra = f" avg={statistics.fmean([int(v) for _, v, _ in q['answers'] if not is_blank(v)]):.2f}"
            print(f"  {q['id']:<5} {q['kind']:<7} answered={answered:<3} blank={blank:<3}{extra}")

    report = build_report(sections, selection_note)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
