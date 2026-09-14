"""Generate per-participant LLM personas from the pre-experiment questionnaire.

Reads the LimeSurvey CSV export (semicolon-delimited), writes structured profiles to
personas/profiles/<user>.json and persona files to prompts/<user>.json consumed by
the standard persona machinery. The participant number is used only for naming,
never inside the persona text.

Usage:
    uv run python scripts/make_personas.py [--questionnaire PATH] [--dry-run]
"""

import argparse
import json
import re
from pathlib import Path

QUESTIONNAIRE = Path("personas/pre_experiment_questionnaire.csv")
PROFILES_DIR = Path("personas/profiles")
PERSONAS_DIR = Path("prompts")

GENDER = {1: "Male", 2: "Female", 3: "Non-binary", 4: "Prefer not to say"}
AGE = {1: "Under 25", 2: "25-34", 3: "35-44", 4: "45-54", 5: "55-64", 6: "65 or older", 7: "Prefer not to say"}
EDUCATION = {
    1: "Secondary education",
    2: "Vocational/technical education",
    3: "Bachelor's degree",
    4: "Master's degree",
    5: "Doctoral degree",
    6: "Prefer not to say",
}
LEAD = {1: "No", 2: "Yes, currently", 3: "Yes, previously"}
YEARS = {1: "None", 2: "< 1 year", 3: "1-3 years", 4: "3-5 years", 5: "> 5 years"}
ML = {1: "not familiar at all", 2: "slightly familiar", 3: "moderately familiar", 4: "quite familiar", 5: "very familiar"}
TOOLS = {1: "No", 2: "Yes"}
FREQUENCY = {1: "never", 2: "rarely", 3: "sometimes", 4: "often", 5: "daily or almost daily"}

LEAD_SENTENCE = {
    1: "You have never held a lead, supervisory, or team-coordination role.",
    2: "You currently hold a lead, supervisory, or team-coordination role.",
    3: "You have previously held a lead, supervisory, or team-coordination role.",
}
YEARS_SENTENCE = {
    1: "You have no prior experience with IT support ticket management.",
    2: "You have less than one year of experience with IT support ticket management.",
    3: "You have one to three years of experience with IT support ticket management.",
    4: "You have three to five years of experience with IT support ticket management.",
    5: "You have more than five years of experience with IT support ticket management.",
}
TOOLS_SENTENCE = {
    1: "You have not used model-assisted tools for classification or labeling before.",
    2: "You have used model-assisted tools for classification or labeling before.",
}
FREQUENCY_SENTENCE = {
    1: "You never classify, route, or label IT support tickets in your own work.",
    2: "You rarely classify, route, or label IT support tickets in your own work.",
    3: "You sometimes classify, route, or label IT support tickets in your own work.",
    4: "You often classify, route, or label IT support tickets in your own work.",
    5: "You classify, route, or label IT support tickets daily or almost daily.",
}

DATA_LINE = re.compile(r"^1;(user\w+);")


def parse_questionnaire(path):
    profiles = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not DATA_LINE.match(line):
            continue
        fields = line.split(";")
        if len(fields) != 11:
            raise RuntimeError(f"Unexpected field count in line: {line[:120]}")
        consent, username, g, a, e, role, lead, y, ml, tools, freq = fields
        codes = {"gender": int(g), "age": int(a), "education": int(e), "lead": int(lead),
                 "years": int(y), "ml": int(ml), "tools": int(tools), "frequency": int(freq)}
        for name, code, book in [
            ("gender", codes["gender"], GENDER),
            ("age", codes["age"], AGE),
            ("education", codes["education"], EDUCATION),
            ("lead", codes["lead"], LEAD),
            ("years", codes["years"], YEARS),
            ("ml", codes["ml"], ML),
            ("tools", codes["tools"], TOOLS),
            ("frequency", codes["frequency"], FREQUENCY),
        ]:
            if code not in book:
                raise RuntimeError(f"{username}: unknown code {code} for {name}")
        profiles.append({
            "participant": username,
            "consent": int(consent),
            "gender_code": codes["gender"],
            "gender": GENDER[codes["gender"]],
            "age_code": codes["age"],
            "age_group": AGE[codes["age"]],
            "education_code": codes["education"],
            "education": EDUCATION[codes["education"]],
            "current_role": role.strip(),
            "lead_code": codes["lead"],
            "lead_role": LEAD[codes["lead"]],
            "years_code": codes["years"],
            "years_it_tickets": YEARS[codes["years"]],
            "ml_code": codes["ml"],
            "tools_code": codes["tools"],
            "model_tools_before": TOOLS[codes["tools"]],
            "frequency_code": codes["frequency"],
            "ticket_frequency": FREQUENCY[codes["frequency"]],
        })
    return profiles


def demographic_clause(p):
    parts = []
    if p["gender"] == "Male":
        parts.append("a man")
    elif p["gender"] == "Female":
        parts.append("a woman")
    elif p["gender"] == "Non-binary":
        parts.append("a non-binary person")
    if p["age_group"] != "Prefer not to say":
        parts.append("aged " + p["age_group"][0].lower() + p["age_group"][1:])
    if p["education"] != "Prefer not to say":
        edu = p["education"]
        parts.append("with a " + edu if edu.endswith("degree") else "with " + edu[0].lower() + edu[1:])
    return parts


def render_persona(p):
    sentences = []
    who = demographic_clause(p)
    if who:
        sentences.append("You are " + ", ".join(who) + ".")
    sentences.append(f"You currently work as: {p['current_role']}.")
    sentences.append(LEAD_SENTENCE[p["lead_code"]])
    sentences.append(YEARS_SENTENCE[p["years_code"]])
    sentences.append(f"You are {ML[p['ml_code']]} with machine learning and model-based systems.")
    sentences.append(TOOLS_SENTENCE[p["tools_code"]])
    sentences.append(FREQUENCY_SENTENCE[p["frequency_code"]])
    return "Persona: " + " ".join(sentences)


def main():
    parser = argparse.ArgumentParser(description="Generate per-participant personas from the questionnaire.")
    parser.add_argument("--questionnaire", default=str(QUESTIONNAIRE))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profiles = parse_questionnaire(args.questionnaire)
    if not profiles:
        raise SystemExit(f"No questionnaire data rows found in {args.questionnaire}")

    for p in profiles:
        description = render_persona(p)
        print(f"--- {p['participant']} ---")
        print(description)
        if args.dry_run:
            continue
        profile_path = PROFILES_DIR / f"{p['participant']}.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps(p, indent=2, ensure_ascii=False), encoding="utf-8")
        persona_path = PERSONAS_DIR / f"{p['participant']}.json"
        persona_path.write_text(
            json.dumps({"name": p["participant"], "description": description}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(f"{len(profiles)} personas processed from {args.questionnaire}")


if __name__ == "__main__":
    main()
