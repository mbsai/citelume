"""
Writes docs/FRAMEWORKS.md from the registry so the human-readable reference never drifts from the code.

    cd backend && python -m frameworks.export_doc
"""
from collections import defaultdict
from pathlib import Path

from frameworks.review_types import ORGANIZATIONS, REVIEW_FRAMEWORKS
from frameworks.rules import GATES, RULES
from frameworks.sources import SOURCES

DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "FRAMEWORKS.md"

# Corrections made when the original rules were checked against their primary sources (November 2026)
AUDIT_CORRECTIONS = [
    ("Narrative said findings were \"synthesized qualitatively\" under SWiM", "SWiM covers quantitative synthesis "
     "without meta-analysis and accepts vote counting by direction of effect; wording now says findings are compared "
     "by direction of effect with no pooled estimate."),
    ("Agreement meter described as \"rather than vote-counting\"", "The meter is direction-of-effect vote counting; "
     "it now says so and states that it ignores effect size and precision (SWiM)."),
    ("RoB 2 concerns rated \"Unclear\"", "\"Unclear\" is the old RoB 1 scale. RoB 2 rates Low / Some concerns / High "
     "per domain; from an abstract the tool now only names the domain and says it is not assessable."),
    ("QUIPS grouped with PROBAST+AI for prediction models", "QUIPS is for prognostic factor studies; PROBAST+AI for "
     "prediction models. They are now routed separately."),
    ("ARRIVE 2.0 used as a risk-of-bias check", "ARRIVE is a reporting guideline; SYRCLE is the animal risk-of-bias "
     "tool. Both are now named for what they are."),
    ("ROBINS-I applied to all non-randomised designs", "ROBINS-I V2 is for interventions; observational exposure "
     "studies now route to ROBINS-E."),
    ("No route for diagnostic accuracy studies", "Added QUADAS-3 (February 2026, replaces QUADAS-2)."),
    ("Any abstract containing \"random\" (e.g. \"random forest\") was treated as a randomised trial",
     "Trial detection now requires randomised/randomized near trial, allocation or assignment."),
    ("No certainty-of-evidence concept", "Added GRADE-informed indicators, labelled as not a GRADE rating."),
    ("Domain rules for CS, physics and chemistry had no sources", "Now cite REFORMS, the ML leakage taxonomy, the "
     "GUM, IUPAC and ACS reporting guidance; mathematics and other conventions are labelled as field conventions."),
]


def render() -> str:
    out = [
        "# Methodology Frameworks & Rules",
        "",
        "_Generated from `backend/frameworks/` by `python -m frameworks.export_doc`. Edit the registry, not this file._",
        "",
        "Rules are applied as framework-informed screening from titles and abstracts. The tool never issues a formal "
        "risk-of-bias or GRADE rating, which require full-text assessment.",
        "",
        "## Corrections from the source audit",
        "",
        "| Before | After |",
        "|---|---|",
    ]
    out += [f"| {before} | {after} |" for before, after in AUDIT_CORRECTIONS]

    out += ["", "## Rules by stage", ""]
    by_stage = defaultdict(list)
    for r in RULES:
        by_stage[r.stage].append(r)
    for stage in ("appraisal", "synthesis", "reporting", "gap"):
        if not by_stage[stage]:
            continue
        out += [f"### {stage.capitalize()}", "", "| ID | Rule | Applies to | Source | Enforced by |", "|---|---|---|---|---|"]
        for r in by_stage[stage]:
            scope = []
            if "*" not in r.domains:
                scope.append("domains: " + ", ".join(r.domains))
            if "*" not in r.study_designs:
                scope.append("designs: " + ", ".join(r.study_designs))
            if "*" not in r.review_types:
                scope.append("reviews: " + ", ".join(r.review_types))
            source = r.citation_label() + (f" ({r.source_locator})" if r.source_locator else "")
            enforced = f"code: `{r.enforcement}`" if r.enforcement != "prompt" else "prompt"
            out.append(f"| {r.rule_id} | {r.text} | {'; '.join(scope) or 'all'} | {source} | {enforced} |")
        out.append("")

    out += ["## Review types", "", "| Framework | Description | Sources |", "|---|---|---|"]
    out += [f"| {f.label} | {f.description} | {', '.join(SOURCES[s].short_name for s in f.source_ids)} |"
            for f in REVIEW_FRAMEWORKS.values()]
    out += ["", "| Organisation | How findings are arranged |", "|---|---|"]
    out += [f"| {o.label} | {o.prompt_instruction} |" for o in ORGANIZATIONS.values()]

    out += ["", "## Deterministic checks", "", "| Check | Where |", "|---|---|"]
    out += [f"| `{gid}` | {where} |" for gid, where in GATES.items()]

    out += ["", "## Sources", ""]
    for kind in ("risk_of_bias", "reporting", "certainty", "review_method", "field_convention", "tool_integrity"):
        group = [s for s in SOURCES.values() if s.kind == kind]
        if not group:
            continue
        out += [f"### {kind.replace('_', ' ').capitalize()}", ""]
        out += [f"- **{s.short_name}**: {s.citation}" + (f" <{s.url}>" if s.url else "") for s in group]
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(render())
    print(f"Wrote {DOC_PATH}")
