"""
RoB 2 for randomised trials (effect of assignment to the intervention): the signalling questions and the decision
algorithms that turn their answers into a judgement for each domain and overall.

Source: Sterne JAC, et al. RoB 2: a revised tool for assessing risk of bias in randomised trials. BMJ
2019;366:l4898, and the RoB 2 full guidance document (Higgins JPT, et al., 2019). The questions are abridged here.
The algorithms follow the guidance's decision figures; they should be checked against the official RoB 2 tool
(riskofbias.info) before results are published, and any difference reported.

Answers: Y (yes), PY (probably yes), PN (probably no), N (no), NI (no information), NA (not applicable).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

ANSWERS = ("Y", "PY", "PN", "N", "NI", "NA")
YES = {"Y", "PY"}
NO = {"N", "PN"}

# (question id, question, domain number)
QUESTIONS: List[Tuple[str, str, int]] = [
    ("1.1", "Was the allocation sequence random?", 1),
    ("1.2", "Was the allocation sequence concealed until participants were enrolled and assigned to interventions?", 1),
    ("1.3", "Did baseline differences between intervention groups suggest a problem with the randomisation process?", 1),
    ("2.1", "Were participants aware of their assigned intervention during the trial?", 2),
    ("2.2", "Were carers and people delivering the interventions aware of participants' assigned intervention?", 2),
    ("2.3", "If Y/PY/NI to 2.1 or 2.2: Were there deviations from the intended intervention that arose because of "
            "the trial context?", 2),
    ("2.4", "If Y/PY to 2.3: Were these deviations likely to have affected the outcome?", 2),
    ("2.5", "If Y/PY/NI to 2.4: Were these deviations from intended intervention balanced between groups?", 2),
    ("2.6", "Was an appropriate analysis used to estimate the effect of assignment to intervention (e.g. intention "
            "to treat)?", 2),
    ("2.7", "If N/PN/NI to 2.6: Was there potential for a substantial impact (on the result) of the failure to "
            "analyse participants in the group to which they were randomised?", 2),
    ("3.1", "Were data for this outcome available for all, or nearly all, participants randomised?", 3),
    ("3.2", "If N/PN/NI to 3.1: Is there evidence that the result was not biased by missing outcome data?", 3),
    ("3.3", "If N/PN to 3.2: Could missingness in the outcome depend on its true value?", 3),
    ("3.4", "If Y/PY/NI to 3.3: Is it likely that missingness in the outcome depended on its true value?", 3),
    ("4.1", "Was the method of measuring the outcome inappropriate?", 4),
    ("4.2", "Could measurement or ascertainment of the outcome have differed between intervention groups?", 4),
    ("4.3", "If N/PN/NI to 4.1 and 4.2: Were outcome assessors aware of the intervention received by study "
            "participants?", 4),
    ("4.4", "If Y/PY/NI to 4.3: Could assessment of the outcome have been influenced by knowledge of intervention "
            "received?", 4),
    ("4.5", "If Y/PY/NI to 4.4: Is it likely that assessment of the outcome was influenced by knowledge of "
            "intervention received?", 4),
    ("5.1", "Were the data that produced this result analysed in accordance with a pre-specified analysis plan that "
            "was finalised before unblinded outcome data were available for analysis?", 5),
    ("5.2", "Is the numerical result being assessed likely to have been selected, on the basis of the results, from "
            "multiple eligible outcome measurements within the outcome domain?", 5),
    ("5.3", "Is the numerical result being assessed likely to have been selected, on the basis of the results, from "
            "multiple eligible analyses of the data?", 5),
]

DOMAINS = {
    1: "Randomisation process",
    2: "Deviations from intended interventions",
    3: "Missing outcome data",
    4: "Measurement of the outcome",
    5: "Selection of the reported result",
}

LOW, SOME, HIGH = "Low", "Some concerns", "High"
_ORDER = {LOW: 0, SOME: 1, HIGH: 2}


def _a(answers: Dict[str, str], qid: str) -> str:
    v = (answers.get(qid) or "NI").strip().upper()
    return v if v in ANSWERS else "NI"


def domain_1(a: Dict[str, str]) -> str:
    q1, q2, q3 = _a(a, "1.1"), _a(a, "1.2"), _a(a, "1.3")
    if q2 in NO:
        return HIGH
    if q2 in YES:
        if q1 in NO:
            return SOME
        return SOME if q3 in YES else LOW
    # Allocation concealment not reported
    return HIGH if q3 in YES else SOME


def domain_2(a: Dict[str, str]) -> str:
    q1, q2, q3, q4, q5 = (_a(a, q) for q in ("2.1", "2.2", "2.3", "2.4", "2.5"))
    q6, q7 = _a(a, "2.6"), _a(a, "2.7")
    # Part 1: deviations from intended interventions
    if q1 in NO and q2 in NO:
        part1 = LOW
    elif q3 in NO or q3 == "NA":
        part1 = LOW
    elif q3 == "NI":
        part1 = SOME
    elif q4 in NO or q4 == "NA":
        part1 = SOME
    elif q5 in NO:
        part1 = HIGH
    else:
        part1 = SOME
    # Part 2: analysis used to estimate the effect of assignment
    if q6 in YES:
        part2 = LOW
    elif q7 in NO:
        part2 = SOME
    else:
        part2 = HIGH
    return max(part1, part2, key=_ORDER.get)


def domain_3(a: Dict[str, str]) -> str:
    q1, q2, q3, q4 = (_a(a, q) for q in ("3.1", "3.2", "3.3", "3.4"))
    if q1 in YES:
        return LOW
    if q2 in YES:
        return LOW
    if q3 in NO:
        return LOW
    if q4 in NO:
        return SOME
    return HIGH


def domain_4(a: Dict[str, str]) -> str:
    q1, q2, q3, q4, q5 = (_a(a, q) for q in ("4.1", "4.2", "4.3", "4.4", "4.5"))
    if q1 in YES or q2 in YES:
        return HIGH
    if q3 in NO:
        result = LOW
    elif q4 in NO:
        result = LOW
    elif q5 in NO:
        result = SOME
    else:
        result = HIGH
    # Without information on whether measurement could differ between groups, the domain is at least some concerns
    if result == LOW and (q1 == "NI" or q2 == "NI"):
        result = SOME
    return result


def domain_5(a: Dict[str, str]) -> str:
    q1, q2, q3 = (_a(a, q) for q in ("5.1", "5.2", "5.3"))
    if q2 in YES or q3 in YES:
        return HIGH
    if q2 in NO and q3 in NO:
        return LOW if q1 in YES else SOME
    return SOME


DOMAIN_RULES = {1: domain_1, 2: domain_2, 3: domain_3, 4: domain_4, 5: domain_5}


def judge(answers: Dict[str, str]) -> Tuple[Dict[int, str], str]:
    """Domain judgements and the overall judgement (High if any domain is High, Low if all are Low, otherwise Some
    concerns; the guidance allows reviewers to judge several Some concerns as High, which is left to the reviewer)."""
    domains = {d: rule(answers) for d, rule in DOMAIN_RULES.items()}
    values = list(domains.values())
    overall = HIGH if HIGH in values else (LOW if all(v == LOW for v in values) else SOME)
    return domains, overall
