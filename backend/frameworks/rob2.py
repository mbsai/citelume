"""
RoB 2 for randomised trials (effect of assignment to the intervention): the signalling questions and the decision
algorithms that turn their answers into a judgement for each domain and overall.

Source: Sterne JAC, et al. RoB 2: a revised tool for assessing risk of bias in randomised trials. BMJ
2019;366:l4898, and the RoB 2 cribsheet and full guidance for parallel-group trials (Higgins JPT, Savovic J, Page MJ,
Sterne JAC, version of 22 August 2019). The questions are abridged here. Each domain function follows the
cribsheet's decision figure for that domain (checked branch by branch against it; see tests/test_risk_of_bias.py).
Only the effect of assignment to intervention (intention to treat) is covered, not the effect of adhering to it.

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


def _asked(answers: Dict[str, str], qid: str) -> str:
    """The answer to a question the algorithm reaches: 'not applicable' there means the information is missing."""
    v = _a(answers, qid)
    return "NI" if v == "NA" else v


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
    q1, q2 = _a(a, "2.1"), _a(a, "2.2")
    # Part 1 (2.1 to 2.5): deviations from intended interventions that arose because of the trial context
    if q1 in NO and q2 in NO:
        part1 = LOW
    else:
        q3 = _asked(a, "2.3")
        if q3 in NO:
            part1 = LOW
        elif q3 == "NI":
            part1 = SOME
        elif _asked(a, "2.4") in NO:
            part1 = SOME
        else:  # 2.4 Y/PY/NI
            part1 = SOME if _asked(a, "2.5") in YES else HIGH  # 2.5 N/PN/NI gives High
    # Part 2 (2.6 and 2.7): the analysis used to estimate the effect of assignment
    if _a(a, "2.6") in YES:
        part2 = LOW
    else:
        part2 = SOME if _asked(a, "2.7") in NO else HIGH  # 2.7 Y/PY/NI gives High
    return max(part1, part2, key=_ORDER.get)


def domain_3(a: Dict[str, str]) -> str:
    if _a(a, "3.1") in YES:
        return LOW
    if _asked(a, "3.2") in YES:  # 3.2 has no NI option; anything else follows the N/PN path
        return LOW
    if _asked(a, "3.3") in NO:
        return LOW
    return SOME if _asked(a, "3.4") in NO else HIGH  # 3.4 Y/PY/NI gives High


def domain_4(a: Dict[str, str]) -> str:
    # 4.1 Y/PY gives High; N/PN/NI moves on to 4.2
    if _a(a, "4.1") in YES:
        return HIGH
    q2 = _a(a, "4.2")
    if q2 in YES:
        return HIGH
    # 4.2 N/PN follows the upper path (best case Low); 4.2 NI follows the lower path (best case Some concerns)
    best = LOW if q2 in NO else SOME
    if _asked(a, "4.3") in NO:
        return best
    if _asked(a, "4.4") in NO:
        return best
    return SOME if _asked(a, "4.5") in NO else HIGH  # 4.5 Y/PY/NI gives High on both paths


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
