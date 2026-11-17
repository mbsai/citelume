"""
Title and abstract screening by the model, for systematic-style and scoping reviews.

Word matching (services/paper_selection.py) decides which records could be eligible; it cannot tell a trial of
vitamin D on fractures from a trial of vitamin D on muscle strength that mentions fractures in passing. Here the
model screens those candidates against the review protocol's criteria, in score order and in parallel batches,
until the study limit is filled.

For every record the model answers each criterion separately (yes, no or unclear) with a short reason. A fixed rule
then decides: any "no" excludes, and "unclear" counts as include (a screening stage should not lose studies whose
abstract is silent). Every exclusion names the criterion that failed and is counted under it in the flow diagram.
The answers are stored with the model, the prompt version and the date, so a stored review replays its screening
exactly; the user may overturn any decision, with a reason, and that is recorded too. Without a model the
word-based eligibility stands and the report says so.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator

from models import ApiKeysConfig, PaperRecord, ReviewProtocol, ScreeningDecisionRecord, SelectionReport
from services.llm_service import execute_validated_llm_task
from services.paper_selection import EXCLUSION_STAGE, STUDY_TYPE_LABELS, _listed

SCREENING_PROMPT_VERSION = "citelume-screening/2.0"
# Used when a decision names no criterion (decisions stored before criteria were answered separately)
REASON_AI_SCREEN = "Not about the question on title and abstract screening (judged by the model)"
REASON_USER_EXCLUDED = "Excluded by the user after reviewing the model's screening"
EXCLUSION_STAGE[REASON_AI_SCREEN] = "eligibility"
EXCLUSION_STAGE[REASON_USER_EXCLUDED] = "eligibility"

SCREEN_BATCH = 12            # records per screening call
SCREEN_CONCURRENCY = 4       # calls at the same time
MAX_SCREENED = 240           # at least this many records may be screened per review
SCREEN_ABSTRACT_CHARS = 420
SCREEN_DEADLINE_S = 75.0


def criterion_reason(criterion: str) -> str:
    """The flow-diagram reason for records the model excluded on this criterion."""
    return f"Model screening: does not meet \"{criterion}\""


ROLE_WORDS = {"intervention": "Intervention", "exposure": "Exposure", "outcome": "Outcome",
              "population": "Population", "topic": "Topic", "method": "Method"}


def screening_criteria(protocol: Optional[ReviewProtocol], method: str) -> List[Tuple[str, str]]:
    """(label, question) for each criterion the model answers: one per searched concept, one per population or
    place, and the study design."""
    if protocol is None:
        return []
    out: List[Tuple[str, str]] = []
    for c in protocol.concepts:
        if c.role == "comparison" or not c.name.strip():
            continue
        role = ROLE_WORDS.get(c.role, "Topic")
        names = c.name + (f" (also called {', '.join(c.synonyms)})" if c.synonyms else "")
        if c.role == "outcome":
            question = f"{role}: the study reports results for {names}, as a primary or secondary outcome"
        else:
            question = f"{role}: the study is about {names}, not only mentioning it in passing"
        out.append((f"{role}: {c.name}", question))
    for p in protocol.populations_and_places:
        if p.name.strip():
            kind = "Place" if p.kind == "place" else "Population"
            examples = ", ".join(p.spellings[:4])
            out.append((f"{kind}: {p.name}",
                        f"{kind}: the study is in {p.name} (or a closely related group, e.g. {examples})"))
    designs = protocol.eligibility.study_designs
    if designs:
        labels = ", ".join(STUDY_TYPE_LABELS.get(d, d).lower() for d in designs)
        out.append(("Design: " + labels, f"Design: the study is one of these: {labels}"))
    elif method == "systematic_prisma":
        out.append(("Design: primary study", "Design: a primary study reporting its own data (not a review, "
                                             "editorial or study protocol)"))
    return out


class ScreeningDecision(BaseModel):
    n: int
    answers: List[str] = Field(default_factory=list)
    reason: str = ""

    @field_validator("answers", mode="before")
    @classmethod
    def _normalise(cls, v):
        if isinstance(v, str):
            v = [x for x in v.replace(";", ",").split(",") if x.strip()]
        out = []
        for a in v or []:
            a = str(a.get("answer", "") if isinstance(a, dict) else a).strip().lower()
            if a.startswith("y") or a == "met":
                out.append("yes")
            elif a.startswith("no") and not a.startswith("not sure") and not a.startswith("not clear"):
                out.append("no")
            else:
                out.append("unclear")
        return out


class ScreeningBatchOutput(BaseModel):
    decisions: List[ScreeningDecision] = Field(default_factory=list)


SYSTEM_PROMPT = (
    "You are screening records for a systematic review, on title and abstract only.\n"
    "For each record answer every criterion, in the order given, with 'yes' (clearly met), 'no' (clearly not met: "
    "the study is about something else, or the topic is only background or mentioned in passing) or 'unclear' "
    "(the title and abstract do not say). Do not guess: use 'unclear' rather than 'no' when the abstract is silent. "
    "Judge each criterion on its own: a review or editorial about the intervention still meets the intervention "
    "criterion; only the design criterion is about what kind of document it is. "
    "Give a reason in under 15 words, naming the criterion that fails if any.\n"
    'Return ONLY valid JSON: {"decisions": [{"n": 1, "answers": ["yes", "no", "unclear"], "reason": "..."}]}'
)


# When several criteria fail, the reason named follows the usual order for reporting exclusions: design, then
# population or place, then intervention, exposure or topic, then outcome
REASON_ORDER = ("Design", "Population", "Place", "Intervention", "Exposure", "Topic", "Method", "Outcome")


def _rank(label: str) -> int:
    head = label.split(":", 1)[0].strip()
    return REASON_ORDER.index(head) if head in REASON_ORDER else len(REASON_ORDER)


def decide(answers: List[str], criteria: List[str]) -> Tuple[bool, str]:
    """The fixed rule: excluded if any criterion is answered "no"; the reason named is the failing criterion that
    comes first in REASON_ORDER (then in the order given)."""
    failed = [label for label, a in zip(criteria, answers) if a == "no"]
    if not failed:
        return True, ""
    return False, min(failed, key=lambda lb: (_rank(lb), criteria.index(lb)))


async def _screen_batch(query: str, batch: Sequence[PaperRecord], config: ApiKeysConfig,
                        criteria: List[Tuple[str, str]]) -> Optional[Dict[str, ScreeningDecisionRecord]]:
    n_crit = len(criteria)
    lines = [f"{i}. {p.title} ({p.year or 'n.d.'}). Abstract: {(p.abstract or '')[:SCREEN_ABSTRACT_CHARS]}"
             for i, p in enumerate(batch, 1)]
    crit_lines = [f"{k}. {q}" for k, (_, q) in enumerate(criteria, 1)]

    def validate(out: ScreeningBatchOutput) -> List[str]:
        got = {d.n for d in out.decisions}
        missing = [i for i in range(1, len(batch) + 1) if i not in got]
        errs = [f"Give a decision for every record; missing: {missing}."] if missing else []
        wrong = [d.n for d in out.decisions if len(d.answers) != n_crit]
        if wrong:
            errs.append(f"Each record needs exactly {n_crit} answers, one per criterion; records {wrong} do not.")
        return errs

    def repair(out: ScreeningBatchOutput, _errors: List[str]) -> Optional[ScreeningBatchOutput]:
        # A short answer list is completed with "unclear" (which never excludes); a record left out is not guessed
        fixed = out.model_copy(deep=True)
        for d in fixed.decisions:
            d.answers = (d.answers + ["unclear"] * n_crit)[:n_crit]
        return fixed if {d.n for d in fixed.decisions} >= set(range(1, len(batch) + 1)) else None

    out, rec = await execute_validated_llm_task(
        task_id="Task_S",
        task_name=f"Task S: Title/abstract screening by criterion ({len(batch)} records)",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=(f"Review question: {query}\n\nCriteria:\n" + "\n".join(crit_lines)
                     + "\n\nRecords:\n" + "\n".join(lines)),
        response_model=ScreeningBatchOutput,
        config=config,
        custom_validator=validate,
        max_retries=1,
        temperature=0.0,
        max_tokens=(30 + 6 * n_crit) * len(batch) + 150,
        deadline_s=None if config.llm_provider in ("ollama", "custom") else SCREEN_DEADLINE_S,
        repair=repair,
    )
    if out is None or rec.degraded_to_deterministic:
        return None
    labels = [label for label, _ in criteria]
    decisions: Dict[str, ScreeningDecisionRecord] = {}
    for d in out.decisions:
        if 1 <= d.n <= len(batch):
            p = batch[d.n - 1]
            answers = (d.answers + ["unclear"] * n_crit)[:n_crit]
            keep, failed = decide(answers, labels)
            decisions[p.id] = ScreeningDecisionRecord(record_id=p.id, title=p.title or "", include=keep,
                                                      reason=d.reason[:160], answers=answers, failed_criterion=failed)
    return decisions


async def screen_and_limit(
    candidates: List[PaperRecord],
    report: SelectionReport,
    query: str,
    config: ApiKeysConfig,
    limit: int,
    criteria: Optional[List[Tuple[str, str]]] = None,
) -> List[PaperRecord]:
    """
    `candidates` are the word-eligible records in score order (paper_selection with a large limit). Screens them in
    waves, against `criteria` (from `screening_criteria`), until `limit` are included, then numbers the included
    studies and updates the report. Returns them.
    """
    criteria = criteria or [("Question", f"The study addresses the question: {query}")]
    included: List[PaperRecord] = []
    excluded: List[PaperRecord] = []
    log: List[ScreeningDecisionRecord] = []
    screened = 0
    failed_batches = 0
    pos = 0
    # Larger reviews may need more records screened to fill their limit
    cap = max(MAX_SCREENED, 2 * limit)
    while len(included) < limit and pos < len(candidates) and screened < cap:
        wave = []
        while len(wave) < SCREEN_CONCURRENCY and pos < len(candidates) and screened + sum(map(len, wave)) < cap:
            wave.append(candidates[pos:pos + SCREEN_BATCH])
            pos += SCREEN_BATCH
        results = await asyncio.gather(*[_screen_batch(query, b, config, criteria) for b in wave])
        for batch, decisions in zip(wave, results):
            screened += len(batch)
            if decisions is None:
                failed_batches += 1
                included.extend(batch)  # the word-based decision stands for this batch
                log.extend(ScreeningDecisionRecord(record_id=p.id, title=p.title or "", include=True, screened=False,
                                                   reason="Not screened: the model gave no usable answer")
                           for p in batch)
                continue
            for p in batch:
                d = decisions.get(p.id) or ScreeningDecisionRecord(
                    record_id=p.id, title=p.title or "", include=True, screened=False,
                    reason="Not screened: the model left this record out of its answer")
                log.append(d)
                (included if d.include else excluded).append(p)
        if failed_batches and failed_batches == len(wave) and not excluded:
            # No model available: the word-based eligibility stands for every remaining candidate too
            rest = candidates[pos:]
            included.extend(rest)
            log.extend(ScreeningDecisionRecord(record_id=p.id, title=p.title or "", include=True, screened=False,
                                               reason="Not screened: no model available") for p in rest)
            pos = len(candidates)
            break
    model = f"{config.llm_provider}:{config.llm_model}" if any(d.screened for d in log) else ""
    if model:
        report.screening_criteria = [label for label, _ in criteria]
        report.screening_prompt_version = SCREENING_PROMPT_VERSION
        report.screening_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return apply_screening_decisions(candidates, report, log, limit, model)


def apply_screening_decisions(
    candidates: List[PaperRecord],
    report: SelectionReport,
    log: List[ScreeningDecisionRecord],
    limit: int,
    model: str = "",
) -> List[PaperRecord]:
    """
    Applies recorded screening decisions to the candidates (in score order): numbers the included studies up to
    `limit`, lists the rest, and updates the report. It calls no model, so a stored corpus replays its screening
    exactly from the decisions it kept.
    """
    by_id = {d.record_id: d for d in log}
    # The user's decision, where they reviewed one, overrides the model's
    included = [p for p in candidates if p.id in by_id and by_id[p.id].effective_include]
    excluded = [p for p in candidates if p.id in by_id and not by_id[p.id].effective_include]
    unscreened = [p for p in candidates if p.id not in by_id]
    not_answered = sum(1 for d in log if not d.screened)
    for p in excluded:
        d = by_id[p.id]
        p.relevance = "off_topic"
        p.relevance_note = (f"Excluded by the user: {d.override_reason}" if d.user_override is False
                            else f"Excluded at screening ({d.failed_criterion or 'the question'}): {d.reason}")

    chosen = included[:limit]
    over = included[limit:] + unscreened
    for i, p in enumerate(chosen, start=1):
        p.citation_index = i
        p.selection_reason = "Meets the eligibility criteria and passed title/abstract screening"
    report.screening_decisions = list(log)
    report.screening_model = model
    report.included = len(chosen)
    report.target_count = limit
    report.eligible = len(included) + len(unscreened)
    # Each exclusion is counted under the criterion that failed (or as the user's decision)
    for p in excluded:
        d = by_id[p.id]
        reason = (REASON_USER_EXCLUDED if d.user_override is False
                  else criterion_reason(d.failed_criterion) if d.failed_criterion else REASON_AI_SCREEN)
        report.excluded[reason] = report.excluded.get(reason, 0) + 1
        report.exclusion_stage[reason] = "eligibility"
    report.eligible_over_limit = [_listed(p) for p in over]
    report.included_by_reason = {"Meets the eligibility criteria": len(chosen)}
    for p in chosen:
        d = by_id.get(p.id)
        if d is not None and d.user_override is True and not d.include:
            p.selection_reason = f"Included by the user after the model excluded it: {d.override_reason}"
    overturned = [d for d in log if d.user_override is not None and d.user_override != d.include]
    if overturned:
        report.notes.append(f"The user reviewed the model's screening and overturned {len(overturned)} decision(s), "
                            "each with a recorded reason.")
    # The selection step's notes about the limit are replaced by the ones below
    report.notes = [n for n in report.notes if not n.startswith("All ") and "met the eligibility criteria;" not in n]
    if any(d.screened for d in log):
        report.notes.append(
            f"The model screened {len(log)} word-eligible records on title and abstract and excluded {len(excluded)} "
            "that did not address the question" + (f"; {not_answered} record(s) could not be screened and were kept"
                                                   if not_answered else "") + ".")
    else:
        report.notes.append("Title/abstract screening by the model was not available, so eligibility rests on the "
                            "word-based criteria alone.")
    if over:
        report.notes.append(
            f"{report.eligible} studies met the eligibility criteria; the {len(chosen)} with the highest scores were "
            f"synthesised and the other {len(over)} are listed as eligible but not synthesised"
            + (f" ({len(unscreened)} of them were not screened by the model because the limit was already filled)"
               if unscreened else "") + ".")
    else:
        report.notes.append(f"All {len(chosen)} eligible studies were included (the limit is {limit}).")
    return chosen
