"""
Title and abstract screening by the model for a systematic-style (PRISMA) review.

Word matching (services/paper_selection.py) decides which records could be eligible; it cannot tell a trial of
vitamin D on fractures from a trial of vitamin D on muscle strength that mentions fractures in passing. Here the
model screens those candidates against the question, in score order and in parallel batches, until the study limit
is filled. Each exclusion is counted at the eligibility stage of the PRISMA flow, so nothing disappears silently.
If no model is available, or screening fails, the word-based eligibility stands and the report says so.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator

from models import ApiKeysConfig, PaperRecord, ScreeningDecisionRecord, SelectionReport
from services.llm_service import execute_validated_llm_task
from services.paper_selection import EXCLUSION_STAGE, _listed

REASON_AI_SCREEN = "Not about the question on title and abstract screening (judged by the model)"
EXCLUSION_STAGE[REASON_AI_SCREEN] = "eligibility"

SCREEN_BATCH = 15            # records per screening call
SCREEN_CONCURRENCY = 4       # calls at the same time
MAX_SCREENED = 240           # at most this many records are screened per review
SCREEN_ABSTRACT_CHARS = 420
SCREEN_DEADLINE_S = 60.0


class ScreeningDecision(BaseModel):
    n: int
    include: Literal["yes", "no"] = "yes"
    reason: str = ""

    @field_validator("include", mode="before")
    @classmethod
    def _yes_no(cls, v):
        return "no" if str(v).strip().lower() in ("no", "exclude", "false", "n") else "yes"


class ScreeningBatchOutput(BaseModel):
    decisions: List[ScreeningDecision] = Field(default_factory=list)


SYSTEM_PROMPT = (
    "You are screening records for a systematic review, on title and abstract only.\n"
    "Include a record ('yes') if it is a primary study that reports results for the question's outcome (as a "
    "primary or secondary outcome) for the question's intervention or exposure, in the question's population or a "
    "closely related one. Exclude ('no') records that do not study that outcome (it is only background or "
    "mentioned in passing), study a different intervention or exposure, or a clearly different population. When "
    "the abstract is unclear but the study could report the outcome, include it.\n"
    'Return ONLY valid JSON: {"decisions": [{"n": 1, "include": "yes|no", "reason": "under 12 words"}]}'
)


async def _screen_batch(query: str, batch: Sequence[PaperRecord], config: ApiKeysConfig
                        ) -> Optional[Dict[str, Tuple[bool, str]]]:
    lines = [f"{i}. {p.title} ({p.year or 'n.d.'}). Abstract: {(p.abstract or '')[:SCREEN_ABSTRACT_CHARS]}"
             for i, p in enumerate(batch, 1)]

    def validate(out: ScreeningBatchOutput) -> List[str]:
        got = {d.n for d in out.decisions}
        missing = [i for i in range(1, len(batch) + 1) if i not in got]
        return [f"Give a decision for every record; missing: {missing}."] if missing else []

    out, rec = await execute_validated_llm_task(
        task_id="Task_S",
        task_name=f"Task S: Title/abstract screening ({len(batch)} records)",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"Review question: {query}\n\nRecords:\n" + "\n".join(lines),
        response_model=ScreeningBatchOutput,
        config=config,
        custom_validator=validate,
        max_retries=1,
        temperature=0.0,
        max_tokens=40 * len(batch) + 120,
        deadline_s=None if config.llm_provider in ("ollama", "custom") else SCREEN_DEADLINE_S,
    )
    if out is None or rec.degraded_to_deterministic:
        return None
    decisions = {}
    for d in out.decisions:
        if 1 <= d.n <= len(batch):
            decisions[batch[d.n - 1].id] = (d.include == "yes", d.reason[:120])
    return decisions


async def screen_and_limit(
    candidates: List[PaperRecord],
    report: SelectionReport,
    query: str,
    config: ApiKeysConfig,
    limit: int,
) -> List[PaperRecord]:
    """
    `candidates` are the word-eligible records in score order (paper_selection with a large limit). Screens them in
    waves until `limit` are included, then numbers the included studies and updates the report. Returns them.
    """
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
        results = await asyncio.gather(*[_screen_batch(query, b, config) for b in wave])
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
                keep, reason = decisions.get(p.id, (True, ""))
                log.append(ScreeningDecisionRecord(record_id=p.id, title=p.title or "", include=keep, reason=reason))
                (included if keep else excluded).append(p)
        if failed_batches and failed_batches == len(wave) and not excluded:
            # No model available: the word-based eligibility stands for every remaining candidate too
            rest = candidates[pos:]
            included.extend(rest)
            log.extend(ScreeningDecisionRecord(record_id=p.id, title=p.title or "", include=True, screened=False,
                                               reason="Not screened: no model available") for p in rest)
            pos = len(candidates)
            break
    model = f"{config.llm_provider}:{config.llm_model}" if any(d.screened for d in log) else ""
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
    included = [p for p in candidates if p.id in by_id and by_id[p.id].include]
    excluded = [p for p in candidates if p.id in by_id and not by_id[p.id].include]
    unscreened = [p for p in candidates if p.id not in by_id]
    not_answered = sum(1 for d in log if not d.screened)
    for p in excluded:
        p.relevance = "off_topic"
        p.relevance_note = f"Excluded at screening: {by_id[p.id].reason}".strip(": ")

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
    if excluded:
        report.excluded[REASON_AI_SCREEN] = len(excluded)
        report.exclusion_stage[REASON_AI_SCREEN] = "eligibility"
    report.eligible_over_limit = [_listed(p) for p in over]
    report.included_by_reason = {"Meets the eligibility criteria": len(chosen)}
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
