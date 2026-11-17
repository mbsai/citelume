"""
Risk of bias that can be checked: suggested judgements for the reviewer to confirm.

* Only studies whose full text was available are assessed (open access, or a PDF the user supplied); the others are
  "Not assessed (abstract only)": an abstract rarely reports randomisation, blinding or missing data.
* Every answer must come with a quote from the study. Code checks that the quote is in the study's text; an answer
  whose quote is not found counts as "No information".
* Randomised trials: the model answers the RoB 2 signalling questions; each domain and the overall judgement follow
  from those answers by the coded RoB 2 algorithm (frameworks/rob2.py), not from the model.
* Other designs: the model judges each domain of a design-matched tool (frameworks/risk_of_bias.py), with a quote;
  the overall judgement follows from the domains by a fixed rule.

Runs only in a systematic-style review, one study per call, alongside the synthesis.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field, field_validator

from frameworks import rob2
from frameworks.risk_of_bias import overall_judgement, tool_for
from models import ApiKeysConfig, PaperRecord, RiskOfBiasDomain, SignallingAnswer, StudyRiskOfBias
from services.fulltext import repair_ligatures
from services.llm_service import execute_validated_llm_task

ROB_PROMPT_VERSION = "citelume-rob/2.0"
ROB_CONCURRENCY = 4
# Methods and results carry most of what RoB 2 asks (allocation, blinding, participant flow, missing data)
EXCERPT_CHARS = {"methods": 9000, "results": 7000, "discussion": 1500, "body": 12000}
MIN_QUOTE_CHARS = 20
ROB_DEADLINE_S = 120.0


# ---------------------------------------------------------------------------------------------------------------
# Quote checking
# ---------------------------------------------------------------------------------------------------------------
def _norm(text: str) -> str:
    t = repair_ligatures(text or "").lower()
    t = t.replace("·", ".").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"[‐-―−]", "-", t)
    # Hyphens are dropped on both sides, so a word split across a PDF line ("supplementa-\ntion") still matches
    t = re.sub(r"-\s*", "", t)
    return re.sub(r"\s+", " ", t).strip()


def quote_found(quote: str, source: str) -> bool:
    """True when the quote (or each part of it around an ellipsis) appears in the study's text."""
    q = _norm(quote).strip(" \"'")
    if len(q) < MIN_QUOTE_CHARS:
        return False
    src = _norm(source)
    parts = [p.strip(" .") for p in re.split(r"\.\.\.|…|\[\.\.\.\]", q) if p.strip(" .")]
    return bool(parts) and all(len(p) >= 12 and p in src for p in parts)


def source_text(p: PaperRecord) -> str:
    return " ".join([p.title or "", p.abstract or ""] + list(p.full_text_sections.values()))


def _excerpt(p: PaperRecord) -> str:
    parts = [f"Abstract: {p.abstract or ''}"]
    for section, budget in EXCERPT_CHARS.items():
        text = p.full_text_sections.get(section) or ""
        if text:
            parts.append(f"{section.capitalize()}: {text[:budget]}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------------------------------------------
# Model replies
# ---------------------------------------------------------------------------------------------------------------
class _Answer(BaseModel):
    id: str
    answer: str = "NI"
    quote: str = ""

    @field_validator("answer", mode="before")
    @classmethod
    def _answer(cls, v):
        v = str(v or "").strip().upper().replace(" ", "")
        aliases = {"YES": "Y", "PROBABLYYES": "PY", "PROBABLYNO": "PN", "NO": "N", "NOINFORMATION": "NI",
                   "NOTAPPLICABLE": "NA"}
        v = aliases.get(v, v)
        return v if v in rob2.ANSWERS else "NI"


class _Rob2Reply(BaseModel):
    answers: List[_Answer] = Field(default_factory=list)
    main_concern: str = ""


class _Domain(BaseModel):
    judgement: str = "No information"
    quote: str = ""


class _DomainReply(BaseModel):
    domains: List[_Domain] = Field(default_factory=list)
    main_concern: str = ""


ROB2_PROMPT = (
    "You are assessing risk of bias in one randomised trial with the RoB 2 tool (effect of assignment to the "
    "intervention), using only the text provided.\n"
    "Answer every signalling question with Y, PY, PN, N, NI (no information) or NA (not applicable, when the "
    "question's condition is not met). For every answer other than NI and NA, give a quote of 1-2 sentences copied "
    "exactly from the text that supports it. If the text does not say, answer NI. Do not judge the domains: they "
    "are computed from your answers. Give main_concern in under 25 words.\n"
    'Return ONLY JSON: {"answers": [{"id": "1.1", "answer": "Y", "quote": "..."}], "main_concern": "..."}'
)

DOMAIN_PROMPT = (
    "You are assessing risk of bias in one study using only the text provided. For each domain, in the order "
    "given, judge Low, Some concerns, High or No information, and give a quote of 1-2 sentences copied exactly "
    "from the text that supports the judgement. If the text does not say, judge No information and leave the quote "
    "empty. Give main_concern in under 25 words.\n"
    'Return ONLY JSON: {"domains": [{"judgement": "Low", "quote": "..."}], "main_concern": "..."}'
)


def _judgement(v: str) -> str:
    v = (v or "").strip().lower()
    if v.startswith("low"):
        return "Low"
    if v.startswith("high") or v.startswith("serious") or v.startswith("critical"):
        return "High"
    if "concern" in v or v.startswith("moderate") or v.startswith("some"):
        return "Some concerns"
    return "No information"


def _stamp(rob: StudyRiskOfBias, config: ApiKeysConfig) -> StudyRiskOfBias:
    rob.assessed_by = f"{config.llm_provider}:{config.llm_model}"
    rob.prompt_version = ROB_PROMPT_VERSION
    rob.assessed_on = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return rob


def not_assessed(p: PaperRecord, why: str) -> StudyRiskOfBias:
    tool = tool_for(_design(p))
    return StudyRiskOfBias(tool=tool.label, overall="Not assessed", reason=why, basis="abstract")


def _design(p: PaperRecord) -> str:
    # PubMed's publication type is the most reliable sign of a randomised trial; then the study type by wording
    if any("randomized controlled trial" in t.lower() for t in (p.publication_types or [])):
        return "rct"
    return p.study_type or "primary_other"


async def _assess_trial(p: PaperRecord, config: ApiKeysConfig) -> Optional[StudyRiskOfBias]:
    questions = "\n".join(f"{qid}. {q}" for qid, q, _ in rob2.QUESTIONS)
    out, rec = await execute_validated_llm_task(
        task_id="Task_RoB",
        task_name=f"Risk of bias (RoB 2): [{p.citation_index}]",
        system_prompt=ROB2_PROMPT,
        user_prompt=f"Signalling questions:\n{questions}\n\nTrial: {p.title}\n\n{_excerpt(p)}",
        response_model=_Rob2Reply,
        config=config,
        max_retries=1,
        temperature=0.0,
        max_tokens=2200,
        deadline_s=None if config.llm_provider in ("ollama", "custom") else ROB_DEADLINE_S,
    )
    if out is None or rec.degraded_to_deterministic:
        return None
    src = source_text(p)
    given = {a.id.strip(): a for a in out.answers}
    signalling: List[SignallingAnswer] = []
    answers: Dict[str, str] = {}
    for qid, question, _ in rob2.QUESTIONS:
        a = given.get(qid)
        answer, quote = (a.answer, a.quote.strip()) if a else ("NI", "")
        verified = quote_found(quote, src) if quote else False
        # An answer the text cannot be shown to support counts as no information
        if answer not in ("NI", "NA") and not verified:
            answer = "NI"
        answers[qid] = answer
        signalling.append(SignallingAnswer(id=qid, question=question, answer=answer, quote=quote[:400],
                                           quote_verified=verified))
    domains, overall = rob2.judge(answers)
    rob_domains = []
    for d, name in rob2.DOMAINS.items():
        support = next((s for s in signalling if s.id.startswith(f"{d}.") and s.quote_verified), None)
        rob_domains.append(RiskOfBiasDomain(domain=name, judgement=domains[d], quote=support.quote if support else "",
                                            quote_verified=bool(support)))
    tool = tool_for("rct")
    return StudyRiskOfBias(tool=tool.label, domains=rob_domains, overall=overall, reason=out.main_concern[:220],
                           basis="full text", signalling=signalling)


async def _assess_other(p: PaperRecord, config: ApiKeysConfig) -> Optional[StudyRiskOfBias]:
    tool = tool_for(_design(p))
    domain_lines = "\n".join(f"{k}. {d}" for k, d in enumerate(tool.domains, 1))
    out, rec = await execute_validated_llm_task(
        task_id="Task_RoB",
        task_name=f"Risk of bias ({tool.label}): [{p.citation_index}]",
        system_prompt=DOMAIN_PROMPT,
        user_prompt=f"Tool: {tool.label}\nDomains:\n{domain_lines}\n\nStudy: {p.title}\n\n{_excerpt(p)}",
        response_model=_DomainReply,
        config=config,
        max_retries=1,
        temperature=0.0,
        max_tokens=1400,
        deadline_s=None if config.llm_provider in ("ollama", "custom") else ROB_DEADLINE_S,
    )
    if out is None or rec.degraded_to_deterministic:
        return None
    src = source_text(p)
    domains = []
    for k, name in enumerate(tool.domains):
        d = out.domains[k] if k < len(out.domains) else _Domain()
        judgement, quote = _judgement(d.judgement), d.quote.strip()
        verified = quote_found(quote, src) if quote else False
        if judgement != "No information" and not verified:
            judgement = "No information"
        domains.append(RiskOfBiasDomain(domain=name, judgement=judgement, quote=quote[:400], quote_verified=verified))
    return StudyRiskOfBias(tool=tool.label, domains=domains, overall=overall_judgement([d.judgement for d in domains]),
                           reason=out.main_concern[:220], basis="full text")


async def assess_risk_of_bias(papers: Sequence[PaperRecord], config: ApiKeysConfig) -> Dict[str, StudyRiskOfBias]:
    """Suggested judgements for every study (record id -> judgement). Studies without full text, and studies whose
    assessment failed, are marked Not assessed with the reason."""
    has_model = bool((config.llm_api_key or "").strip()) or config.llm_provider in ("ollama", "custom")
    sem = asyncio.Semaphore(ROB_CONCURRENCY)

    async def one(p: PaperRecord) -> StudyRiskOfBias:
        if not p.full_text_sections:
            return not_assessed(p, "Not assessed: only the abstract was available, and risk of bias needs the full text.")
        if not has_model:
            return not_assessed(p, "Not assessed: no AI model was configured.")
        async with sem:
            rob = await (_assess_trial(p, config) if _design(p) == "rct" else _assess_other(p, config))
        if rob is None:
            return not_assessed(p, "Not assessed: the model gave no usable answer for this study.")
        return _stamp(rob, config)

    results = await asyncio.gather(*[one(p) for p in papers if p.relevance != "off_topic"])
    return {p.id: r for p, r in zip([p for p in papers if p.relevance != "off_topic"], results)}


def apply_edit(rob: StudyRiskOfBias, domain: str, judgement: str, reason: str) -> StudyRiskOfBias:
    """The reviewer's change to one domain, recorded; the overall judgement is recomputed by the same rule."""
    from models import RiskOfBiasEdit

    target = next((d for d in rob.domains if d.domain == domain), None)
    if target is None:
        raise ValueError(f"No domain named {domain!r} for this study.")
    rob.edits.append(RiskOfBiasEdit(domain=domain, before=target.judgement, after=judgement, reason=reason.strip(),
                                    at=datetime.now(timezone.utc).isoformat(timespec="seconds")))
    target.judgement = judgement
    values = [d.judgement for d in rob.domains]
    if rob.signalling:  # RoB 2 overall rule
        rob.overall = "High" if "High" in values else ("Low" if all(v == "Low" for v in values) else "Some concerns")
    else:
        rob.overall = overall_judgement(values)
    rob.status = "edited by the user"
    return rob
