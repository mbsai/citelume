import math
import re
from typing import List, Dict, Optional, Tuple
from services.question_parsing import GENERIC_CONCEPT_WORDS, concept_blocks, concept_mentions, is_method_concept
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from models import (
    PaperRecord,
    HFResourceRecord,
    GraphNode,
    GraphEdge,
    LiteratureGraph,
    UniversalQuestionUnderstanding,
    QueryDecompositionPlan,
    SynthesisEvidencePackage,
    StructuredEvidenceMatrix,
    KnowledgeGapMatrix,
)
from services.corpus import _is_preprint_record, _normalize_title, merge_duplicates  # noqa: F401 (re-exported)
from services.network_engine import build_multilayer_literature_network
from services.paper_selection import MIN_CONSTRAINT_MATCHES, select_studies


def _topic_relevance(concept: str, title: str, abstract: str) -> str:
    """
    Rule-based topic check for the question's main concept (the model's screening refines it when available):
    in the title or mentioned at least twice in the abstract -> "addresses"; once -> "partly"; never -> "off_topic".
    A metadata-only record (no abstract) is judged on its title alone, without penalising the missing abstract.
    """
    if not concept:
        return "addresses"

    def mentions(text: str) -> int:
        return concept_mentions(concept, text)

    if mentions(title):
        return "addresses"
    if not abstract.strip():
        return "partly"
    n = mentions(abstract)
    return "addresses" if n >= 2 else ("partly" if n == 1 else "off_topic")


def core_topic_concepts(phrases: List[str], limit: int = 3) -> List[str]:
    """The question's leading subject concepts (not methods, not purely generic words) used to screen studies."""
    core = [p for p in phrases if not is_method_concept(p) and any(w not in GENERIC_CONCEPT_WORDS for w in p.split())]
    return (core or list(phrases))[:limit]


def _topic_relevance_multi(concepts: List[str], title: str, abstract: str,
                           main_alternatives: Optional[List[str]] = None) -> Tuple[str, List[str]]:
    """
    Topic check against several concepts. With one concept it is the single-concept rule. With more, a study that
    covers at least two of them addresses the question, one that covers only the first (main) concept partly does,
    and anything else is off topic. Returns the level and the concepts found.
    """
    if len(concepts) <= 1:
        c = concepts[0] if concepts else ""
        level = _topic_relevance(c, title, abstract)
        return level, ([c] if c and level != "off_topic" else [])
    found = [c for c in concepts if concept_mentions(c, f"{title} {abstract}")]
    mains = set(main_alternatives or concepts[:1])
    # Fully on topic: one of the main concepts AND another part of the question (two side-by-side alternatives,
    # e.g. BCAA and arginine in a cancer paper, are not enough)
    if any(c in mains for c in found) and any(c not in mains for c in found):
        return "addresses", found
    if any(c in mains for c in found):
        return "partly", found
    return "off_topic", found


def deduplicate_and_rank_papers(
    raw_papers: List[PaperRecord],
    query: str,
    max_total: int = 20,
    understanding: Optional[UniversalQuestionUnderstanding] = None,
    stats: Optional[Dict[str, int]] = None,
    report_out: Optional[list] = None,
    method: str = "narrative",
) -> List[PaperRecord]:
    """
    Merges duplicate records, then applies the CiteLume selection protocol (eligibility, 100-point score and
    balanced selection; see services/paper_selection.py and docs/RANKING.md). `report_out`, when given, receives
    the SelectionReport describing what was excluded and why each study was included.

    Phase 6 Multi-Criteria Study Deduplication & Selection:
    1. Deduplicates across DOI, arXiv ID, PMID, S2 ID, and normalized title + first-author
       matching to link preprint and published versions of the same underlying study.
    2. Scores candidate studies across 7 dimensions (direct relevance, subquestion coverage,
       methodological suitability, evidence independence, contradictory evidence preservation,
       foundational relevance, and recency balance) rather than citation count alone.
    3. Caps at max_total (20 for Standard Synthesis, 60 for Deep Research) without artificial padding.
    """
    unique_papers = merge_duplicates(raw_papers)
    if stats is not None:
        # Reported in the PRISMA-style flow: records left after duplicate removal, before selection
        stats["unique_records"] = len(unique_papers)
    if not unique_papers:
        return []

    # Eligibility, scoring and balanced selection follow the documented protocol (services/paper_selection.py)
    chosen, report = select_studies(unique_papers, query, max_total, understanding, stats,
                                    records_found=len(raw_papers), method=method)
    if report_out is not None:
        report_out.append(report)
    return chosen


def build_literature_graph(
    papers: List[PaperRecord],
    hf_resources: List[HFResourceRecord],
    query: str = "",
    mode: str = "standard",
    decomposition_plan: Optional[QueryDecompositionPlan] = None,
    synthesis_package: Optional[SynthesisEvidencePackage] = None,
    evidence_matrix: Optional[StructuredEvidenceMatrix] = None,
    gap_matrix: Optional[KnowledgeGapMatrix] = None,
    context_papers: Optional[List[PaperRecord]] = None,
) -> LiteratureGraph:
    """
    Universal, Multilayer Scientific Literature Mapping and Knowledge-Relationship Engine (Section 4):
    Delegates to `build_multilayer_literature_network` to construct Layer A (Bibliographic),
    Layer B (Semantic/Conceptual), Layer C (Scientific Claims, Shared Datasets & Gaps),
    Layer D (Research Evolution & Lineage), and Artifact Links with deterministic verification.
    """
    return build_multilayer_literature_network(
        papers=papers,
        hf_resources=hf_resources,
        query=query,
        mode=mode,
        decomposition_plan=decomposition_plan,
        synthesis_package=synthesis_package,
        evidence_matrix=evidence_matrix,
        gap_matrix=gap_matrix,
        context_papers=context_papers,
    )
