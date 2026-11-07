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
from services.network_engine import build_multilayer_literature_network


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def _is_preprint_record(p: PaperRecord) -> bool:
    v_low = (p.venue or "").lower()
    s_low = (p.source or "").lower()
    if s_low in ("arxiv", "biorxiv") or "biorxiv" in v_low or "medrxiv" in v_low or "arxiv" in v_low:
        return True
    if p.doi and p.doi.startswith("10.1101/"):
        return True
    return False


# Keep only studies matching the question's place/population constraints once at least this many match
MIN_CONSTRAINT_MATCHES = 5


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
) -> List[PaperRecord]:
    """
    Phase 6 Multi-Criteria Study Deduplication & Selection:
    1. Deduplicates across DOI, arXiv ID, PMID, S2 ID, and normalized title + first-author
       matching to link preprint and published versions of the same underlying study.
    2. Scores candidate studies across 7 dimensions (direct relevance, subquestion coverage,
       methodological suitability, evidence independence, contradictory evidence preservation,
       foundational relevance, and recency balance) rather than citation count alone.
    3. Caps at max_total (20 for Standard Synthesis, 60 for Deep Research) without artificial padding.
    """
    merged_by_key: Dict[str, PaperRecord] = {}
    doi_to_key: Dict[str, str] = {}
    arxiv_to_key: Dict[str, str] = {}
    pmid_to_key: Dict[str, str] = {}
    s2_to_key: Dict[str, str] = {}
    title_to_key: Dict[str, str] = {}

    for p in raw_papers:
        norm_title = _normalize_title(p.title)
        if not norm_title:
            continue

        clean_doi = p.doi.strip().lower() if p.doi else None
        clean_arxiv = re.sub(r"v\d+$", "", p.arxiv_id.strip()) if p.arxiv_id else None
        clean_pmid = p.pmid.strip() if p.pmid else None
        clean_s2 = p.s2_id.strip() if p.s2_id else None

        existing_key = None
        if clean_doi and clean_doi in doi_to_key:
            existing_key = doi_to_key[clean_doi]
        elif clean_arxiv and clean_arxiv in arxiv_to_key:
            existing_key = arxiv_to_key[clean_arxiv]
        elif clean_pmid and clean_pmid in pmid_to_key:
            existing_key = pmid_to_key[clean_pmid]
        elif clean_s2 and clean_s2 in s2_to_key:
            existing_key = s2_to_key[clean_s2]
        elif norm_title in title_to_key:
            existing_key = title_to_key[norm_title]

        is_prep = _is_preprint_record(p)

        if existing_key and existing_key in merged_by_key:
            curr = merged_by_key[existing_key]
            curr_was_prep = _is_preprint_record(curr)
            if p.source and p.source not in curr.all_sources:
                curr.all_sources.append(p.source)
            if not curr.doi and clean_doi:
                curr.doi = clean_doi
            elif clean_doi and not clean_doi.startswith("10.1101/") and (curr.doi or "").startswith("10.1101/"):
                # Prefer peer-reviewed journal DOI over bioRxiv preprint DOI
                curr.doi = clean_doi
                curr.venue = p.venue or curr.venue
            if not curr.arxiv_id and clean_arxiv:
                curr.arxiv_id = clean_arxiv
            if not curr.pmid and clean_pmid:
                curr.pmid = clean_pmid
            if not curr.s2_id and clean_s2:
                curr.s2_id = clean_s2
            if len(p.abstract or "") > len(curr.abstract or ""):
                curr.abstract = p.abstract
            if p.citation_count > curr.citation_count:
                curr.citation_count = p.citation_count
            if not curr.pdf_url and p.pdf_url:
                curr.pdf_url = p.pdf_url
            curr.references = list(dict.fromkeys(curr.references + p.references))
            curr.cited_by = list(dict.fromkeys(curr.cited_by + p.cited_by))

            # Track preprint + published linkage for the same underlying study
            if (curr_was_prep and not is_prep) or (not curr_was_prep and is_prep):
                curr.version_status = "preprint_with_published_link"
        else:
            key = p.id or f"study:{len(merged_by_key) + 1}"
            p.canonical_study_id = f"study:{norm_title[:32]}"
            p.version_status = "preprint" if is_prep else "published"
            if clean_doi:
                p.doi = clean_doi
            if clean_arxiv:
                p.arxiv_id = clean_arxiv
            if not p.all_sources and p.source:
                p.all_sources = [p.source]
            merged_by_key[key] = p
            title_to_key[norm_title] = key
            if clean_doi:
                doi_to_key[clean_doi] = key
            if clean_arxiv:
                arxiv_to_key[clean_arxiv] = key
            if clean_pmid:
                pmid_to_key[clean_pmid] = key
            if clean_s2:
                s2_to_key[clean_s2] = key

    unique_papers = list(merged_by_key.values())
    if stats is not None:
        # Reported in the PRISMA-style flow: records left after duplicate removal, before selection
        stats["unique_records"] = len(unique_papers)
    if not unique_papers:
        return []

    # 1. Direct Relevance via TF-IDF cosine similarity
    corpus = [f"{p.title}. {p.abstract or ''}" for p in unique_papers]
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=3500, ngram_range=(1, 2))
        mat = vec.fit_transform(corpus + [query])
        q_vec = mat[-1]
        doc_mat = mat[:-1]
        sims = cosine_similarity(doc_mat, q_vec).flatten()
    except Exception:
        sims = np.ones(len(unique_papers)) * 0.5

    # Subquestion keyword map for subquestion coverage tracking
    sq_keywords: Dict[str, List[str]] = {}
    if understanding and understanding.subquestions:
        for sq in understanding.subquestions:
            toks = [w.lower() for w in re.findall(r"[a-zA-Z0-9-]{4,}", sq.question_text)]
            sq_keywords[sq.id] = toks[:6]

    author_year_counts: Dict[str, int] = {}

    for idx, p in enumerate(unique_papers):
        text_low = f"{p.title} {p.abstract or ''}".lower()
        direct_rel = float(sims[idx])

        # 2. Subquestion coverage
        matched_sqs: List[str] = []
        for sq_id, kw_list in sq_keywords.items():
            if any(kw in text_low for kw in kw_list[:4]):
                matched_sqs.append(sq_id)
        if not matched_sqs:
            matched_sqs = ["SQ1"]
        p.matched_subquestion_ids = matched_sqs
        sq_score = min(0.18, len(matched_sqs) * 0.06)

        # 3. Methodological & Evidence Richness (abstract completeness, quantitative/methodological signals)
        has_abstract = 0.10 if len(p.abstract or "") > 140 else 0.0
        has_quant_or_proof = 0.06 if re.search(
            r"(\b\d+(?:\.\d+)?%|\bp\s*[<=>]\s*0\.\d+|\bauroc\b|\bauc\b|\btheorem\b|\bproof\b|\bmic\b|\bic50\b)",
            text_low,
        ) else 0.0
        method_score = has_abstract + has_quant_or_proof

        # 4. Contradictory / Robustness Evidence Preservation (ensure negative/conflicting studies are kept)
        contra_boost = 0.08 if re.search(
            r"\b(contradict|negative|null|fail|limitation|out-of-distribution|challenge|bottleneck|counterexample)\b",
            text_low,
        ) else 0.0

        # 5. Foundational Impact (bounded log scale so high citations don't drown out relevant new work)
        cite_boost = min(math.log10(max(p.citation_count, 1) + 1) * 0.09, 0.32)

        # 6. Recency & Peer-Review Balance
        recency_boost = 0.05 if (p.year and p.year >= 2023) else (0.03 if (p.year and p.year <= 2018 and p.citation_count >= 50) else 0.0)
        pub_boost = 0.04 if p.version_status in ("published", "preprint_with_published_link") else 0.0

        # 7. Evidence Independence (diminishing return if multiple papers share identical first-author + year)
        first_auth_key = f"{(p.authors[0].split()[-1].lower() if p.authors else 'anon')}_{p.year or 0}"
        seen_cnt = author_year_counts.get(first_auth_key, 0)
        author_year_counts[first_auth_key] = seen_cnt + 1
        independence_penalty = -0.08 * seen_cnt if seen_cnt > 0 else 0.0

        total_score = (
            direct_rel
            + sq_score
            + method_score
            + contra_boost
            + cite_boost
            + recency_boost
            + pub_boost
            + independence_penalty
        )
        p.relevance_score = round(max(total_score, 0.01), 4)
        p.selection_breakdown = {
            "direct_relevance": round(direct_rel, 3),
            "subquestion_coverage": round(sq_score, 3),
            "methodological_richness": round(method_score, 3),
            "contradictory_preservation": round(contra_boost, 3),
            "foundational_impact": round(cite_boost, 3),
            "recency_and_publication": round(recency_boost + pub_boost, 3),
            "independence_adjustment": round(independence_penalty, 3),
        }

    # Place/population constraints from the question: matching studies rank first; non-matching ones are dropped
    # when enough matching studies exist, otherwise kept and flagged so the synthesis can leave them out
    constraints = understanding.concepts.required_constraints if understanding else []
    topic_concepts = core_topic_concepts(understanding.concepts.concept_phrases or []) if understanding else []
    main_concept = topic_concepts[0] if topic_concepts else ""
    # Concepts the question lists side by side with the main one ("X and Y metabolism") count as main too
    first_block = (concept_blocks(understanding.normalized_query, topic_concepts) or [[]])[0] if understanding else []
    if constraints or main_concept:
        for p in unique_papers:
            text = f"{p.title} {p.abstract or ''}".lower()
            missing = [c.name for c in constraints if not any(sp in text for sp in c.spellings)]
            topic, found = _topic_relevance_multi(topic_concepts, p.title, p.abstract or "", first_block)
            if missing or topic == "off_topic":
                p.relevance = "off_topic"
                p.relevance_note = (f"Title/abstract does not mention {', '.join(missing)}" if missing
                                    else f"Not about {' / '.join(topic_concepts)}")
            else:
                p.relevance = topic
                p.relevance_note = (f"Matches {', '.join(c.name for c in constraints)}; " if constraints else "") + (
                    f"covers {', '.join(found)}" if topic == "addresses" else
                    f"mentions {main_concept} only" + (" in passing" if len(topic_concepts) == 1 else ""))
        matching = [p for p in unique_papers if p.relevance != "off_topic"]
        if stats is not None:
            stats["constraint_excluded"] = 0
        if len(matching) >= MIN_CONSTRAINT_MATCHES:
            if stats is not None:
                stats["constraint_excluded"] = len(unique_papers) - len(matching)
            unique_papers = matching

    unique_papers.sort(key=lambda x: (x.relevance != "off_topic", x.relevance_score), reverse=True)
    top_papers = unique_papers[:max_total]

    for i, p in enumerate(top_papers, start=1):
        p.citation_index = i

    return top_papers


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
