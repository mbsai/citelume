from typing import List, Optional, Dict, Any, Literal
import re

from pydantic import BaseModel, Field, field_validator


class ApiKeysConfig(BaseModel):
    llm_provider: Literal[
        "gemini",
        "openai",
        "anthropic",
        "openrouter",
        "huggingface",
        "ollama",
        "custom",
    ] = "openrouter"
    llm_api_key: str = ""
    llm_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    llm_base_url: str = ""
    # OpenRouter only: models tried in order when the primary model is rate-limited, down, or errors
    llm_fallback_models: List[str] = Field(default_factory=list)
    provider_keys: Dict[str, str] = Field(default_factory=dict)
    provider_models: Dict[str, str] = Field(default_factory=dict)
    provider_base_urls: Dict[str, str] = Field(default_factory=dict)

    # Literature & Hub API Keys (optional where public APIs exist, used when provided)
    pubmed_api_key: str = ""
    s2_api_key: str = ""
    # OpenAlex (fallback for Semantic Scholar) requires a free API key for regular use: openalex.org/settings/api
    openalex_api_key: str = ""
    hf_api_key: str = ""
    arxiv_api_key: str = ""
    biorxiv_api_key: str = ""

    # Enabled toggles
    enable_pubmed: bool = True
    enable_arxiv: bool = True
    enable_biorxiv: bool = True
    enable_s2: bool = True
    enable_hf: bool = True
    # Fetch open-access full text (Europe PMC, bioRxiv/medRxiv, arXiv, open PDFs) for the included studies
    enable_fulltext: bool = True


class ProviderModelsRequest(BaseModel):
    llm_provider: str = "gemini"
    llm_api_key: str = ""
    llm_base_url: str = ""
    hf_api_key: str = ""


class TestLlmRequest(BaseModel):
    llm_provider: str = "gemini"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_base_url: str = ""
    hf_api_key: str = ""


class TestSourceKeyRequest(BaseModel):
    source: Literal["pubmed", "s2", "arxiv", "biorxiv", "huggingface"]
    api_key: str = ""


class SearchRequest(BaseModel):
    query: str
    mode: Literal["standard", "deep_research"] = "standard"
    max_per_source: int = Field(default=8, ge=3, le=35)
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    config: ApiKeysConfig = Field(default_factory=ApiKeysConfig)
    # Client-generated id used to poll /api/search-progress/{progress_id} while the search runs
    progress_id: Optional[str] = None
    # Field chosen in the UI; None means auto-detect from the query
    domain: Optional["DomainCategoryType"] = None
    # Deep Research review type: how the review is reported, and how its findings section is organised
    review_framework: Literal["narrative", "systematic_prisma", "scoping"] = "narrative"
    organization: Literal["thematic", "chronological", "methodological", "theoretical"] = "thematic"


class ExpandNodeRequest(BaseModel):
    paper_id: str
    title: str
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    s2_id: Optional[str] = None
    config: ApiKeysConfig = Field(default_factory=ApiKeysConfig)


# ============================================================================
# PHASE 2 & 3: Universal Question Understanding, Domain & Methodological Profiles
# ============================================================================

ResearchIntentType = Literal[
    "descriptive",
    "explanatory_mechanistic",
    "causal_intervention",
    "associational",
    "predictive",
    "comparative",
    "methodological",
    "theoretical",
    "exploratory_mapping",
    "gap_discovery",
]

DomainCategoryType = Literal[
    "biomedical_clinical",
    "computer_science_ai",
    "chemistry",
    "physics",
    "mathematics",
    "engineering_materials",
    "social_sciences",
    "humanities_interdisciplinary",
    "interdisciplinary_biomedical_ai",
]

MethodologicalProfileType = Literal[
    "observational",
    "experimental",
    "clinical_intervention",
    "predictive_modeling",
    "ml_benchmarking",
    "omics_high_throughput",
    "theoretical_mathematical",
    "simulation_based",
    "qualitative",
    "literature_conceptual_review",
]


class AmbiguousTermNote(BaseModel):
    term: str
    candidate_meanings: List[str] = Field(default_factory=list)
    resolved_meaning: str = ""


class EntityRelationship(BaseModel):
    source_entity: str
    relation: str
    target_entity: str


class RequiredConstraint(BaseModel):
    """A place or population the question is restricted to, with the spellings papers may use."""
    kind: Literal["place", "population", "model_system"] = "place"
    name: str
    spellings: List[str] = Field(default_factory=list)


class TechnicalConceptExtraction(BaseModel):
    main_entities: List[str] = Field(default_factory=list)
    # Multi-word concepts kept intact for searching (e.g. "coronary artery disease")
    concept_phrases: List[str] = Field(default_factory=list)
    # Places / populations every search must include and every included study should match
    required_constraints: List[RequiredConstraint] = Field(default_factory=list)
    # Experimental systems the question asks to cover (mouse models, organoids, human samples); targeted, not required
    model_systems: List[RequiredConstraint] = Field(default_factory=list)
    normalized_terms: List[str] = Field(default_factory=list)
    synonyms_and_aliases: Dict[str, List[str]] = Field(default_factory=dict)
    entity_relationships: List[EntityRelationship] = Field(default_factory=list)
    outcomes_or_targets: List[str] = Field(default_factory=list)
    experimental_or_theoretical_context: str = ""
    constraints: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)


class ResearchSubquestion(BaseModel):
    id: str
    question_text: str
    relationship_to_main: str = ""
    category: Literal[
        "direct_evidence",
        "contextual_evidence",
        "comparative_evidence",
        "validation_robustness",
    ] = "direct_evidence"
    priority: Literal["high", "medium", "exploratory"] = "high"
    required_evidence: str = ""
    depends_on: List[str] = Field(default_factory=list)


class UniversalQuestionUnderstanding(BaseModel):
    # True when primary_domain was chosen by the user in the UI rather than auto-detected
    domain_selected_by_user: bool = False
    raw_query: str
    normalized_query: str
    question_complexity: Literal["narrow", "moderate", "broad_interdisciplinary"] = "moderate"
    primary_intent: ResearchIntentType = "descriptive"
    secondary_intents: List[ResearchIntentType] = Field(default_factory=list)
    primary_domain: DomainCategoryType = "biomedical_clinical"
    secondary_domains: List[DomainCategoryType] = Field(default_factory=list)
    active_methodological_profiles: List[MethodologicalProfileType] = Field(default_factory=list)
    ambiguous_terms: List[AmbiguousTermNote] = Field(default_factory=list)
    concepts: TechnicalConceptExtraction = Field(default_factory=TechnicalConceptExtraction)
    subquestions: List[ResearchSubquestion] = Field(default_factory=list)


# ============================================================================
# PHASE 4 & 5: Universal Search Planner & Source-Specific Query Compilation
# ============================================================================

SearchFacetCode = Literal[
    "F1_direct_evidence",
    "F2_foundational_contextual",
    "F3_alternative_contradictory",
    "F4_recent_developments",
    "F5_gap_verification",
    "F6_citation_network_expansion",
]


class FacetApiQueries(BaseModel):
    pubmed: str = ""
    semantic_scholar: str = ""
    arxiv: str = ""
    biorxiv_medrxiv: List[str] = Field(default_factory=list)
    hugging_face: str = ""
    hf_resource_targets: List[str] = Field(default_factory=lambda: ["models", "datasets"])


class SearchFacet(BaseModel):
    facet_code: str = "F1_direct_evidence"
    facet_name: str
    facet_intent: str = ""
    linked_subquestion_ids: List[str] = Field(default_factory=list)
    expansion_reason: str = "Primary retrieval"
    priority_rank: int = 1
    queries: FacetApiQueries = Field(default_factory=FacetApiQueries)


class UniversalSearchPlan(BaseModel):
    retrieval_budget_papers: int = 20
    active_facet_codes: List[str] = Field(default_factory=list)
    subquestion_coverage_map: Dict[str, List[str]] = Field(default_factory=dict)
    preferred_sources: List[str] = Field(default_factory=list)
    adaptive_expansion_triggered: bool = False
    expansion_notes: List[str] = Field(default_factory=list)


class QueryDecompositionPlan(BaseModel):
    # Backward-compatible fields expected by existing UI & callers
    detected_domain: str = "BIOMEDICAL/AI"
    extracted_technical_entities: List[str] = Field(default_factory=list)
    search_facets: List[SearchFacet] = Field(default_factory=list)

    # New Phase 2-5 structured objects
    question_understanding: Optional[UniversalQuestionUnderstanding] = None
    search_plan: Optional[UniversalSearchPlan] = None
    active_profile_summary: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# PHASE 7 & 8: Shared Scientific Evidence Representation & Section Schemas
# ============================================================================

ReportingStatusType = Literal[
    "Reported",
    "Not reported",
    "Not applicable",
    "Not assessed",
    "Unavailable",
    "Uncertain",
]

ProvenanceSourceType = Literal[
    "full_text",
    "abstract_derived",
    "metadata_only",
    "inferred",
    "unverified",
]

VerificationStatusType = Literal[
    "passed_validation",
    "passed_with_disclosed_limitations",
    "partially_verified",
    "failed_verification",
]


class ModelCapabilityProfile(BaseModel):
    provider: str = "gemini"
    model_id: str = "gemini-2.5-flash"
    capability_source: Literal["provider_metadata", "documented_config", "conservative_default"] = "documented_config"
    supports_native_json_schema: bool = True
    supports_json_object_mode: bool = True
    supports_tool_calling: bool = True
    supports_streaming: bool = True
    context_window_tokens: int = 128000
    max_output_tokens: int = 4096
    recommended_study_batch_size: int = 15
    provider_constraints: List[str] = Field(default_factory=list)
    observed_quality_tier: Literal["tier_1_high_structured", "tier_2_standard", "tier_3_constrained_local"] = "tier_1_high_structured"


class TaskExecutionRecord(BaseModel):
    task_id: str
    task_name: str
    prompt_version: str = "v2.1"
    model_used: str = ""
    attempts_used: int = 1
    validation_passed: bool = True
    verification_status: VerificationStatusType = "passed_validation"
    validation_errors: List[str] = Field(default_factory=list)
    degraded_to_deterministic: bool = False
    # True when the model hit its output token limit; callers split the work instead of resending the same prompt
    output_truncated: bool = False
    # Last schema-valid candidate that failed quality gates (enables section-level partial reuse; never serialized)
    last_candidate: Optional[Dict[str, Any]] = Field(default=None, exclude=True)


class TaskCStudyExtractionItem(BaseModel):
    citation_index: int
    stance: Literal["supporting", "mixed", "contradicting", "neutral"] = "supporting"
    study_design: str = "Empirical / Analytical Study"
    methodology: str = ""
    dataset_or_system: str = ""
    sample_or_scale: str = "Not reported"
    sample_reporting_status: ReportingStatusType = "Not reported"
    quantitative_result: str = "Not reported"
    key_findings: str = ""
    limitations: str = ""
    # Does the study address the research question (topic, place, population)?
    addresses_question: Literal["yes", "partly", "no"] = "yes"
    relevance_reason: str = ""


class TaskCStudyBatchOutput(BaseModel):
    study_extractions: List[TaskCStudyExtractionItem] = Field(default_factory=list)


class TaskDEFGClaimItem(BaseModel):
    claim_id: str = "claim:1"
    proposition: str
    claim_category: Literal[
        "association",
        "causal_intervention",
        "predictive_performance",
        "mechanistic",
        "theoretical",
        "methodological_comparison",
    ] = "methodological_comparison"
    supporting_indices: List[int] = Field(default_factory=list)
    opposing_indices: List[int] = Field(default_factory=list)
    applicability_scope: str = "Not reported"
    methodological_caveats: str = "Not reported"


class TaskDEFGClaimsOutput(BaseModel):
    comparable_groups_summary: str = ""
    claims: List[TaskDEFGClaimItem] = Field(default_factory=list)


class TaskHIGapAssignmentItem(BaseModel):
    row_idx: int
    col_idx: int
    paper_indices: List[int] = Field(default_factory=list)
    gap_category: str = "Evidence absence"
    why_gap_exists: str = ""
    proposed_hypothesis: str = ""
    verification_query: str = ""
    impact_level: Literal["High", "Medium", "Exploratory"] = "High"


class TaskHIGapOutput(BaseModel):
    rows: List[str] = Field(default_factory=list)
    cols: List[str] = Field(default_factory=list)
    assignments: List[TaskHIGapAssignmentItem] = Field(default_factory=list)


class TaskJSynthesisSectionItem(BaseModel):
    heading: str
    content: str
    cited_indices: List[int] = Field(default_factory=list)


class TaskJSectionOutput(BaseModel):
    """One section of a Deep Research review, written in its own call."""
    content: str
    # Short descriptive title the model proposes for a body subsection (B.n)
    heading: str = ""


class TaskJNarrativeSynthesisOutput(BaseModel):
    executive_answer: str
    consensus_verdict: str = "Context-Dependent Evidence Agreement"
    consensus_summary: str = ""
    sections: List[TaskJSynthesisSectionItem] = Field(default_factory=list)


class ElicitExtraction(BaseModel):
    methodology: str = ""
    dataset_or_system: str = ""
    key_findings: str = ""
    limitations: str = ""
    code_or_hf_link: str = ""
    # Extended Phase 7-8 domain-adaptive & provenance fields
    study_design: str = "Empirical / Computational Study"
    sample_or_scale: str = "Not reported"
    sample_reporting_status: ReportingStatusType = "Not reported"
    quantitative_result: str = "Not reported"
    provenance_type: ProvenanceSourceType = "abstract_derived"
    evidence_UNCERTAINTY_note: str = ""


_LATEX_WRAPPERS = re.compile(r"\\(?:ensuremath|mathrm|mathit|mathbf|mathcal|textrm|textit|textbf|text|operatorname|emph)\s*")
_LATEX_COMMAND = re.compile(r"\\([a-zA-Z]+)")


def strip_latex_markup(text: str) -> str:
    """
    Turns LaTeX markup common in arXiv / Semantic Scholar titles and abstracts into plain text
    ("\\ensuremath{\\mathrm{g}-2}" -> "g-2", "$\\mu$" -> "mu") so it does not leak into theme labels or prompts.
    """
    if not text or ("\\" not in text and "$" not in text):
        return text
    t = _LATEX_WRAPPERS.sub("", text)
    t = _LATEX_COMMAND.sub(r"\1", t)
    t = re.sub(r"[${}]", "", t)
    return re.sub(r"\s+", " ", t).strip()


class PaperRecord(BaseModel):
    id: str
    canonical_study_id: str = ""
    version_status: Literal["published", "preprint", "preprint_with_published_link", "unknown"] = "unknown"
    citation_index: int = 1
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    source: str = ""
    all_sources: List[str] = Field(default_factory=list)
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    s2_id: Optional[str] = None
    abstract: str = ""

    @field_validator("title", "abstract", mode="before")
    @classmethod
    def _plain_text(cls, v):
        return strip_latex_markup(v) if isinstance(v, str) else v

    citation_count: int = 0
    influential_citation_count: int = 0
    url: str = ""
    pdf_url: Optional[str] = None
    references: List[str] = Field(default_factory=list)
    cited_by: List[str] = Field(default_factory=list)
    sub_query: Optional[str] = None
    matched_subquestion_ids: List[str] = Field(default_factory=list)
    relevance_score: float = 0.0
    selection_breakdown: Dict[str, float] = Field(default_factory=dict)
    cluster_id: int = 0
    cluster_label: str = "Core Literature"
    stance: Literal["supporting", "mixed", "contradicting", "neutral"] = "supporting"
    stance_reason: str = ""
    # Whether the study addresses the question (place/population constraints, then the model's screening).
    # Off-topic studies stay in the bibliography but are left out of the synthesis and agreement meter.
    relevance: Literal["addresses", "partly", "off_topic", "unassessed"] = "unassessed"
    relevance_note: str = ""
    # Open-access full text, when available (see services/fulltext.py). The section text is used for extraction
    # and checks on the server but left out of API responses to keep them (and the browser history) small.
    full_text_source: str = ""
    full_text_sections: Dict[str, str] = Field(default_factory=dict, exclude=True)
    is_citation_context_only: bool = False
    elicit: ElicitExtraction = Field(default_factory=ElicitExtraction)


class StudyEntity(BaseModel):
    study_id: str
    paper_id: str
    citation_index: int
    title: str
    year: Optional[int] = None
    study_design: str = ""
    publication_status: str = "published"
    provenance: ProvenanceSourceType = "abstract_derived"
    methodological_appraisal: str = ""
    limitations: str = ""


class ResultEntity(BaseModel):
    result_id: str
    study_id: str
    citation_index: int
    outcome_or_measure: str
    finding_text: str
    quantitative_estimate: str = "Not reported"
    reporting_status: ReportingStatusType = "Reported"
    provenance: ProvenanceSourceType = "abstract_derived"
    source_passage_hint: str = "Abstract"


class ClaimEntity(BaseModel):
    claim_id: str
    proposition: str
    claim_category: Literal[
        "association",
        "causal_intervention",
        "predictive_performance",
        "mechanistic",
        "theoretical",
        "methodological_comparison",
    ] = "methodological_comparison"
    supporting_indices: List[int] = Field(default_factory=list)
    opposing_indices: List[int] = Field(default_factory=list)
    applicability_scope: str = ""
    methodological_caveats: str = ""
    provenance: ProvenanceSourceType = "abstract_derived"
    verification_state: VerificationStatusType = "passed_validation"


GapCategoryType = Literal[
    "Evidence absence",
    "Methodological limitations",
    "Inconsistent findings",
    "Population or contextual limitations",
    "Validation deficiencies",
    "Mechanistic uncertainty",
    "Translational deficiencies",
    "Measurement/standardization limitations",
    "Theoretical or formal unresolved problems",
]

GapTaxonomyCode = Literal["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"]

CellCoverageClassificationType = Literal[
    "Direct evidence identified",
    "Partial evidence identified",
    "Indirect evidence identified",
    "Conflicting evidence identified",
    "No relevant evidence identified within the examined corpus",
    "Not assessed",
    "Insufficient information to classify",
]

GapVerificationStateType = Literal[
    "Candidate — Not Yet Verified",
    "Corroborated Within Search Scope",
    "Partially Addressed",
    "Contradicted / Resolved by Retrieved Evidence",
    "Inconclusive",
    "Verification Unavailable",
]

GapRecordDistinctionType = Literal[
    "knowledge_gap",
    "methodological_limitation",
    "research_opportunity",
    "resolved_by_verification",
]

GapPriorityCategoryType = Literal[
    "High relevance to the research question",
    "Moderate relevance",
    "Contextual or exploratory relevance",
    "Insufficient information to prioritize",
]


class GapEntity(BaseModel):
    gap_id: str
    description: str
    gap_category: GapCategoryType = "Evidence absence"
    supporting_observations: List[str] = Field(default_factory=list)
    related_paper_indices: List[int] = Field(default_factory=list)
    verification_queries: List[str] = Field(default_factory=list)
    verification_status: Literal[
        "verified_in_retrieved_corpus",
        "targeted_search_checked",
        "candidate_requires_broader_search",
    ] = "verified_in_retrieved_corpus"
    quality_gate_status: VerificationStatusType = "passed_validation"
    scope_limitation: str = (
        "Assessed relative to the retrieved study corpus; absence in selected papers does not alone prove universal absence across all literature."
    )
    resolution_approaches: List[str] = Field(default_factory=list)


class RelationshipEntity(BaseModel):
    relationship_id: str
    source_id: str
    target_id: str
    relationship_type: Literal[
        "cites",
        "shares_references",
        "co_cited",
        "semantically_related",
        "addresses_same_question",
        "supports_claim",
        "challenges_claim",
        "uses_same_dataset",
        "extends_prior_work",
        "model_implementation",
    ] = "semantically_related"
    provenance: Literal["bibliometric_metadata", "lexical_tfidf", "evidence_verified", "llm_inferred"] = "lexical_tfidf"
    weight: float = 0.5
    evidence_note: str = ""


# ============================================================================
# SECTION 1: Claim-Centered Scientific Synthesis Records
# ============================================================================

EvidenceAccessType = Literal["full_text", "abstract", "metadata_only", "supplementary"]

FindingOriginType = Literal["directly_reported", "author_interpretation", "llm_interpretation"]

FindingDirectionType = Literal[
    "positive",
    "negative",
    "null_or_nonsignificant",
    "mixed_or_conditional",
    "not_applicable",
    "unclear",
]

ClaimEvidenceType = Literal[
    "observed_association",
    "predictive_relationship",
    "experimental_causal_effect",
    "proposed_mechanism",
    "validated_mechanism",
    "proved_mathematical_result",
    "computational_performance_comparison",
    "theoretical_or_model_based",
    "descriptive_observation",
]

ClaimVerificationState = Literal[
    "Verified",
    "Verified with qualifications",
    "Partially supported",
    "Unsupported",
    "Not verifiable from accessible sources",
]

QuestionAnswerabilityType = Literal[
    "answered",
    "partially_answered",
    "not_answerable_from_selected_evidence",
    "no_eligible_evidence",
]


class AtomicFinding(BaseModel):
    """One source-grounded result extracted from one study (Phase 3)."""

    result_id: str
    study_id: str
    citation_index: int
    subquestion_id: str = "Q0"
    proposition: str = ""
    study_evidence_type: str = "Not reported"
    claim_evidence_type: ClaimEvidenceType = "descriptive_observation"
    context_or_system: str = "Not reported"
    exposure_or_method: str = "Not reported"
    comparator: str = "Not reported"
    outcome: str = "Not reported"
    observed_result: str = "Unavailable"
    direction: FindingDirectionType = "unclear"
    quantitative_estimates: List[str] = Field(default_factory=list)
    uncertainty: str = "Not reported"
    authors_conclusion: str = "Not reported"
    methodological_limitations: str = "Not reported"
    source_document_id: str = ""
    source_passage: str = ""
    evidence_access: EvidenceAccessType = "abstract"
    origin: FindingOriginType = "directly_reported"
    field_status: Dict[str, ReportingStatusType] = Field(default_factory=dict)
    dataset_tokens: List[str] = Field(default_factory=list)
    question_relevance: float = 0.0
    excluded_reason: str = ""


class EvidenceCluster(BaseModel):
    """Findings judged sufficiently comparable to be considered together (Phase 4 Steps 2-3)."""

    cluster_id: str
    theme_label: str
    proposition: str
    claim_evidence_type: ClaimEvidenceType = "descriptive_observation"
    subquestion_id: str = "Q0"
    result_ids: List[str] = Field(default_factory=list)
    study_indices: List[int] = Field(default_factory=list)
    supporting_result_ids: List[str] = Field(default_factory=list)
    challenging_result_ids: List[str] = Field(default_factory=list)
    inconclusive_result_ids: List[str] = Field(default_factory=list)
    contextual_result_ids: List[str] = Field(default_factory=list)
    independent_study_groups: List[List[int]] = Field(default_factory=list)
    overlap_notes: List[str] = Field(default_factory=list)
    comparability_notes: List[str] = Field(default_factory=list)
    question_relevance: float = 0.0


class ClaimVerificationResult(BaseModel):
    """Provenance/verification state (Phase 7). Not a statistical confidence estimate."""

    structural_passed: bool = True
    factual_passed: bool = True
    inference_passed: bool = True
    state: ClaimVerificationState = "Verified with qualifications"
    issues: List[str] = Field(default_factory=list)
    qualifications: List[str] = Field(default_factory=list)


class SynthesisClaim(BaseModel):
    """Structured claim record that is the sole source of Section 1 narrative text (Phase 4 Step 6)."""

    claim_id: str
    claim_text: str
    research_question_id: str = "Q0"
    theme_label: str = ""
    cluster_id: str = ""
    evidence_type: ClaimEvidenceType = "descriptive_observation"
    supporting_result_ids: List[str] = Field(default_factory=list)
    challenging_result_ids: List[str] = Field(default_factory=list)
    inconclusive_result_ids: List[str] = Field(default_factory=list)
    contextual_result_ids: List[str] = Field(default_factory=list)
    relevant_study_indices: List[int] = Field(default_factory=list)
    supporting_indices: List[int] = Field(default_factory=list)
    challenging_indices: List[int] = Field(default_factory=list)
    inconclusive_indices: List[int] = Field(default_factory=list)
    key_quantitative_results: List[str] = Field(default_factory=list)
    comparability_limitations: List[str] = Field(default_factory=list)
    methodological_limitations: List[str] = Field(default_factory=list)
    qualified_interpretation: str = ""
    independent_support_count: int = 0
    text_origin: Literal["deterministic_template", "llm_refined"] = "deterministic_template"
    verification: ClaimVerificationResult = Field(default_factory=ClaimVerificationResult)


class SynthesisQualityCheck(BaseModel):
    name: str
    passed: bool = True
    reasons: List[str] = Field(default_factory=list)


class SynthesisQualityReport(BaseModel):
    checks: List[SynthesisQualityCheck] = Field(default_factory=list)
    all_passed: bool = True
    regenerated_sections: List[str] = Field(default_factory=list)
    narrative_origin: Literal["llm_verified", "llm_partially_verified", "deterministic"] = "deterministic"
    workflow_version: str = "section1-claim-synthesis/1.0"
    prompt_versions: Dict[str, str] = Field(default_factory=dict)
    model_used: str = ""


class SynthesisEvidencePackage(BaseModel):
    workflow_version: str = "section1-claim-synthesis/1.0"
    domain_appraisal_profile: str = "universal_conservative"
    appraisal_considerations: List[str] = Field(default_factory=list)
    question_answerability: QuestionAnswerabilityType = "partially_answered"
    findings: List[AtomicFinding] = Field(default_factory=list)
    clusters: List[EvidenceCluster] = Field(default_factory=list)
    claims: List[SynthesisClaim] = Field(default_factory=list)
    excluded_claims: List[SynthesisClaim] = Field(default_factory=list)
    quality: SynthesisQualityReport = Field(default_factory=SynthesisQualityReport)
    disclosures: List[str] = Field(default_factory=list)


# ============================================================================
# SECTION 2: Universal, Question-Adaptive, Source-Grounded Evidence Matrix
# ============================================================================

FieldPriorityType = Literal["Essential", "Recommended", "Optional", "Not applicable"]

ComparabilityStatusType = Literal[
    "Directly comparable",
    "Comparable with qualifications",
    "Related but not directly comparable",
    "Not comparable",
    "Insufficient information",
]

DocumentRelationType = Literal[
    "primary_document",
    "duplicate_document",
    "preprint_or_published_version",
    "follow_up_or_secondary_report",
    "shared_dataset_independent_study",
    "multi_experiment_publication",
]

EvidenceUnitRelationshipType = Literal[
    "single_publication",
    "merged_preprint_and_published",
    "follow_up_or_secondary_analysis",
    "shared_cohort_or_dataset",
    "multi_experiment_unit",
]

AnalysisTierType = Literal[
    "primary",
    "secondary",
    "development",
    "internal_validation",
    "external_validation",
    "exploratory",
    "theoretical_derivation",
]

ExtractionMethodType = Literal[
    "bibliographic_api",
    "deterministic_pattern",
    "llm_structured_extraction",
    "hybrid_verified",
]


class FieldProvenance(BaseModel):
    """Phase 9: Traceable source provenance for extracted evidence fields."""

    field_name: str
    source_document_id: str
    source_section: str = "Abstract"
    page_or_span: str = "Abstract"
    supporting_excerpt: str = ""
    extraction_method: ExtractionMethodType = "deterministic_pattern"
    extraction_timestamp: str = ""
    verification_status: ClaimVerificationState = "Verified"
    access_level: EvidenceAccessType = "abstract"
    reporting_status: ReportingStatusType = "Reported"


class DerivedQuantityRecord(BaseModel):
    """Phase 7: Explicitly labeled derived or converted quantity (never overwrites original text)."""

    quantity_name: str
    original_inputs: Dict[str, str] = Field(default_factory=dict)
    formula_used: str = ""
    derived_value: str = ""
    unit: str = ""
    calculation_status: Literal["computed", "not_computable_missing_inputs"] = "computed"


class QuantitativeResultDetail(BaseModel):
    """Phase 8: Structured quantitative extraction preserving complete scientific context."""

    raw_reported_text: str = "Not reported"
    normalized_value: str = "Not reported"
    metric_type: str = "Not reported"
    unit: str = "Not reported"
    direction: FindingDirectionType = "unclear"
    comparator: str = "Not reported"
    conditions_or_setting: str = "Not reported"
    sample_size_or_scale: str = "Not reported"
    time_point_or_horizon: str = "Not reported"
    uncertainty_interval: str = "Not reported"
    statistical_test: str = "Not reported"
    adjustment_status: str = "Not reported"
    p_value: str = "Not reported"
    analysis_tier: AnalysisTierType = "primary"
    reporting_status: ReportingStatusType = "Not reported"
    derived_quantities: List[DerivedQuantityRecord] = Field(default_factory=list)


class MethodologicalAppraisalRecord(BaseModel):
    """Phase 11: Method-aware critical appraisal separated from factual extraction."""

    appraisal_framework: str = "Conservative Methodological & Provenance Appraisal"
    appraisal_completeness: Literal[
        "full_text_complete",
        "preliminary_abstract_only",
        "metadata_only_minimal",
    ] = "preliminary_abstract_only"
    source_reported_limitations: List[str] = Field(default_factory=list)
    independent_methodological_concerns: List[str] = Field(default_factory=list)
    applicability_concerns: List[str] = Field(default_factory=list)
    internal_validity_notes: str = "Preliminary assessment (abstract-level access)"
    external_validity_notes: str = "Scope restricted to reported study system/dataset"
    reproducibility_transparency: str = "Not reported in abstract"


class ComparabilityAssessment(BaseModel):
    """Phase 10: Explicit comparability classification across studies and results."""

    comparability_group_id: str = "CG-1"
    comparability_group_label: str = "General Domain Evidence"
    status: ComparabilityStatusType = "Comparable with qualifications"
    comparable_peer_indices: List[int] = Field(default_factory=list)
    shared_dimensions: List[str] = Field(default_factory=list)
    divergent_dimensions: List[str] = Field(default_factory=list)
    comparability_rationale: str = ""
    warnings: List[str] = Field(default_factory=list)


class ExtractionFieldSpec(BaseModel):
    """Phase 5: Query-adaptive extraction field definition."""

    field_key: str
    display_label: str
    dimension: Literal[
        "Identification",
        "Research Objective",
        "Context",
        "Methodology",
        "Evidence",
        "Limitations",
        "Relevance",
        "Provenance",
        "Domain Specific",
    ] = "Domain Specific"
    priority: FieldPriorityType = "Essential"
    domain_module: str = "universal"
    description: str = ""


class ExtractionPlan(BaseModel):
    """Phase 5 & 6: Query-adaptive extraction plan derived before building the evidence matrix."""

    plan_id: str = "EP-1"
    query: str = ""
    primary_domain: str = "Multidisciplinary / General Science"
    secondary_domains: List[str] = Field(default_factory=list)
    domain_module_activated: str = "universal_fallback"
    study_types_represented: List[str] = Field(default_factory=list)
    essential_fields: List[ExtractionFieldSpec] = Field(default_factory=list)
    recommended_fields: List[ExtractionFieldSpec] = Field(default_factory=list)
    optional_fields: List[ExtractionFieldSpec] = Field(default_factory=list)
    not_applicable_fields: List[ExtractionFieldSpec] = Field(default_factory=list)
    visible_domain_columns: List[ExtractionFieldSpec] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    """Phase 3 Level A: Publication / Document Record."""

    document_id: str
    citation_index: int
    canonical_title: str
    authors: List[str] = Field(default_factory=list)
    publication_year: Optional[int] = None
    venue_or_source: str = ""
    doi: Optional[str] = None
    pmid: Optional[str] = None
    arxiv_id: Optional[str] = None
    s2_id: Optional[str] = None
    source_databases: List[str] = Field(default_factory=list)
    publication_status: Literal[
        "published",
        "preprint",
        "preprint_with_published_link",
        "conference_proceedings",
        "working_paper_or_report",
        "unknown",
    ] = "published"
    document_relation: DocumentRelationType = "primary_document"
    linked_document_ids: List[str] = Field(default_factory=list)
    abstract_available: bool = True
    full_text_available: bool = False
    supplementary_available: bool = False
    evidence_access_level: EvidenceAccessType = "abstract"
    retrieval_timestamp: str = ""
    source_url: str = ""
    pdf_url: Optional[str] = None
    retraction_or_correction_status: str = (
        "No retraction notice detected in retrieved metadata (external retraction watch not queried)"
    )


class GranularFindingRecord(BaseModel):
    """Phase 3 Level C: Individual Result / Finding Record within a study."""

    result_id: str
    contribution_id: str
    source_document_id: str
    citation_index: int
    subquestion_addressed: str = "Q0"
    experiment_or_component_label: str = "Primary Reported Analysis"
    outcome_or_phenomenon: str = "Primary study outcome"
    measurement_or_analytical_method: str = "Not reported"
    experimental_or_theoretical_conditions: str = "Not reported"
    comparator: str = "Not reported"
    factual_source_text: str = ""
    normalized_summary: str = ""
    llm_interpretation: str = ""
    quantitative_detail: QuantitativeResultDetail = Field(default_factory=QuantitativeResultDetail)
    direction_of_effect: FindingDirectionType = "unclear"
    statistical_or_mathematical_qualifications: str = "Not reported"
    extraction_status: ReportingStatusType = "Reported"
    provenance: FieldProvenance = Field(
        default_factory=lambda: FieldProvenance(field_name="finding", source_document_id="")
    )
    linked_section1_claim_ids: List[str] = Field(default_factory=list)


class EvidenceUnitRecord(BaseModel):
    """Phase 3 Level B: Underlying Study / Research Contribution Record."""

    contribution_id: str
    primary_document_id: str
    associated_document_ids: List[str] = Field(default_factory=list)
    citation_index: int
    relationship_type: EvidenceUnitRelationshipType = "single_publication"
    overlap_or_linkage_note: str = "Independent study record"
    # Dimension 2: Research Objective
    research_objective: str = "Not reported"
    subquestions_addressed: List[str] = Field(default_factory=list)
    contribution_type: str = "Empirical / Computational Study"
    # Dimension 3: Context / System / Population / Domain Setting
    context_or_system: str = "Not reported"
    population_or_material_or_regime: str = "Not reported"
    sample_or_scale: str = "Not reported"
    sample_reporting_status: ReportingStatusType = "Not reported"
    # Dimension 4: Methodology / Analytical Approach
    methodological_approach: str = "Not reported"
    experimental_or_analytical_design: str = "Not reported"
    key_interventions_or_models: str = "Not reported"
    comparators_or_baselines: str = "Not reported"
    key_assumptions_or_controls: str = "Not reported"
    data_sources_or_benchmarks: str = "Not reported"
    validation_approach: str = "Not reported"
    code_or_artifact_availability: str = "Not reported"
    # Domain-Adaptive Fields (key -> value and key -> status)
    domain_specific_fields: Dict[str, str] = Field(default_factory=dict)
    domain_field_statuses: Dict[str, ReportingStatusType] = Field(default_factory=dict)
    # Dimension 5: Extracted Results (Level C)
    results: List[GranularFindingRecord] = Field(default_factory=list)
    # Dimension 6: Methodological Appraisal & Limitations
    appraisal: MethodologicalAppraisalRecord = Field(default_factory=MethodologicalAppraisalRecord)
    # Dimension 7: Relevance & Comparability
    relevance_to_query_score: float = 0.0
    relevance_rationale: str = ""
    comparability: ComparabilityAssessment = Field(default_factory=ComparabilityAssessment)
    linked_claim_ids: List[str] = Field(default_factory=list)
    # Dimension 8: Field-level Provenance Map
    field_provenance: Dict[str, FieldProvenance] = Field(default_factory=dict)


class EvidenceMatrixRow(BaseModel):
    """Phase 12: Flattened + Hierarchical Row representation for UI table & exports."""

    row_id: str
    citation_index: int
    document: DocumentRecord
    evidence_unit: EvidenceUnitRecord
    primary_finding: GranularFindingRecord
    additional_findings_count: int = 0
    stance: Literal["supporting", "mixed", "contradicting", "neutral"] = "supporting"
    verification_status: ClaimVerificationState = "Verified"


class EvidenceTableValidationReport(BaseModel):
    """Phase 16: Multi-layer validation report for Section 2 Structured Evidence Matrix."""

    structural_valid: bool = True
    bibliographic_valid: bool = True
    factual_provenance_valid: bool = True
    scientific_comparability_valid: bool = True
    section1_consistency_valid: bool = True
    checks_performed: List[str] = Field(default_factory=list)
    issues_detected: List[str] = Field(default_factory=list)
    repaired_fields_count: int = 0


class StructuredEvidenceMatrix(BaseModel):
    """Top-level Section 2 Universal, Question-Adaptive, Source-Grounded Evidence Matrix."""

    matrix_version: str = "section2-evidence-matrix/2.0"
    extraction_plan: ExtractionPlan = Field(default_factory=ExtractionPlan)
    documents: List[DocumentRecord] = Field(default_factory=list)
    evidence_units: List[EvidenceUnitRecord] = Field(default_factory=list)
    granular_findings: List[GranularFindingRecord] = Field(default_factory=list)
    rows: List[EvidenceMatrixRow] = Field(default_factory=list)
    comparability_groups: List[ComparabilityAssessment] = Field(default_factory=list)
    validation_report: EvidenceTableValidationReport = Field(default_factory=EvidenceTableValidationReport)
    summary_disclosures: List[str] = Field(default_factory=list)




class SharedEvidenceBase(BaseModel):
    studies: List[StudyEntity] = Field(default_factory=list)
    results: List[ResultEntity] = Field(default_factory=list)
    claims: List[ClaimEntity] = Field(default_factory=list)
    gaps: List[GapEntity] = Field(default_factory=list)
    relationships: List[RelationshipEntity] = Field(default_factory=list)
    task_audit_trail: List[TaskExecutionRecord] = Field(default_factory=list)
    # Section 1 claim-centered synthesis records (reusable by Sections 2-4)
    atomic_findings: List["AtomicFinding"] = Field(default_factory=list)
    evidence_clusters: List["EvidenceCluster"] = Field(default_factory=list)


class QualityGateReport(BaseModel):
    overall_verification_status: VerificationStatusType = "passed_validation"
    retrieval_valid: bool = True
    extraction_valid: bool = True
    synthesis_valid: bool = True
    gap_valid: bool = True
    network_valid: bool = True
    # Model-Agnostic Quality Gates (Section 6)
    schema_valid: bool = True
    source_integrity_valid: bool = True
    numerical_integrity_valid: bool = True
    claim_attribution_valid: bool = True
    inference_validity_valid: bool = True
    evidence_consistency_valid: bool = True
    cross_section_consistency_valid: bool = True
    correction_retries_used: int = 0
    degradation_events: List[str] = Field(default_factory=list)
    model_capability_used: Optional[ModelCapabilityProfile] = None
    task_audit_trail: List[TaskExecutionRecord] = Field(default_factory=list)
    checks_passed: List[str] = Field(default_factory=list)
    warnings_and_Notices: List[str] = Field(default_factory=list)


class HFResourceRecord(BaseModel):
    id: str
    resource_type: Literal["model", "dataset"] = "model"
    pipeline_tag: str = ""
    downloads: int = 0
    likes: int = 0
    author: str = ""
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    linked_arxiv_ids: List[str] = Field(default_factory=list)
    url: str = ""
    last_modified: str = ""
    evidence_note: str = "Community model/dataset artifact (not peer-reviewed scientific evidence unless backed by linked publication)."


class BibliographyEntry(BaseModel):
    index: int
    paper_id: str
    title: str
    authors_str: str
    year_str: str
    venue: str
    source: str
    identifier_str: str
    url: str
    apa_formatted: str
    ieee_formatted: str
    bibtex: str
    # Section 1 bibliographic integrity metadata (derived only from retrieved records)
    stable_key: str = ""
    canonical_identifier: str = ""
    evidence_access: str = "Abstract"
    publication_type: str = "Peer-reviewed publication"
    linked_version_note: str = ""
    retraction_check: str = "Not checked — no retraction/correction metadata service is configured"
    cited_in_synthesis: bool = False


class ConsensusBreakdown(BaseModel):
    supporting_pct: int = 0
    mixed_pct: int = 0
    contradicting_pct: int = 0
    total_evaluated: int = 0
    consensus_verdict: str = ""
    consensus_summary: str = ""


class SynthesisSection(BaseModel):
    heading: str
    content: str
    cited_indices: List[int] = Field(default_factory=list)


# ============================================================================
# SECTION 4: Universal, Multilayer Literature Network Map & Knowledge Graph
# ============================================================================

NetworkLayerType = Literal[
    "bibliographic",
    "semantic_conceptual",
    "scientific_evidence",
    "research_evolution",
    "artifact_link",
]

NodeInclusionRoleType = Literal[
    "included_synthesis_evidence",
    "citation_context_publication",
    "candidate_publication",
    "supplementary_verification_source",
    "synthesized_claim_node",
    "verified_gap_node",
    "research_theme_node",
    "community_artifact",
]

NodeImportanceRoleType = Literal[
    "foundational_contribution",
    "citation_hub_in_context",
    "bridge_publication",
    "recent_methodological_contribution",
    "evidence_central_publication",
    "standard_included_study",
    "contextual_or_artifact_node",
]


class GraphNode(BaseModel):
    id: str
    label: str
    full_title: str
    node_type: Literal[
        "paper",
        "evidence_unit",
        "scientific_claim",
        "research_theme",
        "knowledge_gap",
        "hf_model",
        "hf_dataset",
    ] = "paper"
    citation_index: Optional[int] = None
    year: int = 2024
    citations: int = 0
    source: str = ""
    cluster_id: int = 0
    cluster_label: str = ""
    bibliographic_cluster_id: Optional[int] = None
    semantic_cluster_id: Optional[int] = None
    stance: str = "supporting"
    url: str = ""
    is_citation_context_only: bool = False
    x_timeline: float = 0.5
    y_timeline: float = 0.5
    x_force: float = 0.5
    y_force: float = 0.5
    pagerank: float = 0.1
    # Phase 4, 11, 13, 14 structured canonical node metadata
    canonical_entity_id: str = ""
    source_ids: List[str] = Field(default_factory=list)
    publication_status: str = "published"
    inclusion_role: NodeInclusionRoleType = "included_synthesis_evidence"
    importance_roles: List[NodeImportanceRoleType] = Field(default_factory=list)
    importance_rationale: str = ""
    domain: str = "universal"
    research_theme_ids: List[str] = Field(default_factory=list)
    related_evidence_unit_ids: List[str] = Field(default_factory=list)
    related_claim_ids: List[str] = Field(default_factory=list)
    related_gap_ids: List[str] = Field(default_factory=list)
    in_degree_citations: int = 0
    out_degree_citations: int = 0
    betweenness_centrality: float = 0.0
    is_isolated_in_bibliographic: bool = False
    metadata_completeness: str = "Abstract & bibliographic IDs available"
    provenance: str = "Retrieved source metadata"


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    edge_type: Literal[
        "citation",
        "bibliographic_coupling",
        "co_citation",
        "semantic",
        "evidence_relationship",
        "research_evolution",
        "model_link",
    ] = "semantic"
    provenance: Literal["bibliometric_metadata", "lexical_tfidf", "evidence_verified", "llm_inferred"] = (
        "lexical_tfidf"
    )
    weight: float = 0.5
    label: str = ""
    # Phase 3, 5-9, 12, 14 structured multilayer edge metadata
    relationship_subtype: str = "similar_research_question"
    network_layer: NetworkLayerType = "semantic_conceptual"
    directed: bool = False
    raw_weight: float = 0.5
    normalized_weight: float = 0.5
    weighting_method: str = "TF-IDF cosine similarity"
    construction_method: str = "Deterministic pairwise computation"
    relationship_explanation: str = ""
    citation_universe_scope: str = ""
    supporting_source_ids: List[str] = Field(default_factory=list)
    supporting_result_ids: List[str] = Field(default_factory=list)
    supporting_claim_ids: List[str] = Field(default_factory=list)
    verification_status: str = "Verified"
    metadata_coverage: str = "Complete for compared endpoints"
    analysis_version: str = "section4-multilayer-network/2.0"


class NetworkThemeSummary(BaseModel):
    """Phase 10 & 18: Structured research theme / community summary."""

    theme_id: str
    theme_label: str
    clustering_layer: Literal["semantic_conceptual", "bibliographic", "descriptive_small_graph"] = (
        "semantic_conceptual"
    )
    member_publication_ids: List[str] = Field(default_factory=list)
    member_citation_indices: List[int] = Field(default_factory=list)
    supporting_concepts: List[str] = Field(default_factory=list)
    related_claim_ids: List[str] = Field(default_factory=list)
    related_gap_ids: List[str] = Field(default_factory=list)
    interpretation: str = ""
    verification_status: str = "Grounded in member publications"


class NetworkCoverageStats(BaseModel):
    """Phase 19: Explicit coverage statistics for network reproducibility and transparency."""

    included_studies_count: int = 0
    studies_with_reference_lists: int = 0
    studies_missing_reference_lists: int = 0
    context_publications_count: int = 0
    direct_citation_edges_count: int = 0
    bibliographic_coupling_edges_count: int = 0
    co_citation_edges_count: int = 0
    semantic_edges_count: int = 0
    evidence_edges_count: int = 0
    evolution_edges_count: int = 0
    isolated_included_studies_count: int = 0
    citation_universe_description: str = "Included synthesis publications and retrieved reference/citing lists"


class NetworkInterpretationReport(BaseModel):
    """Phase 16 & 17: Grounded scientific interpretation of the multilayer literature network."""

    domain_profile: str = "universal_fallback"
    research_themes_summary: str = ""
    intellectual_foundations_summary: str = ""
    methodological_relationships_summary: str = ""
    scientific_evidence_relationships_summary: str = ""
    cross_theme_connections_summary: str = ""
    research_development_summary: str = ""
    network_limitations_and_disclosures: List[str] = Field(default_factory=list)


class GraphMetadataRecord(BaseModel):
    """Phase 14: Reproducible graph construction and parameter metadata."""

    graph_id: str = "GRAPH-1"
    research_question_id: str = "Q0"
    research_mode: str = "standard"
    included_study_count: int = 0
    context_document_count: int = 0
    data_sources: List[str] = Field(default_factory=list)
    retrieval_date: str = ""
    network_layers_present: List[NetworkLayerType] = Field(default_factory=list)
    similarity_methods: Dict[str, str] = Field(default_factory=dict)
    similarity_thresholds: Dict[str, float] = Field(default_factory=dict)
    clustering_method: str = "Layer-specific TF-IDF KMeans / Connected Components with small-graph safeguards"
    clustering_parameters: Dict[str, Any] = Field(default_factory=dict)
    layout_method: str = "Deterministic Spring Force-Directed (seed=42) + Chronological Impact Timeline"
    layout_parameters: Dict[str, Any] = Field(default_factory=lambda: {"seed": 42, "iterations": 60})
    random_seed: int = 42
    graph_generation_version: str = "section4-multilayer-network/2.0"
    source_coverage_limitations: List[str] = Field(default_factory=list)


class NetworkValidationReport(BaseModel):
    """Phase 19: Multi-level validation report for Section 4 Literature Network Map."""

    bibliographic_valid: bool = True
    mathematical_network_valid: bool = True
    scientific_validity_valid: bool = True
    coverage_valid: bool = True
    small_network_safeguards_applied: bool = True
    checks_performed: List[str] = Field(default_factory=list)
    issues_detected: List[str] = Field(default_factory=list)
    repaired_elements_count: int = 0


class LiteratureGraph(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    clusters: List[Dict[str, Any]] = Field(default_factory=list)
    # Upgraded Section 4 Multilayer Scientific Literature Mapping fields
    metadata: GraphMetadataRecord = Field(default_factory=GraphMetadataRecord)
    coverage_stats: NetworkCoverageStats = Field(default_factory=NetworkCoverageStats)
    themes: List[NetworkThemeSummary] = Field(default_factory=list)
    interpretation: NetworkInterpretationReport = Field(default_factory=NetworkInterpretationReport)
    validation_report: NetworkValidationReport = Field(default_factory=NetworkValidationReport)


class GapOpportunity(BaseModel):
    id: str
    method_row: str
    domain_col: str
    status: Literal["gap", "sparse", "well_covered"] = "gap"
    coverage_classification: CellCoverageClassificationType = (
        "No relevant evidence identified within the examined corpus"
    )
    independent_unit_count: int = 0
    evidence_unit_ids: List[str] = Field(default_factory=list)
    result_ids: List[str] = Field(default_factory=list)
    claim_ids: List[str] = Field(default_factory=list)
    known_limitations: List[str] = Field(default_factory=list)
    evidence_access_summary: str = "Abstract-derived"
    gap_category: GapCategoryType = "Evidence absence"
    secondary_gap_categories: List[GapCategoryType] = Field(default_factory=list)
    verification_status: str = "Candidate — Not Yet Verified"
    scope_disclaimer: str = (
        "Reflects coverage within the selected study set; absence in selected papers does not prove universal field-wide absence."
    )
    paper_count: int = 0
    paper_indices: List[int] = Field(default_factory=list)
    why_gap_exists: str = ""
    proposed_hypothesis: str = ""
    verification_search_query: str = ""
    suggested_hf_models: List[str] = Field(default_factory=list)
    impact_level: Literal["High", "Medium", "Exploratory"] = "High"
    linked_gap_id: Optional[str] = None


class SubquestionCoverageRecord(BaseModel):
    """Phase 4: Research-question-driven evidence coverage per subquestion."""

    subquestion_id: str
    subquestion_text: str
    required_evidence_types: List[str] = Field(default_factory=list)
    coverage_status: Literal[
        "adequately_addressed",
        "partially_addressed",
        "unresolved_in_selected_corpus",
        "conflicting_evidence",
        "not_assessable",
    ] = "partially_addressed"
    relevant_study_indices: List[int] = Field(default_factory=list)
    relevant_unit_ids: List[str] = Field(default_factory=list)
    relevant_result_ids: List[str] = Field(default_factory=list)
    applicable_claim_ids: List[str] = Field(default_factory=list)
    methodological_approaches_present: List[str] = Field(default_factory=list)
    contextual_coverage_summary: str = ""
    missing_information: List[str] = Field(default_factory=list)
    access_limitations: str = "Abstract-level evidence unless full text noted"


class SupplementaryVerificationRecord(BaseModel):
    """
    Phase 6: Supplementary literature record inspected during targeted gap verification.
    Kept strictly separate from the primary 20/60 synthesis study set.
    """

    verification_source_id: str
    gap_id: str
    query_executed: str
    source_database: str = "Semantic Scholar / PubMed / arXiv"
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    persistent_id: str = ""
    url: str = ""
    abstract_excerpt: str = ""
    relevance_to_gap: Literal[
        "directly_addresses",
        "partially_addresses",
        "contradicts_or_complicates",
        "does_not_address",
        "insufficient_source_access",
    ] = "partially_addresses"
    assessment_note: str = ""


class AdversarialGapCheckResult(BaseModel):
    """Phase 15: 10-point adversarial challenge verifying a candidate gap's assumptions."""

    already_addressed_by_publication: bool = False
    arises_from_search_strategy: bool = False
    arises_from_selection_limit: bool = True
    full_text_might_contain_evidence: bool = True
    single_study_limitation_only: bool = False
    scientifically_relevant_to_query: bool = True
    confuses_novel_method_with_unanswered_question: bool = False
    misinterprets_null_result: bool = False
    terminology_variant_risk: bool = False
    proposed_study_resolves_uncertainty: bool = True
    challenge_summary: str = ""
    disposition: Literal[
        "retain_as_verified_gap",
        "retain_with_narrowed_scope",
        "reclassify_as_partially_addressed",
        "reclassify_as_research_opportunity",
        "invalidate_and_archive",
    ] = "retain_as_verified_gap"


class GapPriorityEvaluation(BaseModel):
    """Phase 11: Explainable, non-numerical priority evaluation separated from verification status."""

    priority_category: GapPriorityCategoryType = "High relevance to the research question"
    relevance_to_question: str = "Directly affects a core research subquestion"
    scientific_importance: str = "Resolving this uncertainty clarifies domain evidence interpretation"
    consequence_of_uncertainty: str = "Conclusions remain conditional without this evidence"
    evidence_deficiency_severity: str = "Substantial deficiency within examined corpus"
    research_feasibility: str = "Qualified — depends on domain experimental/computational resources"
    potential_contribution: str = "Provides discriminating empirical or theoretical evidence"
    priority_rationale: str = ""


class StructuredKnowledgeGap(BaseModel):
    """Phase 10: Complete machine-readable KnowledgeGap object."""

    # Identification
    gap_id: str
    gap_title: str
    gap_statement: str
    record_distinction: GapRecordDistinctionType = "knowledge_gap"
    research_question_id: str = "Q0"
    subquestion_ids: List[str] = Field(default_factory=list)
    matrix_cell_id: str = ""

    # Classification
    taxonomy_code: GapTaxonomyCode = "G1"
    primary_gap_type: GapCategoryType = "Evidence absence"
    secondary_gap_types: List[GapCategoryType] = Field(default_factory=list)
    domain_profile: str = "universal_fallback"
    methodological_profile: str = "general_scientific"

    # Evidence justification
    current_knowledge: str = ""
    unresolved_issue: str = ""
    why_it_matters: str = ""
    origin_type: Literal[
        "A_missing_direct_evidence",
        "B_limitations_in_existing_evidence",
        "C_unresolved_contradictions",
        "D_insufficient_validation",
        "E_unexplored_relevant_conditions",
        "F_unresolved_theoretical_questions",
    ] = "A_missing_direct_evidence"
    supporting_study_indices: List[int] = Field(default_factory=list)
    supporting_study_ids: List[str] = Field(default_factory=list)
    supporting_evidence_unit_ids: List[str] = Field(default_factory=list)
    supporting_result_ids: List[str] = Field(default_factory=list)
    related_claim_ids: List[str] = Field(default_factory=list)
    methodological_limitations: List[str] = Field(default_factory=list)
    contradictory_evidence_ids: List[str] = Field(default_factory=list)

    # Verification (Phases 6 & 7)
    verification_question: str = ""
    verification_queries: List[str] = Field(default_factory=list)
    verification_search_ids: List[str] = Field(default_factory=list)
    verification_source_ids: List[str] = Field(default_factory=list)
    verification_outcome: str = ""
    verification_status: GapVerificationStateType = "Candidate — Not Yet Verified"
    search_scope: str = ""
    search_date: str = ""
    source_access_limitations: str = (
        "Assessed via abstract/metadata APIs; full-text methods and paywalled supplements were not inspected."
    )

    # Research implications & Priority (Phases 8, 10, 11)
    suggested_research_question: str = ""
    evidence_needed: str = ""
    recommended_study_or_approach: str = ""
    expected_knowledge_contribution: str = ""
    priority_assessment: GapPriorityCategoryType = "High relevance to the research question"
    priority_rationale: str = ""
    priority_evaluation: GapPriorityEvaluation = Field(default_factory=GapPriorityEvaluation)
    adversarial_check: AdversarialGapCheckResult = Field(default_factory=AdversarialGapCheckResult)

    # Quality and audit
    source_provenance: List[str] = Field(default_factory=list)
    generated_by_workflow_version: str = "section3-gap-engine/2.0"
    verification_method: Literal[
        "corpus_and_targeted_supplementary_api",
        "corpus_evidence_appraisal",
        "cached_supplementary_verification",
    ] = "corpus_and_targeted_supplementary_api"
    review_status: Literal["verified_active", "partially_addressed", "resolved_archived", "opportunity_only"] = (
        "verified_active"
    )
    last_updated: str = ""


class ResearchOpportunityRecord(BaseModel):
    """
    Phase 8: Explicitly separated Research Opportunity (e.g., applying a novel architecture
    or Hugging Face model) that is distinct from an unresolved scientific knowledge gap.
    """

    opportunity_id: str
    title: str
    description: str
    related_gap_id: Optional[str] = None
    related_subquestion_ids: List[str] = Field(default_factory=list)
    suggested_hf_models_or_datasets: List[str] = Field(default_factory=list)
    rationale: str = ""
    distinction_note: str = (
        "Classified as a constructive research opportunity rather than a verified knowledge deficiency."
    )


class GapMatrixValidationReport(BaseModel):
    """Phase 16: Cross-section quality gate report for Section 3 Knowledge Gap Matrix."""

    structural_valid: bool = True
    evidence_grounding_valid: bool = True
    verification_integrity_valid: bool = True
    scientific_validity_valid: bool = True
    cross_section_consistency_valid: bool = True
    checks_performed: List[str] = Field(default_factory=list)
    issues_detected: List[str] = Field(default_factory=list)
    repaired_or_downgraded_count: int = 0


class KnowledgeGapMatrix(BaseModel):
    matrix_version: str = "section3-gap-matrix/2.0"
    domain_profile: str = "universal_fallback"
    row_dimension_title: str = "Methodologies & Architectures"
    col_dimension_title: str = "Evaluation Domains, Datasets & Outcomes"
    rows: List[str] = Field(default_factory=list)
    cols: List[str] = Field(default_factory=list)
    cells: List[List[GapOpportunity]] = Field(default_factory=list)
    top_opportunities: List[GapOpportunity] = Field(default_factory=list)
    # Upgraded Section 3 Universal Evidence-Gap Identification, Verification & Prioritization fields
    subquestion_coverage: List[SubquestionCoverageRecord] = Field(default_factory=list)
    structured_gaps: List[StructuredKnowledgeGap] = Field(default_factory=list)
    resolved_or_invalidated_gaps: List[StructuredKnowledgeGap] = Field(default_factory=list)
    research_opportunities: List[ResearchOpportunityRecord] = Field(default_factory=list)
    supplementary_verification_records: List[SupplementaryVerificationRecord] = Field(default_factory=list)
    flagged_claim_ids_for_review: List[str] = Field(default_factory=list)
    validation_report: GapMatrixValidationReport = Field(default_factory=GapMatrixValidationReport)
    corpus_adequacy_explanation: str = ""
    disclosures: List[str] = Field(default_factory=list)


class ExportRequest(BaseModel):
    format: Literal[
        "txt",
        "md",
        "pdf",
        "csv",
        "json",
        "graphml",
        "graph_nodes_csv",
        "graph_edges_csv",
        "graph_json",
    ] = "md"
    query: str
    mode: str = "standard"
    executive_answer: str
    sections: List[SynthesisSection] = Field(default_factory=list)
    consensus: ConsensusBreakdown = Field(default_factory=ConsensusBreakdown)
    papers: List[PaperRecord] = Field(default_factory=list)
    bibliography: List[BibliographyEntry] = Field(default_factory=list)
    gap_matrix: Optional[KnowledgeGapMatrix] = None
    graph: Optional[LiteratureGraph] = None
    hf_resources: List[HFResourceRecord] = Field(default_factory=list)
    review_protocol: Optional["ReviewProtocolReport"] = None
    include_evidence_table: bool = True
    include_gap_matrix: bool = True
    include_network_summary: bool = True
    include_hf_models: bool = True
    synthesis_evidence: Optional[SynthesisEvidencePackage] = None
    evidence_matrix: Optional[StructuredEvidenceMatrix] = None


class AppliedRule(BaseModel):
    rule_id: str
    text: str
    stage: str
    sources: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    enforcement: str = "prompt"


class PrismaFlowCounts(BaseModel):
    identified_by_source: Dict[str, int] = Field(default_factory=dict)
    identified_total: int = 0
    duplicates_removed: int = 0
    screened: int = 0
    excluded_at_screening: int = 0
    included: int = 0


class ReviewProtocolReport(BaseModel):
    framework_id: str = "narrative"
    framework_label: str = ""
    organization_id: str = "thematic"
    organization_label: str = ""
    framework_citations: List[str] = Field(default_factory=list)
    disclosure: str = ""
    flow: Optional[PrismaFlowCounts] = None
    search_strategies: Dict[str, List[str]] = Field(default_factory=dict)
    eligibility_criteria: List[str] = Field(default_factory=list)
    rules_applied: List[AppliedRule] = Field(default_factory=list)


class ResearchResponse(BaseModel):
    query: str
    mode: Literal["standard", "deep_research"] = "standard"
    sub_queries: List[str] = Field(default_factory=list)
    decomposition_plan: Optional[QueryDecompositionPlan] = None
    evidence_base: Optional[SharedEvidenceBase] = None
    quality_report: Optional[QualityGateReport] = None
    synthesis_evidence: Optional[SynthesisEvidencePackage] = None
    evidence_matrix: Optional[StructuredEvidenceMatrix] = None
    source_stats: Dict[str, int] = Field(default_factory=dict)
    executive_answer: str = ""
    sections: List[SynthesisSection] = Field(default_factory=list)
    consensus: ConsensusBreakdown = Field(default_factory=ConsensusBreakdown)
    papers: List[PaperRecord] = Field(default_factory=list)
    bibliography: List[BibliographyEntry] = Field(default_factory=list)
    hf_resources: List[HFResourceRecord] = Field(default_factory=list)
    graph: LiteratureGraph = Field(default_factory=LiteratureGraph)
    gap_matrix: KnowledgeGapMatrix = Field(default_factory=KnowledgeGapMatrix)
    warnings: List[str] = Field(default_factory=list)
    review_protocol: Optional[ReviewProtocolReport] = None


# SearchRequest refers to DomainCategoryType, which is defined further down this module
SearchRequest.model_rebuild()
ExportRequest.model_rebuild()
