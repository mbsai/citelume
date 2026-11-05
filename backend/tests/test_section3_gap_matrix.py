import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    ElicitExtraction,
    HFResourceRecord,
    PaperRecord,
    QueryDecompositionPlan,
    SupplementaryVerificationRecord,
    SynthesisEvidencePackage,
)
from services.claim_synthesis import (
    build_and_verify_claims,
    cluster_comparable_findings,
    extract_atomic_findings,
)
from services.evidence_table_engine import build_structured_evidence_matrix
from services.gap_matrix_engine import (
    build_universal_knowledge_gap_matrix,
    clear_gap_verification_cache,
    set_test_gap_verifier,
)


def _sample_amp_papers() -> list[PaperRecord]:
    return [
        PaperRecord(
            id="amp-1",
            canonical_study_id="study-amp-1",
            version_status="published",
            citation_index=1,
            title="DeepAMP: Transformer Screening of Antimicrobial Peptides on DBAASP",
            authors=["Alice Zhang", "Bob Chen"],
            year=2025,
            venue="Bioinformatics",
            source="PubMed",
            doi="10.1093/bioinformatics/amp1",
            abstract=(
                "We present DeepAMP for sequence-based antimicrobial peptide prediction. "
                "On the DBAASP 5-fold cross-validation benchmark (n = 3,800 peptides), DeepAMP achieved AUROC of 0.93 compared with SVM baselines. "
                "However, in vivo murine toxicity and serum stability were not evaluated."
            ),
            matched_subquestion_ids=["Q1"],
            relevance_score=0.94,
            stance="supporting",
            elicit=ElicitExtraction(
                methodology="Transformer Encoder",
                dataset_or_system="DBAASP In Silico Benchmark",
                key_findings="AUROC 0.93 on DBAASP cross-validation",
                limitations="In silico only; no in vivo toxicity validation",
                study_design="Computational Benchmark Study",
                sample_or_scale="n = 3,800 peptides",
                sample_reporting_status="Reported",
                quantitative_result="AUROC 0.93",
                provenance_type="abstract_derived",
            ),
        ),
        PaperRecord(
            id="amp-2",
            canonical_study_id="study-amp-2",
            version_status="published",
            citation_index=2,
            title="GraphAMP: Equivariant Graph Networks for Peptide MIC Prediction",
            authors=["Carlos Rivera", "Dana Lee"],
            year=2024,
            venue="JCIM",
            source="Semantic Scholar",
            doi="10.1021/acs.jcim.4c001",
            abstract=(
                "GraphAMP predicts peptide minimum inhibitory concentration using 3D conformational graphs. "
                "On an internal test split (n = 1,500 sequences), GraphAMP reached RMSE of 0.42, whereas performance dropped on non-homologous sequences. "
                "Experimental wet-lab synthesis remains necessary."
            ),
            matched_subquestion_ids=["Q1"],
            relevance_score=0.87,
            stance="mixed",
            elicit=ElicitExtraction(
                methodology="Equivariant Graph Neural Network",
                dataset_or_system="Internal Conformational Dataset",
                key_findings="RMSE 0.42 on internal split; lower on non-homologous sequences",
                limitations="Drop on out-of-distribution sequences; no wet-lab synthesis",
                study_design="Computational Benchmark Study",
                sample_or_scale="n = 1,500 sequences",
                sample_reporting_status="Reported",
                quantitative_result="RMSE 0.42",
                provenance_type="abstract_derived",
            ),
        ),
    ]


class TestSection3KnowledgeGapMatrix(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        clear_gap_verification_cache()
        set_test_gap_verifier(None)

    def tearDown(self) -> None:
        clear_gap_verification_cache()
        set_test_gap_verifier(None)

    async def test_01_candidate_gap_supported_by_selected_evidence_and_section_integration(self):
        """
        Fixture 1: Verifies candidate gap generation from Sections 1 & 2, G1-G9 classification,
        separation of Research Opportunities from Knowledge Gaps, and Cross-Section ID resolution.
        """
        papers = _sample_amp_papers()
        query = "Deep learning models for antimicrobial peptide screening and in vivo validation"
        plan = QueryDecompositionPlan(
            detected_domain="AI & Molecular Biology",
            extracted_technical_entities=["antimicrobial peptide", "deep learning", "in vivo"],
        )
        findings = extract_atomic_findings(query, papers, plan)
        clusters = cluster_comparable_findings(query, papers, findings)
        claims, excluded = build_and_verify_claims(query, papers, findings, clusters)
        pkg = SynthesisEvidencePackage(findings=findings, clusters=clusters, claims=claims, excluded_claims=excluded)
        ev_matrix = build_structured_evidence_matrix(query, papers, plan, pkg)

        hf_res = [
            HFResourceRecord(
                id="facebook/esm2_t33_650M_UR50D",
                resource_type="model",
                pipeline_tag="feature-extraction",
                downloads=125000,
                url="https://huggingface.co/facebook/esm2_t33_650M_UR50D",
            )
        ]

        gap_matrix = await build_universal_knowledge_gap_matrix(
            query=query,
            papers=papers,
            hf_resources=hf_res,
            decomposition_plan=plan,
            synthesis_package=pkg,
            evidence_matrix=ev_matrix,
        )

        # Must have active corroborated gaps
        self.assertGreaterEqual(len(gap_matrix.structured_gaps), 1)
        g1 = gap_matrix.structured_gaps[0]
        self.assertEqual(g1.verification_status, "Corroborated Within Search Scope")
        self.assertEqual(g1.taxonomy_code, "G5")
        self.assertEqual(g1.primary_gap_type, "Validation deficiencies")
        self.assertIn("UNIT-1", g1.supporting_evidence_unit_ids)
        self.assertIn("R-1-1", g1.supporting_result_ids)

        # Must separate Hugging Face model transfer into research_opportunities (Phase 8)
        self.assertGreaterEqual(len(gap_matrix.research_opportunities), 1)
        self.assertIn("facebook/esm2_t33_650M_UR50D", gap_matrix.research_opportunities[0].suggested_hf_models_or_datasets)

        # Validation report must pass all 5 gates
        val = gap_matrix.validation_report
        self.assertTrue(val.structural_valid)
        self.assertTrue(val.evidence_grounding_valid)
        self.assertTrue(val.verification_integrity_valid)
        self.assertTrue(val.scientific_validity_valid)
        self.assertTrue(val.cross_section_consistency_valid)

    async def test_02_candidate_gap_invalidated_or_partially_addressed_by_supplementary_search(self):
        """
        Fixture 2 & 3 (Case B & Case E):
        - GAP-1 is directly resolved by supplementary search -> moves to resolved_or_invalidated_gaps,
          removed from active structured_gaps, flags related Section 1 claims, and does NOT expand primary papers.
        - GAP-2 is partially addressed by supplementary search -> status becomes 'Partially Addressed'.
        """
        papers = _sample_amp_papers()
        query = "Antimicrobial peptide screening validation"
        findings = extract_atomic_findings(query, papers)
        clusters = cluster_comparable_findings(query, papers, findings)
        claims, excluded = build_and_verify_claims(query, papers, findings, clusters)
        pkg = SynthesisEvidencePackage(findings=findings, clusters=clusters, claims=claims, excluded_claims=excluded)
        ev_matrix = build_structured_evidence_matrix(query, papers, None, pkg)

        def _mock_verifier(gap_id: str, verif_query: str) -> list[SupplementaryVerificationRecord]:
            if gap_id == "GAP-1":
                return [
                    SupplementaryVerificationRecord(
                        verification_source_id="SUPP-PMID-998877",
                        gap_id=gap_id,
                        query_executed=verif_query,
                        source_database="PubMed",
                        title="Multi-Center In Vivo Murine Validation of DeepAMP and GraphAMP Peptides",
                        authors=["S. Patel", "Alice Zhang"],
                        year=2025,
                        persistent_id="PMID:998877",
                        url="https://pubmed.ncbi.nlm.nih.gov/998877/",
                        abstract_excerpt="Directly validates DeepAMP and GraphAMP in vivo across 3 independent murine infection models.",
                        relevance_to_gap="directly_addresses",
                        assessment_note="Directly resolves the external/in vivo validation deficiency.",
                    )
                ]
            if gap_id == "GAP-2":
                return [
                    SupplementaryVerificationRecord(
                        verification_source_id="SUPP-ARXIV-2501",
                        gap_id=gap_id,
                        query_executed=verif_query,
                        source_database="arXiv",
                        title="Preliminary Standardized Benchmark for Peptide Predictors",
                        authors=["L. Gomez"],
                        year=2025,
                        persistent_id="arXiv:2501.00112",
                        relevance_to_gap="partially_addresses",
                        assessment_note="Harmonizes in silico splits but does not standardize MIC assay conditions.",
                    )
                ]
            return []

        set_test_gap_verifier(_mock_verifier)

        gap_matrix = await build_universal_knowledge_gap_matrix(
            query=query,
            papers=papers,
            hf_resources=[],
            synthesis_package=pkg,
            evidence_matrix=ev_matrix,
        )

        # Primary papers count must remain strictly unchanged (2 papers)
        self.assertEqual(len(papers), 2)

        # GAP-1 must be archived in resolved_or_invalidated_gaps and NOT in active structured_gaps
        active_ids = {g.gap_id for g in gap_matrix.structured_gaps}
        resolved_ids = {g.gap_id for g in gap_matrix.resolved_or_invalidated_gaps}
        self.assertNotIn("GAP-1", active_ids)
        self.assertIn("GAP-1", resolved_ids)
        self.assertEqual(
            gap_matrix.resolved_or_invalidated_gaps[0].verification_status,
            "Contradicted / Resolved by Retrieved Evidence",
        )

        # GAP-2 must be retained with status 'Partially Addressed'
        gap2 = next(g for g in gap_matrix.structured_gaps if g.gap_id == "GAP-2")
        self.assertEqual(gap2.verification_status, "Partially Addressed")

        # Supplementary verification records must be preserved separately
        self.assertEqual(len(gap_matrix.supplementary_verification_records), 2)

    async def test_03_missing_information_and_empty_corpus_cases(self):
        """
        Fixture 4 (Case G & Case H):
        - Metadata-only records yield 'Inconclusive' verification status and 'Insufficient information to prioritize'.
        - Empty corpus yields 0 fabricated gaps with an explicit scope explanation.
        """
        # Case H: Empty corpus
        empty_matrix = await build_universal_knowledge_gap_matrix(
            query="Quantum gravity in biological membranes",
            papers=[],
            hf_resources=[],
        )
        self.assertEqual(len(empty_matrix.structured_gaps), 0)
        self.assertEqual(empty_matrix.subquestion_coverage[0].coverage_status, "not_assessable")

        # Case G: Metadata-only record
        meta_paper = PaperRecord(
            id="meta-only-1",
            canonical_study_id="meta-only-1",
            version_status="published",
            citation_index=1,
            title="Unindexed Archival Report on Peptide Screening",
            authors=["R. Author"],
            year=2019,
            source="Semantic Scholar",
            abstract="",
            elicit=ElicitExtraction(
                study_design="Empirical Study",
                sample_reporting_status="Unavailable",
                provenance_type="metadata_only",
            ),
        )
        meta_matrix = await build_universal_knowledge_gap_matrix(
            query="Peptide screening archival report",
            papers=[meta_paper],
            hf_resources=[],
        )
        self.assertEqual(meta_matrix.subquestion_coverage[0].coverage_status, "not_assessable")
        for g in meta_matrix.structured_gaps:
            if g.supporting_study_indices == [1]:
                self.assertEqual(g.verification_status, "Inconclusive")
                self.assertEqual(g.priority_assessment, "Insufficient information to prioritize")

    async def test_04_mathematical_established_proof_vs_open_conjecture(self):
        """
        Fixture 5 (Case F & G9):
        - When a mathematical proposition already has a complete proof in the corpus and no open conjectures,
          do not fabricate unresolved gaps.
        - When a mathematical paper states an open conjecture in higher dimensions, classify as G9
          ('Theoretical or formal unresolved problems').
        """
        proven_paper = PaperRecord(
            id="math-proven",
            canonical_study_id="math-proven",
            version_status="published",
            citation_index=1,
            title="Complete Proof of the Sharp Cohomology Vanishing Theorem on Projective Schemes",
            authors=["E. Noether"],
            year=2024,
            venue="Annals of Mathematics",
            source="arXiv",
            abstract="We establish a sharp vanishing theorem and complete proof for sheaf cohomology on projective varieties.",
            elicit=ElicitExtraction(
                methodology="Spectral sequence proof",
                dataset_or_system="Projective varieties",
                study_design="Formal Mathematical Proof",
            ),
        )
        proven_matrix = await build_universal_knowledge_gap_matrix(
            query="Formal proof of cohomology vanishing theorem on projective varieties",
            papers=[proven_paper],
            hf_resources=[],
        )
        self.assertEqual(len(proven_matrix.structured_gaps), 0)
        self.assertIn("directly addressed by established evidence/proofs", proven_matrix.corpus_adequacy_explanation)

        # Open mathematical conjecture paper -> G9 gap
        open_math_paper = PaperRecord(
            id="math-open",
            canonical_study_id="math-open",
            version_status="published",
            citation_index=1,
            title="Partial Bounds and Open Conjectures for Non-Abelian Cohomology in Higher Dimensions",
            authors=["A. Grothendieck"],
            year=2024,
            venue="Inventiones Mathematicae",
            source="arXiv",
            abstract=(
                "We derive conditional bounds for rank-2 bundles, while the general conjecture in higher dimensions "
                "remains an open problem with unresolved counterexample regimes."
            ),
            elicit=ElicitExtraction(
                methodology="Cohomological obstruction theory",
                dataset_or_system="Non-abelian bundles in dimension >= 4",
                study_design="Formal Mathematical Analysis",
            ),
        )
        open_matrix = await build_universal_knowledge_gap_matrix(
            query="Formal proof and open conjecture for non-abelian cohomology bounds",
            papers=[open_math_paper],
            hf_resources=[],
        )
        self.assertGreaterEqual(len(open_matrix.structured_gaps), 1)
        self.assertEqual(open_matrix.structured_gaps[0].taxonomy_code, "G9")
        self.assertEqual(
            open_matrix.structured_gaps[0].primary_gap_type,
            "Theoretical or formal unresolved problems",
        )


if __name__ == "__main__":
    unittest.main()
