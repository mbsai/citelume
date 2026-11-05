import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    ElicitExtraction,
    PaperRecord,
    QueryDecompositionPlan,
    SynthesisEvidencePackage,
)
from services.claim_synthesis import (
    extract_atomic_findings,
    cluster_comparable_findings,
    build_and_verify_claims,
)
from services.evidence_table_engine import (
    build_query_adaptive_extraction_plan,
    build_structured_evidence_matrix,
    export_evidence_matrix_to_csv,
    export_evidence_matrix_to_json,
)


def _sample_papers() -> list[PaperRecord]:
    return [
        PaperRecord(
            id="s2-amp-1",
            canonical_study_id="study-amp-1",
            version_status="published",
            citation_index=1,
            title="DeepAMP-Transformer: Sequence-Based Antimicrobial Peptide Screening and Wet-Lab Validation",
            authors=["Alice Zhang", "Bob Chen"],
            year=2025,
            venue="Nature Machine Intelligence",
            source="PubMed",
            all_sources=["PubMed", "Semantic Scholar"],
            doi="10.1038/s42256-025-00001-x",
            pmid="39001122",
            abstract=(
                "We introduce DeepAMP-Transformer to screen antimicrobial peptides against multidrug-resistant pathogens. "
                "On the DBAASP holdout benchmark (n = 4,200 peptides), DeepAMP-Transformer achieved AUROC of 0.94 and accuracy of 91.2% compared with Random Forest baselines. "
                "In secondary wet-lab in vitro synthesis of 24 candidate peptides, 18 peptides demonstrated MIC <= 4 ug/mL against E. coli (p < 0.001, 95% CI 1.8 to 4.2). "
                "However, in vivo systemic toxicity and serum proteolytic stability remain key limitations requiring future investigation."
            ),
            citation_count=48,
            url="https://pubmed.ncbi.nlm.nih.gov/39001122/",
            matched_subquestion_ids=["Q1"],
            relevance_score=0.95,
            stance="supporting",
            elicit=ElicitExtraction(
                methodology="Transformer Encoder + Contrastive Pretraining",
                dataset_or_system="DBAASP Benchmark & E. coli ATCC 25922",
                key_findings="AUROC 0.94 on DBAASP; MIC <= 4 ug/mL in vitro",
                limitations="In vivo toxicity and proteolytic stability not evaluated",
                code_or_hf_link="https://huggingface.co/biolab/DeepAMP-Transformer",
                study_design="Computational Model + In Vitro Assay",
                sample_or_scale="n = 4,200 peptides",
                sample_reporting_status="Reported",
                quantitative_result="AUROC 0.94; accuracy 91.2%; MIC <= 4 ug/mL",
                provenance_type="abstract_derived",
            ),
        ),
        PaperRecord(
            id="s2-amp-2",
            canonical_study_id="study-amp-2",
            version_status="preprint_with_published_link",
            citation_index=2,
            title="GraphAMP: Graph Neural Networks on DBAASP for Peptide Activity Prediction",
            authors=["Alice Zhang", "Carlos Rivera"],
            year=2024,
            venue="bioRxiv",
            source="bioRxiv",
            all_sources=["bioRxiv", "Semantic Scholar"],
            doi="10.1093/bioinformatics/btae111",
            arxiv_id="2403.09999",
            abstract=(
                "We evaluate GraphAMP on the DBAASP benchmark dataset using 5-fold cross-validation. "
                "GraphAMP achieved AUROC of 0.89 and F1-score of 0.86 compared to CNN baselines. "
                "A limitation of this study is the lack of wet-lab experimental synthesis and potential sequence homology leakage across splits."
            ),
            citation_count=19,
            url="https://www.biorxiv.org/content/10.1101/2024.03.09999",
            matched_subquestion_ids=["Q1"],
            relevance_score=0.88,
            stance="mixed",
            elicit=ElicitExtraction(
                methodology="Equivariant Graph Neural Network (GNN)",
                dataset_or_system="DBAASP Benchmark & E. coli ATCC 25922",
                key_findings="AUROC 0.89 and F1-score 0.86 on DBAASP",
                limitations="Computational only; potential homology leakage",
                study_design="Computational Model + In Vitro Assay",
                sample_or_scale="Not reported",
                sample_reporting_status="Not reported",
                quantitative_result="AUROC 0.89; F1 0.86",
                provenance_type="abstract_derived",
            ),
        ),
        PaperRecord(
            id="s2-amp-3",
            canonical_study_id="study-amp-3",
            version_status="published",
            citation_index=3,
            title="Metadata-Only Legacy Record of Peptide Screening",
            authors=["D. Legacy"],
            year=2020,
            venue="Journal of Peptide Science",
            source="Semantic Scholar",
            abstract="",
            citation_count=5,
            url="https://example.org/legacy-amp",
            matched_subquestion_ids=["Q2"],
            relevance_score=0.52,
            stance="neutral",
            elicit=ElicitExtraction(
                methodology="",
                dataset_or_system="",
                key_findings="",
                limitations="",
                study_design="Empirical Study",
                sample_or_scale="Unavailable",
                sample_reporting_status="Unavailable",
                quantitative_result="Unavailable",
                provenance_type="metadata_only",
            ),
        ),
    ]


class TestSection2EvidenceMatrix(unittest.TestCase):
    def test_three_level_model_and_multi_result_extraction(self):
        papers = _sample_papers()
        query = "AI models and tools for antimicrobial peptide screening, limitations, and wet-lab validation"
        matrix = build_structured_evidence_matrix(query=query, papers=papers)

        # Verify Level A (Documents), Level B (Evidence Units), Level C (Granular Findings)
        self.assertEqual(len(matrix.documents), 3)
        self.assertEqual(len(matrix.evidence_units), 3)
        self.assertEqual(len(matrix.rows), 3)
        # Study 1 has multiple quantitative sentences in abstract -> >= 2 Level C findings
        unit1 = matrix.evidence_units[0]
        self.assertGreaterEqual(len(unit1.results), 2)
        self.assertEqual(unit1.relationship_type, "multi_experiment_unit")

        # Check derived quantity on Study 1 (91.2% -> 0.9120 proportion) without overwriting raw text
        has_derived = any(len(r.quantitative_detail.derived_quantities) > 0 for r in unit1.results)
        self.assertTrue(has_derived)
        self.assertIn("91.2%", unit1.results[0].factual_source_text)

        # Study 3 is metadata-only -> extraction status must be Unavailable
        unit3 = matrix.evidence_units[2]
        self.assertEqual(matrix.documents[2].evidence_access_level, "metadata_only")
        self.assertEqual(unit3.results[0].extraction_status, "Unavailable")
        self.assertEqual(unit3.comparability.status, "Insufficient information")

    def test_domain_specific_extraction_plans_across_disciplines(self):
        # 1. Clinical query
        plan_clin = build_query_adaptive_extraction_plan(
            "Randomized clinical trial of early norepinephrine in septic shock patient mortality"
        )
        self.assertEqual(plan_clin.domain_module_activated, "biomedical_clinical")
        self.assertTrue(any(f.priority == "Not applicable" for f in plan_clin.not_applicable_fields))

        # 2. Pure Mathematics query -> experimental sample size / p-value marked Not applicable
        plan_math = build_query_adaptive_extraction_plan(
            "Formal proof and cohomology bounds for algebraic varieties over Banach spaces"
        )
        self.assertEqual(plan_math.domain_module_activated, "mathematics")
        na_keys = {f.field_key for f in plan_math.not_applicable_fields}
        self.assertIn("experimental_sample_size_pvalue", na_keys)

        math_paper = PaperRecord(
            id="math-1",
            canonical_study_id="math-1",
            version_status="published",
            citation_index=1,
            title="Cohomology Vanishing Theorems on Projective Schemes",
            authors=["E. Noether"],
            year=2024,
            venue="Annals of Mathematics",
            source="arXiv",
            arxiv_id="2401.12345",
            abstract="We establish a sharp vanishing theorem for sheaf cohomology under regularity hypotheses on projective varieties.",
            elicit=ElicitExtraction(
                methodology="Spectral sequence construction",
                dataset_or_system="Projective varieties over algebraically closed fields",
                study_design="Formal Mathematical Proof",
            ),
        )
        math_matrix = build_structured_evidence_matrix(
            query="Formal proof and cohomology bounds for algebraic varieties",
            papers=[math_paper],
        )
        qd = math_matrix.evidence_units[0].results[0].quantitative_detail
        self.assertEqual(qd.sample_size_or_scale, "Not applicable")
        self.assertEqual(qd.p_value, "Not applicable")
        self.assertEqual(
            math_matrix.evidence_units[0].domain_field_statuses.get("experimental_sample_size_pvalue"),
            "Not applicable",
        )

        # 3. Chemistry & Materials Science
        plan_chem = build_query_adaptive_extraction_plan(
            "Perovskite electrocatalyst synthesis and Faraday efficiency for CO2 reduction"
        )
        self.assertEqual(plan_chem.domain_module_activated, "chemistry_materials")

        # 4. Physics
        plan_phys = build_query_adaptive_extraction_plan(
            "Quantum entanglement renormalization in condensed matter superconductivity"
        )
        self.assertEqual(plan_phys.domain_module_activated, "physics")

        # 5. Social Sciences
        plan_soc = build_query_adaptive_extraction_plan(
            "Difference-in-differences econometric analysis of minimum wage public policy on labor market"
        )
        self.assertEqual(plan_soc.domain_module_activated, "social_sciences_humanities")

    def test_appraisal_separation_comparability_and_section1_consistency(self):
        papers = _sample_papers()
        query = "Deep learning models for antimicrobial peptide screening on DBAASP"
        plan = QueryDecompositionPlan(
            detected_domain="AI & Molecular Biology",
            extracted_technical_entities=["antimicrobial peptide", "DBAASP", "deep learning"],
        )
        findings = extract_atomic_findings(query, papers, plan)
        clusters = cluster_comparable_findings(query, papers, findings)
        claims, excluded = build_and_verify_claims(query, papers, findings, clusters)
        pkg = SynthesisEvidencePackage(
            findings=findings,
            clusters=clusters,
            claims=claims,
            excluded_claims=excluded,
        )
        matrix = build_structured_evidence_matrix(
            query=query,
            papers=papers,
            decomposition_plan=plan,
            synthesis_package=pkg,
        )

        # Validation report must pass all 5 checks
        val = matrix.validation_report
        self.assertTrue(val.structural_valid)
        self.assertTrue(val.bibliographic_valid)
        self.assertTrue(val.factual_provenance_valid)
        self.assertTrue(val.scientific_comparability_valid)
        self.assertTrue(val.section1_consistency_valid)

        # Study 1 and Study 2 share design & system -> Directly comparable or Comparable with qualifications
        comp1 = matrix.evidence_units[0].comparability
        self.assertIn(comp1.status, ("Directly comparable", "Comparable with qualifications"))
        self.assertIn(2, comp1.comparable_peer_indices)

        # Study 2 has version_status="preprint_with_published_link" -> merged_preprint_and_published
        self.assertEqual(matrix.evidence_units[1].relationship_type, "merged_preprint_and_published")

        # CSV and JSON exports must serialize cleanly and preserve provenance columns
        csv_text = export_evidence_matrix_to_csv(matrix)
        self.assertIn("Citation_Ref,Document_ID,Contribution_ID", csv_text)
        self.assertIn("DeepAMP-Transformer", csv_text)

        json_text = export_evidence_matrix_to_json(matrix)
        parsed = json.loads(json_text)
        self.assertEqual(parsed["matrix_version"], "section2-evidence-matrix/2.0")
        self.assertEqual(len(parsed["rows"]), 3)


if __name__ == "__main__":
    unittest.main()
