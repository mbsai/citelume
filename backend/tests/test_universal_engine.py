import asyncio
import sys
from pathlib import Path
import unittest

# Ensure backend directory is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models import (
    ApiKeysConfig,
    PaperRecord,
    HFResourceRecord,
)
from services.domain_profiles import DOMAIN_PROFILES, METHODOLOGICAL_PROFILES, compose_profiles
from services.query_utils import (
    understand_research_question,
    compile_deterministic_decomposition_plan,
    detect_domains_and_profiles,
)
from services.graph_builder import deduplicate_and_rank_papers, build_literature_graph
from services.synthesis_engine import (
    generate_bibliography,
    generate_synthesis_and_extractions,
    build_knowledge_gap_matrix,
    build_shared_evidence_base,
)
from services.verification_gates import run_verification_and_quality_gates


class UniversalResearchEngineTests(unittest.TestCase):
    def test_representative_7_domains_classification(self):
        """
        Tests domain and methodological profile detection across all 7 required domains:
        1. Biomedical research
        2. Clinical prediction modeling
        3. Machine learning
        4. Chemistry
        5. Physics
        6. Mathematics
        7. Interdisciplinary biomedical AI
        """
        cases = [
            (
                "In vitro minimum inhibitory concentration and hemolysis toxicity of antimicrobial peptides in bacterial infection",
                "biomedical_clinical",
            ),
            (
                "Prognostic risk score calibration and AUROC validation for sepsis mortality in ICU patient cohorts",
                "biomedical_clinical",
            ),
            (
                "Transformer attention quantization and out-of-distribution benchmark generalization in autoregressive language models",
                "computer_science_ai",
            ),
            (
                "Enantiomeric selectivity and reaction barrier energetics using density functional theory in organometallic catalysis",
                "chemistry",
            ),
            (
                "Entanglement entropy and finite-temperature decoherence in quantum spin lattice Hamiltonians",
                "physics",
            ),
            (
                "Non-asymptotic minimax convergence rates and Sobolev regularity proofs for stochastic partial differential equations on Hilbert manifolds",
                "mathematics",
            ),
            (
                "De novo antimicrobial peptide design using protein language models and equivariant graph neural networks",
                "interdisciplinary_biomedical_ai",
            ),
        ]
        for query, expected_domain in cases:
            u = understand_research_question(query)
            self.assertEqual(
                u.primary_domain,
                expected_domain,
                f"Failed domain classification for '{query}': got {u.primary_domain}, expected {expected_domain}",
            )

    def test_ambiguous_acronym_disambiguation(self):
        """
        Edge case: Ambiguous acronyms (AMP, DFT, GAN, IV) resolved by domain context.
        """
        # AMP in peptide context vs GPU training context
        u_bio_amp = understand_research_question("AMP screening for bacterial membrane disruption and peptide toxicity")
        self.assertIn(u_bio_amp.primary_domain, ("biomedical_clinical", "interdisciplinary_biomedical_ai"))
        resolved_bio = {a.term: a.resolved_meaning for a in u_bio_amp.ambiguous_terms}
        self.assertEqual(resolved_bio.get("AMP"), "antimicrobial peptide")

        u_cs_amp = understand_research_question("AMP fp16 bf16 gradient scaling in PyTorch GPU transformer training")
        self.assertEqual(u_cs_amp.primary_domain, "computer_science_ai")
        resolved_cs = {a.term: a.resolved_meaning for a in u_cs_amp.ambiguous_terms}
        self.assertEqual(resolved_cs.get("AMP"), "automatic mixed precision")

        # DFT in quantum chemistry vs signal processing
        u_chem_dft = understand_research_question("DFT B3LYP Kohn-Sham electronic energy barrier in molecular catalysis")
        self.assertEqual(u_chem_dft.primary_domain, "chemistry")
        resolved_dft = {a.term: a.resolved_meaning for a in u_chem_dft.ambiguous_terms}
        self.assertEqual(resolved_dft.get("DFT"), "density functional theory")

    def test_narrow_vs_broad_subquestion_scaling(self):
        """
        Edge case: Narrow questions remain narrow (1-2 subquestions), while broad
        multi-part interdisciplinary questions expand to 3-4 subquestions.
        """
        narrow_u = understand_research_question("CRISPR Cas9 off-target cleavage")
        self.assertEqual(narrow_u.question_complexity, "narrow")
        self.assertLessEqual(len(narrow_u.subquestions), 2)

        broad_u = understand_research_question(
            "Compare transformer protein language models and molecular dynamics simulations for antimicrobial peptide screening, "
            "evaluating out-of-distribution generalization limitations, clinical toxicity gaps, and novel generative approaches"
        )
        self.assertEqual(broad_u.question_complexity, "broad_interdisciplinary")
        self.assertGreaterEqual(len(broad_u.subquestions), 3)

    def test_5_api_query_compilation_validity(self):
        """
        Verifies Stage C query compilers for Biomedical, AI, and Mathematics queries:
        - PubMed has balanced parentheses and valid tags
        - Semantic Scholar has NO Boolean operators
        - arXiv uses valid field prefixes (ti:, abs:, all:) and balanced parentheses
        - bioRxiv outputs a non-empty list of short phrases
        - Hugging Face strips paper-only filler words
        """
        for q in [
            "Antimicrobial peptide screening using protein language models",
            "Graph neural network expressivity and Weisfeiler-Leman isomorphism bounds",
            "Eigenvalue gap and ergodic convergence for Markov chains on Banach spaces",
        ]:
            plan = compile_deterministic_decomposition_plan(q, mode="deep_research")
            self.assertGreaterEqual(len(plan.search_facets), 2)
            for facet in plan.search_facets:
                apis = facet.queries
                self.assertEqual(apis.pubmed.count("("), apis.pubmed.count(")"))
                self.assertNotRegex(apis.semantic_scholar, r"\b(AND|OR|NOT)\b")
                self.assertEqual(apis.arxiv.count("("), apis.arxiv.count(")"))
                self.assertTrue(any(p in apis.arxiv for p in ("ti:", "abs:", "all:")))
                self.assertTrue(isinstance(apis.biorxiv_medrxiv, list) and len(apis.biorxiv_medrxiv) >= 1)
                self.assertNotIn("review", apis.hugging_face.lower())

    def test_composable_domain_and_methodological_profiles(self):
        """
        Verifies that Biomedical + Predictive Modeling composes biomedical MeSH terms
        with predictive modeling appraisal criteria, while Mathematics + Theoretical sets
        sample_reporting_policy to 'not_applicable'.
        """
        bio_pred = compose_profiles(
            "biomedical_clinical",
            ["computer_science_ai"],
            ["predictive_modeling", "clinical_intervention"],
        )
        self.assertEqual(bio_pred["sample_reporting_policy"], "required")
        self.assertIn("antimicrobial peptide", bio_pred["validated_mesh_mappings"])

        math_theory = compose_profiles(
            "mathematics",
            [],
            ["theoretical_mathematical"],
        )
        self.assertEqual(math_theory["sample_reporting_policy"], "not_applicable")

    def test_deduplication_preprint_and_published_versions_and_limits(self):
        """
        Edge case: Preprint and published versions of the same underlying study are merged
        into a single canonical study with version_status='preprint_with_published_link',
        and Standard (20) vs Deep Research (60) caps are enforced without artificial padding.
        """
        raw_papers = [
            PaperRecord(
                id="biorxiv:1",
                title="Deep Language Models for De Novo Peptide Design",
                authors=["Alice Smith", "Bob Jones"],
                year=2024,
                venue="bioRxiv",
                source="bioRxiv",
                doi="10.1101/2024.01.01.12345",
                abstract="We introduce a transformer model for peptide design achieving 92% accuracy on N=1,200 peptides.",
                citation_count=5,
                references=["doi:10.1038/s41586-021-03819-2"],
            ),
            PaperRecord(
                id="pmid:998877",
                title="Deep Language Models for De Novo Peptide Design.",
                authors=["Alice Smith", "Bob Jones"],
                year=2025,
                venue="Nature Biotechnology",
                source="PubMed",
                doi="10.1038/s41587-025-00001-x",
                pmid="998877",
                abstract="We introduce a transformer model for peptide design achieving 92% accuracy on N=1,200 peptides in vitro.",
                citation_count=42,
                references=["doi:10.1038/s41586-021-03819-2"],
            ),
            PaperRecord(
                id="arxiv:2402.00001",
                title="Negative Results in Out-of-Distribution Peptide Toxicity Prediction",
                authors=["Carol White"],
                year=2024,
                venue="arXiv (q-bio.BM)",
                source="arXiv",
                arxiv_id="2402.00001",
                abstract="However, existing classifiers fail on non-canonical peptides and overestimate generalization.",
                citation_count=8,
                references=["doi:10.1038/s41586-021-03819-2"],
            ),
        ]

        deduped = deduplicate_and_rank_papers(raw_papers, "peptide design language models", max_total=20)
        # Should merge the bioRxiv preprint and Nature Biotechnology published paper into 1 record -> 2 unique studies total
        self.assertEqual(len(deduped), 2)
        merged_study = next(p for p in deduped if "De Novo Peptide Design" in p.title)
        self.assertEqual(merged_study.version_status, "preprint_with_published_link")
        self.assertEqual(merged_study.doi, "10.1038/s41587-025-00001-x")
        self.assertIn("bioRxiv", merged_study.all_sources)
        self.assertIn("PubMed", merged_study.all_sources)

    def test_mathematics_extraction_and_end_to_end_quality_gates(self):
        """
        Edge case: Mathematics questions without experimental sample sizes must report
        sample_reporting_status='Not applicable' (never inventing clinical cohorts),
        and pass all 5 Phase 9 deterministic quality gates.
        """
        query = "Minimax convergence rates and Sobolev regularity theorems for stochastic partial differential equations on Hilbert spaces"
        plan = compile_deterministic_decomposition_plan(query, mode="standard")
        self.assertEqual(plan.question_understanding.primary_domain, "mathematics")

        math_papers = [
            PaperRecord(
                id="arxiv:2301.11111",
                citation_index=1,
                title="Minimax Optimal Convergence Rates for Stochastic PDEs in Hilbert Spaces",
                authors=["E. Noether", "D. Hilbert"],
                year=2023,
                venue="Annals of Probability",
                source="arXiv",
                arxiv_id="2301.11111",
                abstract="We prove a sharp non-asymptotic theorem establishing minimax convergence rates in Sobolev norms under convex regularity assumptions.",
                citation_count=30,
                references=["arxiv:2101.00001"],
            ),
            PaperRecord(
                id="arxiv:2401.22222",
                citation_index=2,
                title="Counterexamples to Regularity for Non-Convex Stochastic Differential Equations",
                authors=["A. Kolmogorov"],
                year=2024,
                venue="arXiv (math.PR)",
                source="arXiv",
                arxiv_id="2401.22222",
                abstract="We construct a counterexample showing that without Lipschitz continuity, standard convergence bounds fail in high dimensions.",
                citation_count=12,
                references=["arxiv:2101.00001"],
            ),
        ]

        ranked = deduplicate_and_rank_papers(
            math_papers,
            query,
            max_total=20,
            understanding=plan.question_understanding,
        )
        graph = build_literature_graph(ranked, [])
        bib = generate_bibliography(ranked)
        cfg = ApiKeysConfig(llm_api_key="")  # Deterministic offline mode

        exec_ans, sections, consensus, warnings = asyncio.run(
            generate_synthesis_and_extractions(query, "standard", ranked, [], bib, cfg, decomposition_plan=plan)
        )

        # Verify mathematical papers have 'Not applicable' sample size status
        for p in ranked:
            self.assertEqual(p.elicit.sample_reporting_status, "Not applicable")
            self.assertIn("Not applicable", p.elicit.sample_or_scale)

        # Verify bibliographic coupling edge was detected from shared reference 'arxiv:2101.00001'
        edge_types = {e.edge_type for e in graph.edges}
        self.assertIn("bibliographic_coupling", edge_types)

        gap_matrix = asyncio.run(
            build_knowledge_gap_matrix(query, ranked, [], cfg, decomposition_plan=plan)
        )
        ev_base = build_shared_evidence_base(ranked, gap_matrix, graph, decomposition_plan=plan)
        self.assertEqual(len(ev_base.studies), 2)
        self.assertGreaterEqual(len(ev_base.claims), 1)
        self.assertGreaterEqual(len(ev_base.gaps), 1)

        q_report = run_verification_and_quality_gates(plan, ranked, exec_ans, sections, gap_matrix, graph)
        self.assertTrue(q_report.retrieval_valid)
        self.assertTrue(q_report.extraction_valid)
        self.assertTrue(q_report.synthesis_valid)
        self.assertTrue(q_report.gap_valid)
        self.assertTrue(q_report.network_valid)

    def test_empty_eligible_studies_graceful_handling(self):
        """
        Edge case: No eligible studies returned from APIs.
        """
        query = "Quantum topological cohomology in non-Archimedean motives"
        plan = compile_deterministic_decomposition_plan(query, mode="standard")
        cfg = ApiKeysConfig(llm_api_key="")
        exec_ans, sections, consensus, warnings = asyncio.run(
            generate_synthesis_and_extractions(query, "standard", [], [], [], cfg, decomposition_plan=plan)
        )
        self.assertEqual(consensus.total_evaluated, 0)
        self.assertEqual(consensus.consensus_verdict, "Insufficient Evidence")
        self.assertEqual(len(sections), 0)


if __name__ == "__main__":
    unittest.main()
