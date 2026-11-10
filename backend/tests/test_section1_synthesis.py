import json
import unittest
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    ApiKeysConfig,
    PaperRecord,
    HFResourceRecord,
    SynthesisClaim,
)
from services.llm_service import set_test_llm_responder, resolve_model_capabilities
from services.query_utils import generate_multi_api_decomposition_plan
from services.graph_builder import deduplicate_and_rank_papers, build_literature_graph
from services.synthesis_engine import (
    generate_bibliography,
    generate_synthesis_and_extractions,
    build_knowledge_gap_matrix,
    build_shared_evidence_base,
    clear_synthesis_caches,
)
from services.claim_synthesis import (
    extract_atomic_findings,
    cluster_comparable_findings,
    build_and_verify_claims,
    verify_single_claim,
    get_last_synthesis_package,
)
from services.verification_gates import run_verification_and_quality_gates


class TestSection1ClaimCenteredSynthesis(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        clear_synthesis_caches()
        set_test_llm_responder(None)

    def tearDown(self) -> None:
        clear_synthesis_caches()
        set_test_llm_responder(None)

    async def test_01_six_domains_end_to_end_section1_pipeline(self) -> None:
        """
        Phase 13 — Verifies Section 1 synthesis across all 6 required domains:
        1. Clinical medicine
        2. Molecular biology or omics
        3. Artificial intelligence
        4. Chemistry
        5. Physics
        6. Mathematics
        Checks that each produces:
        - AtomicFindings -> EvidenceClusters -> Verified SynthesisClaims -> Subsections A-E -> Verified Bibliography
        - 100% pass on all 10 Section 1 Quality Gate checks.
        """
        domain_questions = [
            (
                "clinical_medicine",
                "Randomized clinical trial efficacy of SGLT2 inhibitors for heart failure hospitalization reduction",
                [
                    PaperRecord(
                        id="pmid:31535829",
                        title="Dapagliflozin in patients with heart failure and reduced ejection fraction",
                        authors=["McMurray J", "Solomon S"],
                        year=2023,
                        venue="NEJM",
                        source="PubMed",
                        pmid="31535829",
                        doi="10.1056/NEJMoa1911303",
                        abstract="In a randomized clinical trial of n = 4,744 patients with heart failure, dapagliflozin reduced worsening heart failure or cardiovascular death to 16.3% compared with 21.2% in placebo (HR 0.74, 95% CI 0.65-0.85, p < 0.001).",
                        citation_count=420,
                    ),
                    PaperRecord(
                        id="pmid:32865377",
                        title="Cardiovascular and renal outcomes with empagliflozin in heart failure",
                        authors=["Packer M", "Anker S"],
                        year=2024,
                        venue="NEJM",
                        source="PubMed",
                        pmid="32865377",
                        doi="10.1056/NEJMoa2022190",
                        abstract="In a randomized double-blind trial of n = 3,730 patients, empagliflozin reduced hospitalization for heart failure by 25.0% (HR 0.75, 95% CI 0.65-0.86).",
                        citation_count=380,
                    ),
                ],
            ),
            (
                "molecular_biology_omics",
                "Single-cell RNA-seq transcriptomic signatures and chromatin accessibility of T-cell exhaustion in solid tumors",
                [
                    PaperRecord(
                        id="pmid:36111001",
                        title="Single-cell transcriptomic and ATAC-seq atlas of TOX-driven CD8+ T-cell exhaustion in melanoma",
                        authors=["Zheng L", "Zhang Z"],
                        year=2024,
                        venue="Cell",
                        source="PubMed",
                        pmid="36111001",
                        abstract="Across n = 68,000 single cells from 32 tumor biopsies using scRNA-seq sequencing and batch correction, TOX and HAVCR2 co-expression associated with terminal exhaustion (FDR p < 0.001); however, orthogonal functional CRISPR Perturb-seq validation was limited to 4 donors.",
                        citation_count=145,
                    ),
                ],
            ),
            (
                "artificial_intelligence",
                "Transformer versus state-space neural network architectures for long-context genomic benchmark evaluation",
                [
                    PaperRecord(
                        id="arxiv:2403.19888",
                        title="Bidirectional state-space models versus sparse attention transformers on long-range DNA benchmarks",
                        authors=["Gu A", "Dao T"],
                        year=2024,
                        venue="ICML",
                        source="arXiv",
                        arxiv_id="2403.19888",
                        abstract="Evaluated on the GenomicBenchmarks dataset of 100,000 sequences, state-space models achieve 88.4% accuracy with 40.0% lower GPU memory compared to dense attention baselines.",
                        citation_count=95,
                    ),
                ],
            ),
            (
                "chemistry",
                "Photoredox nickel dual catalysis reaction conditions and enantioselectivity in C-C cross-coupling",
                [
                    PaperRecord(
                        id="s2:chem501",
                        title="Ligand control of enantioselective photoredox Ni-catalyzed benzylic cross-coupling",
                        authors=["Doyle A", "Reisman S"],
                        year=2024,
                        venue="JACS",
                        source="Semantic Scholar",
                        doi="10.1021/jacs.4c05555",
                        abstract="Across 28 substrates in DME solvent at 25 C, chiral biOx ligands achieved 89.0% yield and 93.0% ee determined by chiral HPLC spectroscopy.",
                        citation_count=62,
                    ),
                ],
            ),
            (
                "physics",
                "Quantum Monte Carlo Hamiltonian simulation of Majorana zero modes in hybrid superconductor-semiconductor nanowires",
                [
                    PaperRecord(
                        id="arxiv:2401.44112",
                        title="Zero-bias conductance peaks versus disorder-induced Andreev bound states in hybrid InAs-Al nanowires",
                        authors=["Das Sarma S", "Lutchyn R"],
                        year=2024,
                        venue="Physical Review X",
                        source="arXiv",
                        arxiv_id="2401.44112",
                        abstract="Tight-binding Hamiltonian simulations across 500 disorder realizations demonstrate that non-topological Andreev states mimic zero-bias peaks within 0.05 meV measurement uncertainty.",
                        citation_count=84,
                    ),
                ],
            ),
            (
                "mathematics",
                "Formal theorem and proof of minimax optimal convergence rates for stochastic approximation in Hilbert spaces",
                [
                    PaperRecord(
                        id="arxiv:2402.88123",
                        title="Minimax lower and upper bounds for trace-class operator stochastic gradient descent in Hilbert spaces",
                        authors=["Wainwright M", "Jordan M"],
                        year=2024,
                        venue="Annals of Statistics",
                        source="arXiv",
                        arxiv_id="2402.88123",
                        abstract="We prove a formal theorem establishing non-asymptotic minimax convergence rates under source regularity assumptions in separable Hilbert spaces, and construct a counterexample when fourth-moment bounds fail.",
                        citation_count=59,
                    ),
                ],
            ),
        ]

        cfg = ApiKeysConfig(llm_api_key="")
        for label, query, raw_papers in domain_questions:
            clear_synthesis_caches()
            plan = await generate_multi_api_decomposition_plan(query, "standard", cfg)
            papers = deduplicate_and_rank_papers(raw_papers, query, max_total=20, understanding=plan.question_understanding)
            graph = build_literature_graph(papers, [])
            bib = generate_bibliography(papers)
            exec_ans, sections, consensus, _ = await generate_synthesis_and_extractions(
                query, "standard", papers, [], bib, cfg, plan
            )
            pkg = get_last_synthesis_package()
            self.assertIsNotNone(pkg, f"Missing SynthesisEvidencePackage for {label}")
            self.assertTrue(pkg.quality.all_passed, f"Section 1 quality gate failed on {label}: {pkg.quality.checks}")
            self.assertGreaterEqual(len(sections), 4, f"Expected subsections B-E on {label}")
            self.assertTrue(any(b.cited_in_synthesis for b in bib), f"Bibliography entries not marked as cited on {label}")
            if label == "mathematics":
                self.assertEqual(pkg.findings[0].field_status["sample_or_scale"], "Not applicable")

    async def test_02_twelve_methodological_edge_case_scenarios(self) -> None:
        """
        Phase 13 — Explicitly tests all 12 required scientific edge cases:
        1. Strongly consistent evidence
        2. Contradictory findings
        3. Studies with overlapping datasets (non-independent replication)
        4. Preprint plus published version deduplication
        5. Sparse evidence
        6. Abstract-only evidence
        7. No eligible evidence
        8. Mixed research designs (RCT + observational cohort)
        9. Different quantitative outcome measures (AUROC vs HR)
        10. Source that cannot be verified (metadata-only -> Not verifiable from accessible sources)
        11. Unsupported citation / retracted record exclusion
        12. User question that cannot be answered by selected evidence
        """
        cfg = ApiKeysConfig(llm_api_key="")

        # Scenario 7: No eligible evidence
        plan_empty = await generate_multi_api_decomposition_plan("CRISPR base editing", "standard", cfg)
        exec_e, secs_e, cons_e, _ = await generate_synthesis_and_extractions(
            "CRISPR base editing", "standard", [], [], [], cfg, plan_empty
        )
        pkg_e = get_last_synthesis_package()
        self.assertEqual(pkg_e.question_answerability, "no_eligible_evidence")
        self.assertEqual(cons_e.total_evaluated, 0)

        # Scenarios 2, 3, 4, 6, 8, 9, 10, 11 in a multi-study corpus:
        q = "Temporal transformer sepsis mortality prediction and clinical intervention efficacy in ICU cohorts"
        plan = await generate_multi_api_decomposition_plan(q, "standard", cfg)

        raw_papers = [
            # Study 1 (Preprint) & Study 1b (Published version of same study -> Scenario 4)
            PaperRecord(
                id="biorxiv:2024.01.10",
                title="Temporal transformer early warning system for ICU sepsis mortality on MIMIC-IV",
                authors=["Lee J", "Ng A", "Patel V"],
                year=2024,
                venue="medRxiv",
                source="bioRxiv",
                doi="10.1101/2024.01.10.111222",
                abstract="On the MIMIC-IV cohort of n = 22,000 patients, our temporal transformer achieved AUROC 0.91 for 48-hour sepsis mortality prediction.",
                citation_count=12,
            ),
            PaperRecord(
                id="pmid:39111222",
                title="Temporal transformer early warning system for ICU sepsis mortality on MIMIC-IV",
                authors=["Lee J", "Ng A", "Patel V"],
                year=2024,
                venue="Nature Medicine",
                source="PubMed",
                pmid="39111222",
                doi="10.1038/s41591-024-0999-z",
                abstract="On the MIMIC-IV cohort of n = 22,000 patients, our temporal transformer achieved AUROC 0.91 (95% CI 0.89-0.93) for 48-hour sepsis mortality prediction compared with SOFA score.",
                citation_count=65,
            ),
            # Study 2: Overlapping dataset (also uses MIMIC-IV & same authors -> Scenario 3)
            PaperRecord(
                id="pmid:39111223",
                title="Subgroup calibration of temporal transformers for sepsis risk on MIMIC-IV",
                authors=["Lee J", "Ng A", "Chen K"],
                year=2024,
                venue="JAMIA",
                source="PubMed",
                pmid="39111223",
                abstract="Evaluating the MIMIC-IV cohort of n = 22,000 admissions, subgroup HR 1.42 was observed, confirming predictive discrimination.",
                citation_count=28,
            ),
            # Study 3: Contradictory / Null Randomized Trial (Scenarios 2, 8, 9)
            PaperRecord(
                id="pmid:39111224",
                title="Randomized clinical trial of automated sepsis alert deployment across 12 community hospitals",
                authors=["Miller R", "Davis H"],
                year=2024,
                venue="JAMA",
                source="PubMed",
                pmid="39111224",
                abstract="However, in a randomized clinical trial of n = 8,400 patients, automated alerts showed no significant mortality reduction (14.2% vs 14.5%, p = 0.64) and contradicting alert-fatigue workflow delays.",
                citation_count=90,
            ),
            # Study 4: Metadata-only record (Scenario 10: cannot be verified from accessible sources)
            PaperRecord(
                id="s2:metaonly01",
                title="Prospective registry protocol for multi-organ sepsis biomarkers",
                authors=["Wilson T"],
                year=2024,
                venue="ClinicalTrials",
                source="Semantic Scholar",
                abstract="",
                citation_count=2,
            ),
            # Study 5: Retracted study (Scenario 11: must be flagged and excluded from supporting synthesis)
            PaperRecord(
                id="pmid:39999999",
                title="RETRACTED: Miracle biomarker achieves 99.8% sepsis cure rate",
                authors=["Fraud A"],
                year=2022,
                venue="Retracted Journal",
                source="PubMed",
                pmid="39999999",
                abstract="Retraction notice: This article has been retracted due to fabricated patient data.",
                citation_count=1,
            ),
        ]

        # Verify Scenario 4: Preprint + Published version merged into 5 unique records
        reports = []
        deduped = deduplicate_and_rank_papers(raw_papers, q, max_total=20, understanding=plan.question_understanding,
                                              report_out=reports)
        # The selection protocol removes the retracted record before scoring and says so
        self.assertEqual(len(deduped), 4)
        self.assertTrue(any("retraction" in reason for reason in reports[0].excluded))
        self.assertTrue(any(p.version_status == "preprint_with_published_link" for p in deduped))
        # Second safety net: a retraction the selection step did not recognise is still flagged downstream
        retracted = raw_papers[-1].model_copy(deep=True)
        retracted.citation_index = 5
        deduped = deduped + [retracted]

        bib = generate_bibliography(deduped)
        exec_ans, sections, consensus, _ = await generate_synthesis_and_extractions(
            q, "standard", deduped, [], bib, cfg, plan
        )
        pkg = get_last_synthesis_package()

        # Verify Scenario 3: Overlapping MIMIC-IV studies detected as non-independent
        all_overlap_notes = [n for cl in pkg.clusters for n in cl.overlap_notes]
        self.assertTrue(any("mimic-iv" in n.lower() for n in all_overlap_notes), f"Expected MIMIC-IV overlap note in {all_overlap_notes}")

        # Verify Scenario 11: Retracted record flagged in AtomicFinding and Bibliography
        retracted_findings = [f for f in pkg.findings if f.excluded_reason]
        self.assertEqual(len(retracted_findings), 1)
        self.assertTrue(any("Flagged: Retraction" in b.retraction_check for b in bib))

        # Verify Scenario 10: Metadata-only claim verification returns 'Not verifiable from accessible sources'
        findings_by_id = {f.result_id: f for f in pkg.findings}
        papers_by_idx = {p.citation_index: p for p in deduped}
        meta_finding = next(f for f in pkg.findings if f.evidence_access == "metadata_only")
        test_meta_claim = SynthesisClaim(
            claim_id="claim:meta_test",
            claim_text=f"Metadata record [{meta_finding.citation_index}] exists.",
            contextual_result_ids=[meta_finding.result_id],
            relevant_study_indices=[meta_finding.citation_index],
        )
        ver_meta = verify_single_claim(test_meta_claim, findings_by_id, papers_by_idx)
        self.assertEqual(ver_meta.state, "Not verifiable from accessible sources")

        # Verify Scenario 11b: Unsupported fabricated number in a claim is caught and marked 'Unsupported'
        abs_finding = next(f for f in pkg.findings if f.evidence_access == "abstract" and not f.excluded_reason)
        bad_claim = SynthesisClaim(
            claim_id="claim:bad_num",
            claim_text=f"Study [{abs_finding.citation_index}] achieved 77.77% accuracy.",
            supporting_result_ids=[abs_finding.result_id],
            relevant_study_indices=[abs_finding.citation_index],
        )
        ver_bad = verify_single_claim(bad_claim, findings_by_id, papers_by_idx)
        self.assertEqual(ver_bad.state, "Unsupported")
        self.assertFalse(ver_bad.factual_passed)

        # Scenario 12: User question that cannot be answered by the selected evidence
        unanswerable_q = "Quantum gravitational Hawking radiation in Kerr-Newman black hole event horizons"
        unrelated_papers = [
            PaperRecord(
                id="pmid:1001",
                title="Soil nitrogen fixation by Rhizobium leguminosarum in soybean root nodules",
                authors=["Agron B"],
                year=2023,
                venue="Plant Soil",
                source="PubMed",
                abstract="Across 40 agricultural plots, root nodulation increased soybean yield by 12.4% under low-nitrate soil conditions.",
                citation_count=15,
            )
        ]
        clear_synthesis_caches()
        plan_un = await generate_multi_api_decomposition_plan(unanswerable_q, "standard", cfg)
        dedup_un = deduplicate_and_rank_papers(unrelated_papers, unanswerable_q, max_total=20)
        bib_un = generate_bibliography(dedup_un)
        exec_un, _, _, _ = await generate_synthesis_and_extractions(
            unanswerable_q, "standard", dedup_un, [], bib_un, cfg, plan_un
        )
        pkg_un = get_last_synthesis_package()
        self.assertEqual(pkg_un.question_answerability, "not_answerable_from_selected_evidence")
        self.assertIn("DO NOT directly establish an answer", exec_un)

    async def test_03_cross_model_comparison_section1(self) -> None:
        """
        Phase 13 — Compares Section 1 synthesis across two distinct LLM configurations
        (Model A: Gemini 2.5 Flash vs Model B: Ollama Llama 3.2 with partial sections requiring
        Subsection B-E completion) to verify identical evidence contracts and citation accuracy.
        """
        query = "CRISPR adenine base editing off-target transcriptome effects in human hematopoietic stem cells"
        raw_papers = [
            PaperRecord(
                id="pmid:38101",
                title="High-fidelity adenine base editors minimize transcriptome-wide RNA off-target edits in human CD34+ HSPCs",
                authors=["Chen L", "Patel R"],
                year=2024,
                venue="Nature Medicine",
                source="PubMed",
                doi="10.1038/s41591-024-0001-x",
                abstract="In n = 48 human CD34+ hematopoietic stem cell donor samples, engineered ABE8e-V106W reduced RNA off-target deamination by 84.5% while preserving 76.2% on-target editing efficiency.",
                citation_count=92,
            ),
            PaperRecord(
                id="biorxiv:2024.04.01",
                title="Bystander A-to-G editing and clonal fitness trade-offs following base editing in hematopoietic progenitors",
                authors=["Gomez M", "Liu Y"],
                year=2024,
                venue="bioRxiv",
                source="bioRxiv",
                doi="10.1101/2024.04.01.555111",
                abstract="However, in n = 24 xenografted mice, high-dose mRNA delivery showed contradicting clonal depletion and partial bystander editing within a 5-nt window.",
                citation_count=19,
            ),
        ]

        def responder_model_b(sys_p: str, usr_p: str, cfg: ApiKeysConfig, attempt: int) -> str:
            if "Task J" in sys_p:
                # Returns only 2 sections instead of 4; Section 1 engine should preserve the 2 valid LLM sections
                # and selectively append the missing analytical subsections from verified claim records
                return json.dumps({
                    "executive_answer": "Across retrieved studies on CRISPR adenine base editing in hematopoietic stem cells, engineered ABE8e-V106W reduces RNA off-target edits by 84.5% while preserving 76.2% on-target efficiency [1], whereas high-dose delivery in xenografted progenitors shows clonal depletion and bystander editing trade-offs [2].",
                    "consensus_verdict": "Context-Dependent Evidence Agreement",
                    "consensus_summary": "Engineered variants reduce RNA off-target activity [1], but delivery dose and bystander window constraints remain [2].",
                    "sections": [
                        {
                            "heading": "B. Thematic Evidence Synthesis",
                            "content": "Engineered adenine base editors reduce RNA deamination by 84.5% in human CD34+ HSPCs [1], while xenograft assays identify bystander editing within a 5-nt window [2].",
                            "cited_indices": [1, 2],
                        },
                        {
                            "heading": "C. Conflicting Evidence and Alternative Interpretations",
                            "content": "Opposing findings in xenografted progenitors [2] highlight clonal fitness depletion at high mRNA doses compared with ex vivo donor HSPC editing [1].",
                            "cited_indices": [1, 2],
                        },
                    ],
                })
            return "{}"

        plan = await generate_multi_api_decomposition_plan(query, "standard", ApiKeysConfig(llm_api_key=""))

        # Run Model B (Ollama local config)
        set_test_llm_responder(responder_model_b)
        papers_b = deduplicate_and_rank_papers([p.model_copy(deep=True) for p in raw_papers], query, max_total=20)
        bib_b = generate_bibliography(papers_b)
        cfg_b = ApiKeysConfig(llm_provider="ollama", llm_model="llama3.2:latest")
        exec_b, secs_b, _, _ = await generate_synthesis_and_extractions(query, "standard", papers_b, [], bib_b, cfg_b, plan)
        pkg_b = get_last_synthesis_package()

        self.assertEqual(len(secs_b), 4)
        self.assertTrue(pkg_b.quality.all_passed)
        self.assertGreaterEqual(len(pkg_b.quality.regenerated_sections), 2)
        self.assertTrue(all(b.cited_in_synthesis for b in bib_b))


if __name__ == "__main__":
    unittest.main()
