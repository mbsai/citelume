import json
import unittest
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    ApiKeysConfig,
    PaperRecord,
    HFResourceRecord,
)
from services.llm_service import (
    resolve_model_capabilities,
    set_test_llm_responder,
    _LIVE_MODEL_METADATA_CACHE,
)
from services.query_utils import generate_multi_api_decomposition_plan
from services.graph_builder import deduplicate_and_rank_papers, build_literature_graph
from services.synthesis_engine import (
    generate_bibliography,
    generate_synthesis_and_extractions,
    build_knowledge_gap_matrix,
    build_shared_evidence_base,
    clear_synthesis_caches,
)
from services.verification_gates import run_verification_and_quality_gates


BENCHMARK_DOMAIN_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "biomedical_research",
        "domain_expected": "biomedical_clinical",
        "query": "CRISPR adenine base editing off-target transcriptome effects in human hematopoietic stem cells",
        "papers": [
            PaperRecord(
                id="pubmed:38101",
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
        ],
    },
    {
        "case_id": "clinical_prediction_modeling",
        "domain_expected": "interdisciplinary_biomedical_ai",
        "query": "Observational cohort evaluation of transformer early warning scores for ICU sepsis mortality prediction",
        "papers": [
            PaperRecord(
                id="pubmed:38202",
                title="Retrospective cohort validation of temporal transformers for 48-hour ICU sepsis mortality prediction",
                authors=["Kim S", "Adams T"],
                year=2024,
                venue="Lancet Digital Health",
                source="PubMed",
                doi="10.1016/S2589-7500(24)00011-2",
                abstract="In a retrospective observational cohort of n = 14,500 ICU admissions, the temporal model achieved AUROC 0.89 for sepsis mortality prediction; however, external hospital calibration drifted.",
                citation_count=64,
            ),
            PaperRecord(
                id="pubmed:38203",
                title="Dataset shift and false-positive alert fatigue in observational clinical sepsis prediction models",
                authors=["Vance E", "Okafor C"],
                year=2023,
                venue="JAMA Internal Medicine",
                source="PubMed",
                doi="10.1001/jamainternmed.2023.4401",
                abstract="Across n = 6,200 retrospective patient encounters, observational risk scores showed no significant mortality reduction without prospective clinical trial validation.",
                citation_count=110,
            ),
        ],
    },
    {
        "case_id": "machine_learning",
        "domain_expected": "computer_science_ai",
        "query": "SE(3)-equivariant graph neural networks versus score-based diffusion models for 3D molecular conformer generation",
        "papers": [
            PaperRecord(
                id="arxiv:2402.11101",
                title="Torsional flow matching with SE(3)-equivariant message passing for fast molecular conformer generation",
                authors=["Zhang H", "Bronstein M"],
                year=2024,
                venue="ICLR",
                source="arXiv",
                arxiv_id="2402.11101",
                abstract="On the GEOM-Drugs benchmark dataset of 45,000 molecules, equivariant flow matching improves coverage recall to 82.4% with 15.0% lower RMSD.",
                citation_count=78,
            ),
            PaperRecord(
                id="arxiv:2403.22202",
                title="Failure modes of equivariant diffusion on macrocyclic and transition-metal complexes",
                authors=["Keller D", "Smidt T"],
                year=2024,
                venue="NeurIPS",
                source="arXiv",
                arxiv_id="2403.22202",
                abstract="However, when evaluated on out-of-distribution macrocycles, chiral inversion errors contradiciting standard small-molecule benchmarks occur in 28.0% of conformers.",
                citation_count=31,
            ),
        ],
    },
    {
        "case_id": "chemistry",
        "domain_expected": "chemistry",
        "query": "Enantioselective photoredox nickel dual catalysis for C(sp3)-C(sp2) cross-coupling reaction mechanism",
        "papers": [
            PaperRecord(
                id="s2:chem901",
                title="Chiral bis(oxazoline)-nickel and iridium photoredox catalysis enables high-ee benzylic cross-coupling",
                authors=["MacMillan D", "Sato K"],
                year=2024,
                venue="JACS",
                source="Semantic Scholar",
                doi="10.1021/jacs.4c01111",
                abstract="Across 36 aryl bromide substrates, the catalytic assay achieved 91.0% yield and 94.0% enantiomeric excess under blue LED irradiation at 25 C.",
                citation_count=85,
            ),
            PaperRecord(
                id="s2:chem902",
                title="Spectroscopic kinetic analysis reveals competing radical homocoupling in Ni/photoredox cycles",
                authors=["Lloyd-Jones G", "Bauer F"],
                year=2023,
                venue="Angewandte Chemie",
                source="Semantic Scholar",
                doi="10.1002/anie.202309999",
                abstract="Transient absorption spectroscopy demonstrates that sterically hindered tertiary radicals fail to undergo reductive elimination, yielding 34.0% byproduct.",
                citation_count=52,
            ),
        ],
    },
    {
        "case_id": "physics_or_mathematics",
        "domain_expected": "mathematics",
        "query": "Non-asymptotic minimax convergence proof for stochastic gradient descent in Hilbert spaces",
        "papers": [
            PaperRecord(
                id="arxiv:2401.77001",
                title="Optimal non-asymptotic minimax rates for trace-class stochastic operator regression in Hilbert spaces",
                authors=["Bach F", "Maillard O"],
                year=2024,
                venue="Annals of Statistics",
                source="arXiv",
                arxiv_id="2401.77001",
                abstract="We establish a formal theorem and proof showing that averaged stochastic gradient descent achieves minimax optimal convergence rates under source regularity conditions in separable Hilbert spaces.",
                citation_count=49,
            ),
            PaperRecord(
                id="arxiv:2401.77002",
                title="Lower bound counterexamples for heavy-tailed stochastic approximation without strong convexity",
                authors=["Nemirovski A", "Polyak B"],
                year=2023,
                venue="Mathematical Programming",
                source="arXiv",
                arxiv_id="2401.77002",
                abstract="We construct a formal counterexample demonstrating that when noise variance is infinite, standard constant-step schedules fail to converge in Banach spaces.",
                citation_count=67,
            ),
        ],
    },
    {
        "case_id": "interdisciplinary_research",
        "domain_expected": "interdisciplinary_biomedical_ai",
        "query": "Deep learning transformer neural network models for antimicrobial peptide screening and hemolysis toxicity reduction",
        "papers": [
            PaperRecord(
                id="pubmed:39001",
                title="Fine-tuning ESM-2 protein language models for broad-spectrum antimicrobial peptide discovery",
                authors=["Santos A", "Wang J"],
                year=2024,
                venue="Nature Biotechnology",
                source="PubMed",
                doi="10.1038/s41587-024-02111-0",
                arxiv_id="2402.99001",
                abstract="Evaluated on n = 12,000 peptide sequences and in vitro MIC assays, our transformer classifier achieved 93.5% accuracy and identified 18 leads with MIC < 4 ug/mL.",
                citation_count=105,
            ),
            PaperRecord(
                id="biorxiv:2024.06.15",
                title="Generalization bottleneck and hemolytic toxicity of generative antimicrobial peptide models in serum",
                authors=["Novak P", "Roy S"],
                year=2024,
                venue="bioRxiv",
                source="bioRxiv",
                doi="10.1101/2024.06.15.600123",
                abstract="However, when tested in 50.0% human serum, 62.0% of de novo generated cationic peptides failed due to proteolytic degradation and erythrocyte hemolysis.",
                citation_count=27,
            ),
        ],
    },
]


class TestCrossModelQualityGatedBenchmark(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        clear_synthesis_caches()
        set_test_llm_responder(None)

    def tearDown(self) -> None:
        clear_synthesis_caches()
        set_test_llm_responder(None)

    def test_01_model_capability_abstraction_layer(self) -> None:
        """
        Verifies Section 2: Model Capability Abstraction Layer resolves provider metadata
        and documented configurations (context window, study batch sizes, native JSON support).
        """
        gemini_cfg = ApiKeysConfig(llm_provider="gemini", llm_model="gemini-2.5-flash", llm_api_key="test")
        cap_gem = resolve_model_capabilities(gemini_cfg)
        self.assertTrue(cap_gem.supports_native_json_schema)
        self.assertTrue(cap_gem.supports_json_object_mode)
        self.assertGreaterEqual(cap_gem.context_window_tokens, 128000)
        self.assertEqual(cap_gem.recommended_study_batch_size, 20)

        ollama_cfg = ApiKeysConfig(llm_provider="ollama", llm_model="llama3.2:latest")
        cap_oll = resolve_model_capabilities(ollama_cfg)
        self.assertEqual(cap_oll.context_window_tokens, 8192)
        self.assertEqual(cap_oll.recommended_study_batch_size, 5)
        self.assertEqual(cap_oll.observed_quality_tier, "tier_3_constrained_local")

        # Verify live metadata cache override (e.g. from OpenRouter /models endpoint)
        _LIVE_MODEL_METADATA_CACHE["openrouter:deepseek/deepseek-r1:free"] = {
            "context_length": 16384,
            "max_output_tokens": 2048,
            "supports_json_object_mode": False,
        }
        or_cfg = ApiKeysConfig(llm_provider="openrouter", llm_model="deepseek/deepseek-r1:free", llm_api_key="sk-or-v1-test")
        cap_or = resolve_model_capabilities(or_cfg)
        self.assertEqual(cap_or.capability_source, "provider_metadata")
        self.assertEqual(cap_or.context_window_tokens, 16384)
        self.assertEqual(cap_or.recommended_study_batch_size, 8)
        self.assertFalse(cap_or.supports_json_object_mode)

    async def test_02_six_domain_benchmark_across_compliant_and_offline_models(self) -> None:
        """
        Verifies Section 10: Runs all 6 benchmark domains (Biomedical, Clinical Prediction,
        Machine Learning, Chemistry, Mathematics, Interdisciplinary Bio-AI) and evaluates both
        Deterministic Metrics (100% schema, citation, numerical, and inference validity) and
        Semantic Quality Metrics.
        """
        def compliant_llm_responder(sys_p: str, usr_p: str, cfg: ApiKeysConfig, attempt: int) -> str:
            if "Task C" in sys_p:
                is_math = "mathematics" in sys_p
                return json.dumps({
                    "study_extractions": [
                        {
                            "citation_index": 1,
                            "stance": "supporting",
                            "study_design": "Formal Mathematical Proof / Analysis" if is_math else "Empirical / Computational Study",
                            "methodology": "Primary validated methodology",
                            "dataset_or_system": "Formal mathematical setting [Sample size: N/A]" if is_math else "Primary benchmark system",
                            "sample_or_scale": "Not applicable (theoretical/mathematical formulation)" if is_math else "Not reported in abstract",
                            "sample_reporting_status": "Not applicable" if is_math else "Not reported",
                            "quantitative_result": "Not reported in abstract",
                            "key_findings": "Demonstrates primary validated finding from source record.",
                            "limitations": "Subject to stated scope and regularity assumptions.",
                        },
                        {
                            "citation_index": 2,
                            "stance": "contradicting",
                            "study_design": "Formal Mathematical Proof / Analysis" if is_math else "Empirical / Computational Study",
                            "methodology": "Counterexample / Robustness analysis",
                            "dataset_or_system": "Out-of-distribution or counterexample regime",
                            "sample_or_scale": "Not applicable (theoretical/mathematical formulation)" if is_math else "Not reported in abstract",
                            "sample_reporting_status": "Not applicable" if is_math else "Not reported",
                            "quantitative_result": "Not reported in abstract",
                            "key_findings": "Demonstrates failure modes and conditional trade-offs under stress regimes.",
                            "limitations": "Requires broader validation across external regimes.",
                        },
                    ]
                })
            if "Tasks D, E, F, and G" in sys_p:
                return json.dumps({
                    "comparable_groups_summary": "Compared primary supporting evidence [1] against opposing/limiting evidence [2].",
                    "claims": [
                        {
                            "claim_id": "claim:1",
                            "proposition": "Primary methods demonstrate strong performance within canonical regimes [1], whereas opposing analyses reveal failure modes under regime shift [2].",
                            "claim_category": "methodological_comparison",
                            "supporting_indices": [1],
                            "opposing_indices": [2],
                            "applicability_scope": "Canonical benchmark and theoretical regimes",
                            "methodological_caveats": "Out-of-distribution generalization and external calibration remain constrained.",
                        }
                    ],
                })
            if "Tasks H & I" in sys_p:
                return json.dumps({
                    "rows": ["Primary Architecture", "Robustness Protocol", "Hybrid Formulation"],
                    "cols": ["Canonical Regime", "Out-of-Distribution Shift", "Prospective Validation"],
                    "assignments": [
                        {
                            "row_idx": 0,
                            "col_idx": 0,
                            "paper_indices": [1],
                            "gap_category": "Validation deficiencies",
                            "why_gap_exists": "Single study coverage in retrieved set.",
                            "proposed_hypothesis": "Replicate [1] across independent multi-center benchmarks.",
                            "verification_query": "benchmark replication validation",
                            "impact_level": "High",
                        },
                        {
                            "row_idx": 1,
                            "col_idx": 1,
                            "paper_indices": [2],
                            "gap_category": "Inconsistent findings",
                            "why_gap_exists": "Opposing findings under distribution shift [2].",
                            "proposed_hypothesis": "Characterize regime boundary between [1] and [2].",
                            "verification_query": "distribution shift failure boundary",
                            "impact_level": "High",
                        },
                    ],
                })
            if "Task J" in sys_p:
                return json.dumps({
                    "executive_answer": "Synthesizing across the retrieved studies reveals that primary approaches succeed within canonical regimes [1], while opposing evidence highlights degradation under out-of-distribution conditions [2].",
                    "consensus_verdict": "Context-Dependent Evidence Agreement",
                    "consensus_summary": "Evidence supports efficacy in standard settings [1] but identifies critical boundary limitations [2].",
                    "sections": [
                        {
                            "heading": "1. Core Scientific Claims & Supporting Evidence",
                            "content": "Primary investigations establish foundational efficacy within the target regime [1].",
                            "cited_indices": [1],
                        },
                        {
                            "heading": "2. Opposing Evidence, Failure Modes & Methodological Caveats",
                            "content": "Critical counter-evidence demonstrates failure modes and regime sensitivity that bound generalizability [2].",
                            "cited_indices": [1, 2],
                        },
                    ],
                })
            return "{}"

        configs_to_test = [
            ("Tier1_Gemini_Structured", ApiKeysConfig(llm_provider="gemini", llm_model="gemini-2.5-flash", llm_api_key="AIza-test"), compliant_llm_responder),
            ("Offline_Deterministic_Engine", ApiKeysConfig(llm_provider="gemini", llm_api_key=""), None),
        ]

        for cfg_name, cfg, responder in configs_to_test:
            set_test_llm_responder(responder)
            for case in BENCHMARK_DOMAIN_CASES:
                clear_synthesis_caches()
                plan = await generate_multi_api_decomposition_plan(case["query"], "standard", ApiKeysConfig(llm_api_key=""))
                self.assertEqual(plan.question_understanding.primary_domain, case["domain_expected"], f"Failed domain detection on {case['case_id']}")

                papers = deduplicate_and_rank_papers(
                    [p.model_copy(deep=True) for p in case["papers"]],
                    case["query"],
                    max_total=20,
                    understanding=plan.question_understanding,
                )
                hf_res = [HFResourceRecord(id="org/model-v1", resource_type="model", pipeline_tag="text-generation", downloads=500)]
                graph = build_literature_graph(papers, hf_res)
                bib = generate_bibliography(papers)

                exec_ans, sections, consensus, warnings = await generate_synthesis_and_extractions(
                    query=case["query"],
                    mode="standard",
                    papers=papers,
                    hf_resources=hf_res,
                    bibliography=bib,
                    config=cfg,
                    decomposition_plan=plan,
                )
                gap_matrix = await build_knowledge_gap_matrix(
                    query=case["query"],
                    papers=papers,
                    hf_resources=hf_res,
                    config=cfg,
                    decomposition_plan=plan,
                )
                eb = build_shared_evidence_base(papers, gap_matrix, graph, plan)
                q_report = run_verification_and_quality_gates(
                    plan=plan,
                    papers=papers,
                    executive_answer=exec_ans,
                    sections=sections,
                    gap_matrix=gap_matrix,
                    graph=graph,
                    evidence_base=eb,
                    task_audit_trail=eb.task_audit_trail,
                    model_capability=resolve_model_capabilities(cfg),
                )

                # Deterministic Acceptance Criteria (100% required across all models & domains)
                self.assertTrue(q_report.schema_valid, f"{cfg_name}/{case['case_id']} failed schema_valid: {q_report.warnings_and_Notices}")
                self.assertTrue(q_report.source_integrity_valid, f"{cfg_name}/{case['case_id']} failed source_integrity_valid")
                self.assertTrue(q_report.numerical_integrity_valid, f"{cfg_name}/{case['case_id']} failed numerical_integrity_valid: {q_report.warnings_and_Notices}")
                self.assertTrue(q_report.claim_attribution_valid, f"{cfg_name}/{case['case_id']} failed claim_attribution_valid")
                self.assertTrue(q_report.inference_validity_valid, f"{cfg_name}/{case['case_id']} failed inference_validity_valid")
                self.assertTrue(q_report.gap_valid, f"{cfg_name}/{case['case_id']} failed gap_valid")
                self.assertTrue(q_report.cross_section_consistency_valid, f"{cfg_name}/{case['case_id']} failed cross_section_consistency_valid")
                self.assertIn(
                    q_report.overall_verification_status,
                    ("passed_validation", "passed_with_disclosed_limitations"),
                )

    async def test_03_controlled_error_correction_and_graceful_degradation(self) -> None:
        """
        Verifies Section 6 & 7:
        1. Model Config A (Self-Correcting Local Model): On Attempt 1, hallucinates an ungrounded number ('99.9%'),
           an out-of-range citation ('[99]'), and an invalid causal claim on observational data.
           Receives exact validation errors on Attempt 2 and self-corrects (`correction_retries_used > 0`, `passed_with_disclosed_limitations`).
        2. Model Config B (Persistently Failing Model): Always outputs fabricated numbers / broken JSON.
           Verifies bounded retries (max 2 retries = 3 total attempts) and graceful degradation to deterministic
           source-grounded synthesis without accepting hallucinated facts or switching paid providers.
        """
        case = BENCHMARK_DOMAIN_CASES[1]  # Clinical prediction modeling (Observational ICU cohort)
        plan = await generate_multi_api_decomposition_plan(case["query"], "standard", ApiKeysConfig(llm_api_key=""))

        # --- Subtest A: Self-Correcting Model on Retry ---
        def self_correcting_responder(sys_p: str, usr_p: str, cfg: ApiKeysConfig, attempt: int) -> str:
            if "ONE section" in sys_p:
                # Sections are written as separate calls; a grounded section needs no correction
                return json.dumps({"content": "While AUROC 0.89 is reported retrospectively [1], dataset shift limits "
                                              "external generalizability and no mortality reduction was shown [2]."})
            if attempt == 1:
                # Deliberately fail quality gates on Attempt 1:
                if "Task C" in sys_p:
                    # Fabricate 99.9% not in abstract
                    return json.dumps({
                        "study_extractions": [
                            {
                                "citation_index": 1,
                                "stance": "supporting",
                                "study_design": "Observational Cohort / Epidemiological",
                                "methodology": "Transformer",
                                "dataset_or_system": "ICU cohort",
                                "sample_or_scale": "n = 14,500",
                                "sample_reporting_status": "Reported",
                                "quantitative_result": "AUROC 99.9%",
                                "key_findings": "Achieved 99.9% accuracy.",
                                "limitations": "Retrospective.",
                            }
                        ]
                    })
                if "Tasks D, E, F, and G" in sys_p:
                    # Unqualified causal_intervention claim on observational cohort
                    return json.dumps({
                        "comparable_groups_summary": "Observational comparison",
                        "claims": [
                            {
                                "claim_id": "claim:1",
                                "proposition": "The score definitively causes survival in ICU patients [1].",
                                "claim_category": "causal_intervention",
                                "supporting_indices": [1],
                                "opposing_indices": [2],
                            }
                        ],
                    })
                if "Task J" in sys_p:
                    # Out-of-bounds citation [99]
                    return json.dumps({
                        "executive_answer": "The model works well [1, 99].",
                        "consensus_verdict": "Moderate",
                        "consensus_summary": "Summary [1].",
                        "sections": [{"heading": "Sec 1", "content": "Details [1, 2, 99].", "cited_indices": [1, 2]}],
                    })
                if "Tasks H & I" in sys_p:
                    # Out-of-bounds paper_indices [99] in gap assignment
                    return json.dumps({
                        "rows": ["Temporal Transformer", "Clinical Alerting", "Calibration Protocol"],
                        "cols": ["Retrospective AUROC", "External Calibration", "Prospective Trial"],
                        "assignments": [{"row_idx": 0, "col_idx": 0, "paper_indices": [99], "gap_category": "Validation deficiencies"}],
                    })
            # On Attempt 2 (after receiving CONTROLLED ERROR CORRECTION prompt), return grounded JSON:
            self.assertIn("CONTROLLED ERROR CORRECTION", usr_p)
            if "Task C" in sys_p:
                return json.dumps({
                    "study_extractions": [
                        {
                            "citation_index": 1,
                            "stance": "supporting",
                            "study_design": "Observational Cohort / Epidemiological",
                            "methodology": "Temporal transformer",
                            "dataset_or_system": "ICU cohort (n = 14,500)",
                            "sample_or_scale": "n = 14,500",
                            "sample_reporting_status": "Reported",
                            "quantitative_result": "AUROC 0.89",
                            "key_findings": "Achieved AUROC 0.89 for sepsis mortality prediction.",
                            "limitations": "External hospital calibration drifted.",
                        },
                        {
                            "citation_index": 2,
                            "stance": "contradicting",
                            "study_design": "Observational Cohort / Epidemiological",
                            "methodology": "Observational risk score evaluation",
                            "dataset_or_system": "Retrospective encounters (n = 6,200)",
                            "sample_or_scale": "n = 6,200",
                            "sample_reporting_status": "Reported",
                            "quantitative_result": "Not reported in abstract",
                            "key_findings": "Showed no significant mortality reduction without prospective trial validation.",
                            "limitations": "Alert fatigue and dataset shift.",
                        },
                    ]
                })
            if "Tasks D, E, F, and G" in sys_p:
                return json.dumps({
                    "comparable_groups_summary": "Observational predictive comparison",
                    "claims": [
                        {
                            "claim_id": "claim:1",
                            "proposition": "Temporal transformers achieve AUROC 0.89 in retrospective ICU cohorts [1], but observational risk scores show calibration drift and require prospective trial validation [2].",
                            "claim_category": "predictive_performance",
                            "supporting_indices": [1],
                            "opposing_indices": [2],
                            "applicability_scope": "Retrospective ICU cohorts",
                            "methodological_caveats": "Observational design cannot establish causal mortality reduction.",
                        }
                    ],
                })
            if "Task J" in sys_p:
                return json.dumps({
                    "executive_answer": "In retrospective ICU cohorts, temporal transformers achieve AUROC 0.89 [1], whereas opposing observational analyses demonstrate calibration drift and lack of proven mortality reduction [2].",
                    "consensus_verdict": "Context-Dependent Predictive Evidence",
                    "consensus_summary": "Predictive discrimination is high [1], but clinical impact requires prospective trials [2].",
                    "sections": [
                        {
                            "heading": "1. Predictive Discrimination vs Clinical Calibration",
                            "content": "While AUROC 0.89 is reported retrospectively [1], dataset shift limits external generalizability [2].",
                            "cited_indices": [1, 2],
                        }
                    ],
                })
            return json.dumps({
                "rows": ["Temporal Transformer", "Clinical Alerting", "Calibration Protocol"],
                "cols": ["Retrospective AUROC", "External Calibration", "Prospective Trial"],
                "assignments": [{"row_idx": 0, "col_idx": 0, "paper_indices": [1], "gap_category": "Validation deficiencies"}],
            })

        set_test_llm_responder(self_correcting_responder)
        papers_a = deduplicate_and_rank_papers([p.model_copy(deep=True) for p in case["papers"]], case["query"], max_total=20)
        graph_a = build_literature_graph(papers_a, [])
        bib_a = generate_bibliography(papers_a)
        cfg_ollama = ApiKeysConfig(llm_provider="ollama", llm_model="llama3.2:latest")

        exec_a, secs_a, _, _ = await generate_synthesis_and_extractions(
            case["query"], "standard", papers_a, [], bib_a, cfg_ollama, plan
        )
        gap_a = await build_knowledge_gap_matrix(case["query"], papers_a, [], cfg_ollama, plan)
        eb_a = build_shared_evidence_base(papers_a, gap_a, graph_a, plan)
        qr_a = run_verification_and_quality_gates(
            plan, papers_a, exec_a, secs_a, gap_a, graph_a, eb_a, eb_a.task_audit_trail, resolve_model_capabilities(cfg_ollama)
        )

        self.assertGreater(qr_a.correction_retries_used, 0)
        self.assertEqual(qr_a.overall_verification_status, "passed_with_disclosed_limitations")
        self.assertTrue(qr_a.numerical_integrity_valid)
        self.assertTrue(qr_a.inference_validity_valid)
        self.assertNotIn("99.9", exec_a)

        # --- Subtest B: Persistently Failing Model -> Graceful Deterministic Degradation ---
        clear_synthesis_caches()
        set_test_llm_responder(lambda s, u, c, a: '{"broken_json": "hallucinated 88.88% [999]"')
        papers_b = deduplicate_and_rank_papers([p.model_copy(deep=True) for p in case["papers"]], case["query"], max_total=20)
        graph_b = build_literature_graph(papers_b, [])
        bib_b = generate_bibliography(papers_b)
        cfg_custom = ApiKeysConfig(llm_provider="openrouter", llm_model="custom/unreliable-7b", llm_api_key="sk-or-v1-test")

        exec_b, secs_b, _, warnings_b = await generate_synthesis_and_extractions(
            case["query"], "standard", papers_b, [], bib_b, cfg_custom, plan
        )
        gap_b = await build_knowledge_gap_matrix(case["query"], papers_b, [], cfg_custom, plan)
        eb_b = build_shared_evidence_base(papers_b, gap_b, graph_b, plan)
        qr_b = run_verification_and_quality_gates(
            plan, papers_b, exec_b, secs_b, gap_b, graph_b, eb_b, eb_b.task_audit_trail, resolve_model_capabilities(cfg_custom)
        )

        self.assertTrue(any("Quality Gate Degradation Disclosure" in w for w in warnings_b))
        self.assertGreater(len(qr_b.degradation_events), 0)
        self.assertEqual(qr_b.overall_verification_status, "passed_with_disclosed_limitations")
        self.assertTrue(qr_b.numerical_integrity_valid)
        self.assertTrue(qr_b.source_integrity_valid)
        self.assertNotIn("88.88", exec_b)

    def test_04_provider_selection_preservation_and_diagnostics(self):
        """
        Verifies that explicit provider selection is strictly preserved and not overridden
        by residual keys or URLs from other providers (e.g. OpenRouter key left in field),
        and that connection failure diagnostics report exact actionable messages.
        """
        from services.llm_service import detect_provider_from_credentials, fetch_provider_models, test_llm_connection
        from models import TestLlmRequest
        import asyncio

        # 1. When user selects 'gemini' but residual OpenRouter key is in field, provider MUST remain 'gemini'
        rec = detect_provider_from_credentials("gemini", "sk-or-v1-residual-openrouter-key", "https://openrouter.ai/api/v1")
        self.assertEqual(rec["provider"], "gemini")

        # 2. When user selects 'ollama', provider MUST remain 'ollama'
        rec_ollama = detect_provider_from_credentials("ollama", "sk-or-v1-key", "")
        self.assertEqual(rec_ollama["provider"], "ollama")

        # 3. fetch_provider_models preserves requested provider
        res = asyncio.run(fetch_provider_models("gemini", "sk-or-v1-key", ""))
        self.assertEqual(res["detected_provider"], "gemini")

        # 4. test_llm_connection with missing key provides explicit informative diagnostic
        test_req = TestLlmRequest(llm_provider="gemini", llm_api_key="", llm_model="gemini-2.5-flash")
        test_res = asyncio.run(test_llm_connection(test_req))
        self.assertFalse(test_res["success"])
        self.assertIn("Please enter a valid API key for GEMINI", test_res["message"])


if __name__ == "__main__":
    unittest.main()
