# Methodology Frameworks & Rules

_Generated from `backend/frameworks/` by `python -m frameworks.export_doc`. Edit the registry, not this file._

Rules are applied as framework-informed screening from titles and abstracts. The tool never issues a formal risk-of-bias or GRADE rating, which require full-text assessment.

## Corrections from the source audit

| Before | After |
|---|---|
| Narrative said findings were "synthesized qualitatively" under SWiM | SWiM covers quantitative synthesis without meta-analysis and accepts vote counting by direction of effect; wording now says findings are compared by direction of effect with no pooled estimate. |
| Agreement meter described as "rather than vote-counting" | The meter is direction-of-effect vote counting; it now says so and states that it ignores effect size and precision (SWiM). |
| RoB 2 concerns rated "Unclear" | "Unclear" is the old RoB 1 scale. RoB 2 rates Low / Some concerns / High per domain; from an abstract the tool now only names the domain and says it is not assessable. |
| QUIPS grouped with PROBAST+AI for prediction models | QUIPS is for prognostic factor studies; PROBAST+AI for prediction models. They are now routed separately. |
| ARRIVE 2.0 used as a risk-of-bias check | ARRIVE is a reporting guideline; SYRCLE is the animal risk-of-bias tool. Both are now named for what they are. |
| ROBINS-I applied to all non-randomised designs | ROBINS-I V2 is for interventions; observational exposure studies now route to ROBINS-E. |
| No route for diagnostic accuracy studies | Added QUADAS-3 (February 2026, replaces QUADAS-2). |
| Any abstract containing "random" (e.g. "random forest") was treated as a randomised trial | Trial detection now requires randomised/randomized near trial, allocation or assignment. |
| No certainty-of-evidence concept | Added GRADE-informed indicators, labelled as not a GRADE rating. |
| Domain rules for CS, physics and chemistry had no sources | Now cite REFORMS, the ML leakage taxonomy, the GUM, IUPAC and ACS reporting guidance; mathematics and other conventions are labelled as field conventions. |

## Rules by stage

### Appraisal

| ID | Rule | Applies to | Source | Enforced by |
|---|---|---|---|---|
| B-rct | Randomised trials: check reporting on the RoB 2 domains - randomisation process, deviations from intended interventions, missing outcome data, outcome measurement, selection of the reported result. | domains: biomedical_clinical, interdisciplinary_biomedical_ai; designs: rct | RoB 2 (RoB 2: five bias domains) | prompt |
| B-nrsi | Non-randomised intervention studies: confounding is the main threat; check how it was controlled. | domains: biomedical_clinical, interdisciplinary_biomedical_ai; designs: nrsi_intervention | ROBINS-I V2 (ROBINS-I V2: bias due to confounding) | prompt |
| B-exposure | Observational exposure studies: check confounding control, participant selection and exposure classification. | domains: biomedical_clinical, interdisciplinary_biomedical_ai; designs: observational_exposure | ROBINS-E (ROBINS-E: bias domains) | prompt |
| B-prediction | Prediction models: check participants and data sources, predictors, outcome and analysis; prefer external validation over internal performance. | domains: biomedical_clinical, interdisciplinary_biomedical_ai; designs: prediction_model | PROBAST+AI (PROBAST+AI: four domains (development and evaluation)) | prompt |
| B-prognostic | Prognostic factor studies: check participation, attrition, factor and outcome measurement, confounding, and analysis. | domains: biomedical_clinical, interdisciplinary_biomedical_ai; designs: prognostic_factor | QUIPS (QUIPS: six bias domains) | prompt |
| B-diagnostic | Diagnostic accuracy studies: check participants, index test, target condition and analysis. | domains: biomedical_clinical, interdisciplinary_biomedical_ai; designs: diagnostic_accuracy | QUADAS-3 (QUADAS-3: four risk-of-bias domains) | prompt |
| B-animal | Animal studies: check randomisation, blinding and attrition (bias) and whether the ARRIVE Essential 10 items are reported. | domains: biomedical_clinical, interdisciplinary_biomedical_ai; designs: animal | SYRCLE RoB; ARRIVE v2 (SYRCLE: 10 items over 6 bias types; ARRIVE 2.0: Essential 10) | prompt |
| B-omics | High-throughput studies: check multiple-testing correction, batch effects and orthogonal validation of hits. | domains: biomedical_clinical, interdisciplinary_biomedical_ai, omics_high_throughput; designs: omics, omics_high_throughput | High-throughput omics convention | prompt |
| CS-leakage | Check for leakage between training and test data (duplicates, temporal or similarity leakage, preprocessing on the full dataset). | domains: computer_science_ai, interdisciplinary_biomedical_ai | ML leakage taxonomy; REFORMS (Leakage taxonomy (8 types); REFORMS: data preprocessing and evaluation) | prompt |
| CS-reproducibility | Note whether code, data, model weights and computing details are available. | domains: computer_science_ai, interdisciplinary_biomedical_ai | REFORMS (REFORMS: computational reproducibility) | prompt |
| CH-purity | Note whether compound purity is documented for compounds used in bioassays. | domains: chemistry | JOC compound purity guidance (JOC author guidelines: >=95% purity for compounds in physical measurements or bioassays) | prompt |
| CH-computation | For computed results, report the level of theory and basis set; separate calculated from measured values. | domains: chemistry | IUPAC QM data reporting | prompt |
| PH-components | Check that all significant uncertainty components are evaluated and combined, not only the statistical scatter. | domains: physics, engineering_materials | GUM (JCGM 100:2008) (GUM: combined standard uncertainty) | prompt |
| MA-verification | Note whether a proof is peer reviewed, preprint-only, or formally verified (e.g. Lean, Coq). | domains: mathematics | Mathematics publishing convention | prompt |
| MA-no-sample | Sample sizes do not apply to proofs; never invent them. | domains: mathematics | CiteLume source-fidelity policy | code: `sample_policy_gate` |
| EN-repeatability | Check repeatability: number of specimens or runs and the test standard used. | domains: engineering_materials | Engineering testing convention | prompt |
| SS-identification | Distinguish correlational designs from causal identification strategies (RCT, IV, DiD, RDD). | domains: social_sciences | Social-science identification convention | prompt |
| HU-provenance | Assess source provenance and corpus representativeness. | domains: humanities_interdisciplinary | Humanities scholarship convention | prompt |
| M-observational-a | Observational studies: assess confounding control and how participants were selected. | designs: observational | ROBINS-E (ROBINS-E: confounding; selection of participants) | prompt |
| M-experimental-a | Experiments: check positive and negative controls, replicates and assay sensitivity. | designs: experimental | Experimental science convention | prompt |
| M-intervention-a | Intervention trials: check randomisation, blinding, statistical power, primary endpoints and adverse events. | designs: clinical_intervention | RoB 2 (RoB 2: five bias domains) | prompt |
| M-predictive-a | Prediction models: check external validation, calibration, discrimination and leakage between development and test data. | designs: predictive_modeling | PROBAST+AI; ML leakage taxonomy (PROBAST+AI: analysis domain) | prompt |
| M-benchmark-a | Benchmarks: check how data were split, fairness of baselines and completeness of ablations. | designs: ml_benchmarking | REFORMS; ML leakage taxonomy (REFORMS: data and evaluation) | prompt |
| M-theory-a | Theoretical work: check assumptions, regularity conditions, asymptotic regime and proof completeness. | designs: theoretical_mathematical | Mathematics publishing convention | prompt |
| M-simulation-a | Simulations: check numerical resolution, convergence and the accuracy of the model or force field used. | designs: simulation_based | Computational simulation convention | prompt |
| M-qualitative-a | Qualitative studies: check sampling, data collection, analysis approach and researcher reflexivity. | designs: qualitative | COREQ; SRQR (COREQ: research team and reflexivity, study design, analysis and findings) | prompt |
| M-review-a | Reviews: check the protocol, search comprehensiveness, study selection and risk-of-bias assessment. | designs: literature_conceptual_review | AMSTAR 2 (AMSTAR 2: critical domains) | prompt |

### Synthesis

| ID | Rule | Applies to | Source | Enforced by |
|---|---|---|---|---|
| U-causal | Do not state causal effects from observational, correlational or predictive evidence; use associational wording. | all | GRADE (GRADE: evidence from observational studies starts at low certainty) | code: `causal_overclaim_gate` |
| U-numbers | Report only numbers that appear in the cited study records; never estimate or pool values. | all | CiteLume source-fidelity policy | code: `ungrounded_number_gate` |
| U-direction | Summarise agreement by direction of effect (supporting / mixed / contradicting), not by statistical significance, and note that this ignores effect size and precision. | all | SWiM (SWiM: synthesis method - vote counting based on direction of effect) | prompt |
| U-heterogeneity | Explain how studies differ (population, setting, design, outcome measures) before drawing conclusions across them. | all | SWiM (SWiM: investigation of heterogeneity) | prompt |
| U-opposing | Cite contradicting or null studies alongside supporting ones. | all | GRADE (GRADE: inconsistency) | code: `omitted_contradiction_gate` |
| U-certainty | State how certain the evidence is, considering risk of bias, inconsistency, indirectness, imprecision and publication bias. | all | GRADE (GRADE: five reasons to rate down certainty) | prompt |
| B-indirectness | Keep in vitro, animal and human findings separate; do not extrapolate preclinical results to patients. | domains: biomedical_clinical, interdisciplinary_biomedical_ai | GRADE (GRADE: indirectness) | prompt |
| CS-benchmarks | Compare methods only on shared datasets, splits and metrics; qualify state-of-the-art claims accordingly. | domains: computer_science_ai, interdisciplinary_biomedical_ai | REFORMS (REFORMS: metrics and uncertainty quantification) | prompt |
| CH-yield | Distinguish isolated yields from crude or conversion yields. | domains: chemistry | Org. Lett. data-reporting standards | prompt |
| CH-conditions | Keep units and reaction conditions (solvent, catalyst, temperature) with every result. | domains: chemistry | Chemistry reporting convention | prompt |
| PH-uncertainty | Report measured values with their stated uncertainty and coverage; do not compare values without their uncertainties. | domains: physics, engineering_materials | GUM (JCGM 100:2008) (GUM: standard and expanded uncertainty) | prompt |
| PH-approximation | Distinguish exact results from perturbative or numerical approximations and state the parameter regime and assumptions. | domains: physics | Physics reporting convention | prompt |
| MA-status | Distinguish proved theorems from conditional results and conjectures. | domains: mathematics | Mathematics publishing convention | prompt |
| MA-assumptions | State the hypotheses each result depends on. | domains: mathematics | Mathematics publishing convention | prompt |
| EN-conditions | Compare performance only under stated, standardised test conditions. | domains: engineering_materials | Engineering testing convention | prompt |
| SS-context | Do not generalise effects across institutional or demographic contexts without evidence. | domains: social_sciences | Social-science identification convention | prompt |
| HU-interpretation | Preserve competing scholarly interpretations and their historical context. | domains: humanities_interdisciplinary | Humanities scholarship convention | prompt |
| M-observational-s | Describe observational findings as associations ('is associated with'), not effects. | designs: observational | GRADE (GRADE: observational evidence starts at low certainty) | prompt |
| M-experimental-s | Scope experimental conclusions to the model system that was tested. | designs: experimental | GRADE (GRADE: indirectness) | prompt |
| M-intervention-s | Distinguish surrogate markers from patient-relevant outcomes. | designs: clinical_intervention | GRADE (GRADE: indirectness of outcomes) | prompt |
| M-predictive-s | Distinguish internal validation performance from external or prospective validation. | designs: predictive_modeling | PROBAST+AI (PROBAST+AI: model development vs evaluation) | prompt |
| M-benchmark-s | Qualify state-of-the-art claims by the split method and out-of-distribution robustness. | designs: ml_benchmarking | REFORMS (REFORMS: generalisability) | prompt |
| M-omics-s | Distinguish primary screening hits from independently confirmed results. | designs: omics_high_throughput | High-throughput omics convention | prompt |
| M-theory-s | State the assumptions each theoretical result depends on; empirical sample sizes do not apply. | designs: theoretical_mathematical | Mathematics publishing convention | code: `sample_policy_gate` |
| M-simulation-s | Distinguish simulated quantities from experimentally measured values. | designs: simulation_based | Computational simulation convention | prompt |
| M-qualitative-s | Present qualitative findings as interpretations grounded in the analysed data. | designs: qualitative | SRQR | prompt |
| M-review-s | Do not count overlapping reviews, or a review and its own included studies, as independent evidence. | designs: literature_conceptual_review | CiteLume source-fidelity policy | prompt |

### Reporting

| ID | Rule | Applies to | Source | Enforced by |
|---|---|---|---|---|
| U-citations | Every claim must cite study numbers [n] that exist in the provided records. | all | CiteLume source-fidelity policy | code: `citation_range_gate` |
| U-preprint | Flag preprints as not peer reviewed. | all | CiteLume source-fidelity policy | prompt |
| CS-artifacts | Model cards and repository claims are community artefacts, not peer-reviewed evidence. | domains: computer_science_ai, interdisciplinary_biomedical_ai | CiteLume source-fidelity policy | prompt |
| CS-slr | Software-engineering systematic reviews: document the protocol, search, study selection and quality assessment. | domains: computer_science_ai, interdisciplinary_biomedical_ai; reviews: systematic_prisma | Kitchenham & Charters SLR guidelines (Kitchenham & Charters: planning, conducting, reporting) | prompt |
| RV-prisma-flow | Report the flow of records: identified per source, duplicates removed, screened, included. | reviews: systematic_prisma | PRISMA 2020 (PRISMA 2020 flow diagram) | prompt |
| RV-prisma-search | Report the full search strategy used for every source. | reviews: systematic_prisma, scoping | PRISMA 2020 (PRISMA 2020: search strategy) | prompt |
| RV-prisma-certainty | Assess risk of bias and certainty of evidence for the main findings. | reviews: systematic_prisma | PRISMA 2020; GRADE (PRISMA 2020: risk of bias and certainty assessment) | prompt |
| RV-scoping | Chart and map the evidence (what exists, where, how studied); risk-of-bias appraisal is optional. | reviews: scoping | PRISMA-ScR (PRISMA-ScR: charting the data; critical appraisal (optional)) | prompt |
| RV-narrative | State the review's aim, how the literature was found, reference key statements, and present key data with its level of evidence. | reviews: narrative | SANRA (SANRA: six items) | prompt |

## Review types

| Framework | Description | Sources |
|---|---|---|
| Narrative review | Expert synthesis of the literature without a protocol-driven search; quality judged against SANRA. | SANRA, Grant & Booth review typology |
| Systematic-style review (PRISMA 2020 reporting) | Documented search, screening flow, design-matched appraisal and certainty of evidence, reported to PRISMA 2020. | PRISMA 2020, GRADE, Grant & Booth review typology |
| Scoping review (PRISMA-ScR reporting) | Maps what evidence exists, where and how it was studied, and where gaps remain; appraisal is optional. | PRISMA-ScR, Grant & Booth review typology |

| Organisation | How findings are arranged |
|---|---|
| Thematic | Organise section B by theme: one paragraph per recurring finding or issue, synthesising across studies. |
| Chronological | Organise section B chronologically: describe how findings and methods developed over time, period by period, naming the years of each period. Explain what changed, not just a list of papers by date. |
| Methodological | Organise section B by research method: group studies by design or approach, and compare what each kind of method found and its strengths and weaknesses. |
| Theoretical | Organise section B by theory: group studies by the theory, model or conceptual framework they use or test, and compare how well the evidence supports each. |

## Deterministic checks

| Check | Where |
|---|---|
| `causal_overclaim_gate` | services/claim_synthesis.py (_OVERCLAIM_CAUSAL_RE) and services/verification_gates.py |
| `ungrounded_number_gate` | services/verification_gates.py and the Task J validator in services/synthesis_engine.py |
| `omitted_contradiction_gate` | Task J validator in services/synthesis_engine.py |
| `sample_policy_gate` | verify_study_extraction_against_source in services/verification_gates.py |
| `citation_range_gate` | Task C / Task J validators in services/synthesis_engine.py |

## Sources

### Risk of bias

- **RoB 2**: Sterne JAC, et al. RoB 2: a revised tool for assessing risk of bias in randomised trials. BMJ 2019;366:l4898. <https://doi.org/10.1136/bmj.l4898>
- **ROBINS-I V2**: ROBINS-I V2: Risk Of Bias In Non-randomized Studies - of Interventions, version 2 (launched November 2024). riskofbias.info. <https://sites.google.com/site/riskofbiastool/welcome/robins-i-v2>
- **ROBINS-E**: Higgins JPT, et al. A tool to assess risk of bias in non-randomized follow-up studies of exposure effects (ROBINS-E). Environ Int 2024;186:108602. <https://doi.org/10.1016/j.envint.2024.108602>
- **PROBAST+AI**: Moons KGM, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods. BMJ 2025. <https://ora.ox.ac.uk/objects/uuid:8b15c087-459b-43fa-ad03-1f66ed6ba370>
- **QUIPS**: Hayden JA, et al. Assessing bias in studies of prognostic factors. Ann Intern Med 2013. <https://abstracts.cochrane.org/2015-vienna/systematic-reviews-prognostic-studies-2-assessing-risk-bias-studies-prognostic-factors>
- **QUADAS-3**: Whiting PF, et al. QUADAS-3: a revised tool for the quality assessment of diagnostic accuracy studies. Ann Intern Med 2026;179(4). <https://doi.org/10.7326/ANNALS-25-02104>
- **SYRCLE RoB**: Hooijmans CR, et al. SYRCLE's risk of bias tool for animal studies. BMC Med Res Methodol 2014;14:43. <https://doi.org/10.1186/1471-2288-14-43>
- **AMSTAR 2**: Shea BJ, et al. AMSTAR 2: a critical appraisal tool for systematic reviews that include randomised or non-randomised studies of healthcare interventions, or both. BMJ 2017;358:j4008. <https://pmc.ncbi.nlm.nih.gov/articles/5833365>
- **ML leakage taxonomy**: Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. Patterns 2023;4(9):100804. <https://doi.org/10.1016/j.patter.2023.100804>

### Reporting

- **SWiM**: Campbell M, et al. Synthesis without meta-analysis (SWiM) in systematic reviews: reporting guideline. BMJ 2020;368:l6890. <https://doi.org/10.1136/bmj.l6890>
- **PRISMA 2020**: Page MJ, et al. The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. BMJ 2021;372:n71. <https://doi.org/10.1136/bmj.n71>
- **PRISMA-ScR**: Tricco AC, et al. PRISMA extension for scoping reviews (PRISMA-ScR): checklist and explanation. Ann Intern Med 2018;169(7):467-473. <https://doi.org/10.7326/M18-0850>
- **ARRIVE v2**: Percie du Sert N, et al. The ARRIVE guidelines 2.0: updated guidelines for reporting animal research. 2020. <https://arriveguidelines.org/news/new-arrive-guidelines-20-released>
- **COREQ**: Tong A, Sainsbury P, Craig J. Consolidated criteria for reporting qualitative research (COREQ): a 32-item checklist for interviews and focus groups. Int J Qual Health Care 2007;19:349-357. <https://researchnow.flinders.edu.au/en/publications/consolidated-criteria-for-reporting-qualitative-research-coreq-a-/>
- **SRQR**: O'Brien BC, et al. Standards for reporting qualitative research: a synthesis of recommendations. Acad Med 2014;89(9):1245-1251. <https://www.casrai.org/guides/coreq-srqr-checklist-reporting-qualitative-research>
- **REFORMS**: Kapoor S, et al. REFORMS: Consensus-based recommendations for machine-learning-based science. Sci Adv 2024;10:eadk3452. <https://reforms.cs.princeton.edu/>
- **GUM (JCGM 100:2008)**: JCGM 100:2008. Evaluation of measurement data - Guide to the expression of uncertainty in measurement (GUM). Joint Committee for Guides in Metrology; also ISO/IEC Guide 98-3:2008. <https://www.bipm.org/en/committees/jc/jcgm/publications>
- **IUPAC QM data reporting**: Albright TA, Halevi EA. Presentation of the results of quantum mechanical computations in organic chemistry (IUPAC recommendations). Pure Appl Chem 1992;64(8):1203-1209. <https://publications.iupac.org/pac/64/8/1203/index.html>
- **Org. Lett. data-reporting standards**: Organic Letters editorial (2023) proposing standards for reporting reaction data, including distinguishing isolated from crude yields. <https://pubs.acs.org/journal/orlef7>
- **JOC compound purity guidance**: The Journal of Organic Chemistry, Author Guidelines: compounds used for physical measurements or bioassays should be documented at >=95% purity. <https://pubs.acs.org/journal/joceah>

### Certainty

- **GRADE**: Guyatt GH, et al. What is "quality of evidence" and why is it important to clinicians? BMJ 2008;336(7651):995-998. <https://pmc.ncbi.nlm.nih.gov/articles/PMC2364804>

### Review method

- **SANRA**: Baethge C, Goldbeck-Wood S, Mertens S. SANRA - a scale for the quality assessment of narrative review articles. Res Integr Peer Rev 2019. <https://doi.org/10.1186/s41073-019-0064-8>
- **Grant & Booth review typology**: Grant MJ, Booth A. A typology of reviews: an analysis of 14 review types and associated methodologies. Health Info Libr J 2009;26(2):91-108. <https://doi.org/10.1111/j.1471-1842.2009.00848.x>
- **Kitchenham & Charters SLR guidelines**: Kitchenham B, Charters S. Guidelines for performing Systematic Literature Reviews in Software Engineering, v2.3. EBSE Technical Report EBSE-2007-01, Keele University and Durham University, 2007. <https://ebse.webspace.durham.ac.uk/ebse-bibliography/lessons-from-applying-the-systematic-literature-review-process-within-the-software-engineering-domain/>

### Field convention

- **Literature review organisation guides**: University library writing guides on organising a literature review (chronological, thematic, methodological, theoretical), e.g. Stony Brook University Libraries. <https://guides.library.stonybrook.edu/literature-review/organize>
- **Mathematics publishing convention**: Accepted practice in mathematics: results are classified by proof status (proved, conditional, conjectured) and peer-review or formal-verification status. No single citable framework.
- **Physics reporting convention**: Accepted practice in physics: state parameter regimes, approximations and symmetry assumptions. No single citable framework.
- **Chemistry reporting convention**: Accepted practice in chemistry: preserve units and conditions (solvent, catalyst, temperature) when comparing results. No single citable framework.
- **Engineering testing convention**: Accepted practice in engineering and materials science: compare performance only under stated, standardised test conditions.
- **Social-science identification convention**: Accepted practice in empirical social science: distinguish correlational designs from causal identification strategies (RCT, IV, DiD, RDD).
- **Humanities scholarship convention**: Accepted practice in humanities scholarship: assess provenance and corpus representativeness; preserve competing interpretations.
- **Experimental science convention**: Accepted practice for laboratory experiments: positive and negative controls, independent replicates, and stated assay sensitivity.
- **Computational simulation convention**: Accepted practice for numerical simulation: report resolution and convergence checks and validate the model against measurements.
- **High-throughput omics convention**: Accepted practice for high-throughput assays: multiple-testing correction, batch-effect control, and orthogonal validation of screening hits.

### Tool integrity

- **CiteLume source-fidelity policy**: This tool's own policy: every statement must be traceable to the retrieved study records; nothing is inferred beyond them.
