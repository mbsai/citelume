"""
Primary sources behind every methodology rule in `frameworks.rules`.

Citations were checked against the publishers' or developers' records (November 2026). Where a detail such as a
volume or page range could not be confirmed it is omitted rather than guessed. `field_convention` entries mark rules
that reflect accepted practice in a discipline without a single citable framework; `tool_integrity` entries are this
tool's own source-fidelity policies. Neither kind is presented to users as an external standard.
"""
from dataclasses import dataclass
from typing import Dict, Literal

SourceKind = Literal[
    "risk_of_bias",
    "reporting",
    "certainty",
    "review_method",
    "field_convention",
    "tool_integrity",
]


@dataclass(frozen=True)
class FrameworkSource:
    source_id: str
    short_name: str
    citation: str
    year: int
    url: str
    kind: SourceKind


def _src(source_id: str, short_name: str, citation: str, year: int, url: str, kind: SourceKind) -> FrameworkSource:
    return FrameworkSource(source_id, short_name, citation, year, url, kind)


SOURCES: Dict[str, FrameworkSource] = {
    s.source_id: s
    for s in [
        # --- Evidence synthesis, certainty and review reporting -------------------------------------------------
        _src("swim", "SWiM", "Campbell M, et al. Synthesis without meta-analysis (SWiM) in systematic reviews: "
             "reporting guideline. BMJ 2020;368:l6890.", 2020, "https://doi.org/10.1136/bmj.l6890", "reporting"),
        _src("grade", "GRADE", "Guyatt GH, et al. What is \"quality of evidence\" and why is it important to "
             "clinicians? BMJ 2008;336(7651):995-998.", 2008, "https://pmc.ncbi.nlm.nih.gov/articles/PMC2364804",
             "certainty"),
        _src("prisma2020", "PRISMA 2020", "Page MJ, et al. The PRISMA 2020 statement: an updated guideline for "
             "reporting systematic reviews. BMJ 2021;372:n71.", 2021, "https://doi.org/10.1136/bmj.n71", "reporting"),
        _src("prisma_scr", "PRISMA-ScR", "Tricco AC, et al. PRISMA extension for scoping reviews (PRISMA-ScR): "
             "checklist and explanation. Ann Intern Med 2018;169(7):467-473.", 2018,
             "https://doi.org/10.7326/M18-0850", "reporting"),
        _src("sanra", "SANRA", "Baethge C, Goldbeck-Wood S, Mertens S. SANRA - a scale for the quality assessment "
             "of narrative review articles. Res Integr Peer Rev 2019.", 2019,
             "https://doi.org/10.1186/s41073-019-0064-8", "review_method"),
        _src("grant_booth", "Grant & Booth review typology", "Grant MJ, Booth A. A typology of reviews: an analysis "
             "of 14 review types and associated methodologies. Health Info Libr J 2009;26(2):91-108.", 2009,
             "https://doi.org/10.1111/j.1471-1842.2009.00848.x", "review_method"),
        _src("lit_review_org", "Literature review organisation guides", "University library writing guides on "
             "organising a literature review (chronological, thematic, methodological, theoretical), e.g. Stony "
             "Brook University Libraries.", 2024, "https://guides.library.stonybrook.edu/literature-review/organize",
             "field_convention"),
        # --- Risk-of-bias and reporting tools for biomedical study designs -------------------------------------
        _src("rob2", "RoB 2", "Sterne JAC, et al. RoB 2: a revised tool for assessing risk of bias in randomised "
             "trials. BMJ 2019;366:l4898.", 2019, "https://doi.org/10.1136/bmj.l4898", "risk_of_bias"),
        _src("robins_i_v2", "ROBINS-I V2", "ROBINS-I V2: Risk Of Bias In Non-randomized Studies - of Interventions, "
             "version 2 (launched November 2024). riskofbias.info.", 2024,
             "https://sites.google.com/site/riskofbiastool/welcome/robins-i-v2", "risk_of_bias"),
        _src("robins_e", "ROBINS-E", "Higgins JPT, et al. A tool to assess risk of bias in non-randomized follow-up "
             "studies of exposure effects (ROBINS-E). Environ Int 2024;186:108602.", 2024,
             "https://doi.org/10.1016/j.envint.2024.108602", "risk_of_bias"),
        _src("probast_ai", "PROBAST+AI", "Moons KGM, et al. PROBAST+AI: an updated quality, risk of bias, and "
             "applicability assessment tool for prediction models using regression or artificial intelligence "
             "methods. BMJ 2025.", 2025, "https://ora.ox.ac.uk/objects/uuid:8b15c087-459b-43fa-ad03-1f66ed6ba370",
             "risk_of_bias"),
        _src("quips", "QUIPS", "Hayden JA, et al. Assessing bias in studies of prognostic factors. "
             "Ann Intern Med 2013.", 2013,
             "https://abstracts.cochrane.org/2015-vienna/systematic-reviews-prognostic-studies-2-assessing-risk-bias-studies-prognostic-factors",
             "risk_of_bias"),
        _src("quadas3", "QUADAS-3", "Whiting PF, et al. QUADAS-3: a revised tool for the quality assessment of "
             "diagnostic accuracy studies. Ann Intern Med 2026;179(4).", 2026,
             "https://doi.org/10.7326/ANNALS-25-02104", "risk_of_bias"),
        _src("syrcle", "SYRCLE RoB", "Hooijmans CR, et al. SYRCLE's risk of bias tool for animal studies. "
             "BMC Med Res Methodol 2014;14:43.", 2014, "https://doi.org/10.1186/1471-2288-14-43", "risk_of_bias"),
        _src("arrive2", "ARRIVE v2", "Percie du Sert N, et al. The ARRIVE guidelines 2.0: updated guidelines for "
             "reporting animal research. 2020.", 2020,
             "https://arriveguidelines.org/news/new-arrive-guidelines-20-released", "reporting"),
        _src("amstar2", "AMSTAR 2", "Shea BJ, et al. AMSTAR 2: a critical appraisal tool for systematic reviews that "
             "include randomised or non-randomised studies of healthcare interventions, or both. BMJ 2017;358:j4008.",
             2017, "https://pmc.ncbi.nlm.nih.gov/articles/5833365", "risk_of_bias"),
        # --- Qualitative research ---------------------------------------------------------------------------------
        _src("coreq", "COREQ", "Tong A, Sainsbury P, Craig J. Consolidated criteria for reporting qualitative research "
             "(COREQ): a 32-item checklist for interviews and focus groups. Int J Qual Health Care 2007;19:349-357.",
             2007, "https://researchnow.flinders.edu.au/en/publications/consolidated-criteria-for-reporting-qualitative-research-coreq-a-/",
             "reporting"),
        _src("srqr", "SRQR", "O'Brien BC, et al. Standards for reporting qualitative research: a synthesis of "
             "recommendations. Acad Med 2014;89(9):1245-1251.", 2014,
             "https://www.casrai.org/guides/coreq-srqr-checklist-reporting-qualitative-research", "reporting"),
        # --- Computer science & machine learning -----------------------------------------------------------------
        _src("reforms", "REFORMS", "Kapoor S, et al. REFORMS: Consensus-based recommendations for "
             "machine-learning-based science. Sci Adv 2024;10:eadk3452.", 2024, "https://reforms.cs.princeton.edu/",
             "reporting"),
        _src("ml_leakage", "ML leakage taxonomy", "Kapoor S, Narayanan A. Leakage and the reproducibility crisis in "
             "machine-learning-based science. Patterns 2023;4(9):100804.", 2023,
             "https://doi.org/10.1016/j.patter.2023.100804", "risk_of_bias"),
        _src("kitchenham", "Kitchenham & Charters SLR guidelines", "Kitchenham B, Charters S. Guidelines for "
             "performing Systematic Literature Reviews in Software Engineering, v2.3. EBSE Technical Report "
             "EBSE-2007-01, Keele University and Durham University, 2007.", 2007,
             "https://ebse.webspace.durham.ac.uk/ebse-bibliography/lessons-from-applying-the-systematic-literature-review-process-within-the-software-engineering-domain/",
             "review_method"),
        # --- Physical sciences ------------------------------------------------------------------------------------
        _src("gum", "GUM (JCGM 100:2008)", "JCGM 100:2008. Evaluation of measurement data - Guide to the expression "
             "of uncertainty in measurement (GUM). Joint Committee for Guides in Metrology; also ISO/IEC Guide "
             "98-3:2008.", 2008, "https://www.bipm.org/en/committees/jc/jcgm/publications", "reporting"),
        _src("iupac_qm", "IUPAC QM data reporting", "Albright TA, Halevi EA. Presentation of the results of quantum "
             "mechanical computations in organic chemistry (IUPAC recommendations). Pure Appl Chem "
             "1992;64(8):1203-1209.", 1992, "https://publications.iupac.org/pac/64/8/1203/index.html", "reporting"),
        _src("orglett_data", "Org. Lett. data-reporting standards", "Organic Letters editorial (2023) proposing "
             "standards for reporting reaction data, including distinguishing isolated from crude yields.", 2023,
             "https://pubs.acs.org/journal/orlef7", "reporting"),
        _src("joc_purity", "JOC compound purity guidance", "The Journal of Organic Chemistry, Author Guidelines: "
             "compounds used for physical measurements or bioassays should be documented at >=95% purity.", 2024,
             "https://pubs.acs.org/journal/joceah", "reporting"),
        # --- Disciplinary conventions and the tool's own integrity policies --------------------------------------
        _src("math_convention", "Mathematics publishing convention", "Accepted practice in mathematics: results are "
             "classified by proof status (proved, conditional, conjectured) and peer-review or formal-verification "
             "status. No single citable framework.", 2026, "", "field_convention"),
        _src("physics_convention", "Physics reporting convention", "Accepted practice in physics: state parameter "
             "regimes, approximations and symmetry assumptions. No single citable framework.", 2026, "",
             "field_convention"),
        _src("chem_convention", "Chemistry reporting convention", "Accepted practice in chemistry: preserve units "
             "and conditions (solvent, catalyst, temperature) when comparing results. No single citable framework.",
             2026, "", "field_convention"),
        _src("engineering_convention", "Engineering testing convention", "Accepted practice in engineering and "
             "materials science: compare performance only under stated, standardised test conditions.", 2026, "",
             "field_convention"),
        _src("social_convention", "Social-science identification convention", "Accepted practice in empirical social "
             "science: distinguish correlational designs from causal identification strategies (RCT, IV, DiD, RDD).",
             2026, "", "field_convention"),
        _src("humanities_convention", "Humanities scholarship convention", "Accepted practice in humanities "
             "scholarship: assess provenance and corpus representativeness; preserve competing interpretations.",
             2026, "", "field_convention"),
        _src("experimental_convention", "Experimental science convention", "Accepted practice for laboratory "
             "experiments: positive and negative controls, independent replicates, and stated assay sensitivity.",
             2026, "", "field_convention"),
        _src("simulation_convention", "Computational simulation convention", "Accepted practice for numerical "
             "simulation: report resolution and convergence checks and validate the model against measurements.",
             2026, "", "field_convention"),
        _src("omics_convention", "High-throughput omics convention", "Accepted practice for high-throughput assays: "
             "multiple-testing correction, batch-effect control, and orthogonal validation of screening hits.",
             2026, "", "field_convention"),
        _src("tool_integrity", "CiteLume source-fidelity policy", "This tool's own policy: every statement must "
             "be traceable to the retrieved study records; nothing is inferred beyond them.", 2026, "",
             "tool_integrity"),
    ]
}
