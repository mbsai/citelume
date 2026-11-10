"""The study selection protocol: eligibility, scoring and balanced selection (docs/RANKING.md)."""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import PaperRecord, SearchAspect
from services.graph_builder import deduplicate_and_rank_papers
from services.paper_selection import CRITERIA, POINTS, bm25_scores, classify_study_type, percentile_ranks
from services.query_utils import compile_deterministic_decomposition_plan as plan_for

YEAR = datetime.now(timezone.utc).year
LONG = " This abstract is long enough to be assessed by the selection protocol."


def paper(i, title, abstract="", year=None, cites=None, types=(), authors=None, by=(), source="PubMed"):
    return PaperRecord(
        id=f"p{i}", title=title, abstract=(abstract + LONG) if abstract is not None else "", year=year or YEAR - 3,
        citation_count=cites or 0, citation_source="iCite" if cites is not None else "", source=source,
        publication_types=list(types), authors=authors or [f"Author{i} A"], retrieved_by=list(by), pmid=str(1000 + i),
    )


def select(papers, query, n=20, mode="standard"):
    plan = plan_for(query, mode)
    reports = []
    chosen = deduplicate_and_rank_papers(papers, query, n, plan.question_understanding, {}, reports)
    return chosen, reports[0], plan


class TestStudyType(unittest.TestCase):
    def test_types_from_source_and_wording(self):
        cases = {
            "evidence_synthesis": paper(1, "Statins after MI: a systematic review and meta-analysis"),
            "rct": paper(2, "Vitamin D for flu", "Participants were randomly assigned to vitamin D or placebo."),
            "review": paper(3, "Vitamin D and immunity", "Several randomized trials exist. In this review we discuss them."),
            "case_report": paper(4, "A rare case of gentamicin toxicity"),
            "controlled_observational": paper(5, "Sleep and dementia", "A prospective cohort of 2,000 adults."),
            "descriptive_observational": paper(6, "CAD in Kerala", "A cross-sectional survey of prevalence."),
            "correction": paper(7, "Correction: Statins after MI"),
            "protocol": paper(8, "Exercise for depression: study protocol for a randomised trial"),
            "conference_abstract": paper(9, "0067 The effect of aging on memory"),
        }
        for want, p in cases.items():
            self.assertEqual(classify_study_type(p), want, p.title)
        self.assertEqual(classify_study_type(paper(10, "Trial X", types=["Randomized Controlled Trial"])), "rct")
        # A year at the start of a title is not meeting-abstract numbering
        self.assertEqual(classify_study_type(paper(11, "2019 ESC Guidelines for chronic coronary syndromes")), "evidence_synthesis")


class TestScoring(unittest.TestCase):
    def test_points_add_up_to_100_and_match_published_criteria(self):
        self.assertEqual(sum(POINTS.values()), 100)
        self.assertEqual(sum(c.max_points for c in CRITERIA), 100)

    def test_breakdown_sums_to_total_and_stays_in_range(self):
        ps = [paper(i, f"Gut microbiome study {i}", "The gut microbiome was profiled.", year=YEAR - i, cites=i * 10)
              for i in range(1, 9)]
        chosen, _, _ = select(ps, "gut microbiome")
        for p in chosen:
            b = p.selection_breakdown
            self.assertAlmostEqual(b["relevance"] + b["evidence"] + b["impact"] + b["recency"] + b["agreement"], b["total"], delta=0.3)
            for key, cap in POINTS.items():
                self.assertTrue(0 <= b[key] <= cap, (key, b[key]))

    def test_each_criterion_moves_the_score_the_right_way(self):
        base = dict(abstract="The gut microbiome was profiled in adults.", year=YEAR - 5, cites=10)
        ps = [
            paper(1, "Gut microbiome in adults", **base),
            paper(2, "Gut microbiome in adults: a systematic review", **base),                      # stronger study type
            paper(3, "Gut microbiome in adults B", **{**base, "cites": 900}),                         # more citations
            paper(4, "Gut microbiome in adults C", **{**base, "year": YEAR - 1}),                     # newer
            paper(5, "Gut microbiome in adults D", by=["Core search", "Mechanisms", "Reviews"], **base),  # more searches
            paper(6, "Obesity in adults", "Diet and the gut microbiome were recorded.", year=YEAR - 5, cites=10),  # weaker match
        ] + [paper(10 + i, f"Gut microbiome filler {i}", **{**base, "cites": 5 + i}) for i in range(6)]
        chosen, _, _ = select(ps, "gut microbiome")
        s = {p.id: p.selection_breakdown for p in chosen}
        self.assertGreater(s["p2"]["evidence"], s["p1"]["evidence"])
        self.assertGreater(s["p3"]["impact"], s["p1"]["impact"])
        self.assertGreater(s["p4"]["recency"], s["p1"]["recency"])
        self.assertGreater(s["p5"]["agreement"], s["p1"]["agreement"])
        self.assertLess(s["p6"]["relevance"], s["p1"]["relevance"])

    def test_new_and_uncited_studies_get_the_middle_impact_value(self):
        ps = [paper(i, f"Gut microbiome study {i}", "The gut microbiome.", year=YEAR - 6, cites=50 * i) for i in range(1, 7)]
        ps.append(paper(7, "Gut microbiome new study", "The gut microbiome.", year=YEAR, cites=0))
        ps.append(paper(8, "Gut microbiome preprint", "The gut microbiome.", year=YEAR - 6, cites=None))
        chosen, _, _ = select(ps, "gut microbiome")
        s = {p.id: p.selection_breakdown for p in chosen}
        self.assertEqual(s["p7"]["impact"], POINTS["impact"] / 2)
        self.assertEqual(s["p8"]["impact"], POINTS["impact"] / 2)

    def test_bm25_and_percentiles(self):
        scores = bm25_scores("gut microbiome diet", ["gut microbiome and diet", "kidney stones", "microbiome"])
        self.assertTrue(scores[0] > scores[2] > scores[1] == 0)
        self.assertEqual(percentile_ranks([1, None, 3, 3]), [1 / 3, None, 2.5 / 3, 2.5 / 3])

    def test_same_input_gives_same_order(self):
        make = lambda: [paper(i, f"Gut microbiome study {i}", "The gut microbiome.", year=YEAR - (i % 4), cites=i * 3) for i in range(1, 30)]
        a = [p.id for p in select(make(), "gut microbiome")[0]]
        b = [p.id for p in select(list(reversed(make())), "gut microbiome")[0]]
        self.assertEqual(a, b)


class TestEligibility(unittest.TestCase):
    def test_exclusions_are_counted_by_reason(self):
        good = [paper(i, f"Coronary artery disease in India study {i}", "Indian adults with coronary artery disease.") for i in range(1, 8)]
        bad = [
            paper(20, "Coronary artery disease in American Indian adults", "American Indian adults with coronary artery disease."),
            paper(21, "Coronary artery disease in Indiana", "Adults in Indiana with coronary artery disease."),
            paper(22, "Fatty liver disease in India", "Indian adults with NAFLD."),
            paper(23, "Correction: Coronary artery disease in India", "Indian adults with coronary artery disease."),
        ]
        chosen, rep, _ = select(good + bad, "What is the state of Coronary artery disease in India")
        ids = {p.id for p in chosen}
        self.assertEqual(ids, {p.id for p in good})
        self.assertEqual(rep.excluded.get("Outside the place or population asked about"), 2)
        self.assertEqual(rep.excluded.get("Not about the topic of the question"), 1)
        self.assertEqual(sum(v for k, v in rep.excluded.items() if "correction" in k), 1)
        self.assertEqual((rep.unique_records, rep.eligible, rep.included), (11, 7, 7))

    def test_full_matches_rank_ahead_of_partial_matches(self):
        full = [paper(i, f"Sleep and memory in older adults {i}", "Sleep and memory in older adults.", cites=1) for i in range(1, 7)]
        partial = paper(9, "Sleep duration in older adults: a meta-analysis", "Sleep in older adults.", cites=5000, year=YEAR)
        chosen, _, _ = select(full + [partial], "How does sleep affect memory in older adults?")
        self.assertEqual(chosen[-1].id, "p9")
        self.assertEqual(chosen[-1].relevance, "partly")


class TestBalancedSelection(unittest.TestCase):
    def test_review_cap_and_author_limit(self):
        reviews = [paper(i, f"Sleep and memory in older adults: a review {i}", "We review sleep and memory in older adults.",
                         cites=500, types=["Review"]) for i in range(1, 16)]
        primary = [paper(100 + i, f"Sleep and memory in older adults cohort {i}", "A prospective cohort of sleep and memory in older adults.",
                         cites=5) for i in range(1, 21)]
        same_author = [paper(200 + i, f"Sleep and memory in older adults trial {i}", "Participants were randomly assigned. Sleep and memory in older adults.",
                             cites=800, authors=["Smith J"], year=YEAR - 2) for i in range(1, 6)]
        chosen, rep, _ = select(reviews + primary + same_author, "How does sleep affect memory in older adults?")
        self.assertEqual(len(chosen), 20)
        self.assertEqual(sum(1 for p in chosen if p.study_type == "review"), 6)       # 30% of 20
        self.assertEqual(sum(1 for p in chosen if p.authors == ["Smith J"]), 2)
        self.assertEqual(rep.included_by_study_type["Narrative review"], 6)

    def test_reviews_of_all_kinds_leave_room_for_primary_studies(self):
        metas = [paper(i, f"Vitamin D and COVID-19: a meta-analysis {i}", "Vitamin D and COVID-19 pooled.", cites=900) for i in range(1, 26)]
        primary = [paper(100 + i, f"Vitamin D and COVID-19 cohort {i}", "A prospective cohort of vitamin D and COVID-19.", cites=1) for i in range(1, 16)]
        chosen, _, _ = select(metas + primary, "Does vitamin D prevent COVID-19?")
        self.assertEqual(sum(1 for p in chosen if p.study_type == "evidence_synthesis"), 12)   # 60% of 20
        self.assertEqual(sum(1 for p in chosen if p.study_type == "controlled_observational"), 8)

    def test_each_aspect_gets_its_best_study(self):
        ps = [paper(i, f"Gut microbiome composition {i}", "Gut microbiome diversity.", cites=400) for i in range(1, 26)]
        ps.append(paper(99, "Gut microbiome transplant therapy", "Gut microbiome. Faecal transplant therapy trial.", cites=0, year=YEAR - 9))
        plan = plan_for("gut microbiome")
        u = plan.question_understanding
        u.aspects = u.aspects[:2] + [SearchAspect(label="Therapy", terms=["transplant", "therapy"])]
        reports = []
        chosen = deduplicate_and_rank_papers(ps, "gut microbiome", 20, u, {}, reports)
        picked = next(p for p in chosen if p.id == "p99")
        self.assertEqual(picked.selection_reason, "Best match for the aspect: Therapy")
        self.assertEqual(reports[0].included_by_aspect["Therapy"], 1)

    def test_null_results_are_kept_for_effect_questions(self):
        pos = [paper(i, f"Vitamin D prevents COVID-19 infection {i}", "Vitamin D reduced COVID-19 infection.", cites=300) for i in range(1, 26)]
        null = paper(77, "Vitamin D and COVID-19 infection", "Vitamin D was not associated with COVID-19 infection.", cites=0, year=YEAR - 9)
        chosen, _, _ = select(pos + [null], "Does vitamin D prevent COVID-19?")
        picked = next(p for p in chosen if p.id == "p77")
        self.assertEqual(picked.selection_reason, "Kept as a null or contrary result")

    def test_numbers_follow_rank_order(self):
        ps = [paper(i, f"Gut microbiome study {i}", "The gut microbiome.", cites=i) for i in range(1, 12)]
        chosen, _, _ = select(ps, "gut microbiome")
        self.assertEqual([p.citation_index for p in chosen], list(range(1, 12)))
        totals = [p.selection_breakdown["total"] for p in chosen]
        self.assertEqual(totals, sorted(totals, reverse=True))


class TestRaisedDotDecimals(unittest.TestCase):
    def test_lancet_style_decimals_are_read_as_decimals(self):
        from services.verification_gates import extract_nontrivial_numbers, study_source_text
        p = PaperRecord(id="x", title="Vitamin D", abstract="Odds ratio 0·70 [0·55-0·89] at ages 1·00-15·99 years.")
        self.assertIn("0.70 [0.55-0.89]", p.abstract)
        p.full_text_sections = {"results": "Hazard ratio 4·82."}
        self.assertIn("4.82", study_source_text(p))
        self.assertEqual([n for n in extract_nontrivial_numbers("OR 0.70 and 15.99") if n not in study_source_text(p)], [])


class TestPublishedProtocol(unittest.TestCase):
    def test_ranking_document_matches_the_code(self):
        from services.ranking_doc import DOC_PATH, render
        self.assertEqual(DOC_PATH.read_text(encoding="utf-8"), render(),
                         "docs/RANKING.md is out of date: run `python -m services.ranking_doc` from backend/")


if __name__ == "__main__":
    unittest.main()
