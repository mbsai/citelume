"""
Golden test for the whole search pipeline: fixed fake database results in, the included studies and the shape of
the response out. It guards refactoring (the corpus layer and per-module endpoints must not change what a search
returns). The expected values live in tests/fixtures/golden_search.json; after an intended change in behaviour,
regenerate them with `UPDATE_GOLDEN=1 python -m unittest tests.test_golden_search` and review the diff.
"""
import asyncio
import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
from models import ApiKeysConfig, PaperRecord, SearchRequest
from services import query_utils, synthesis_engine
from services.gap_matrix_engine import clear_gap_verification_cache
from services.llm_service import set_test_llm_responder

import tempfile

# Corpora written by these runs go to a temporary folder, not the real cache
os.environ["CORPUS_DIR"] = tempfile.mkdtemp(prefix="citelume_test_corpora_")

GOLDEN = Path(__file__).parent / "fixtures" / "golden_search.json"
QUERY = "Does vitamin D supplementation reduce fractures in older adults?"
YEAR = datetime.now(timezone.utc).year
PAD = " Participants were followed for several years and outcomes were recorded in older adults."


def _records():
    """About 50 records as the databases would return them: trials, cohorts, reviews, partial matches, off-topic
    records, a correction, a record without an abstract, and the same studies from two databases."""
    out = []

    def add(i, title, abstract, source="PubMed", year_back=3, cites=None, types=(), doi=True, pmid=True):
        out.append(dict(
            id=f"{source[:2].lower()}{i}", title=title, abstract=abstract, source=source, year=YEAR - year_back,
            authors=[f"Author{i} A", f"Writer{i} B"], citation_count=cites or 0,
            citation_source="iCite" if cites is not None else "", publication_types=list(types),
            doi=f"10.1000/study{i}" if doi else None, pmid=str(5000 + i) if pmid and source == "PubMed" else None,
            venue="Journal of Bone Research", url=f"https://example.org/{i}",
        ))

    for i in range(1, 15):
        add(i, f"Vitamin D supplementation and fractures in older adults: randomised trial {i}",
            f"Older adults were randomly assigned to vitamin D supplementation or placebo. Fractures occurred in "
            f"{10 + i}.5% versus {11 + i}.2%." + (" There was no significant difference." if i % 3 == 0 else "") + PAD,
            year_back=1 + i % 9, cites=20 * i, types=["Randomized Controlled Trial"])
    for i in range(15, 22):
        add(i, f"Vitamin D supplementation and fracture risk in older adults: cohort study {i}",
            "A prospective cohort of older adults taking vitamin D supplementation was followed for fractures." + PAD,
            year_back=2 + i % 6, cites=5 * i)
    for i in range(22, 29):
        add(i, f"Vitamin D supplementation for preventing fractures in older adults: a systematic review and "
               f"meta-analysis {i}", "We pooled randomised trials of vitamin D supplementation and fractures in older "
               "adults." + PAD, year_back=1 + i % 5, cites=40 * i, types=["Meta-Analysis"])
    for i in range(29, 33):
        add(i, f"Vitamin D and bone health in older adults: a narrative review {i}",
            "In this review we discuss vitamin D supplementation, fractures and falls in older adults." + PAD,
            year_back=2, cites=15, types=["Review"])
    for i in range(33, 39):
        add(i, f"Vitamin D supplementation and muscle strength in older adults {i}",
            "Vitamin D supplementation was given to older adults and grip strength was measured." + PAD, cites=8)
    for i in range(39, 44):
        add(i, f"Vitamin D supplementation and fractures in children {i}",
            "Children received vitamin D supplementation and fractures were recorded over two years of follow-up.",
            cites=12)
    for i in range(44, 48):
        add(i, f"Statin therapy after myocardial infarction {i}",
            "Adults received statins after myocardial infarction and cardiovascular events were recorded." + PAD,
            cites=30)
    add(48, "Correction: Vitamin D supplementation and fractures in older adults",
        "This corrects the article on vitamin D supplementation and fractures in older adults." + PAD)
    add(49, "Vitamin D supplementation and fractures in older adults: a registry report", "", cites=3)
    # The same studies as indexed by a second database (merged by DOI), and one preprint
    for i in (1, 2, 3, 15, 22):
        src = dict(out[i - 1])
        src.update(id=f"se{i}", source="Semantic Scholar", pmid=None, citation_count=src["citation_count"] + 7,
                   citation_source="Semantic Scholar")
        out.append(src)
    add(50, "Vitamin D supplementation and fractures in older adults: a preprint trial",
        "Older adults were randomly assigned to vitamin D supplementation or placebo and fractures were counted." + PAD,
        source="bioRxiv", year_back=1, pmid=False)
    return out


RECORDS = _records()


def _by_source(source):
    async def fake(*_args, **_kwargs):
        return [PaperRecord(**r) for r in RECORDS if r["source"] == source]
    return fake


async def _none(*_args, **_kwargs):
    return 0


async def _empty(*_args, **_kwargs):
    return []


def run_search(mode="standard", framework="narrative", llm_key="", responder=None):
    """Runs the pipeline on the fixed records with no network access."""
    query_utils._DECOMPOSITION_CACHE.clear()
    synthesis_engine._STUDY_EXTRACTION_CACHE.clear()
    clear_gap_verification_cache()
    set_test_llm_responder(responder)
    cfg = ApiKeysConfig(llm_provider="openrouter", llm_api_key=llm_key, enable_hf=False, enable_fulltext=False)
    req = SearchRequest(query=QUERY, mode=mode, review_framework=framework, config=cfg)
    patches = [
        mock.patch.object(main, "search_pubmed", _by_source("PubMed")),
        mock.patch.object(main, "search_semantic_scholar", _by_source("Semantic Scholar")),
        mock.patch.object(main, "search_biorxiv", _by_source("bioRxiv")),
        mock.patch.object(main, "search_arxiv", _by_source("arXiv")),
        mock.patch.object(main, "enrich_citation_counts", _none),
        mock.patch.object(main, "expand_paper_neighbors", _empty),
        mock.patch.object(main.citation_chasing, "_post_batch", _empty),
        mock.patch.object(main, "add_mesh_headings", _none),
    ]
    for p in patches:
        p.start()
    try:
        return asyncio.run(main._run_research_search(req))
    finally:
        for p in patches:
            p.stop()
        set_test_llm_responder(None)


def snapshot(result) -> dict:
    """What must not change when the code is reorganised."""
    sel = result.selection_report
    return {
        "included_ids": [p.id for p in result.papers],
        "citation_indices": [p.citation_index for p in result.papers],
        "study_types": [p.study_type for p in result.papers],
        "records_found": sel.records_found,
        "unique_records": sel.unique_records,
        "eligible": sel.eligible,
        "excluded": dict(sel.excluded),
        "included_by_study_type": dict(sel.included_by_study_type),
        "section_headings": [s.heading for s in result.sections],
        "bibliography_count": len(result.bibliography),
        "evidence_rows": len(result.evidence_matrix.rows) if result.evidence_matrix else 0,
        "graph_paper_nodes": sum(1 for n in result.graph.nodes if n.node_type == "paper"),
        "gap_ids": [g.gap_id for g in result.gap_matrix.structured_gaps],
        "response_fields": sorted(k for k, v in result.model_dump().items() if v not in (None, [], {}, "")),
        "source_stats": {k: v for k, v in result.source_stats.items() if v},
    }


CASES = {
    "standard": dict(mode="standard"),
    "deep_research": dict(mode="deep_research"),
    "prisma": dict(mode="deep_research", framework="systematic_prisma"),
}


class TestGoldenSearch(unittest.TestCase):
    maxDiff = None

    def test_search_output_matches_the_golden_record(self):
        actual = {name: snapshot(run_search(**kw)) for name, kw in CASES.items()}
        if os.environ.get("UPDATE_GOLDEN"):
            GOLDEN.parent.mkdir(exist_ok=True)
            GOLDEN.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n")
        expected = json.loads(GOLDEN.read_text())
        for name in CASES:
            self.assertEqual(actual[name], expected[name], name)

    def test_a_model_planning_the_searches_does_not_change_the_selection(self):
        """Same records, with and without a model: the model's plan (different synonyms and aspects) may change the
        searches, but selection reads the rule-based parse, so the included studies are identical."""
        def responder(system_prompt, _user_prompt, _config, _attempt):
            if system_prompt.startswith("You plan literature searches"):
                return json.dumps({
                    "concepts": [{"name": "vitamin D", "synonyms": ["cholecalciferol", "muscle strength"]},
                                 {"name": "fractures", "synonyms": ["grip strength", "bone health"]}],
                    "breadth": "focused",
                    "aspects": [{"label": "Muscle outcomes", "terms": ["muscle", "grip strength"],
                                 "question": "Does vitamin D improve muscle strength?"},
                                {"label": "Narrative reviews", "terms": ["review", "discuss"],
                                 "question": "What do narrative reviews say?"}],
                })
            return None  # every other step falls back to the rule-based text

        # The PRISMA case is the sensitive one: with the model's synonyms the muscle-strength studies would count
        # as addressing the question and be included (this test fails if selection reads the model's plan)
        for case in (CASES["standard"], CASES["prisma"]):
            without = run_search(**case)
            with_model = run_search(llm_key="test-key", responder=responder, **case)
            self.assertNotEqual(with_model.decomposition_plan.question_understanding.planned_by, "rules")
            self.assertEqual([p.id for p in with_model.papers], [p.id for p in without.papers])
            self.assertEqual(with_model.selection_report.excluded, without.selection_report.excluded)
            self.assertEqual(with_model.selection_report.concepts_source, "rules")
            self.assertEqual([p.selection_breakdown for p in with_model.papers],
                             [p.selection_breakdown for p in without.papers])

    def test_the_search_does_not_change_the_records_it_was_given(self):
        import copy
        from services.corpus import merge_duplicates
        raw = [PaperRecord(**r) for r in RECORDS]
        before = copy.deepcopy([r.model_dump() for r in raw])
        merged = merge_duplicates(raw)
        self.assertEqual([r.model_dump() for r in raw], before)
        self.assertLess(len(merged), len(raw))
        self.assertEqual([m.id for m in merge_duplicates(raw)], [m.id for m in merged])

    def test_stored_corpus_replays_to_the_same_selection(self):
        from services import corpus_store
        from services.replay import check
        for name, kw in CASES.items():
            result = run_search(**kw)
            corpus_store._MEMORY.clear()  # force the read from disk
            corpus = corpus_store.load(result.corpus_id)
            self.assertIsNotNone(corpus, name)
            self.assertEqual(corpus.included_ids, [p.id for p in result.papers], name)
            self.assertEqual(corpus.content_hash, result.corpus_hash, name)
            # every record is accounted for: a ranked candidate or excluded with a reason
            accounted = {c.record_id for c in corpus.candidates} | set(corpus.excluded)
            self.assertEqual(accounted, {r.id for r in corpus.records}, name)
            ok, lines = check(corpus)
            self.assertTrue(ok, f"{name}: " + " | ".join(lines))
        self.assertIsNone(corpus_store.load("../../etc/passwd"))

    def test_a_search_is_repeatable(self):
        a, b = run_search(), run_search()
        self.assertEqual(snapshot(a), snapshot(b))
        self.assertEqual(a.corpus_hash, b.corpus_hash)  # same records and inputs, same selection
        self.assertNotEqual(a.corpus_id, b.corpus_id)


if __name__ == "__main__":
    unittest.main()
