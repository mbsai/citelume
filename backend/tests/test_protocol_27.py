"""Selection protocol 2.7: OpenAlex as a full source and the ClinicalTrials.gov registry (26 Nov 2026). No network."""
import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import SearchConcept
from services import clinicaltrials_client as ct
from services import openalex_client as oa
from services.query_planning import _boolean_concept_block, build_search_plan
from services.query_utils import understand_research_question


def study(nct, refs=(), results=False, design=None):
    return {"hasResults": results, "protocolSection": {
        "identificationModule": {"nctId": nct, "briefTitle": f"Trial {nct}"},
        "descriptionModule": {"briefSummary": "Melatonin versus placebo in the ICU."},
        "designModule": design or {"studyType": "INTERVENTIONAL", "phases": ["PHASE3"],
                                   "designInfo": {"allocation": "RANDOMIZED"}, "enrollmentInfo": {"count": 80}},
        "statusModule": {"completionDateStruct": {"date": "2015-06"}},
        "referencesModule": {"references": [{"pmid": p, "type": t} for p, t in refs]}}}


class TestClinicalTrials(unittest.TestCase):
    def test_what_a_registered_trial_contributes(self):
        pmids, records, neither = ct.parse_studies([
            study("NCT1", refs=[("111", "RESULT"), ("222", "BACKGROUND"), ("333", "DERIVED")]),
            study("NCT2", results=True),
            study("NCT3"),
        ])
        self.assertEqual(pmids, ["111"])            # only the trial's own results publication
        self.assertEqual([r.id for r in records], ["nct:NCT2"])  # posted results, no publication
        self.assertIn("randomised", records[0].abstract)
        self.assertIn("80 participants", records[0].abstract)
        self.assertEqual(records[0].year, 2015)
        self.assertEqual(neither, 1)

    def test_linked_publications_come_from_pubmed(self):
        from models import PaperRecord
        seen = {}

        async def fake_pubmed(query, n, key="", year_from=None, year_to=None, broaden=True, **k):
            seen.update(query=query, year_to=year_to)
            return [PaperRecord(id="pmid:111", pmid="111", title="Melatonin RCT", source="PubMed")]

        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"studies": [study("NCT1", refs=[("111", "RESULT")])]}

        class Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, *a, **k):
                return Resp()

        with mock.patch.object(ct, "search_pubmed", fake_pubmed), mock.patch.object(ct.httpx, "AsyncClient", Client):
            recs = asyncio.run(ct.search_clinicaltrials('("melatonin")', year_to=2019))
        self.assertEqual(seen["query"], "111[uid] OR 111[uid]")
        self.assertEqual(seen["year_to"], 2019)
        self.assertEqual((recs[0].source, recs[0].pmid), ("ClinicalTrials.gov", "111"))


class TestOpenAlex(unittest.TestCase):
    def test_record(self):
        r = oa.record({"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/X", "title": "Atrazine and Aspergillus",
                       "publication_year": 2020, "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/123"},
                       "abstract_inverted_index": {"Atrazine": [0], "degraded": [1]}, "type": "article"})
        self.assertEqual((r.id, r.doi, r.pmid, r.source, r.abstract), ("oa:W1", "10.1/x", "123", "OpenAlex",
                                                                       "Atrazine degraded"))


class TestPlans(unittest.TestCase):
    def test_boolean_strings(self):
        self.assertEqual(_boolean_concept_block(SearchConcept(name="intensive care unit", synonyms=["ICU"])),
                         '("intensive care unit" OR (intensive AND care AND unit) OR icu)')
        u = understand_research_question("Melatonin for sleep in the intensive care unit")
        facets = build_search_plan(u, "deep_research", "")
        core, reviews, aspect = facets[0], facets[1], facets[-1]
        self.assertTrue(core.queries.openalex and core.queries.clinicaltrials)
        self.assertTrue(reviews.queries.openalex)
        self.assertEqual((reviews.queries.clinicaltrials, aspect.queries.openalex), ("", ""))


if __name__ == "__main__":
    unittest.main()
