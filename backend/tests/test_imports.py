"""Session 3d: search strings for other databases, imported exports and the user's own PDFs."""
import base64
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import main
from services import corpus_store, demo_mode, search_counts
from services.importers import manual_search_strings, parse_export, parse_pdf
from services.replay import check
from tests import test_golden_search as golden
from tests.test_corpus_endpoints import CONFIG, fake_sources
from tests.test_review_protocol import protocol_for

FIX = Path(__file__).parent / "fixtures" / "imports"
LONG = ("Older adults were randomly assigned to vitamin D supplementation or placebo and fractures were recorded. "
        "Methods. The trial enrolled participants aged 65 years and over in primary care. Results. Fractures "
        "occurred less often than expected in both groups. ") * 6


def make_pdf(title: str, doi: str = "", body: str = LONG, text: bool = True) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    if text:
        y = 800
        for line in [title, f"https://doi.org/{doi}" if doi else "", "Abstract", body[:90], body[90:180]] + [
                body[i:i + 95] for i in range(180, len(body), 95)]:
            c.drawString(40, y, line)
            y -= 14
            if y < 60:
                c.showPage()
                y = 800
    else:
        c.rect(50, 50, 400, 600, fill=1)  # an image-like page with no text layer
    c.save()
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


class TestParsers(unittest.TestCase):
    def test_every_format(self):
        cases = {"scopus.ris": ("ris", "Scopus", 3), "wos.txt": ("wos", "Web of Science", 2),
                 "pubmed.nbib": ("medline", "PubMed", 2), "records.bib": ("bibtex", "Imported", 1),
                 "scopus.csv": ("csv", "Scopus", 1)}
        for name, (fmt, source, n) in cases.items():
            records, got_fmt, got_source = parse_export((FIX / name).read_bytes(), name)
            self.assertEqual((got_fmt, got_source, len(records)), (fmt, source, n), name)
            first = records[0]
            self.assertTrue(first.title and first.year and first.abstract, name)
            self.assertTrue(first.doi and first.doi.startswith("10.1000/"), name)
            self.assertIn("(imported by the user)", first.retrieved_by[0])
        wos = parse_export((FIX / "wos.txt").read_bytes(), "wos.txt")[0][0]
        self.assertEqual(wos.title, "Vitamin D supplementation and fractures in older adults: a randomised trial in Japan")
        self.assertEqual(wos.authors, ["Tanaka, Hiroshi", "Sato, Mai"])
        nbib = parse_export((FIX / "pubmed.nbib").read_bytes(), "x.nbib")[0][0]
        self.assertEqual((nbib.pmid, nbib.doi), ("99000001", "10.1000/nbib-new-1"))
        self.assertIn("usual care", nbib.abstract)
        with self.assertRaises(ValueError):
            parse_export(b"just some words", "notes.doc")

    def test_search_strings_for_scopus_and_web_of_science(self):
        out = manual_search_strings([("vitamin D supplementation", ["cholecalciferol"]), ("fractures", [])],
                                    [["older adults", "elderly"]], 2010, 2020)
        scopus, wos = out[0]["query"], out[1]["query"]
        self.assertEqual(scopus, 'TITLE-ABS-KEY(("vitamin D supplementation" OR cholecalciferol) AND fractures AND '
                                 '("older adults" OR elderly)) AND PUBYEAR > 2009 AND PUBYEAR < 2021')
        self.assertTrue(wos.startswith('TS=(("vitamin D supplementation"'))
        self.assertTrue(wos.endswith("AND PY=(2010-2020)"))

    def test_pdfs(self):
        rec = parse_pdf(make_pdf("Vitamin D and fractures: a randomised trial in older adults", "10.1000/study3"), "a.pdf")
        self.assertEqual(rec.doi, "10.1000/study3")
        self.assertTrue(rec.full_text_sections)
        with self.assertRaisesRegex(ValueError, "no readable text layer"):
            parse_pdf(make_pdf("x", text=False), "scan.pdf")
        with self.assertRaisesRegex(ValueError, "not a PDF"):
            parse_pdf(b"hello", "x.pdf")


class TestImportsInTheCorpus(unittest.TestCase):
    def setUp(self):
        golden.query_utils._DECOMPOSITION_CACHE.clear()
        self.client = TestClient(main.app)

    def upload(self, name, source=""):
        r = self.client.post("/api/imports", json={"filename": name, "content_base64": b64((FIX / name).read_bytes()),
                                                   "source": source, "search_string": "TITLE-ABS-KEY(test)"})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def build(self, framework="systematic_prisma", **extra):
        p = protocol_for("deep_research", framework).model_dump()
        with fake_sources():
            r = self.client.post("/api/corpus", json={"query": golden.QUERY, "mode": "deep_research",
                                                      "protocol": p, "config": CONFIG, **extra})
        return r

    def test_imported_records_join_the_pool_and_replay(self):
        imp = self.upload("scopus.ris")
        self.assertEqual((imp["source"], imp["records"]), ("Scopus", 3))
        r = self.build(import_ids=[imp["import_id"]])
        self.assertEqual(r.status_code, 200, r.text)
        s = r.json()
        self.assertEqual(s["source_stats"]["Scopus (imported by the user)"], 3)
        ids = [p["id"] for p in s["included"]]
        # The two new trials are eligible and included; the duplicate of trial 2 merged into the PubMed record
        new = [p for p in s["included"] if p["doi"] in ("10.1000/scopus-new-1", "10.1000/scopus-new-2")]
        self.assertEqual(len(new), 2)
        self.assertEqual(sum(1 for p in s["included"] if p["doi"] == "10.1000/study2"), 1)
        merged = next(p for p in s["included"] if p["doi"] == "10.1000/study2")
        self.assertIn("Scopus", merged["all_sources"])
        corpus = corpus_store.load(s["corpus_id"])
        self.assertEqual(corpus.imports[0].search_string, "TITLE-ABS-KEY(test)")
        self.assertTrue(check(corpus)[0])
        self.assertEqual(len(ids), len(set(ids)))

    def test_imports_show_in_the_flow_and_search_strategy(self):
        imp = self.upload("wos.txt")
        s = self.build(import_ids=[imp["import_id"]]).json()
        with fake_sources():
            full = self.client.post(f"/api/corpus/{s['corpus_id']}/analyse", json={"config": CONFIG}).json()
        flow = full["review_protocol"]["flow"]
        self.assertEqual(flow["identified_by_source"]["Web of Science (imported by the user)"], 2)
        self.assertIn("TITLE-ABS-KEY(test)", str(full["review_protocol"]["search_strategies"]))

    def test_a_pdf_attaches_full_text_to_its_study(self):
        pdf = self.client.post("/api/imports/pdf", json={
            "filename": "trial3.pdf",
            "content_base64": b64(make_pdf("Vitamin D supplementation and fractures in older adults: randomised trial 3",
                                           "10.1000/study3"))}).json()
        s = self.build(user_pdfs=[{"pdf_id": pdf["pdf_id"]}]).json()
        corpus = corpus_store.load(s["corpus_id"])
        target = next(r.id for r in corpus.records if r.doi == "10.1000/study3")
        self.assertEqual(corpus.user_pdfs[0].matched_record_id, target)
        self.assertIn(target, corpus.user_full_texts)
        self.assertNotIn("User PDF", " ".join(p.source for p in corpus.records))  # merged, not a new study
        self.assertEqual(len(corpus.records), len(corpus_store.load(self.build().json()["corpus_id"]).records))
        run = main._run_from_corpus(corpus, main.ApiKeysConfig(**CONFIG))
        p3 = next(p for p in run.papers if p.id == target)
        self.assertTrue(p3.full_text_source.startswith("PDF supplied by the user"))
        self.assertTrue(check(corpus)[0])

    def test_a_pdf_can_be_included_regardless_of_the_criteria_with_a_reason(self):
        body = "Children with asthma received inhaled steroids; lung function was measured. " * 12
        pdf = self.client.post("/api/imports/pdf", json={
            "filename": "asthma.pdf", "content_base64": b64(make_pdf("Inhaled steroids in children with asthma", body=body))}).json()
        no_reason = self.build(user_pdfs=[{"pdf_id": pdf["pdf_id"], "force_include": True}])
        self.assertEqual(no_reason.status_code, 400)
        s = self.build(user_pdfs=[{"pdf_id": pdf["pdf_id"], "force_include": True,
                                   "reason": "Key trial named by the supervisor"}]).json()
        added = [p for p in s["included"] if p["title"].startswith("Inhaled steroids")]
        self.assertEqual(len(added), 1)
        self.assertIn("Key trial named by the supervisor", added[0]["selection_reason"])
        self.assertEqual(s["source_stats"]["Full texts supplied by the user"], 1)
        self.assertTrue(check(corpus_store.load(s["corpus_id"]))[0])

    def test_preview_offers_the_strings_and_the_demo_refuses_uploads(self):
        async def fake_count(*_a, **_k):
            return search_counts.SearchCount(source="PubMed", count=1)
        with mock.patch.object(search_counts, "count_pubmed", fake_count), \
                mock.patch.object(search_counts, "count_semantic_scholar", fake_count), \
                mock.patch.object(search_counts, "count_preprints", fake_count), \
                mock.patch.object(search_counts, "count_arxiv", fake_count):
            pv = self.client.post("/api/protocol/preview", json={"protocol": protocol_for().model_dump()}).json()
        self.assertEqual([m["database"] for m in pv["manual_searches"]], ["Scopus", "Web of Science"])
        self.assertTrue(pv["imports_allowed"])
        with mock.patch.object(demo_mode, "demo_enabled", return_value=True):
            r = self.client.post("/api/imports", json={"filename": "scopus.ris",
                                                       "content_base64": b64((FIX / "scopus.ris").read_bytes())})
            self.assertEqual(r.status_code, 403)
            r = self.client.post("/api/imports/pdf", json={"filename": "a.pdf", "content_base64": b64(make_pdf("x" * 40))})
            self.assertEqual(r.status_code, 403)


if __name__ == "__main__":
    unittest.main()
