import sys
import unittest
from pathlib import Path

# Ensure backend directory is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models import PaperRecord, TaskCStudyExtractionItem
from services.fulltext import canonical_section, parse_arxiv_html, parse_jats, parse_pdf_text
from services.claim_synthesis import extract_atomic_findings
from services.evidence_table_engine import build_structured_evidence_matrix
from services.synthesis_engine import _full_text_excerpt
from services.verification_gates import verify_study_extraction_against_source

JATS = """<article xmlns:xlink="http://www.w3.org/1999/xlink"><front><article-meta><title-group>
<article-title>Prevalence of CAD in Kerala</article-title></title-group></article-meta></front><body>
<sec><title>Background</title><p>CAD is common in India.</p></sec>
<sec><title>Methods</title><sec><title>Study population</title><p>We surveyed 5167 adults aged 20-79 years.</p></sec></sec>
<sec><title>Results</title><p>The age-adjusted prevalence of CAD was 12.5% in men.</p></sec>
<sec><title>Strengths and limitations</title><p>Self-reported history may misclassify cases.</p></sec>
</body></article>"""

ARXIV_HTML = """<html><body><article>
<h2 class="ltx_title">1 Introduction</h2><p class="ltx_p">Leakage inflates scores.</p>
<h2 class="ltx_title">3 Proposed taxonomy</h2><p class="ltx_p">We define eight leakage types.</p>
<h3 class="ltx_title">3.1 Label leakage</h3><p class="ltx_p">Labels leak through features.</p>
<h2 class="ltx_title">5 Experiments</h2><p class="ltx_p">Accuracy fell by 14.2 points without leakage.</p>
<math>x^2</math>
<h2 class="ltx_title">References</h2><p>[1] Some citation.</p>
</article></body></html>"""

PDF_TEXT = """Title of paper
Abstract
We study spiking networks.
1. Introduction
Spiking networks are efficient.
2. Methods
We simulated 1000 neurons.
3. Results
Energy fell by 37.5%.
4. Discussion
Hardware limits apply.
References
[1] A. Author."""


class TestFullTextParsing(unittest.TestCase):
    def test_section_names_cover_non_medical_headings(self):
        self.assertEqual(canonical_section("2.3 Statistical analysis"), "methods")
        self.assertEqual(canonical_section("5 Experiments"), "results")
        self.assertEqual(canonical_section("Strengths and limitations"), "limitations")
        self.assertIsNone(canonical_section("Data leakage: definition"))

    def test_jats_sections(self):
        s = parse_jats(JATS)
        self.assertIn("5167 adults", s["methods"])  # nested <sec> inherits its parent's section
        self.assertIn("12.5%", s["results"])
        self.assertIn("misclassify", s["limitations"])

    def test_arxiv_html_keeps_unmatched_sections_and_skips_references(self):
        s = parse_arxiv_html(ARXIV_HTML)
        self.assertIn("eight leakage types", s["methods"])  # "Proposed ..." heading
        self.assertIn("14.2 points", s["results"])
        self.assertNotIn("Some citation", " ".join(s.values()))
        self.assertNotIn("x^2", " ".join(s.values()))

    def test_pdf_sections_stop_at_references(self):
        s = parse_pdf_text(PDF_TEXT)
        self.assertIn("1000 neurons", s["methods"])
        self.assertIn("37.5%", s["results"])
        self.assertNotIn("A. Author", s["discussion"])


class TestFullTextUse(unittest.TestCase):
    def setUp(self):
        self.p = PaperRecord(id="1", title="CAD in Kerala", source="PubMed", citation_index=1,
                             abstract="A survey of CAD prevalence in Kerala.")
        self.p.full_text_source = "Europe PMC open access (PMC1)"
        self.p.full_text_sections = parse_jats(JATS)

    def test_extraction_prompt_gets_full_text_excerpt(self):
        excerpt = _full_text_excerpt(self.p)
        self.assertIn("Full text (Europe PMC open access (PMC1))", excerpt)
        self.assertIn("12.5%", excerpt)

    def test_number_check_accepts_figures_from_full_text(self):
        item = TaskCStudyExtractionItem(citation_index=1, quantitative_result="Prevalence 12.5% in men",
                                        key_findings="CAD prevalence was 12.5%")
        self.assertEqual(verify_study_extraction_against_source(item, self.p), [])
        item.quantitative_result = "Prevalence 44.4%"
        self.assertTrue(verify_study_extraction_against_source(item, self.p))

    def test_evidence_table_reads_and_labels_full_text(self):
        matrix = build_structured_evidence_matrix(query="CAD prevalence in Kerala", papers=[self.p])
        doc, unit = matrix.documents[0], matrix.evidence_units[0]
        self.assertEqual(doc.evidence_access_level, "full_text")
        self.assertTrue(doc.full_text_available)
        first = unit.results[0]
        self.assertIn("12.5%", first.factual_source_text)  # result sentence taken from the full text's Results
        self.assertTrue(first.provenance.source_section.startswith("Results ("))
        self.assertEqual(first.provenance.access_level, "full_text")
        self.assertEqual(unit.appraisal.appraisal_completeness, "full_text_complete")

    def test_full_text_figures_survive_table_validation(self):
        # The validation step used to check figures against the abstract only and strip full-text figures
        matrix = build_structured_evidence_matrix(query="CAD prevalence in Kerala", papers=[self.p])
        q = matrix.evidence_units[0].results[0].quantitative_detail
        self.assertEqual(q.reporting_status, "Reported")
        self.assertIn("12.5", q.normalized_value)

    def test_sample_size_and_limitations_read_from_full_text(self):
        unit = build_structured_evidence_matrix(query="CAD prevalence in Kerala", papers=[self.p]).evidence_units[0]
        self.assertIn("5167 adults", unit.sample_or_scale)
        self.assertTrue(any("misclassify" in lim for lim in unit.appraisal.source_reported_limitations))

    def test_rule_based_findings_read_full_text_figures(self):
        finding = extract_atomic_findings("CAD prevalence in Kerala", [self.p])[0]
        self.assertEqual(finding.evidence_access, "full_text")
        self.assertTrue(any("12.5" in q for q in finding.quantitative_estimates))

    def test_full_text_is_not_sent_in_api_responses(self):
        self.assertNotIn("full_text_sections", self.p.model_dump())
        self.assertEqual(self.p.model_dump()["full_text_source"], "Europe PMC open access (PMC1)")


if __name__ == "__main__":
    unittest.main()
