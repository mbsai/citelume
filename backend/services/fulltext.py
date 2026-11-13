"""
Open-access full-text retrieval for the studies included in a review.

Sources, tried in order for each study:
  1. Europe PMC full-text XML (JATS) for open-access articles in PubMed Central;
  2. bioRxiv / medRxiv JATS XML via the bioRxiv API for preprints (DOI 10.1101/...);
  3. arXiv HTML (recent papers), then the arXiv PDF;
  4. any open-access PDF link already found by the search (e.g. Semantic Scholar's openAccessPdf).

Only openly available text is fetched; paywalled articles stay at abstract level. Text is split into canonical
sections (methods, results, discussion, limitations, conclusion) and cached on disk, so repeat searches do not
download it again.
"""
import asyncio
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import httpx

from models import PaperRecord, normalize_decimal_points

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "fulltext"
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
BIORXIV_DETAILS = "https://api.biorxiv.org/details/{server}/{doi}"
HEADERS = {"User-Agent": "CiteLume/1.0 (open-access literature synthesis)"}

# Concurrent downloads per search, per-request timeout, and the overall time allowed for the whole batch
MAX_CONCURRENT_FETCHES = 10
REQUEST_TIMEOUT_S = 12.0
BATCH_TIMEOUT_S = 25.0
# Characters kept per canonical section (enough for extraction; keeps cache files and memory small)
MAX_SECTION_CHARS = 20000
MAX_PDF_PAGES = 30

SECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("limitations", re.compile(r"\blimitation|strengths? and weaknesses|strengths? and limitations", re.I)),
    ("methods", re.compile(r"\bmethod|material|study design|participants|patients and|population|data source|"
                           r"experimental (setup|design|procedure)|procedure|setting|statistical analys|approach|"
                           r"proposed|architecture|algorithm|implementation|setup|dataset|data and|model\b", re.I)),
    ("results", re.compile(r"\bresult|finding|outcome|experiment|evaluation|benchmark|performance|"
                           r"main theorem|simulation", re.I)),
    ("discussion", re.compile(r"\bdiscussion", re.I)),
    ("conclusion", re.compile(r"\bconclu|summary", re.I)),
    ("introduction", re.compile(r"\bintroduction|background", re.I)),
]


def canonical_section(title: str) -> Optional[str]:
    for name, pattern in SECTION_PATTERNS:
        if pattern.search(title or ""):
            return name
    return None


# PDF text extraction often loses ligatures (ffi, fi, ff, fl, ffl): either the ligature character itself comes
# through, or a NUL character stands in its place ("e\x00cacy"). The ligature characters are expanded; for a NUL the
# candidate that turns the word into a known word start is chosen ("efficacy", "significant", "different"),
# and "fi" (the most common ligature) otherwise.
_LIGATURE_CHARS = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}
_LIGATURE_WORDS = (
    # ffi / fi
    "efficac", "efficien", "sufficien", "deficien", "difficul", "office", "officia", "coefficien", "affili",
    "traffic", "significan", "specific", "specifi", "first", "field", "fifth", "find", "final", "define",
    "definit", "confirm", "identifi", "classifi", "quantifi", "modifi", "benefit", "profile", "magnifi", "fibr",
    "figur", "filter", "fixed", "fitted", "fitness", "fiber",
    # ff
    "effect", "differ", "offer", "affect", "staff", "suffer", "afford", "effort", "diffus", "buffer", "offspring",
    # fl / ffl
    "flow", "influen", "reflect", "conflict", "fluid", "fluor", "inflam", "fluctu", "flexib", "flag", "flat",
    "floor", "afflict", "baffl", "shuffl",
)
_NUL_WORD = re.compile(r"([A-Za-z]*)\x00([A-Za-z]*)")


def repair_ligatures(text: str) -> str:
    if not text:
        return text or ""
    for k, v in _LIGATURE_CHARS.items():
        text = text.replace(k, v)
    if "\x00" not in text:
        return text

    def fix(m: "re.Match") -> str:
        before, after = m.group(1), m.group(2)
        for lig in ("ffi", "ffl", "fi", "ff", "fl"):
            word = (before + lig + after).lower()
            start, end = len(before), len(before) + len(lig)
            for known in _LIGATURE_WORDS:
                at = word.find(known)
                # the known word part must contain the whole ligature, so "first" cannot justify "ffirst"
                if at != -1 and at <= start and at + len(known) >= end:
                    return before + lig + after
        return before + "fi" + after

    return _NUL_WORD.sub(fix, text).replace("\x00", "")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", repair_ligatures(text or "")).strip()


def _add(sections: Dict[str, str], name: Optional[str], text: str) -> None:
    if not name or not text:
        return
    merged = f"{sections.get(name, '')} {text}".strip()
    sections[name] = normalize_decimal_points(merged[:MAX_SECTION_CHARS])


# --------------------------------------------------------------------------------------------- parsers
def parse_jats(xml_text: str) -> Dict[str, str]:
    """Sections from JATS XML (Europe PMC, bioRxiv): top-level <sec> titles map to canonical sections."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}
    body = next((el for el in root.iter() if el.tag.split("}")[-1] == "body"), None)
    if body is None:
        return {}
    sections: Dict[str, str] = {}

    def walk(sec, inherited: Optional[str]) -> None:
        title_el = next((c for c in sec if c.tag.split("}")[-1] == "title"), None)
        name = canonical_section("".join(title_el.itertext())) if title_el is not None else None
        # Unrecognised headings (common outside medicine) keep their text as general body content
        name = name or inherited or "body"
        for child in sec:
            tag = child.tag.split("}")[-1]
            if tag == "sec":
                walk(child, name)
            elif tag == "p":
                _add(sections, name, _clean("".join(child.itertext())))

    for sec in body:
        if sec.tag.split("}")[-1] == "sec":
            walk(sec, None)
    return sections


class _ArxivHTMLParser(HTMLParser):
    """Collects text under h2-h4 headings of arXiv's LaTeXML HTML, skipping maths, scripts and references."""

    def __init__(self) -> None:
        super().__init__()
        self.sections: Dict[str, str] = {}
        self.current: Optional[str] = None
        self.in_heading = False
        self.heading = ""
        self.skip_depth = 0
        self.buffer: List[str] = []

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class", "") or ""
        if tag in ("script", "style", "math") or "ltx_bibliography" in cls:
            self.skip_depth += 1
        elif tag in ("h2", "h3", "h4"):
            self._flush()
            self.in_heading, self.heading = True, ""

    def handle_endtag(self, tag):
        if tag in ("script", "style", "math") and self.skip_depth:
            self.skip_depth -= 1
        elif tag in ("h2", "h3", "h4") and self.in_heading:
            self.in_heading = False
            name = canonical_section(self.heading)
            # A sub-heading without its own match stays in its parent section; an unmatched top heading is body text
            if name:
                self.current = name
            elif tag == "h2":
                self.current = None if re.search(r"reference|bibliograph|acknowledg|appendix", self.heading, re.I) \
                    else "body"

    def handle_data(self, data):
        if self.skip_depth:
            return
        if self.in_heading:
            self.heading += data
        elif self.current:
            self.buffer.append(data)

    def _flush(self):
        _add(self.sections, self.current, _clean(" ".join(self.buffer)))
        self.buffer = []

    def close(self):
        super().close()
        self._flush()


def parse_arxiv_html(html_text: str) -> Dict[str, str]:
    parser = _ArxivHTMLParser()
    parser.feed(html_text)
    parser.close()
    return parser.sections


_PDF_HEADING = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?|[IVX]+\.)?\s*(abstract|introduction|background|methods?|materials and methods|"
    r"methodology|study design|results?|findings|discussion|limitations?|conclusions?)\s*$",
    re.I | re.M,
)


def parse_pdf_text(text: str) -> Dict[str, str]:
    """Sections from PDF text using heading lines such as "2. Methods" or "RESULTS"."""
    sections: Dict[str, str] = {}
    matches = list(_PDF_HEADING.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        refs = re.search(r"^\s*(references|bibliography|acknowledg)", text[m.end():end], re.I | re.M)
        chunk = text[m.end(): m.end() + refs.start()] if refs else text[m.end():end]
        _add(sections, canonical_section(m.group(1)), _clean(chunk))
    return sections


def _pdf_to_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        return repair_ligatures("\n".join((page.extract_text() or "") for page in reader.pages[:MAX_PDF_PAGES]))
    except Exception:
        return ""


# --------------------------------------------------------------------------------------------- fetchers
async def _europe_pmc(client: httpx.AsyncClient, p: PaperRecord) -> Optional[Tuple[str, Dict[str, str]]]:
    if p.pmid:
        q = f"EXT_ID:{p.pmid} AND SRC:MED"
    elif p.doi:
        q = f'DOI:"{p.doi}"'
    else:
        return None
    r = await client.get(EPMC_SEARCH, params={"query": q, "format": "json", "resultType": "core", "pageSize": "1"})
    if r.status_code != 200:
        return None
    hits = (r.json().get("resultList") or {}).get("result") or []
    if hits and not (p.abstract or "").strip() and hits[0].get("abstractText"):
        # Records that arrived without an abstract (common from Semantic Scholar) get Europe PMC's copy
        p.abstract = _clean(re.sub(r"<[^>]+>", " ", hits[0]["abstractText"]))
    if not hits or hits[0].get("isOpenAccess") != "Y" or not hits[0].get("pmcid"):
        return None
    x = await client.get(EPMC_FULLTEXT.format(pmcid=hits[0]["pmcid"]))
    if x.status_code != 200:
        return None
    sections = parse_jats(x.text)
    return (f"Europe PMC open access ({hits[0]['pmcid']})", sections) if sections else None


async def _biorxiv(client: httpx.AsyncClient, p: PaperRecord) -> Optional[Tuple[str, Dict[str, str]]]:
    if not (p.doi or "").startswith("10.1101/"):
        return None
    for server in ("biorxiv", "medrxiv"):
        r = await client.get(BIORXIV_DETAILS.format(server=server, doi=p.doi))
        if r.status_code != 200:
            continue
        versions = r.json().get("collection") or []
        url = (versions[-1].get("jatsxml") if versions else None)
        if not url:
            continue
        x = await client.get(url)
        if x.status_code == 200:
            sections = parse_jats(x.text)
            if sections:
                return f"{server} full text (JATS)", sections
    return None


async def _arxiv(client: httpx.AsyncClient, p: PaperRecord) -> Optional[Tuple[str, Dict[str, str]]]:
    if not p.arxiv_id:
        return None
    r = await client.get(f"https://arxiv.org/html/{p.arxiv_id}")
    if r.status_code == 200 and "ltx_" in r.text:
        sections = parse_arxiv_html(r.text)
        if sections:
            return "arXiv HTML", sections
    for url in (f"https://arxiv.org/pdf/{p.arxiv_id}", f"https://arxiv.org/pdf/{p.arxiv_id}v1"):
        r = await client.get(url)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/pdf"):
            sections = parse_pdf_text(_pdf_to_text(r.content))
            if sections:
                return "arXiv PDF", sections
    return None


async def _open_pdf(client: httpx.AsyncClient, p: PaperRecord) -> Optional[Tuple[str, Dict[str, str]]]:
    url = p.pdf_url or ""
    if not url.startswith("http") or "arxiv.org" in url or "biorxiv.org" in url:
        return None
    r = await client.get(url)
    if r.status_code != 200 or not r.headers.get("content-type", "").startswith("application/pdf"):
        return None
    sections = parse_pdf_text(_pdf_to_text(r.content))
    return ("Open-access PDF", sections) if sections else None


FETCHERS = (_europe_pmc, _biorxiv, _arxiv, _open_pdf)


# --------------------------------------------------------------------------------------------- cache + driver
def _cache_key(p: PaperRecord) -> Optional[str]:
    ident = p.pmid or p.doi or p.arxiv_id or p.s2_id
    return hashlib.sha1(ident.lower().encode()).hexdigest() if ident else None


def _cache_read(key: str) -> Optional[dict]:
    path = CACHE_DIR / f"{key}.json"
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except (OSError, ValueError):
        return None


def _cache_write(key: str, payload: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{key}.json").write_text(json.dumps(payload))
    except OSError:
        pass


async def _fetch_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, p: PaperRecord) -> None:
    key = _cache_key(p)
    if not key:
        return
    cached = _cache_read(key)
    if cached is not None:
        # A cached miss ({"source": ""}) is remembered too, so closed-access papers are not re-requested
        if cached.get("source"):
            p.full_text_source, p.full_text_sections = cached["source"], cached["sections"]
        if cached.get("abstract") and not (p.abstract or "").strip():
            p.abstract = cached["abstract"]
        return
    had_abstract = bool((p.abstract or "").strip())
    async with sem:
        for fetch in FETCHERS:
            try:
                found = await fetch(client, p)
            except (httpx.HTTPError, ValueError, KeyError):
                found = None
            if found and any(v for k, v in found[1].items() if k != "introduction"):
                p.full_text_source, p.full_text_sections = found
                _cache_write(key, {"source": found[0], "sections": found[1],
                                   "abstract": "" if had_abstract else (p.abstract or "")})
                return
    _cache_write(key, {"source": "", "sections": {}, "abstract": "" if had_abstract else (p.abstract or "")})


async def attach_open_access_full_texts(papers: Sequence[PaperRecord]) -> int:
    """Fetches open-access full text for the given studies in place; returns how many now have full text."""
    sem = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, headers=HEADERS, follow_redirects=True) as client:
        tasks = [asyncio.create_task(_fetch_one(client, sem, p)) for p in papers]
        done, pending = await asyncio.wait(tasks, timeout=BATCH_TIMEOUT_S)
        for t in pending:
            t.cancel()
    return sum(1 for p in papers if p.full_text_source)
