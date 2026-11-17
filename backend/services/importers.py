"""
Records and full texts the user brings: exports from databases CiteLume cannot search (Scopus, Web of Science,
Embase and others) and their own PDFs.

* `manual_search_strings` writes the protocol's core search in Scopus and Web of Science syntax for the user to run.
* `parse_export` reads an export (RIS, BibTeX, CSV/tab-delimited, Web of Science or PubMed tagged text) into
  records. Imported records then go through the same duplicate merging, eligibility and scoring as searched ones.
* `parse_pdf` reads a PDF's text layer (no OCR) into a record with its full text, for matching to a study in the
  pool or adding as a study identified by other methods.

Exports and PDFs are the user's licensed material: they are accepted only in local installs (never on the public
demo) and are kept in memory for a limited time until a corpus is built from them.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import time
import uuid
from typing import Dict, List, Optional, Tuple

from models import PaperRecord
from services.corpus import _normalize_title
from services.fulltext import _clean, _pdf_to_text, parse_pdf_text, repair_ligatures

MAX_EXPORT_BYTES = 30 * 1024 * 1024
MAX_PDF_BYTES = 40 * 1024 * 1024
MIN_PDF_TEXT_CHARS = 500
HOLD_SECONDS = 6 * 3600


# ---------------------------------------------------------------------------------------------------------------
# Search strings for databases CiteLume cannot query
# ---------------------------------------------------------------------------------------------------------------
def _quoted(term: str) -> str:
    t = re.sub(r'["()]', " ", term).strip()
    t = re.sub(r"\s+", " ", t)
    return f'"{t}"' if " " in t or "-" in t else t


def _block(terms: List[str]) -> str:
    seen = list(dict.fromkeys(t for t in (_quoted(x) for x in terms) if t))
    return seen[0] if len(seen) == 1 else "(" + " OR ".join(seen) + ")"


def manual_search_strings(concepts: List[Tuple[str, List[str]]], constraints: List[List[str]],
                          year_from: Optional[int] = None, year_to: Optional[int] = None) -> List[Dict[str, str]]:
    """The core search (every concept, any of its names, and every population or place) in Scopus and Web of
    Science syntax. `concepts` are (name, synonyms); `constraints` are lists of spellings."""
    blocks = [_block([n] + list(syns)) for n, syns in concepts if n.strip()]
    blocks += [_block(sp) for sp in constraints if sp]
    inner = " AND ".join(blocks)
    if not inner:
        return []
    scopus = f"TITLE-ABS-KEY({inner})"
    if year_from:
        scopus += f" AND PUBYEAR > {year_from - 1}"
    if year_to:
        scopus += f" AND PUBYEAR < {year_to + 1}"
    wos = f"TS=({inner})"
    if year_from or year_to:
        wos += f" AND PY=({year_from or 1900}-{year_to or 2100})"
    return [
        {"database": "Scopus", "query": scopus,
         "how_to": "Scopus > Advanced document search: paste, run, select all results, Export > RIS or CSV "
                   "(include citation information and abstract)."},
        {"database": "Web of Science", "query": wos,
         "how_to": "Web of Science Core Collection > Advanced search: paste, run, Export > RIS, Plain text or "
                   "Tab delimited (full record). Up to 1,000 records per export."},
    ]


# ---------------------------------------------------------------------------------------------------------------
# Export parsers
# ---------------------------------------------------------------------------------------------------------------
def _record_id(title: str, year: Optional[int], doi: str) -> str:
    key = (doi or "").lower() or f"{_normalize_title(title)}|{year or ''}"
    return "imp-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:14]


def _year(text: str) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", text or "")
    return int(m.group(0)) if m else None


def _doi(text: str) -> str:
    m = re.search(r"10\.\d{4,9}/[^\s\"<>]+", text or "")
    return m.group(0).rstrip(".,;)").lower() if m else ""


def _record(source: str, title: str, abstract: str = "", authors: Optional[List[str]] = None, year=None,
            venue: str = "", doi: str = "", pmid: str = "", url: str = "", types: Optional[List[str]] = None,
            label: str = "") -> Optional[PaperRecord]:
    title = re.sub(r"\s+", " ", (title or "")).strip().rstrip(".")
    if not title:
        return None
    year = year if isinstance(year, int) else _year(str(year or ""))
    doi = _doi(doi) or ""
    return PaperRecord(
        id=_record_id(title, year, doi), title=title, abstract=re.sub(r"\s+", " ", abstract or "").strip(),
        authors=[a.strip() for a in (authors or []) if a.strip()][:20], year=year, venue=venue.strip(),
        doi=doi or None, pmid=(pmid or "").strip() or None, url=url or (f"https://doi.org/{doi}" if doi else ""),
        source=source, all_sources=[source], publication_types=list(types or []),
        retrieved_by=[label or f"{source} export (imported by the user)"],
    )


def _parse_ris(text: str, source: str, label: str) -> List[PaperRecord]:
    out, cur = [], {}
    for line in text.splitlines():
        m = re.match(r"^([A-Z][A-Z0-9])  - ?(.*)$", line)
        if not m:
            continue
        tag, val = m.group(1), m.group(2).strip()
        if tag == "ER":
            r = _record(source, (cur.get("TI") or cur.get("T1") or [""])[0], " ".join(cur.get("AB", cur.get("N2", []))),
                        cur.get("AU", []) + cur.get("A1", []),
                        (cur.get("PY") or cur.get("Y1") or cur.get("DA") or [""])[0],
                        (cur.get("T2") or cur.get("JO") or cur.get("JF") or [""])[0], (cur.get("DO") or [""])[0],
                        "", (cur.get("UR") or [""])[0], cur.get("M3", []), label)
            if r:
                out.append(r)
            cur = {}
        else:
            cur.setdefault(tag, []).append(val)
    return out


def _parse_tagged(text: str, source: str, label: str) -> List[PaperRecord]:
    """Web of Science plain text (two-letter tags, continuation lines indented) and PubMed MEDLINE/NBIB
    ("TAG - value", records separated by blank lines)."""
    out: List[PaperRecord] = []
    cur: Dict[str, List[str]] = {}
    last = None
    medline = bool(re.search(r"^PMID- ", text, re.M))

    def flush():
        if not cur:
            return
        if medline:
            doi = next((v for v in cur.get("LID", []) + cur.get("AID", []) if "[doi]" in v), "")
            r = _record(source, " ".join(cur.get("TI", [])), " ".join(cur.get("AB", [])), cur.get("FAU", cur.get("AU", [])),
                        (cur.get("DP") or [""])[0], (cur.get("JT") or cur.get("TA") or [""])[0], doi,
                        (cur.get("PMID") or [""])[0], "", cur.get("PT", []), label)
        else:
            r = _record(source, " ".join(cur.get("TI", [])), " ".join(cur.get("AB", [])), cur.get("AF", cur.get("AU", [])),
                        (cur.get("PY") or [""])[0], " ".join(cur.get("SO", [])), (cur.get("DI") or [""])[0],
                        (cur.get("PM") or [""])[0], "", cur.get("DT", []), label)
        if r:
            out.append(r)

    for line in text.splitlines():
        if medline:
            m = re.match(r"^([A-Z]{2,4})\s*- (.*)$", line)
            if m:
                last = m.group(1)
                if last == "PMID" and cur:
                    flush()
                    cur = {}
                cur.setdefault(last, []).append(m.group(2).strip())
            elif line.startswith("      ") and last:
                cur[last][-1] += " " + line.strip()
            continue
        if line.startswith("ER"):
            flush()
            cur, last = {}, None
            continue
        m = re.match(r"^([A-Z][A-Z0-9]) (.*)$", line)
        if m:
            last = m.group(1)
            cur.setdefault(last, []).append(m.group(2).strip())
        elif line.startswith("   ") and last:
            # Continuation line: another author, or more of the same field
            if last in ("AU", "AF"):
                cur[last].append(line.strip())
            else:
                cur[last][-1] += " " + line.strip()
    flush()
    return out


def _parse_bibtex(text: str, source: str, label: str) -> List[PaperRecord]:
    out = []
    for entry in re.split(r"\n\s*@", "\n" + text)[1:]:
        fields = {}
        for m in re.finditer(r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|\d+)", entry):
            fields[m.group(1).lower()] = re.sub(r"[{}]", "", m.group(2)).strip('"').strip()
        r = _record(source, fields.get("title", ""), fields.get("abstract", ""),
                    re.split(r"\s+and\s+", fields.get("author", "")), fields.get("year", ""),
                    fields.get("journal", fields.get("booktitle", "")), fields.get("doi", ""),
                    fields.get("pmid", ""), fields.get("url", ""), [], label)
        if r:
            out.append(r)
    return out


_CSV_COLUMNS = {
    "title": ("title", "ti", "article title", "document title"),
    "abstract": ("abstract", "ab"),
    "authors": ("authors", "author", "au", "author full names", "af"),
    "year": ("year", "py", "publication year", "pubyear"),
    "venue": ("source title", "journal", "so", "source"),
    "doi": ("doi", "di"),
    "pmid": ("pubmed id", "pmid", "pm"),
    "url": ("link", "url"),
    "types": ("document type", "dt", "publication type"),
}


def _parse_csv(text: str, source: str, label: str) -> List[PaperRecord]:
    dialect = "excel-tab" if text.split("\n", 1)[0].count("\t") > text.split("\n", 1)[0].count(",") else "excel"
    rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    if not rows:
        return []
    cols = {k.strip().lower().lstrip("﻿"): k for k in rows[0].keys() if k}

    def col(row, name):
        for alias in _CSV_COLUMNS[name]:
            if alias in cols:
                return row.get(cols[alias]) or ""
        return ""

    out = []
    for row in rows:
        authors = re.split(r";\s*|,\s(?=[A-Z][a-z]+,)", col(row, "authors")) if col(row, "authors") else []
        r = _record(source, col(row, "title"), col(row, "abstract"), authors, col(row, "year"), col(row, "venue"),
                    col(row, "doi"), col(row, "pmid"), col(row, "url"), [col(row, "types")] if col(row, "types") else [],
                    label)
        if r:
            out.append(r)
    return out


def detect_format(text: str, filename: str = "") -> str:
    name = filename.lower()
    if re.search(r"^TY  - ", text, re.M):
        return "ris"
    if re.search(r"^PMID- ", text, re.M):
        return "medline"
    if re.search(r"^PT [A-Z]\s*$", text, re.M) or re.search(r"^UT WOS:", text, re.M):
        return "wos"
    if re.search(r"^\s*@\w+\s*\{", text, re.M) or name.endswith(".bib"):
        return "bibtex"
    if name.endswith((".csv", ".tsv", ".txt")) or text.split("\n", 1)[0].count(",") >= 3 or "\t" in text.split("\n", 1)[0]:
        return "csv"
    return "unknown"


def guess_source(text: str, fmt: str) -> str:
    if "scopus.com" in text or "Scopus" in text[:4000]:
        return "Scopus"
    if fmt == "wos" or "WOS:" in text[:20000]:
        return "Web of Science"
    if fmt == "medline":
        return "PubMed"
    if "embase" in text[:20000].lower():
        return "Embase"
    return "Imported"


def parse_export(data: bytes, filename: str = "", source: str = "") -> Tuple[List[PaperRecord], str, str]:
    """Records, format and source label from an exported file."""
    if len(data) > MAX_EXPORT_BYTES:
        raise ValueError("The file is larger than 30 MB; export fewer records at a time.")
    text = data.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
    fmt = detect_format(text, filename)
    source = source.strip() or guess_source(text, fmt)
    label = f"{source} export (imported by the user)"
    parser = {"ris": _parse_ris, "medline": _parse_tagged, "wos": _parse_tagged, "bibtex": _parse_bibtex,
              "csv": _parse_csv}.get(fmt)
    if parser is None:
        raise ValueError("This file's format was not recognised. Export as RIS, BibTeX, CSV, Web of Science plain "
                         "text or PubMed (MEDLINE) format.")
    records = parser(text, source, label)
    if not records:
        raise ValueError("No records with a title were found in this file.")
    # The same record exported twice keeps one copy
    return list({r.id: r for r in records}.values()), fmt, source


# ---------------------------------------------------------------------------------------------------------------
# User PDFs
# ---------------------------------------------------------------------------------------------------------------
def parse_pdf(data: bytes, filename: str = "") -> PaperRecord:
    """A record from a PDF's text layer: DOI, a title guess, the abstract and the full-text sections."""
    if len(data) > MAX_PDF_BYTES:
        raise ValueError("The PDF is larger than 40 MB.")
    if not data.startswith(b"%PDF"):
        raise ValueError("This file is not a PDF.")
    text = repair_ligatures(_pdf_to_text(data) or "")
    if len(text.strip()) < MIN_PDF_TEXT_CHARS:
        raise ValueError("This PDF has no readable text layer (it may be a scanned image). CiteLume cannot read "
                         "scanned PDFs; use a PDF with selectable text.")
    head = text[:4000]
    doi = _doi(head)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in head.splitlines()]
    candidates = [ln for ln in lines[:25] if 25 <= len(ln) <= 250 and not re.search(r"doi|http|©|journal|volume|issn|received", ln, re.I)]
    title = max(candidates[:6], key=len) if candidates else (filename.rsplit(".", 1)[0] or "Uploaded PDF")
    sections = parse_pdf_text(text)
    abstract = sections.get("abstract") or ""
    if not abstract:
        m = re.search(r"abstract[:.\s]+(.{200,2500}?)(?:\n\s*\n|keywords|introduction|background)", text, re.I | re.S)
        abstract = _clean(m.group(1)) if m else _clean(text[:1500])
    body = {k: v for k, v in sections.items() if k != "abstract"} or {"body": _clean(text[:60000])}
    record = _record("User PDF", title, abstract, [], _year(head), "", doi, "", "", [],
                     "Full text supplied by the user")
    record.full_text_source = f"PDF supplied by the user ({filename or 'upload'})"
    record.full_text_sections = body
    return record


def match_pdf(pdf: PaperRecord, pool: List[PaperRecord], pdf_text_head: str = "") -> Optional[PaperRecord]:
    """The record in the pool this PDF is the full text of: same DOI, or a pool title that appears near the start
    of the PDF."""
    if pdf.doi:
        hit = next((p for p in pool if p.doi and p.doi.lower() == pdf.doi.lower()), None)
        if hit:
            return hit
    head = _normalize_title(pdf_text_head or (pdf.title + " " + pdf.abstract[:500]))
    best = None
    for p in pool:
        t = _normalize_title(p.title)
        if len(t) >= 25 and t in head and (best is None or len(t) > len(_normalize_title(best.title))):
            best = p
    return best


# ---------------------------------------------------------------------------------------------------------------
# Holding uploads until a corpus is built
# ---------------------------------------------------------------------------------------------------------------
_HELD: Dict[str, Tuple[float, dict]] = {}


def hold(kind: str, payload: dict) -> str:
    now = time.time()
    for k in [k for k, (t, _) in _HELD.items() if now - t > HOLD_SECONDS]:
        _HELD.pop(k, None)
    key = uuid.uuid4().hex
    _HELD[key] = (now, {"kind": kind, **payload})
    return key


def held(key: str, kind: str) -> Optional[dict]:
    hit = _HELD.get(key or "")
    if not hit or hit[1].get("kind") != kind or time.time() - hit[0] > HOLD_SECONDS:
        return None
    return hit[1]
