"""
MeSH headings for the search concepts (Session 5c).

CiteLume's PubMed searches tag every word [Title/Abstract], which switches off PubMed's Automatic Term Mapping:
"antidepressants" then misses trials of doxepin or trazodone, which PubMed indexes under the pharmacological action
"Antidepressive Agents". For each concept (and each of its other names) this module asks PubMed once how it maps
the words (the `translationset` of an untagged esearch) and keeps only the controlled-vocabulary parts:
`"X"[MeSH Terms]` (which PubMed explodes to narrower headings) and `"X"[Pharmacological Action]`. These are written
into the stored query as one more alternative for the concept, so the query sent to PubMed, stored in the corpus
and reported in the Methods is explicit and the same on every run. If the lookup fails, the query is unchanged.

The parts of a concept that PubMed maps to MeSH are ANDed (each as "its headings OR its words"); parts it does not
map ("early", "analogues"), and single leading words mapped on their own, are left out of this alternative,
because the heading of the rest already names the subject.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import httpx

from models import SearchConcept, SearchFacet
from services.source_health import report_source_failure, retrying_transport

MESH_LOOKUP_VERSION = "citelume-mesh/1.0"
ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_TAG = re.compile(r'"([^"]+)"\[(MeSH Terms|Pharmacological Action)\]')
# Headings too general to say what a study is about
_TOO_GENERAL = {"humans", "animals", "male", "female", "adult", "aged", "child", "infant", "adolescent",
                "middle aged", "young adult", "methods", "therapeutics", "research", "disease", "patients",
                "persons", "population", "time", "time factors"}
MAX_TAGS_PER_WORD = 4
_CACHE: Dict[str, List[str]] = {}


def groups_from_translation(translationset: Sequence[dict], term: str = "") -> List[str]:
    """One OR-group per mapped part of the term: its MeSH headings and pharmacological actions, and the part's own
    words in title or abstract. Only parts of two or more words, and the part the term ends with (the head noun:
    "chronic plaque psoriasis" -> psoriasis), are used: a single leading word mapped on its own often takes another
    sense ("adjuvant" -> "Adjuvants, Pharmaceutic", "plaque" -> "Dental Plaque")."""
    groups: List[str] = []
    term_low = re.sub(r"\s+", " ", (term or "").strip().lower())
    for entry in translationset or []:
        part = (entry.get("from") or "").strip().lower()
        if term_low and len(part.split()) < 2 and not term_low.endswith(part):
            continue
        tags = list(dict.fromkeys((h.lower(), f) for h, f in _TAG.findall(entry.get("to", ""))
                                  if h.lower() not in _TOO_GENERAL))[:MAX_TAGS_PER_WORD]
        if not tags:
            continue
        words = (entry.get("from") or "").strip().lower()
        alts = [f'"{h}"[{f}]' for h, f in tags]
        if words:
            alts.append(f'"{words}"[Title/Abstract]')
        groups.append("(" + " OR ".join(dict.fromkeys(alts)) + ")")
    return groups


async def lookup(term: str, client: httpx.AsyncClient, api_key: str = "") -> List[str]:
    """The MeSH OR-groups for a term ([] if PubMed maps none of it)."""
    key = term.strip().lower()
    if key in _CACHE:
        return _CACHE[key]
    params = {"db": "pubmed", "term": key, "retmax": "0", "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    resp = await client.get(ESEARCH, params=params)
    resp.raise_for_status()
    groups = groups_from_translation((resp.json().get("esearchresult") or {}).get("translationset") or [], key)
    _CACHE[key] = groups
    return groups


def mesh_alternative(groups: Sequence[str]) -> str:
    return groups[0] if len(groups) == 1 else "(" + " AND ".join(groups) + ")"


async def add_mesh_headings(concepts: List[SearchConcept], facets: List[SearchFacet], api_key: str = "",
                            block=None) -> int:
    """Looks up each concept's MeSH headings, stores them on the concept (`mesh`), and rewrites the concept's
    block in every facet's PubMed query to include them. Returns the number of concepts given headings. Never
    raises: a failed lookup leaves the queries as they were."""
    from services.query_planning import _pubmed_concept_block
    block = block or _pubmed_concept_block
    added = 0
    try:
        async with httpx.AsyncClient(transport=retrying_transport(max_retries=1), timeout=15.0) as client:
            for c in concepts:
                if c.mesh:
                    continue
                before = block(c)
                alts: List[str] = []
                for name in [c.name] + list(c.synonyms):
                    groups = await lookup(name, client, api_key)
                    if groups:
                        alts.append(mesh_alternative(groups))
                c.mesh = list(dict.fromkeys(alts))
                if not c.mesh:
                    continue
                after = block(c)
                for f in facets:
                    if f.queries.pubmed and before in f.queries.pubmed:
                        f.queries.pubmed = f.queries.pubmed.replace(before, after)
                added += 1
    except Exception as exc:
        report_source_failure("PubMed (MeSH lookup)", exc)
    return added
