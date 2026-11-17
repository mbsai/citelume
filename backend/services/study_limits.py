"""
How many studies a review analyses, and how they are chosen, by review type.

* Quick answer: up to 20 studies, chosen by the balanced ranking.
* Narrative review: up to 60 by default (a reasoned sample, chosen by the balanced ranking); the user may set
  another number up to the ceiling.
* Systematic-style and scoping reviews: every study that meets the criteria, up to a ceiling. The ceiling is a cost
  limit, not a method choice: every included study is read by the model. Eligible studies beyond it are listed.

The ceilings are lower on the public demo, where one shared key serves every visitor. Set STUDY_LIMIT_CEILING and
SCOPING_LIMIT_CEILING to change them for a local install.
"""
from __future__ import annotations

import os
from typing import Optional

from services import demo_mode

QUICK_LIMIT = 20
NARRATIVE_LIMIT = 60
DEMO_CEILING = 60
DEFAULT_CEILING = 150
DEFAULT_SCOPING_CEILING = 200
MIN_LIMIT = 5


def _env_int(name: str, default: int) -> int:
    try:
        return max(MIN_LIMIT, int(os.environ.get(name, default)))
    except ValueError:
        return default


def ceiling(mode: str, review_type: str) -> int:
    """The most studies one review may analyse here."""
    if mode != "deep_research":
        return QUICK_LIMIT
    if demo_mode.demo_enabled():
        return DEMO_CEILING
    if review_type == "scoping":
        return _env_int("SCOPING_LIMIT_CEILING", DEFAULT_SCOPING_CEILING)
    return _env_int("STUDY_LIMIT_CEILING", DEFAULT_CEILING)


def selection_method(mode: str, review_type: str) -> str:
    """'narrative' (balanced ranking up to the limit) or an eligibility-based method that takes every eligible
    study: 'systematic_prisma' or 'scoping'."""
    if mode == "deep_research" and review_type in ("systematic_prisma", "scoping"):
        return review_type
    return "narrative"


def resolve_study_limit(mode: str, review_type: str, requested: Optional[int] = None) -> int:
    """The study limit for a review: the user's number if given (within 5 and the ceiling), otherwise the default
    for the review type (all eligible studies, up to the ceiling, for systematic-style and scoping reviews)."""
    top = ceiling(mode, review_type)
    if requested:
        return max(MIN_LIMIT, min(int(requested), top))
    if mode != "deep_research":
        return QUICK_LIMIT
    if review_type in ("systematic_prisma", "scoping"):
        return top
    return min(NARRATIVE_LIMIT, top)


def estimate_minutes(n_studies: int) -> float:
    """A rough time to analyse n studies with a free hosted model (extraction in parallel batches, then the
    writing). Measured runs: 20 studies about 2-3 minutes, 60 about 4-5."""
    return round(1.5 + n_studies / 20, 1)
