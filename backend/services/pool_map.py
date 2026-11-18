"""
The candidate pool as a map: the records found are grouped by their words (TF-IDF of title and abstract, k-means
with a fixed seed, so the same corpus always gives the same clusters), and each cluster shows how many of its records
were included. A large cluster with nothing included is flagged: a part of the literature the searches found but
the review does not draw on, worth a look before the review is reported. Clicking a cluster lists its records.

Built from a stored corpus alone; no model and no network.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from models import Corpus, PaperRecord, PoolCluster, PoolMap, PoolMember

OTHER_METHODS = {"Citation searching", "Reference lists of reviews"}
METHOD = "TF-IDF of title and abstract (1-2 word terms), k-means, k = sqrt(records / 2) between 2 and 12, seed 42"
MIN_RECORDS = 10
FLAG_MIN_SIZE = 5
FLAG_SHARE = 0.05


def _k(n: int) -> int:
    return max(2, min(12, round(math.sqrt(n / 2))))


def build_pool_map(records: Sequence[PaperRecord], included: Sequence[PaperRecord],
                   excluded: Optional[Dict[str, str]] = None, chasing=None,
                   ranks: Optional[Dict[str, int]] = None) -> PoolMap:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    excluded, ranks = excluded or {}, ranks or {}

    def reason(r: PaperRecord) -> str:
        if r.id in excluded:
            return excluded[r.id]
        if r.id in ranks:
            return f"Eligible; ranked {ranks[r.id]}, below the study limit"
        return ""

    by_id = {p.id: p for p in included}
    pool = PoolMap(records=len(records), included=len(included), citation_chasing=chasing,
                   included_with_references=sum(1 for p in included if p.references))
    members = [PoolMember(id=r.id, title=r.title or "", year=r.year, included=r.id in by_id,
                          citation_index=by_id[r.id].citation_index if r.id in by_id else None,
                          excluded_reason="" if r.id in by_id else reason(r),
                          chased=bool({r.source, *(r.all_sources or [])} & OTHER_METHODS))
               for r in records]
    texts = [f"{r.title or ''}. {r.abstract or ''}" for r in records]
    labels = None
    if len(records) >= MIN_RECORDS:
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2), min_df=2)
            mat = vec.fit_transform(texts)
            k = min(_k(len(records)), mat.shape[0])
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(mat)
            terms = np.array(vec.get_feature_names_out())
            overall = np.asarray(mat.mean(axis=0)).ravel()
            pool.method = METHOD
        except ValueError:  # too few distinct words to cluster
            labels = None
    if labels is None:
        pool.method = "one group (too few records or words to cluster)"
        pool.clusters = [PoolCluster(cluster_id=1, label="All records", size=len(members),
                                     included=sum(m.included for m in members), members=members)]
        return pool
    clusters: List[PoolCluster] = []
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        # The terms most characteristic of the cluster: its mean weight above the pool's mean
        lift = np.asarray(mat[idx].mean(axis=0)).ravel() - overall
        top = [str(terms[i]) for i in lift.argsort()[::-1][:4] if lift[i] > 0]
        group = [members[i] for i in idx]
        group.sort(key=lambda m: (not m.included, m.citation_index or 0, -(m.year or 0), m.title.lower()))
        clusters.append(PoolCluster(cluster_id=0, label=", ".join(top[:3]) or "Other records", terms=top,
                                    size=len(group), included=sum(m.included for m in group), members=group))
    clusters.sort(key=lambda c: (-c.size, c.label))
    flag_size = max(FLAG_MIN_SIZE, math.ceil(FLAG_SHARE * len(records)))
    for i, c in enumerate(clusters, start=1):
        c.cluster_id = i
        c.flagged = c.included == 0 and c.size >= flag_size
    pool.clusters = clusters
    flagged = [c for c in clusters if c.flagged]
    if flagged:
        pool.note = (f"{len(flagged)} large {'group' if len(flagged) == 1 else 'groups'} of records "
                     f"({', '.join(c.label for c in flagged)}) had no included study. Check whether they are off "
                     "topic or a part of the question the selection missed.")
    return pool


def pool_map_for(corpus: Corpus, included: Sequence[PaperRecord]) -> PoolMap:
    return build_pool_map(corpus.records, included, corpus.excluded, corpus.citation_chasing,
                          ranks={c.record_id: c.rank for c in corpus.candidates})
