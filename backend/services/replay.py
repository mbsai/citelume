"""
Replay a stored corpus: re-run study selection from the records and inputs it holds and check that the same
studies come out.

    python -m services.replay path/to/corpus.json.gz

Exit status 0 when the included studies and the content hash match what the corpus recorded, 1 otherwise. No model
and no network are used.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

from models import Corpus
from services.corpus import corpus_hash, replay_selection
from services.corpus_store import read_file
from services.paper_selection import PROTOCOL_VERSION


def check(corpus: Corpus) -> Tuple[bool, List[str]]:
    """Returns whether the replay reproduced the corpus, and the lines describing the comparison."""
    lines = [
        f"Corpus {corpus.corpus_id} created {corpus.created_at}",
        f"Question: {corpus.query}",
        f"Records: {len(corpus.records)} after merging duplicates ({corpus.records_found} found); "
        f"method: {corpus.selection_method}; study limit: {corpus.study_limit}",
        f"Selection protocol: stored {corpus.protocol_version}, running {PROTOCOL_VERSION}",
    ]
    replayed = replay_selection(corpus)
    same_ids = replayed == list(corpus.included_ids)
    replayed_hash = corpus_hash(corpus.model_copy(update={"included_ids": replayed}))
    same_hash = replayed_hash == corpus.content_hash
    lines.append(f"Included studies: stored {len(corpus.included_ids)}, replayed {len(replayed)} - "
                 + ("identical, in the same order" if same_ids else "DIFFERENT"))
    if not same_ids:
        stored, again = set(corpus.included_ids), set(replayed)
        if stored - again:
            lines.append(f"  only in the stored selection: {sorted(stored - again)}")
        if again - stored:
            lines.append(f"  only in the replayed selection: {sorted(again - stored)}")
        if stored == again:
            lines.append("  same studies in a different order")
        if corpus.protocol_version != PROTOCOL_VERSION:
            lines.append("  the corpus was made with a different protocol version, so a difference is expected")
    lines.append(f"Content hash: stored {corpus.content_hash}, replayed {replayed_hash} - "
                 + ("match" if same_hash else "MISMATCH"))
    return same_ids and same_hash, lines


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip())
        return 2
    ok, lines = check(read_file(Path(argv[1])))
    print("\n".join(lines))
    print("REPRODUCED" if ok else "NOT REPRODUCED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
