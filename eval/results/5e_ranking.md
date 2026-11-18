# Ranking for screening (development set only)

After Session 5d, 17 included studies were eligible but never screened: the model screens at most 300 records per
review in rank order, and eligible studies were ordered by tier first (every concept matched in words) and score
second, so a strong study matching only part of the question in words was pushed behind every full match.

Offline comparison on the stored 5d development corpora (no model, no cost): included studies among the first K
eligible records, in screening order. 77 eligible included studies in all.

| Order | Top 300 (the screening cap) | Top 60 | Top 20 |
|---|---|---|---|
| Tier, then score (selection protocol 2.4) | 60 | 38 | 20 |
| Score alone (2.5, chosen) | **74** | **50** | **24** |
| Score, partial matches -5 points | 74 | 45 | 20 |
| Score, partial matches -10 points | 74 | 40 | 20 |
| Score without citation impact and recency | 73 | 45 | 18 |
| Relevance and evidence only | 73 | 40 | 24 |

The 2.5 row is measured by re-running the selection code on the stored records (`select_studies`), the others by
re-sorting the stored scores. The score already rewards matching every concept (concept-match points), so a
separate tier is not needed. Narrative selection is unchanged.

Change: selection protocol citelume-selection/2.5 orders eligible studies in systematic-style and scoping reviews by
score alone, and records that order as each study's rank. The screening cap was not raised: with this order, 74 of
77 eligible included studies are already inside it.

Expected effect, not yet measured with the model: in the 5d run the model screened out 3 of the 60 included studies
it saw; at that rate, about 70 of 141 included studies (50%) would be selected, against 55 (39%).
