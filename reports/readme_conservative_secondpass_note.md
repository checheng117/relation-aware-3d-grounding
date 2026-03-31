# README Conservative Second-Pass Note

## Updated sections

The README was updated in these sections:

- `Current project stage`
- `Limitations`
- `Future work`

The update is intentionally small and only reframes the current stage from "co-adaptation existence check" to "co-adaptation gain-retention check".

## Authoritative result files used

The README framing is grounded in:

- `outputs/20260331_180150_minimal_coadaptation/coadapted_shortlist_table.csv`
- `outputs/20260331_180150_minimal_coadaptation/coadapted_reranker_table.csv`
- `outputs/20260331_180150_minimal_coadaptation/shortlist_rerank_combined_table_coadaptation.csv`

Key numbers reflected in README:

- `improved_shortlist_plus_reference_rerank = 0.1090`
- `reselected_shortlist_plus_first_retrained_rerank = 0.1154`
- `reselected_shortlist_plus_second_retrained_rerank = 0.1090`

## Why the framing now points to conservative second-pass tuning

The artifact tables already show:

1. shortlist strengthening is real
2. reranker rebalance is real
3. co-adaptation signal is real (intermediate `0.1154`)
4. naive second-pass reranker retraining did not preserve that gain

So the narrow next question is not whether co-adaptation exists. It is whether second-pass reranker adaptation can be made conservative enough to retain the shortlist-side gain.
