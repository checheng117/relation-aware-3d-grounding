# README Coadaptation Note

## What was updated

The README was updated in the current phase-history section by revising:

- `Recent validated findings`
- `Updated bottleneck after shortlist strengthening`
- `Current project stage`

The limitations / future-work wording was also tightened so the next step is explicitly a **minimal alternating shortlist/reranker co-adaptation experiment** rather than more isolated reranker work or broader scope expansion.

After the co-adaptation run completed, a short follow-up sentence was added to the `Current project stage` / `Limitations` discussion so README reflects the new evidence precisely:

- shortlist re-selection under the first retrained reranker reached **0.1154**
- the second reranker pass returned the final combined row to **0.1090**
- the framing therefore still points to shortlist–reranker co-adaptation, but now more specifically to preserving shortlist-side gains during reranker retraining

## Authoritative result files used

The README update was grounded in these validated artifact tables:

- `outputs/20260331_160556_fix_combined_nloss/shortlist_rerank_combined_table_fixed.csv`
- `outputs/20260331_170659_official_shortlist_strengthening/shortlist_retrieval_table_official.csv`
- `outputs/20260331_170659_official_shortlist_strengthening/shortlist_rerank_combined_table_official.csv`
- `outputs/20260331_173836_rerank_rebalance/rerank_rebalance_table.csv`
- `outputs/20260331_173836_rerank_rebalance/shortlist_rerank_combined_table_rebalance.csv`
- `outputs/20260331_180150_minimal_coadaptation/coadapted_shortlist_table.csv`
- `outputs/20260331_180150_minimal_coadaptation/coadapted_reranker_table.csv`
- `outputs/20260331_180150_minimal_coadaptation/shortlist_rerank_combined_table_coadaptation.csv`

## Why the framing now points to co-adaptation

These artifacts support three sequential claims:

1. corrected reranker gains are real after the fix phase
2. shortlist strengthening is materially real under the corrected objective
3. reranker-only rebalance recovers mismatch but does not move the system past the current `0.1090` combined level

That means the narrow next question is no longer "does shortlist matter?" or "does reranker rebalance matter?". The current evidence now goes one step further:

1. shortlist re-selection under the retrained reranker can improve the combined score beyond `0.1090`
2. a second reranker pass on that reselected shortlist did not preserve the gain

So the README now points to **co-adaptation quality and gain preservation** as the real bottleneck, which is the framing most consistent with the current artifact tables.
