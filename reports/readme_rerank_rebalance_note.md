# README Rerank Rebalance Note

## What was updated

The README was updated in the current phase-history section by adding or revising:

- `Official shortlist strengthening`
- `Recent validated findings`
- `Updated bottleneck after shortlist strengthening`
- `Current project stage`

The limitations / future-work wording was also tightened so the next step is explicitly **reranker re-balance on the improved shortlist**, not broader scope expansion.

## Authoritative result files used

The README update was grounded in the current validated artifact tables:

- `outputs/20260331_160556_fix_combined_nloss/shortlist_rerank_combined_table_fixed.csv`
- `outputs/20260331_170659_official_shortlist_strengthening/shortlist_retrieval_table_official.csv`
- `outputs/20260331_170659_official_shortlist_strengthening/shortlist_rerank_combined_table_official.csv`
- `reports/official_shortlist_strengthening_summary.md`

## Why this matches current evidence

These files jointly support three claims:

1. corrected reranker gains are real after the March 31 fix phase
2. shortlist strengthening is also now real under corrected end-to-end evaluation
3. the next bottleneck is reranker match to the healthier shortlist distribution, because the improved shortlist helps final accuracy but reused `O_best` is no longer the best-matched reranker on that shortlist

So the README now reflects the current evidence-based stage of the project rather than the earlier "shortlist quality is still the main bottleneck" framing.
