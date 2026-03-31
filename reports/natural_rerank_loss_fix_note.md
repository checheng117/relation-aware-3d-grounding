# Natural Rerank Loss Fix Note

## Root cause

Protocol `N` trains with:

- `shortlist_train_inject_gold: false`

So gold is often absent from the natural top-K shortlist.

Before the fix, the rerank training path still computed full-scene CE on those rows:

- `scripts/train_two_stage_rerank.py`
- `src/rag3d/relation_reasoner/two_stage_rerank.py`
- `src/rag3d/relation_reasoner/losses.py`

When gold was not in the shortlist, the gold logit stayed at `-inf`, so CE was supervising an impossible target. That produced the observed `train_loss_mean = Infinity` in the March 27 natural metrics.

## Fix

The change is semantic, not a clamp:

- `src/rag3d/relation_reasoner/two_stage_rerank.py`
  - `forward_two_stage_rerank(...)` can now return shortlist aux data

- `scripts/train_two_stage_rerank.py`
  - computes a per-sample `gold_in_shortlist` mask during training
  - logs:
    - `rerank_train_valid_fraction`
    - `gold_in_shortlist_rate_train`
    - `gold_in_shortlist_rate_val`
    - `nan_or_inf_batch_count`

- `src/rag3d/relation_reasoner/losses.py`
  - CE and hinge losses now accept `valid_rows`
  - only rows with a realizable rerank target contribute to loss
  - all-invalid batches now reduce to a finite zero-loss no-op instead of generating NaN/Inf

## Fixed rerun

The fixed rerun lives in:

- `outputs/20260331_160556_fix_combined_nloss/`

Relevant artifacts:

- `outputs/20260331_160556_fix_combined_nloss/metrics_rerank_natural.jsonl`
- `outputs/20260331_160556_fix_combined_nloss/oracle_reranker_table_nfix.csv`
- `outputs/20260331_160556_fix_combined_nloss/oracle_reranker_interpretation_nfix.md`

## Results

From the fixed March 31, 2026 rerun:

- `N` best natural val `Acc@1 = 0.1474`
- `N` last natural val `Acc@1 = 0.1218`
- `N` best oracle-shortlist val `Acc@1 = 0.3718`
- `rerank_train_valid_fraction` stayed in `[0.2781, 0.3147]`
- `nan_or_inf_batch_count_max = 0`

Compared with reused protocol `O`:

- `O` best natural val `Acc@1 = 0.1538`
- `O` best oracle-shortlist val `Acc@1 = 0.6731`

## Interpretation

- The old `Infinity` loss was caused by supervising impossible natural-shortlist rows.
- After the fix, `N` trains stably.
- The previous `N` collapse was mostly a bug.
- But `O` still has a slightly better best natural score and a much stronger oracle-shortlist ceiling, so oracle-shortlist training remains the stronger reranker regime in this phase.
