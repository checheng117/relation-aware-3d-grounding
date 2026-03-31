# Rerank Rebalance Summary

## What was implemented

This phase added one narrow reranker rebalance pass on top of the healthier shortlist:

- README evidence update for the current validated stage
- one orchestration entrypoint: `scripts/run_rerank_rebalance_phase.py`
- one natural improved-shortlist reranker retraining run
- corrected combined evaluation against baseline, improved-shortlist reference rerank, improved-shortlist `O_best`, and improved-shortlist retrained rerank

Authoritative output directory:

- `outputs/20260331_173836_rerank_rebalance/`

The official run was executed with:

- `PYTHONHASHSEED=42`

## Prior corrected / improved assets reused

### Improved shortlist asset

- `outputs/20260331_170659_official_shortlist_strengthening/checkpoints/coarse_official_shortlist_strengthening_best_pipeline_natural.pt`

### Prior reranker baselines

- reference rerank:
  - `outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt`
- corrected `O_best`:
  - `outputs/20260331_160556_fix_combined_nloss/checkpoints/rerank_longtrain_oracle_best_natural_two_stage.pt`
- corrected fixed `N_best`:
  - `outputs/20260331_160556_fix_combined_nloss/checkpoints/rerank_longtrain_natural_best_natural_two_stage.pt`

### Corrected evaluation path

- `src/rag3d/evaluation/two_stage_eval.py`
- `fine_only_from_checkpoint=True`

## Did reranker retraining on the improved shortlist work?

Partially.

From `rerank_rebalance_table.csv`:

- `reference_rerank_on_improved_shortlist`: natural Acc@1 `0.1090`, `cond_in_K` `0.1868`
- `O_best_on_improved_shortlist`: natural Acc@1 `0.0897`, `cond_in_K` `0.1538`
- `retrained_rerank_on_improved_shortlist`: natural Acc@1 `0.1090`, `cond_in_K` `0.1868`

So the retrained reranker:

- **recovered** the gap from reused `O_best`
- **matched** the current improved-shortlist reference rerank on the primary metric
- slightly improved secondary ranking metrics:
  - Acc@5 `0.3269 -> 0.3397`
  - MRR `0.2255 -> 0.2323`

But it did **not** establish a new best Acc@1.

## Did the best combined result beat `0.1090`?

No.

From `shortlist_rerank_combined_table_rebalance.csv`:

- `improved_shortlist_plus_reference_rerank = 0.1090`
- `improved_shortlist_plus_retrained_rerank = 0.1090`
- delta: `+0.0000`

So the rebalance run tied the current best combined result but did not exceed it.

## Did `cond_in_K` improve?

It improved relative to reused `O_best`, but not beyond the current reference row.

- reused `O_best` on improved shortlist: `0.1538`
- retrained rerank on improved shortlist: `0.1868`
- reference rerank on improved shortlist: `0.1868`

So reranker rebalance restored conditional quality from the mismatched `O_best` setting, but did not push conditional accuracy past the current improved-shortlist reference baseline.

## New dominant bottleneck after this phase

The dominant issue is now not raw shortlist failure in isolation.

The shortlist is healthier, and reranker rebalance can recover mismatch, but a single reranker-only rebalance run did not produce a new Acc@1 best.

The remaining bottleneck is therefore better described as:

- **retrieval / reranker co-adaptation on the healthier shortlist**

In other words:

- shortlist quality is no longer the old blocker
- reranker mismatch can be corrected partially
- but isolated reranker retraining alone was not enough to move past `0.1090`

## Next best action

The next best action is:

- **revisit retrieval / reranker co-training**

Reason:

- the standalone reranker rebalance run matched the current best but did not beat it
- that is weaker evidence for more isolated reranker-only tuning
- it is still not strong enough to justify widening scope to hybrid routing or phase **D**

So the most evidence-aligned next step is to improve shortlist+rerank co-adaptation while keeping the scope narrow.
