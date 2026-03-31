# Minimal Coadaptation Summary

## What was implemented

This phase ran one lightweight alternating shortlist/reranker update on top of the existing corrected stack:

1. shortlist re-selection using the **first retrained reranker** as the frozen downstream reference
2. one more reranker training pass on top of that reselected shortlist
3. corrected combined evaluation across the baseline, improved-shortlist, and coadapted systems

The implementation reused the existing shortlist-training, reranker-training, and corrected combined-evaluation paths. The only new orchestration code is `scripts/run_minimal_coadaptation_phase.py`.

## Prior assets reused

- corrected fix-phase bundle: `outputs/20260331_160556_fix_combined_nloss/`
- official improved shortlist bundle: `outputs/20260331_170659_official_shortlist_strengthening/`
- first reranker rebalance bundle: `outputs/20260331_173836_rerank_rebalance/`

Concrete checkpoints reused:

- improved shortlist retriever: `coarse_official_shortlist_strengthening_best_pipeline_natural.pt`
- reference rerank baseline: `rerank_k10_stage1_last.pt`
- reused corrected `O_best`: `rerank_longtrain_oracle_best_natural_two_stage.pt`
- first retrained reranker: `rerank_rebalance_improved_natural_best_natural_two_stage.pt`

## What changed

### 1. Shortlist re-selection changed meaningfully

Under the first retrained reranker as the frozen downstream selector:

- Recall@10 stayed flat at **0.5833**
- Recall@20 improved **0.7821 -> 0.8013**
- natural two-stage Acc@1 with the first retrained reranker improved **0.1090 -> 0.1154**
- `cond_in_K` improved **0.1868 -> 0.1978**

So the shortlist half of co-adaptation was real even though the top-10 recall headline did not move.

### 2. The second reranker pass did not help

On the reselected shortlist:

- first retrained reranker: **0.1154**
- second retrained reranker: **0.1090**
- `cond_in_K`: **0.1978 -> 0.1868**

Secondary metrics were mixed rather than uniformly worse:

- Acc@5 improved **0.3526 -> 0.3590**
- MRR moved **0.2377 -> 0.2328**

Training remained numerically stable:

- `nan_or_inf_batch_count = 0` for every logged epoch
- `rerank_train_valid_fraction` stayed roughly **0.77-0.78**

So the problem is not the old invalid-loss bug returning. The issue is that this second reranker pass did not preserve the shortlist-side gain on the primary metric.

## Did the final combined result beat 0.1090?

It depends on which row is treated as the co-adaptation result:

- `reselected_shortlist_plus_first_retrained_rerank` reached **0.1154**, which **does** beat the previous **0.1090** reference by **+0.0064**
- the required final system, `reselected_shortlist_plus_second_retrained_rerank`, ended at **0.1090**, which **does not** beat the previous reference

So one alternating round produced a better **intermediate** combined system, but the full shortlist-then-reranker sequence did not improve the final row.

## New dominant bottleneck

The dominant bottleneck is now **preserving shortlist-side co-adaptation gains during reranker retraining**.

This phase shows:

- shortlist re-selection under the adapted reranker can help
- another reranker pass on top of that shortlist is not automatically beneficial

That is a narrower conclusion than "co-adaptation failed". Co-adaptation mattered, but the reranker update step did not convert the gain into a stronger final system.

## Next best action

The most conservative next step is: **stop and write the report**.

Reason:

- the narrow co-adaptation question has now been answered cleanly
- the best new signal came from shortlist re-selection, not from the second reranker pass
- another immediate alternating pass would likely add complexity before the current result is properly analyzed

If work resumes later, the next technical direction should be a more careful retrieval/reranker co-training revisit rather than broader scope expansion.
