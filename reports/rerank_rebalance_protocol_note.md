# Rerank Rebalance Protocol Note

## What “improved shortlist” means in this phase

In this repository, the rerank shortlist is determined entirely by the coarse checkpoint used inside `TwoStageCoarseRerankModel`.

For this phase, "improved shortlist" means:

- coarse checkpoint: `outputs/20260331_170659_official_shortlist_strengthening/checkpoints/coarse_official_shortlist_strengthening_best_pipeline_natural.pt`
- rerank top-K: `K=10`
- natural training condition: `shortlist_train_inject_gold: false`

So the reranker sees the healthier shortlist distribution produced by the official shortlist-strengthening coarse model, not the older shortlist produced by `coarse_geom_recall_last.pt`.

## Reranker variants compared

### A. `reference_rerank_on_improved_shortlist`

- checkpoint: `outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt`
- no new training
- evaluated with the improved shortlist coarse checkpoint

This is the current best combined baseline from the official shortlist bundle:

- `improved_shortlist_plus_reference_rerank = 0.1090`

### B. `O_best_on_improved_shortlist`

- checkpoint: `outputs/20260331_160556_fix_combined_nloss/checkpoints/rerank_longtrain_oracle_best_natural_two_stage.pt`
- no new training
- evaluated with the improved shortlist coarse checkpoint

This tests whether the previously strongest reranker still transfers well once shortlist quality has changed.

### C. `retrained_rerank_on_improved_shortlist`

- mandatory new training run
- coarse checkpoint fixed to the improved shortlist checkpoint
- shortlist construction during training is natural: `shortlist_train_inject_gold: false`
- model selection is by `val_natural_two_stage_acc@1`

Initialization choice for the main run:

- initialize the fine reranker from prior corrected `O_best`

Reason:

- `O_best` is the strongest validated reranker from the corrected fix phase
- the current issue is that it is not well matched to the improved shortlist distribution
- rebalancing it directly against the improved shortlist is the cleanest test of the current hypothesis

### Optional D. `mixed_retrain_rerank`

This is optional only and should not block the phase.

The main line remains the natural improved-shortlist retrain above.

## Why this is the correct next step

The current evidence already answers two earlier questions:

1. stronger rerankers are real under corrected evaluation
2. stronger shortlists are also now real under corrected evaluation

The remaining mismatch is:

- `improved_shortlist_plus_reference_rerank = 0.1090`
- `improved_shortlist_plus_rerank_O_best = 0.0897`
- `cond_in_K` for reused `O_best` fell from `0.2045` to `0.1538`

So the correct next question is no longer whether shortlist matters.

It is:

- can reranker training be re-balanced so it better exploits the healthier shortlist?

That makes improved-shortlist reranker retraining the narrowest and most evidence-aligned next step.
