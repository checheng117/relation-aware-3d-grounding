# Shortlist Selection Alignment Note

## Previous shortlist-stage selection behavior

The repo already had two different shortlist-stage selection patterns:

1. Historical coarse promotion scripts could rank candidates by coarse retrieval metrics such as `recall@20` / `recall@10`.
2. `scripts/train_coarse_stage1.py` + `src/rag3d/training/checkpoint_selection.py` already supported `val_two_stage_selection`, but the prior long-train recipe pointed that selector at the older stage-1 reranker checkpoint rather than the corrected March 31 `O_best` reranker.

That meant shortlist training could still be optimized against a downstream objective, but not against the **corrected strongest reranker reference** that motivated this phase.

## Corrected shortlist-stage selection behavior

The official shortlist-strengthening run uses:

- `val_two_stage_selection.enabled: true`
- `reference_rerank_checkpoint: outputs/20260331_160556_fix_combined_nloss/checkpoints/rerank_longtrain_oracle_best_natural_two_stage.pt`
- `reference_label: 20260331_160556_fix_combined_nloss::rerank_O_best`
- primary selector: `val_pipeline_natural_acc@1`

This is now logged explicitly in the coarse metrics JSONL via:

- `val_pipeline_selection_reference_checkpoint`
- `val_pipeline_selection_reference_label`

Implementation path:

- `src/rag3d/training/checkpoint_selection.py`
- `src/rag3d/training/runner.py`
- `scripts/run_official_shortlist_strengthening_phase.py`

## Operational note: cross-process consistency

The repository’s `TextHashEncoder` currently uses Python’s salted `hash()` function.

That means the token mapping changes across Python processes unless `PYTHONHASHSEED` is fixed **before** startup.

For the official bundle, the run was executed under:

- `PYTHONHASHSEED=42`

This was necessary so that:

- the coarse training subprocess
- the parent-process combined evaluator
- the March 31 reranker checkpoints reused in this phase

were all evaluated under one consistent hash mapping.

## Why the new behavior is better aligned

The real objective of this phase is not coarse recall by itself. It is:

- corrected natural two-stage full-scene validation `Acc@1`

with the already-validated reranker in the loop.

Using March 31 `O_best` as the frozen selector reference is better aligned because it answers the actual phase question:

- does a better shortlist help the corrected stronger reranker produce more end-to-end gain?

Using Recall@K alone, or using an older reranker as the selector reference, is weaker because it can choose coarse checkpoints that improve shortlist coverage without improving the corrected deployed pipeline.
