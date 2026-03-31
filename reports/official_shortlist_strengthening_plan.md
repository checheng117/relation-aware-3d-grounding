# Official Shortlist Strengthening Plan

## Scope

This phase stays inside the existing **coarse shortlist -> rerank -> corrected combined evaluation** pipeline.

It does **not** touch:

- phase **D**
- the **hybrid router**
- broader geometry redesign

## 1. Exact scripts and modules to reuse

### Coarse retrieval training

- `scripts/train_coarse_stage1.py`
- `src/rag3d/training/runner.py`
- `src/rag3d/training/checkpoint_selection.py`
- `src/rag3d/relation_reasoner/losses.py`

### Shortlist hard-negative mining / loss definitions

- `src/rag3d/relation_reasoner/losses.py`
  - `same_class_hinge_loss`
  - `spatial_nearby_hinge_loss`
  - `hardest_negative_margin_loss`
  - `compute_batch_training_loss`

### Corrected combined evaluation

- `src/rag3d/evaluation/two_stage_eval.py`
  - `load_two_stage_model(..., fine_only_from_checkpoint=True)`
- `src/rag3d/evaluation/two_stage_rerank_metrics.py`
  - `eval_two_stage_inject_mode`
  - `eval_by_candidate_load_bucket`
- `src/rag3d/evaluation/coarse_recall.py`
  - `eval_coarse_stage1_metrics`

### Prior orchestration pattern to reuse

- `scripts/run_longtrain_shortlist_rerank_phase.py`

This is the closest existing template for:

- timestamped output directories
- generated configs
- combined pipeline tables
- headless plotting
- repro command capture

## 2. Corrected assets to reuse from `outputs/20260331_160556_fix_combined_nloss/`

### Primary reranker reference for selection and main combined eval

- `outputs/20260331_160556_fix_combined_nloss/checkpoints/rerank_longtrain_oracle_best_natural_two_stage.pt`
  - use as the **primary frozen reranker reference**
  - this is the main `O_best` asset for shortlist-stage checkpoint selection
  - this is also the main reranker in the official combined table

### Secondary reranker reference for auxiliary comparison

- `outputs/20260331_160556_fix_combined_nloss/checkpoints/rerank_longtrain_natural_best_natural_two_stage.pt`
  - auxiliary comparison only
  - do **not** use as the main selector

### Historical corrected table / summary for background only

- `outputs/20260331_160556_fix_combined_nloss/shortlist_rerank_combined_table_fixed.csv`
- `reports/fix_combined_and_nloss_summary.md`
- `reports/combined_eval_fix_note.md`
- `reports/natural_rerank_loss_fix_note.md`

### Coarse init checkpoint to reuse

- `outputs/checkpoints_stage1/coarse_geom_recall_last.pt`

This remains the safest coarse initialization point for the new official shortlist run.

### Explicitly not reused as the new starting checkpoint

- `outputs/20260331_160556_fix_combined_nloss/checkpoints/coarse_focused_hardneg_longtrain_best_pipeline_natural.pt`

Reason:

- it was selected against an older reranker reference, not March 31 `O_best`
- the published fix-bundle retrieval/combined artifacts are not fully self-consistent with current live re-evaluation, so the official phase should generate a fresh self-consistent bundle instead of inheriting that coarse artifact as the new base

## 3. Minimal code changes needed

### Required

1. Add an official orchestration script for this phase.
   - new timestamped output root
   - generated coarse config
   - retrieval table/json/md
   - corrected combined eval table/json/md/png
   - report bundle README

2. Make coarse selection provenance explicit in logged metrics.
   - log which frozen reranker checkpoint is used for `val_two_stage_selection`
   - avoid ambiguity about whether selection used stage-1 rerank or corrected `O_best`

3. Adjust the official coarse loss mix only at the config/orchestrator level.
   - keep the same three negative families
   - rebalance weights to be less noisy than the previous focused long-train recipe

### Not required

- no change to reranker training logic
- no new dataset pipeline
- no new dependency
- no router code
- no phase D changes

## 4. Exact experiment order

1. Audit code paths and fix-phase assets.
2. Write this plan.
3. Align shortlist-stage selection to the corrected downstream objective.
   - frozen selector reranker = March 31 `O_best`
4. Run one official coarse shortlist-strengthening training job.
   - init from `coarse_geom_recall_last.pt`
   - focused three-part hard-negative recipe only
   - save `*_best_pipeline_natural.pt`
5. Evaluate shortlist retrieval.
   - baseline coarse vs improved coarse
   - Recall@5/10/20/40
   - shortlist-oriented diagnostics
6. Run corrected combined evaluation.
   - `baseline_reference`
   - `rerank_O_best`
   - `improved_shortlist_plus_reference_rerank`
   - `improved_shortlist_plus_rerank_O_best`
   - optional `improved_shortlist_plus_rerank_N_best`
7. Write interpretation reports, report bundle README, summary, README note.

## 5. Expected output directory

- `outputs/<timestamp>_official_shortlist_strengthening/`

Expected contents:

- `generated_configs/`
- `logs/`
- `checkpoints/`
- `shortlist_retrieval_results_official.json`
- `shortlist_retrieval_table_official.csv`
- `shortlist_retrieval_table_official.md`
- `shortlist_retrieval_interpretation_official.md`
- `shortlist_rerank_combined_results_official.json`
- `shortlist_rerank_combined_table_official.csv`
- `shortlist_rerank_combined_table_official.md`
- `shortlist_rerank_main_figure_official.png`
- `shortlist_rerank_interpretation_official.md`
- `repro_commands.sh`
- `report_bundle/README.md`

## 6. Mandatory vs optional items

### Mandatory

- one official shortlist-strengthening coarse run
- selection driven by corrected natural two-stage full-scene val `Acc@1`
- corrected combined evaluation with `O_best` as the main reranker
- concise reports:
  - `reports/shortlist_selection_alignment_note.md`
  - `reports/official_shortlist_hard_negative_note.md`
  - `reports/official_shortlist_strengthening_summary.md`
  - `reports/readme_official_shortlist_strengthening_note.md`

### Optional

- auxiliary combined row using fixed `N_best`
- additional candidate-load or clutter slice text beyond the required one

## 7. Metrics that will drive checkpoint selection

### Primary selector

- `val_pipeline_natural_acc@1`

Definition:

- natural two-stage full-scene validation `Acc@1`
- current coarse checkpoint + frozen March 31 `O_best` reranker
- evaluated through the corrected `fine_only_from_checkpoint=True` path

### Secondary diagnostics to log but not use as the sole selector

- `val_coarse_recall@5`
- `val_coarse_recall@10`
- `val_coarse_recall@20`
- `val_coarse_recall@40`
- `val_pipeline_natural_shortlist_recall`
- `val_pipeline_natural_cond_acc_in_k`
- `val_pipeline_oracle_acc@1`
- `candidate_load::high` retrieval slice where available

## 8. Audit conclusion

The repo already contains the right corrected building blocks:

- coarse selection by downstream natural two-stage objective
- fixed natural-shortlist rerank loss
- corrected fine-only combined evaluation

So this phase should be implemented as a **narrow orchestration and config alignment pass**, not a project reset.
