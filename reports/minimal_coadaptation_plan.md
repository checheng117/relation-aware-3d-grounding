# Minimal Coadaptation Plan

## Scope

This phase stays inside the existing corrected two-stage stack:

- official improved shortlist from the shortlist-strengthening phase
- first retrained reranker from the rerank-rebalance phase
- one lightweight alternating round:
  - shortlist reselection under the retrained reranker
  - one more reranker pass on top of the reselected shortlist
- corrected combined evaluation

It does **not** widen scope to:

- hybrid routing
- phase **D**
- geometry redesign
- joint-training infrastructure

## 1. Exact scripts and modules to reuse

### Shortlist training / selection

- `scripts/train_coarse_stage1.py`
- `src/rag3d/training/runner.py`
- `src/rag3d/training/checkpoint_selection.py`
- `src/rag3d/relation_reasoner/losses.py`

### Reranker training

- `scripts/train_two_stage_rerank.py`
- `src/rag3d/relation_reasoner/two_stage_rerank.py`
- `src/rag3d/relation_reasoner/losses.py`

### Corrected combined evaluation

- `src/rag3d/evaluation/two_stage_eval.py`
- `src/rag3d/evaluation/two_stage_rerank_metrics.py`
- `src/rag3d/evaluation/coarse_recall.py`

### Closest orchestration templates

- `scripts/run_official_shortlist_strengthening_phase.py`
- `scripts/run_rerank_rebalance_phase.py`

## 2. Prior assets / checkpoints to reuse

### Current best improved shortlist retriever

- `outputs/20260331_170659_official_shortlist_strengthening/checkpoints/coarse_official_shortlist_strengthening_best_pipeline_natural.pt`

### Reference rerank baseline

- `outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt`

### Reused corrected `O_best`

- `outputs/20260331_160556_fix_combined_nloss/checkpoints/rerank_longtrain_oracle_best_natural_two_stage.pt`

### First retrained reranker on improved shortlist

- `outputs/20260331_173836_rerank_rebalance/checkpoints/rerank_rebalance_improved_natural_best_natural_two_stage.pt`

### Optional fixed `N_best`

- `outputs/20260331_160556_fix_combined_nloss/checkpoints/rerank_longtrain_natural_best_natural_two_stage.pt`

## 3. Minimal code changes needed

### Required

1. Add one orchestration script for the minimal alternating round.
   - generate shortlist-reselection config
   - generate second reranker-pass config
   - train one reselected shortlist
   - train one second-pass reranker
   - write focused tables/json/md/figures
   - create `repro_commands.sh` and report bundle README

2. Update README with the current co-adaptation framing.

3. Add plan / protocol / summary notes for this phase.

### Not required

- no new model family
- no alternating loop framework
- no dataset rewrite
- no new dependency
- no router work
- no phase **D** changes

## 4. Exact experiment order

1. Update README to record that shortlist strengthening and reranker rebalance are already validated, and that the current bottleneck is shortlist–reranker co-adaptation.
2. Audit the current shortlist/rerank path and write this plan.
3. Write the minimal co-adaptation protocol note.
4. Reselect shortlist using the first retrained reranker as the frozen downstream reference.
5. Evaluate current improved shortlist vs reselected shortlist under that retrained reranker.
6. Run one more reranker pass on the reselected shortlist.
7. Run corrected combined evaluation across the baseline, improved-shortlist, and coadapted systems.
8. Write summary, refresh the report bundle, and add only a tiny README follow-up if warranted.

## 5. Expected output directory

- `outputs/<timestamp>_minimal_coadaptation/`

Expected contents:

- `generated_configs/`
- `logs/`
- `checkpoints/`
- `metrics_coadapted_shortlist.jsonl`
- `metrics_coadapted_reranker.jsonl`
- `coadapted_shortlist_results.json`
- `coadapted_shortlist_table.csv`
- `coadapted_shortlist_table.md`
- `coadapted_shortlist_interpretation.md`
- `coadapted_reranker_results.json`
- `coadapted_reranker_table.csv`
- `coadapted_reranker_table.md`
- `coadapted_reranker_curves.png`
- `coadapted_reranker_interpretation.md`
- `shortlist_rerank_combined_results_coadaptation.json`
- `shortlist_rerank_combined_table_coadaptation.csv`
- `shortlist_rerank_combined_table_coadaptation.md`
- `shortlist_rerank_main_figure_coadaptation.png`
- `shortlist_rerank_interpretation_coadaptation.md`
- `repro_commands.sh`
- `report_bundle/README.md`

## 6. Mandatory vs optional items

### Mandatory

- README update for the co-adaptation stage
- one real shortlist reselection under the first retrained reranker
- one second reranker pass on top of that reselected shortlist
- corrected combined evaluation including:
  - `baseline_reference`
  - `improved_shortlist_plus_reference_rerank`
  - `improved_shortlist_plus_reused_O_best`
  - `improved_shortlist_plus_first_retrained_rerank`
  - `reselected_shortlist_plus_first_retrained_rerank`
  - `reselected_shortlist_plus_second_retrained_rerank`

### Optional

- one ablation where the shortlist is reselected but the reranker is not trained again
- include fixed `N_best` in a secondary row if cheap

## 7. Primary selection metric

The primary selector throughout this phase remains:

- `natural two-stage full-scene validation Acc@1`

Concretely:

- shortlist reselection uses `val_pipeline_natural_acc@1` with the **first retrained reranker** frozen in the loop
- the second reranker pass uses `val_natural_two_stage_acc@1` on the **reselected shortlist**

## 8. Audit conclusion

The repo already has the exact moving parts needed for one lightweight alternating round:

- shortlist-stage selection already supports a chosen frozen reranker reference
- reranker training already supports a chosen shortlist regime through `coarse_checkpoint`
- impossible natural-shortlist supervision rows are already skipped safely
- corrected combined evaluation already composes coarse + rerank via `fine_only_from_checkpoint=True`

So this phase should be executed as **one clean alternating round**, not as a redesign or a new training framework.
