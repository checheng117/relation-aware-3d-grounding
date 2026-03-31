# Rerank Rebalance Plan

## Scope

This phase stays inside the existing corrected two-stage stack:

- improved coarse shortlist from the official shortlist-strengthening phase
- reranker retraining / rebalance on top of that shortlist
- corrected combined evaluation

It does **not** widen scope to:

- hybrid routing
- phase **D**
- geometry redesign

## 1. Exact scripts and modules to reuse

### Reranker training

- `scripts/train_two_stage_rerank.py`
- `src/rag3d/relation_reasoner/two_stage_rerank.py`
- `src/rag3d/relation_reasoner/losses.py`

### Corrected combined evaluation

- `src/rag3d/evaluation/two_stage_eval.py`
- `src/rag3d/evaluation/two_stage_rerank_metrics.py`
- `src/rag3d/evaluation/coarse_recall.py`

### Closest orchestration templates

- `scripts/run_longtrain_shortlist_rerank_phase.py`
- `scripts/run_official_shortlist_strengthening_phase.py`

## 2. Improved shortlist assets to reuse

The authoritative improved shortlist source is the official shortlist-strengthening bundle:

- `outputs/20260331_170659_official_shortlist_strengthening/checkpoints/coarse_official_shortlist_strengthening_best_pipeline_natural.pt`

This is the improved shortlist generation path because:

- `scripts/train_two_stage_rerank.py` builds the rerank shortlist from `coarse_checkpoint`
- `TwoStageCoarseRerankModel` in `src/rag3d/relation_reasoner/two_stage_rerank.py` uses that coarse checkpoint to produce the top-K rerank candidates

So "training on the improved shortlist" means:

- `coarse_checkpoint = outputs/20260331_170659_official_shortlist_strengthening/checkpoints/coarse_official_shortlist_strengthening_best_pipeline_natural.pt`

## 3. Baseline reranker checkpoints

### Reference rerank baseline

- `outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt`

### Corrected strongest prior reranker

- `outputs/20260331_160556_fix_combined_nloss/checkpoints/rerank_longtrain_oracle_best_natural_two_stage.pt`

### Auxiliary fixed natural reranker

- `outputs/20260331_160556_fix_combined_nloss/checkpoints/rerank_longtrain_natural_best_natural_two_stage.pt`

## 4. Minimal code changes needed

### Required

1. Add one orchestration script for this phase.
   - generate rerank config for improved-shortlist training
   - train one reranker rebalance run
   - write rerank table/json/md/curve
   - run corrected combined evaluation
   - create `repro_commands.sh` and report bundle README

2. Update README with the current validated evidence.

3. Add plan / protocol / summary notes for this phase.

### Not required

- no architectural reranker expansion
- no dataset rewrite
- no new dependency
- no router work
- no phase **D** changes

## 5. Exact experiment order

1. Update README with validated evidence from the fix bundle and official shortlist bundle.
2. Audit improved-shortlist rerank path and write this plan.
3. Write rerank rebalance protocol note.
4. Run one main reranker retraining / finetune experiment on the improved shortlist.
5. Evaluate baseline rerankers on the improved shortlist.
6. Run corrected combined evaluation with the retrained reranker.
7. Write summary and refresh the output report bundle.

## 6. Expected output directory

- `outputs/<timestamp>_rerank_rebalance/`

Expected contents:

- `generated_configs/`
- `logs/`
- `checkpoints/`
- `metrics_rerank_rebalance.jsonl`
- `rerank_rebalance_results.json`
- `rerank_rebalance_table.csv`
- `rerank_rebalance_table.md`
- `rerank_rebalance_curves.png`
- `rerank_rebalance_interpretation.md`
- `shortlist_rerank_combined_results_rebalance.json`
- `shortlist_rerank_combined_table_rebalance.csv`
- `shortlist_rerank_combined_table_rebalance.md`
- `shortlist_rerank_main_figure_rebalance.png`
- `shortlist_rerank_interpretation_rebalance.md`
- `repro_commands.sh`
- `report_bundle/README.md`

## 7. Mandatory vs optional items

### Mandatory

- README evidence update
- one main retrained reranker on the improved shortlist
- corrected combined evaluation including:
  - `baseline_reference`
  - `improved_shortlist_plus_reference_rerank`
  - `improved_shortlist_plus_rerank_O_best`
  - `improved_shortlist_plus_retrained_rerank`
- concise reports:
  - `reports/rerank_rebalance_protocol_note.md`
  - `reports/rerank_rebalance_summary.md`
  - `reports/readme_rerank_rebalance_note.md`

### Optional

- include `improved_shortlist_plus_fixed_N_best`
- include one mixed retrain variant if it is cheap and clean

## 8. Primary model-selection metric

The primary selector remains:

- `val_natural_two_stage_acc@1`

with:

- the improved shortlist coarse checkpoint in the loop
- natural shortlist construction (`shortlist_train_inject_gold: false`)
- corrected evaluation path from `eval_two_stage_inject_mode`

This is already how `scripts/train_two_stage_rerank.py` chooses `*_best_natural_two_stage.pt`.

## 9. Audit conclusion

The repo already has the key capability needed for this phase:

- rerank training can be aligned to a chosen shortlist simply by changing `coarse_checkpoint`
- natural versus oracle shortlist conditions are already controlled by `shortlist_train_inject_gold`
- impossible natural-shortlist rows are already skipped safely
- checkpoint selection is already based on natural two-stage validation Acc@1

So this phase should be executed as a **narrow reranker re-balance pass on top of the improved coarse checkpoint**, not as a redesign.
