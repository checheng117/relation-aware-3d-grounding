# Gain Retention Validation Plan

## Audit snapshot (from current conservative second-pass artifacts)

- Current validated `low_lr_secondpass` config file:
  - `outputs/20260331_183805_conservative_secondpass/generated_configs/low_lr_secondpass.yaml`
- Training/eval CLI path used:
  - training: `scripts/train_two_stage_rerank.py --config <yaml>`
  - phase orchestration: `scripts/run_conservative_secondpass_phase.py`
- Validated seed:
  - `seed: 42`
- Validated low LR:
  - `lr: 5.0e-06`
- Reused co-adapted shortlist asset:
  - `outputs/20260331_180150_minimal_coadaptation/checkpoints/coadapted_shortlist_best_pipeline_natural.pt`
- Fine init asset used by low-LR second-pass:
  - `outputs/20260331_173836_rerank_rebalance/checkpoints/rerank_rebalance_improved_natural_best_natural_two_stage.pt`
- Corrected combined evaluation path:
  - `src/rag3d/evaluation/two_stage_eval.py` (`load_two_stage_model(..., fine_only_from_checkpoint=True)`)
  - `src/rag3d/evaluation/two_stage_rerank_metrics.py` (`eval_two_stage_inject_mode`)

## 1) Exact scripts/modules to reuse

1. `scripts/train_two_stage_rerank.py`
2. `scripts/run_conservative_secondpass_phase.py` (as reference structure)
3. `src/rag3d/evaluation/two_stage_eval.py`
4. `src/rag3d/evaluation/two_stage_rerank_metrics.py`
5. `src/rag3d/datasets/referit3d.py` + `src/rag3d/datasets/collate.py` (same val loader path)

## 2) Exact current low_lr_secondpass setup

From the validated yaml:

- `coarse_checkpoint`: `.../outputs/20260331_180150_minimal_coadaptation/checkpoints/coadapted_shortlist_best_pipeline_natural.pt`
- `fine_init_checkpoint`: `.../outputs/20260331_173836_rerank_rebalance/checkpoints/rerank_rebalance_improved_natural_best_natural_two_stage.pt`
- `epochs: 8`
- `lr: 5.0e-06`
- `seed: 42`
- `fine_tune_mode: full`
- `shortlist_train_inject_gold: false`
- `selection_margin_thresh: 0.15`
- `early_stop_patience: 0`
- `min_delta: 0.0`

## 3) One additional seed

- Additional seed for the narrow validation: `43`
- Rationale: minimal perturbation from validated setup, no broader multi-seed expansion.

## 4) Tiny LR sweep values

Center LR = `5.0e-06`.

Narrow local sweep:

- `2.5e-06` (`0.5x`)
- `5.0e-06` (`1.0x`)
- `1.0e-05` (`2.0x`)

All sweep runs keep the same co-adapted shortlist regime and same conservative second-pass recipe except LR.

## 5) Expected output directory

- `outputs/<timestamp>_gain_retention_validation/`

Key required artifacts:

- `low_lr_secondpass_extra_seed_results.json`
- `low_lr_secondpass_extra_seed_table.csv`
- `low_lr_secondpass_extra_seed_interpretation.md`
- `low_lr_sweep_results.json`
- `low_lr_sweep_table.csv`
- `low_lr_sweep_table.md`
- `low_lr_sweep_plot.png`
- `low_lr_sweep_interpretation.md`
- `shortlist_rerank_combined_results_gain_validation.json`
- `shortlist_rerank_combined_table_gain_validation.csv`
- `shortlist_rerank_combined_table_gain_validation.md`
- `shortlist_rerank_main_figure_gain_validation.png`
- `shortlist_rerank_interpretation_gain_validation.md`
- `repro_commands.sh`
- `report_bundle/README.md`

## 6) Primary and secondary metrics

Primary:

- natural two-stage full-scene validation `Acc@1`

Secondary:

- `cond_in_K` (conditional Acc@1 given gold in shortlist)
- `Acc@5`
- `MRR`
- best epoch / early-stop epoch (if available from metrics rows)

## 7) Why this validation is intentionally narrow

This repo has already validated shortlist strengthening, reranker rebalance, co-adaptation, and gain retention under conservative second-pass tuning. The only open practical question is robustness of the retained-gain strategy near the current best low-LR setting.

So this validation intentionally avoids:

- new model components
- broad hyperparameter search
- hybrid router/geometry scope widening

and only performs:

- one extra-seed low-LR check
- one tiny 3-point local LR sweep

to decide whether to stop experiments and move to writing.
