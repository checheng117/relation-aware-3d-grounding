# Fix Combined Eval And N-Loss Plan

## 1. Current combined-evaluation code path

- Entry point: `scripts/run_longtrain_shortlist_rerank_phase.py`
- Checkpoint discovery:
  - `rerank_nat_best = checkpoints/rerank_longtrain_natural_best_natural_two_stage.pt`
  - `rerank_nat_last = checkpoints/rerank_longtrain_natural_last.pt`
  - `rerank_oracle_best = checkpoints/rerank_longtrain_oracle_best_natural_two_stage.pt`
  - `rerank_oracle_last = checkpoints/rerank_longtrain_oracle_last.pt`
  - `coarse_best = checkpoints/coarse_focused_hardneg_longtrain_best_pipeline_natural.pt`
- Actual current combined-selection logic:
  - At `scripts/run_longtrain_shortlist_rerank_phase.py:429`, `strong_rerank` is hard-coded to `rerank_nat_best` (fallback `rerank_nat_last`).
  - The combined rows at `scripts/run_longtrain_shortlist_rerank_phase.py:431-434` therefore evaluate:
    - `A_baseline_pipeline`
    - `B_stronger_rerank_only` = baseline coarse + `N_best/N_last`
    - `C_focused_coarse_old_rerank`
    - `D_focused_coarse_stronger_rerank` = focused coarse + `N_best/N_last`
- Result: protocol `O` checkpoints are trained, saved, and summarized in `oracle_reranker_results_longtrain.json`, but are never used by the downstream combined table.

## 2. Current reranker-training loss code path

- Entry point: `scripts/train_two_stage_rerank.py`
- Training forward:
  - `scripts/train_two_stage_rerank.py:87` calls `forward_two_stage_rerank(...)`
  - `src/rag3d/relation_reasoner/two_stage_rerank.py:198-204` sets `inject=False` when `model.training` and `shortlist_train_inject_gold=False`
  - `TwoStageCoarseRerankModel.forward(...)` then builds a natural shortlist without forcing gold into top-K
  - `full_logits` is initialized to `-inf` and only shortlisted indices are filled (`two_stage_rerank.py:173-175`)
- Loss:
  - `scripts/train_two_stage_rerank.py:89-96` calls `compute_batch_training_loss(...)`
  - `src/rag3d/relation_reasoner/losses.py:169` calls `grounding_cross_entropy(...)`
  - `grounding_cross_entropy(...)` masks invalid positions to `-inf` and passes full-scene logits to `F.cross_entropy(...)`

## 3. What is actually wrong

### Combined evaluation

- The current “stronger rerank” branch is not the strongest reranker.
- In the verified long-train run `outputs/20260327_231112_longtrain_shortlist_rerank/`:
  - `oracle_reranker_table_longtrain.csv` shows protocol `O` clearly beats `N`
  - `O` reaches natural val `Acc@1 = 0.1538` and oracle val `Acc@1 = 0.6731`
  - `N` has `train_loss_mean = Infinity` for every epoch and degrades to natural val `Acc@1 = 0.0192`
- But `shortlist_rerank_combined_table_longtrain.csv` uses `N_best` as row `B` and therefore does not test the actual strongest reranker.
- Conclusion: the old combined table is a misleading readout of reranker strengthening.

### Natural-shortlist loss

- Under protocol `N`, gold is often absent from the current shortlist by design.
- When gold is absent:
  - the gold index never receives a finite rerank logit
  - the target logit remains `-inf`
  - full-scene CE on that target becomes `inf`
- This is the direct code-level explanation for `train_loss_mean = Infinity` in `metrics_rerank_natural.jsonl`.
- The current loss path does not distinguish valid rerank-supervision rows from impossible rows.

## 4. Minimal fix plan

### A. Combined evaluation

- Patch `scripts/run_longtrain_shortlist_rerank_phase.py` so combined evaluation explicitly evaluates separate reranker representatives instead of one vague `stronger_rerank` alias.
- Required rows:
  - `baseline_reference`
  - `rerank_N_best`
  - `rerank_O_best`
  - `focused_coarse_plus_old_rerank`
  - `focused_coarse_plus_rerank_O_best`
- Optional cheap rows:
  - `rerank_N_last`
  - `rerank_O_last`
- Keep old baseline checkpoint choices unchanged; only fix the evaluation selection/reporting.

### B. Natural-shortlist loss

- Patch the rerank training path so rerank CE is computed only on rows where gold is actually inside the current shortlist.
- Minimal implementation shape:
  - expose shortlist indices / gold-in-shortlist information from the two-stage forward path during training
  - build a per-sample valid mask for rerank supervision
  - skip invalid rows rather than supervising an impossible target
  - keep existing hinge losses only on the same valid rows
- Add explicit metrics:
  - `rerank_train_valid_fraction`
  - `gold_in_shortlist_rate_train`
  - `gold_in_shortlist_rate_val`
  - `nan_or_inf_batch_count`

## 5. Minimal reruns after the fix

- Reuse existing `O` and focused-coarse checkpoints from:
  - `outputs/20260327_231112_longtrain_shortlist_rerank/`
- Rerun only protocol `N` training into a new timestamped bundle.
- Then rerun combined evaluation with explicit `N_best` and `O_best`.

### Planned command set

1. Run fixed `N` reranker training into the new bundle config.
2. Copy or reuse existing `O` / focused-coarse checkpoints and metrics into the same bundle.
3. Run fixed combined evaluation from `scripts/run_longtrain_shortlist_rerank_phase.py` in eval-only mode against that bundle.

Concretely, after the code changes I expect the minimal repro script to contain commands of the form:

```bash
python scripts/train_two_stage_rerank.py --config outputs/<stamp>_fix_combined_nloss/generated_configs/rerank_longtrain_natural.yaml
python scripts/run_longtrain_shortlist_rerank_phase.py --stamp <stamp> --output-tag fix_combined_nloss --skip-train
```

with the new output directory pre-populated with the reused `O` / coarse artifacts from `outputs/20260327_231112_longtrain_shortlist_rerank/`.

## 6. Outputs to generate

### Reports

- `reports/combined_eval_fix_note.md`
- `reports/natural_rerank_loss_fix_note.md`
- `reports/fix_combined_and_nloss_summary.md`

### New experiment bundle

- `outputs/<timestamp>_fix_combined_nloss/`
- `outputs/<timestamp>_fix_combined_nloss/repro_commands.sh`
- `outputs/<timestamp>_fix_combined_nloss/report_bundle/README.md`

### Combined-eval artifacts

- `shortlist_rerank_combined_table_fixed.csv`
- `shortlist_rerank_combined_table_fixed.md`
- `shortlist_rerank_combined_interpretation_fixed.md`

### Natural-reranker-fix artifacts

- `oracle_reranker_results_nfix.json`
- `oracle_reranker_table_nfix.csv`
- `oracle_reranker_interpretation_nfix.md`

### Logs

- fixed `N` training log
- fixed combined-eval log
