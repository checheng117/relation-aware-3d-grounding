# Combined Eval Fix Note

## What was wrong before

The March 27, 2026 long-train combined table in:

- `outputs/20260327_231112_longtrain_shortlist_rerank/shortlist_rerank_combined_table_longtrain.csv`

was misleading for two separate reasons:

1. The historical `B_stronger_rerank_only` row did **not** use the strongest reranker.
   The code path in `scripts/run_longtrain_shortlist_rerank_phase.py` selected:
   - `rerank_longtrain_natural_best_natural_two_stage.pt`
   - fallback `rerank_longtrain_natural_last.pt`
   and never represented `rerank_longtrain_oracle_best_natural_two_stage.pt`.

2. The combined evaluator loaded the full two-stage checkpoint when composing coarse + rerank.
   In practice that allowed `coarse.*` inside the rerank checkpoint to overwrite the requested coarse checkpoint, so the focused-coarse rows were not a clean coarse swap.

## What was changed

The fix is intentionally narrow:

- `scripts/run_longtrain_shortlist_rerank_phase.py`
  - added explicit combined rows such as `rerank_N_best`, `rerank_O_best`, `rerank_N_last`, `rerank_O_last`
  - added `--output-tag` so the fix bundle can live under `outputs/<timestamp>_fix_combined_nloss/`
  - made row names transparent instead of using one vague `stronger_rerank` alias

- `src/rag3d/evaluation/two_stage_eval.py`
  - added `fine_only_from_checkpoint=True` support in `load_two_stage_model(...)`
  - combined eval now loads only `fine.*` from the rerank checkpoint when a separate coarse checkpoint is being tested

## Corrected output

The corrected bundle is:

- `outputs/20260331_160556_fix_combined_nloss/`

The final fixed combined table is:

- `outputs/20260331_160556_fix_combined_nloss/shortlist_rerank_combined_table_fixed.csv`

Main corrected readout from that table:

- `baseline_reference` natural `Acc@1 = 0.0256`
- `rerank_O_best` natural `Acc@1 = 0.0385`
- `focused_coarse_plus_old_rerank` natural `Acc@1 = 0.0192`

## Interpretation

- Yes, the previous combined table used the wrong reranker representative.
- After representing `O_best` correctly, the end-to-end natural pipeline is better than the corrected baseline in this March 31, 2026 bundle.
- The gain is real but still modest in absolute terms, so the reranker-strengthening direction remains promising rather than already solved.
- The focused coarse checkpoint did not provide cleaner evidence than the reranker fix in this pass.
