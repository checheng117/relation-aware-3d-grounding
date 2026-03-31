# Fix Combined And N-Loss Summary

## 1. What was wrong before

There were two blocking issues in the March 27, 2026 long-train readout.

1. The combined table did not represent the actual strongest reranker.
   The historical `stronger_rerank` branch in `scripts/run_longtrain_shortlist_rerank_phase.py` pointed to `N_best/N_last`, not `O_best`.

2. Protocol `N` was trained with an invalid objective on many rows.
   Under natural shortlist training, gold is often absent from top-K. The old loss path still ran CE on those rows, which left the gold logit at `-inf` and produced `Infinity` training loss.

There was also a second combined-eval problem discovered during the audit:

- full two-stage checkpoint loading could overwrite the requested coarse checkpoint, so focused-coarse rows were not a clean coarse/rerank decomposition.

## 2. What was fixed

- Combined eval now exposes explicit reranker rows such as `rerank_N_best` and `rerank_O_best`.
- Combined eval now composes checkpoints correctly by loading only `fine.*` from the rerank checkpoint when a separate coarse checkpoint is being tested.
- Natural-shortlist rerank training now computes loss only on rows where gold is actually inside the current shortlist.
- Natural rerank training now logs:
  - `rerank_train_valid_fraction`
  - `gold_in_shortlist_rate_train`
  - `gold_in_shortlist_rate_val`
  - `nan_or_inf_batch_count`

## 3. Was the old combined table misleading?

Yes.

- It did not test `O_best`, even though `O` was clearly stronger than the broken March 27 `N` run.
- It also did not cleanly isolate focused coarse vs rerank because full checkpoint loading could override `coarse.*`.

So the old `A/B/C/D` table should not be used as a decisive rejection of reranker strengthening.

## 4. Does `O_best` actually improve the combined pipeline?

In the corrected March 31, 2026 fix bundle:

- `baseline_reference` natural `Acc@1 = 0.0256`
- `rerank_O_best` natural `Acc@1 = 0.0385`

So yes, `O_best` improves the corrected end-to-end natural pipeline in this bundle.

That said, the absolute gain is still modest, and natural performance remains far below oracle, so the bottleneck is not solved.

## 5. Can `N` now train stably?

Yes.

From `outputs/20260331_160556_fix_combined_nloss/oracle_reranker_table_nfix.csv`:

- `N` best natural val `Acc@1 = 0.1474`
- `N` last natural val `Acc@1 = 0.1218`
- `nan_or_inf_batch_count_max = 0`
- `rerank_train_valid_fraction` stayed in `[0.2781, 0.3147]`

So the fixed natural-shortlist path is now usable.

## 6. Is reranker strengthening still the right direction?

Yes, cautiously.

- Fixing `N` removed a clear training bug and recovered a strong standalone reranker.
- Representing `O_best` correctly in the combined eval gives a positive downstream gain over the corrected baseline.
- `O` still shows the stronger oracle-shortlist ceiling, which means rerank capacity / supervision quality still matter.

But the corrected combined gains are much smaller than the standalone reranker-only validation numbers. That means shortlist + downstream evaluation alignment is still the main practical bottleneck.

## 7. Next best action

The next best action is not a new broad experiment phase.

It is:

1. Keep the fixed loss path and fixed combined evaluator as the authoritative code path.
2. Re-run future shortlist/rerank checkpoint promotion only through the corrected combined evaluator.
3. Investigate why strong standalone reranker validation (`~0.15`) still translates into only modest corrected combined gains (`~0.04`) under the deployed shortlist path.

Concretely, the next phase should stay focused on shortlist recall + rerank alignment, not on hybrid routing or pure structured replacement.
