# Conservative Second-Pass Summary

## 1) What was implemented

This phase implemented a narrow conservative second-pass reranker tuning pass on top of the fixed co-adapted shortlist:

- README stage update + minimal follow-up
- reranker training entrypoint extension:
  - `fine_tune_mode` (`full`, `attr_rel_heads`, `relation_head_only`)
  - `early_stop_patience`
  - `min_delta`
- orchestration script:
  - `scripts/run_conservative_secondpass_phase.py`
- mandatory conservative variants:
  - `low_lr_secondpass`
  - `short_schedule_secondpass`
  - `partial_freeze_secondpass`
- corrected combined evaluation and report bundle generation under:
  - `outputs/20260331_183805_conservative_secondpass/`

## 2) Prior assets reused

- co-adapted shortlist:
  - `outputs/20260331_180150_minimal_coadaptation/checkpoints/coadapted_shortlist_best_pipeline_natural.pt`
- first retrained reranker (intermediate best pair):
  - `outputs/20260331_173836_rerank_rebalance/checkpoints/rerank_rebalance_improved_natural_best_natural_two_stage.pt`
- naive second-pass reranker (fallback row):
  - `outputs/20260331_180150_minimal_coadaptation/checkpoints/coadapted_reranker_second_best_natural_two_stage.pt`
- corrected baseline/improved-shortlist references:
  - `outputs/checkpoints_stage1/coarse_geom_recall_last.pt`
  - `outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt`
  - `outputs/20260331_170659_official_shortlist_strengthening/checkpoints/coarse_official_shortlist_strengthening_best_pipeline_natural.pt`

## 3) Did any conservative second-pass variant preserve `0.1154`?

Yes.

From `conservative_secondpass_table.csv`:

- `coadapted_intermediate_best`: `0.1154`
- `low_lr_secondpass`: `0.1154`

So `low_lr_secondpass` preserved the intermediate co-adapted gain.

## 4) Did any variant beat naive second-pass `0.1090`?

Yes.

- `naive_secondpass_rerank`: `0.1090`
- `low_lr_secondpass`: `0.1154` (`+0.0064`)

`short_schedule_secondpass` and `partial_freeze_secondpass` did not beat `0.1090`.

## 5) How did `cond_in_K` change?

- intermediate best: `0.1978`
- naive second-pass: `0.1868`
- `low_lr_secondpass`: `0.1978` (retained)
- `short_schedule_secondpass`: `0.1868`
- `partial_freeze_secondpass`: `0.1868`

So the same variant that retained Acc@1 also retained conditional reranker quality.

## 6) New dominant bottleneck after this phase

The bottleneck shifted from "whether retention is possible" to "how robustly retention can be achieved without landing at epoch-0-like behavior".

Observed pattern:

- all best checkpoints were selected at epoch `0`
- low-LR retained gain, but more aggressive/structured updates regressed to `0.1090`

So the main issue is controlled update magnitude and stable retention over training trajectory, not absence of co-adaptation signal.

## 7) Next best action

**One more narrow reranker retention attempt.**

Recommended strict-scope follow-up:

- keep the same co-adapted shortlist fixed
- keep low LR regime
- test a tiny retention-focused schedule around early epochs (e.g., checkpoint-at-step policy and tighter min-delta guard)
- stop after this one additional pass and then finalize report writing

This stays aligned with current evidence and avoids widening to hybrid router or phase **D**.
