# Gain Retention Validation Summary

## 1) What was implemented

This phase ran a strictly narrow robustness check around the validated `low_lr_secondpass` strategy:

1. one extra-seed validation (`seed=43`) with the same low-LR second-pass setup
2. one tiny 3-point LR sweep around `5e-6`:
   - `2.5e-6` (0.5x)
   - `5e-6` (1.0x)
   - `1e-5` (2.0x)
3. corrected combined evaluation table/figure across reference and validation rows

All artifacts are under:

- `outputs/20260331_185900_gain_retention_validation/`

## 2) Prior assets reused

- co-adapted shortlist:
  - `outputs/20260331_180150_minimal_coadaptation/checkpoints/coadapted_shortlist_best_pipeline_natural.pt`
- first retrained reranker (intermediate-best pair):
  - `outputs/20260331_173836_rerank_rebalance/checkpoints/rerank_rebalance_improved_natural_best_natural_two_stage.pt`
- previously validated low-LR second-pass checkpoint:
  - `outputs/20260331_183805_conservative_secondpass/checkpoints/low_lr_secondpass_best_natural_two_stage.pt`
- improved-shortlist reference rerank row:
  - `outputs/20260331_170659_official_shortlist_strengthening/checkpoints/coarse_official_shortlist_strengthening_best_pipeline_natural.pt`
  - `outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt`

## 3) Does the extra seed support low_lr_secondpass?

Not in this run.

From `low_lr_secondpass_extra_seed_table.csv`:

- validated low-LR (seed 42): `0.1154`, `cond_in_K=0.1978`
- extra seed (seed 43): `0.1026`, `cond_in_K=0.1758`

So the extra seed moved below both the retained-gain level (`0.1154`) and the `0.1090` shortlist+reference-rerank benchmark.

## 4) Does the LR sweep suggest a stable local optimum?

Locally stable for the validated seed.

From `low_lr_sweep_table.csv` (seed 42):

- `2.5e-6`: `0.1154`
- `5.0e-6`: `0.1154`
- `1.0e-5`: `0.1154`

`cond_in_K` stays `0.1978` at all three points, and MRR varies only slightly (`0.2378` to `0.2373`).

This indicates a stable local LR regime around the current low-LR setting for seed 42.

## 5) Did any variant beat `0.1154`?

No.

Best value in this phase remains `0.1154` (tie), with no strict improvement above the current retained-gain reference.

## 6) Confidence level for presenting low_lr_secondpass as current best strategy

Conservative confidence:

- **high** confidence for a local LR-stable retained-gain regime at seed 42
- **lower** confidence for seed robustness after the single extra-seed drop to `0.1026`

Practical framing:

- keep `low_lr_secondpass` as the current best narrow strategy
- report that retained gain is not yet fully seed-stable based on this validation

## 7) Continue experiments or stop and write?

Recommendation: **stop additional tuning and move to writing**.

Reason:

- this phase already answered the narrow open robustness question
- no variant exceeded `0.1154`
- adding broader searches would violate the intentionally narrow closure scope

If any follow-up is needed later, it should be a small reproducibility appendix (not another optimization phase).
