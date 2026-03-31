# Official Shortlist Strengthening Summary

## What was implemented

This phase added a narrow official shortlist-strengthening pass on top of the corrected March 31 pipeline:

- one timestamped orchestration entrypoint: `scripts/run_official_shortlist_strengthening_phase.py`
- coarse checkpoint selection aligned to corrected natural two-stage validation `Acc@1`
- a focused three-part shortlist hard-negative recipe
- corrected combined evaluation against baseline, `rerank_O_best`, and improved-shortlist pipelines

Authoritative output directory:

- `outputs/20260331_170659_official_shortlist_strengthening/`

The official run was executed with:

- `PYTHONHASHSEED=42`

so that training and evaluation used one consistent text-hash mapping.

## Corrected assets reused from the fix phase

Primary reused assets from `outputs/20260331_160556_fix_combined_nloss/`:

- `checkpoints/rerank_longtrain_oracle_best_natural_two_stage.pt`
  - reused as the **main shortlist-selection reference**
  - reused as the **main reranker** in corrected combined evaluation

- `checkpoints/rerank_longtrain_natural_best_natural_two_stage.pt`
  - reused for the auxiliary `N_best` combined row

- corrected combined-evaluation loading path
  - `src/rag3d/evaluation/two_stage_eval.py`
  - `fine_only_from_checkpoint=True`

## Did shortlist quality improve?

Yes, materially.

From `shortlist_retrieval_table_official.csv`:

- Recall@5: `0.1282 -> 0.4038`
- Recall@10: `0.2821 -> 0.5833`
- Recall@20: `0.5513 -> 0.7821`
- Recall@40: `0.9167 -> 0.9679`
- candidate-load-high Recall@10: `0.2348 -> 0.5303`

So shortlist quality improved well beyond noise-level movement.

## Did corrected end-to-end natural Acc@1 improve?

Yes.

Main corrected comparison with reused `O_best`:

- `rerank_O_best`: `0.0577`
- `improved_shortlist_plus_rerank_O_best`: `0.0897`
- delta: `+0.0321`

The stronger shortlist also improved the baseline/reference rerank path:

- `baseline_reference`: `0.0513`
- `improved_shortlist_plus_reference_rerank`: `0.1090`
- delta: `+0.0577`

So shortlist strengthening is now working under the corrected end-to-end objective, not only on Recall@K.

## Did `rerank_O_best` benefit more from the stronger shortlist?

Yes in absolute end-to-end terms, but not as cleanly as hoped.

Positive result:

- `rerank_O_best` alone improved from `0.0577` to `0.0897` when paired with the stronger shortlist

Constraint:

- `improved_shortlist_plus_reference_rerank` reached `0.1090`, which is **higher** than `improved_shortlist_plus_rerank_O_best` at `0.0897`
- with `O_best`, conditional accuracy given gold in shortlist fell from `0.2045` to `0.1538`
- with `O_best`, oracle upper bound fell from `0.4295` to `0.1667`

That means the stronger shortlist helped final accuracy, but the old `O_best` reranker is not ideally matched to the new shortlist distribution.

## New dominant bottleneck after this phase

Before this run, shortlist quality was clearly the main blocker.

After this run:

- shortlist recall is much better, but still incomplete at `Recall@10 = 0.5833`
- the sharper issue is now **reranker compatibility / conditional quality on the improved shortlist**

The evidence is that shortlist coverage rose strongly while the reused `O_best` reranker became less effective **once the gold was inside the shortlist**.

So the bottleneck has shifted from "shortlist quality dominates everything" toward a more balanced picture:

- shortlist quality is still not solved
- reranker alignment to the healthier shortlist is now comparatively more limiting

## Next best action

The next best action is:

- **re-balance retrieval vs reranking**

Concretely:

- keep the improved shortlist recipe
- retrain or reselect the reranker against that healthier shortlist under the same corrected natural two-stage objective
- do **not** widen scope yet to hybrid routing or phase D

This is more justified than continuing pure retrieval-only work immediately, because the official run already showed large recall gains but only partial translation through reused `O_best`.
