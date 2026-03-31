# Minimal Coadaptation Protocol Note

## Why this is an alternating co-adaptation experiment

This phase changes both halves of the current two-stage stack, but only once and in sequence:

1. shortlist selection is re-aligned to the **first retrained reranker**
2. the reranker is then trained once more on top of that **reselected shortlist**

That is enough to test whether the current ceiling is caused by incomplete alignment between shortlist and reranker, without introducing any larger joint-training framework.

## Why it is intentionally minimal

The current evidence already shows:

- shortlist strengthening is real
- reranker rebalance is real as mismatch recovery
- reranker-only rebalance did not beat the current `0.1090` combined level

So the narrow next question is not whether either component matters in isolation.

It is whether one small amount of **shortlist–reranker co-adaptation** can push the system past the current best.

For that reason this phase uses:

- one shortlist reselection pass
- one second reranker pass
- one corrected combined evaluation

and does **not** introduce loops, new architectures, or broader scope.

## System variants

### A. `current_best_system`

For the co-adaptation line itself, this means:

- improved shortlist from `outputs/20260331_170659_official_shortlist_strengthening/`
- first retrained reranker from `outputs/20260331_173836_rerank_rebalance/`

This is the current adapted shortlist+rerank state entering the alternating round.

The global `0.1090` reference row from the current repo is still:

- `improved_shortlist_plus_reference_rerank`

and it remains in the final combined table as the system-to-beat.

### B. `shortlist_reselected_for_retrained_reranker`

- coarse training / selection uses the official shortlist recipe
- initialization starts from the current improved shortlist checkpoint
- checkpoint selection uses the **first retrained reranker** as the frozen downstream reference

### C. `reranker_retrained_again_on_reselected_shortlist`

- reranker training uses the reselected shortlist checkpoint as `coarse_checkpoint`
- initialization starts from the **first retrained reranker**
- training remains natural-shortlist (`shortlist_train_inject_gold: false`)
- selection is by natural two-stage validation Acc@1

### D. `final_combined_system`

- reselected shortlist
- second retrained reranker

### Optional E. shortlist-only ablation

- reselected shortlist
- first retrained reranker

This is useful because it shows whether shortlist reselection alone already helps before the second reranker pass.

## What counts as success

Primary success criterion:

- `reselected_shortlist_plus_second_retrained_rerank` beats the current `0.1090` combined reference

Secondary success criteria:

- shortlist reselection changes the selected coarse checkpoint or its corrected end-to-end behavior materially
- the second reranker pass improves over the first retrained reranker on the reselected shortlist
- `cond_in_K` improves further under the reselected shortlist regime

If the final combined system does not beat `0.1090`, the experiment is still informative if it shows which half moved more:

- shortlist reselection
- or the second reranker pass
