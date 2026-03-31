# Official Shortlist Hard-Negative Note

## Exact mining rules used in the current repo

The official shortlist-strengthening run keeps the existing three focused negative families from `src/rag3d/relation_reasoner/losses.py`:

1. `same_class_hinge_loss`
   - for each training row, find the highest-logit valid non-gold object whose normalized class name matches the gold class
   - apply a hinge penalty if that same-class confuser scores too close to or above the gold

2. `spatial_nearby_hinge_loss`
   - for each row, compute 3D center distance from the gold object to other valid objects
   - keep only the nearest `max_neighbors`
   - apply the hinge against the highest-logit object inside that local neighbor set

3. `hardest_negative_margin_loss`
   - for each row, take the highest-logit valid non-gold candidate overall
   - apply a hinge penalty against that current top-ranked false positive

## Sampling / weighting design

The run does **not** create a separate mined-negative minibatch and does **not** replace the ordinary retrieval objective.

Instead, the training loss is:

- full masked cross-entropy over all valid candidates
- plus a focused ranking-margin term
- plus a focused local-spatial hinge
- plus a focused same-class hinge

Official config from `outputs/20260331_170659_official_shortlist_strengthening/generated_configs/coarse_official_shortlist_strengthening.yaml`:

- `ranking_margin`: `margin=0.20`, `lambda=0.25`
- `spatial_nearby_hinge`: `margin=0.20`, `lambda=0.15`, `max_neighbors=3`
- `hard_negative` / same-class hinge: `margin=0.25`, `lambda_hinge=0.35`

Important exclusions:

- no candidate-load weighting in the loss
- no extra negative families beyond the three focused ones
- no parser-conditioned anchor-mining rewrite in this phase

## Verification against the requested priority negatives

1. **Same-class negatives**
   - directly implemented by `same_class_hinge_loss`

2. **Near-anchor but wrong-target negatives**
   - the current coarse training path does **not** have a true anchor-conditioned negative miner
   - the nearest existing focused spatial confuser is `spatial_nearby_hinge_loss`, which is target-local rather than anchor-local
   - this phase deliberately reuses that minimal spatial proxy instead of widening scope into parser-aware coarse mining

3. **Current coarse top-ranked false positives**
   - directly implemented by `hardest_negative_margin_loss`

## What changed relative to the prior long-train shortlist phase

Compared with the previous focused long-train coarse recipe in `scripts/run_longtrain_shortlist_rerank_phase.py`, the official run makes only narrow rebalancing changes:

- `ranking_margin.lambda`: `0.20 -> 0.25`
- `spatial_nearby_hinge.max_neighbors`: `4 -> 3`
- same-class `margin`: `0.30 -> 0.25`
- same-class `lambda_hinge`: `0.75 -> 0.35`

The intent was to keep the same interpretable three-part design while reducing over-dominance from the same-class hinge and shifting a bit more pressure onto the actually top-ranked false positive.

## Why this design is expected to help end-to-end natural performance

This phase is trying to improve shortlist quality for the corrected two-stage objective, not to maximize a coarse-only proxy in isolation.

The chosen design is therefore:

- focused enough to attack the shortlist failure modes that matter most
- conservative enough to avoid an uncontrolled negative soup
- selection-aligned through `val_pipeline_natural_acc@1` with March 31 `O_best`

In practice, this official run did improve shortlist coverage materially:

- Recall@10: `0.2821 -> 0.5833`
- Recall@20: `0.5513 -> 0.7821`
- high-load Recall@10: `0.2348 -> 0.5303`

So the targeted shortlist-strengthening recipe worked as a retrieval intervention, even though reranker compatibility on the new shortlist remains the next limiting issue.
