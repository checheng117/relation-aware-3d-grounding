# Shortlist-aware hard negative design

## Categories (prioritized)

1. **Same-class distractors** — objects sharing normalized class name with the target; primary source of shortlist confusion. **Implementation:** `same_class_hinge_loss` in `losses.py`, enabled in coarse upgrade YAML (`loss.hard_negative`).
2. **Spatial neighbors** — objects close in 3D to the target (non-target). **Implementation:** `spatial_nearby_hinge_loss` (`loss.spatial_nearby_hinge`).
3. **Hardest non-target** — global max logit among masked negatives vs gold. **Implementation:** `hardest_negative_margin_loss` (`loss.ranking_margin`).
4. **Top coarse false positives (conceptual)** — objects ranked above gold by frozen coarse scorer. **Not mined from disk logs in this phase** (would require a prior dump of per-sample coarse rankings); the ranking hinge approximates pushing gold above the strongest negative.

## Mining and injection

- **Coarse stage:** Negatives are **defined online** from the full candidate tensor and `GroundingSample.objects` (class + center). No separate JSONL mine step.
- **Rerank stage:** Shortlist is a **subset** of objects; CE + optional `hard_negative` in YAML operate on **scattered** full-scene logits (same as existing rerank configs). Oracle-shortlist **training** changes **which K indices** are gathered, not the loss code.

## Shortlist utility alignment

- Retrieval is scored with **Recall@K** and stratified slices (`stratified_recall_from_lists`) logged each validation epoch when `val_coarse_recall_ks` is set.
- Principle: improve **P(gold ∈ top-K)** under clutter/high load, not only coarse Acc@1.
