# Hybrid router experiment — repository audit

## 1. What exact existing artifacts can be reused?

| Artifact | Location / pattern | Role |
|----------|-------------------|------|
| **B (raw-text relation) checkpoints** | `outputs/checkpoints/raw_relation_last.pt` (when trained); this workspace also has `outputs/checkpoints_nr3d_geom_first/raw_relation_last.pt`, `outputs/checkpoints_nr3d_warmup/raw_relation_last.pt`, diagnosis runs under `outputs/checkpoints_diagnosis/`, and multi-seed `outputs/20260327_135641_full_train_official/np_*_raw_s*_last.pt` | Frozen scorer for B logits |
| **C (relation-aware) checkpoints** | Same layout with `relation_aware` / `np_*_structured` naming; e.g. `outputs/checkpoints_nr3d_geom_first/relation_aware_last.pt` | Frozen scorer for C logits + anchor distribution |
| **Debug checkpoints** | `outputs/checkpoints/*_debug_last.pt` | Small-manifest sanity runs |
| **Two-stage stack** | Coarse: `outputs/checkpoints_stage1/coarse_geom_recall_last.pt`; full two-stage state: `outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt` (includes `coarse.*` + `fine.*`) | Shortlist + rerank; oracle-shortlist evaluation |
| **Rerank variants** | `outputs/checkpoints_rerank/rerank_full_k10_last.pt`, `outputs/checkpoints_stage1_rerank/rerank_k{5,10,20}_stage1_last.pt` | Alternative frozen rerankers if needed |
| **Manifests** | `data/processed/train_manifest.jsonl`, `val_manifest.jsonl` (from `configs/dataset/referit3d.yaml` `processed_dir`) | Train = router fit, val = held-out metrics |
| **Stratified / bottleneck code** | `src/rag3d/evaluation/stratified_eval.py`, `shortlist_bottleneck.py`, `two_stage_eval.py` | Slices, geometry tags, two-stage metrics |

**Not found as a single dump:** per-sample B/C logits from past `eval_all.py` runs (that script only writes aggregate JSON). For hybrid training, logits must be **recomputed** once from frozen checkpoints or saved by a new script.

## 2. What per-sample signals are already available?

| Signal | Source |
|--------|--------|
| **Logits / Acc@1** | Forward pass of `RawTextRelationModel` / `RelationAwareModel` (`scripts/eval_all.py` pattern) |
| **Parser confidence** | `ParsedUtterance.parser_confidence` from `StructuredRuleParser` / `CachedParser` |
| **Anchor entropy** | `RelationAwareModel` returns `p_anchor`; `rag3d.diagnostics.confidence.anchor_entropy` |
| **Target / top-2 margin** | `logit_top12_margin` in `metrics.py` (used via `augment_meta_with_model_margins`) |
| **Candidate set size** | `meta["n_objects"]` or `tags["n_objects"]` from collate / stratification |
| **Geometry quality proxy** | `SceneObject.geometry_quality`, `geometry_fallback_fraction`, `tags["geometry_high_fallback"]` via `augment_meta_geometry_fallback_tags` |
| **Same-class clutter** | `tags["same_class_clutter"]` from `compute_stratification_tags` / manifest |
| **Candidate load regime** | `tags["candidate_load"]` ∈ {low, medium, high} from `transforms.py` |
| **Relation type / density** | `relation_type_gold` in meta; density proxy = `len(parsed.relation_types)` (+ non-trivial anchor head) from structured parse |
| **Shortlist / recall proxy** | Not in historical metrics JSON; computable by running **coarse** forward (e.g. `coarse_forward` / `CoarseGeomAttributeModel`) and testing whether `target_index` lies in top-K indices (same K as rerank) |

## 3. What minimal new extraction code is required?

One driver script (`scripts/run_hybrid_router_phase.py`) that:

1. Builds `DataLoader`s for train/val manifests with `make_grounding_collate_fn`.
2. Loads frozen B and C (paths CLI-configurable).
3. For each sample, stores **B/C logits** (or sufficient statistics), **parser confidence**, **anchor entropy**, **margins**, **geometry fallback fraction**, **clutter / candidate_load flags**, **coarse top-K hit** for rerank K, and correctness flags for B and C.
4. For **oracle shortlist**, reuses `TwoStageCoarseRerankModel` internals with `_topk_union_target(..., training=True)` so the gold index is **forced** into the shortlist while the fine module stays in **eval** mode (no dropout path).
5. Trains a **small \(\alpha(x)\)** router (linear + sigmoid) on train logits/features with CE on fused logits; evaluates on val.

No changes to model D (attribute-only line) are required for this phase.

## 4. Cleanest train/val protocol for the router experiment

- **Split:** Router parameters are fit only on **train** manifest rows; all reported generalization metrics (B, C, fusion, oracle) on **val**.
- **Leakage:** Scalar normalization (mean/std of router features) is computed on **train** only, then applied to val.
- **Frozen B/C:** No backprop through B or C; only router weights update.
- **Oracle metrics:** Computed on **val** (and optionally train for debugging); they are **upper bounds**, not trainable targets.
- **Two-stage oracle shortlist:** Uses the **same** val loader and frozen two-stage checkpoint as the rest of the shortlist diagnostics for consistency.

This matches standard practice for **stacked / gating** probes on top of frozen base models.
