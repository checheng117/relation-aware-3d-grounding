# Shortlist + rerank upgrade phase — audit and execution plan

## 1. Runnable entrypoints (actual)

| Component | Script / module | Notes |
|-----------|-----------------|-------|
| Coarse (retrieval) training | `scripts/train_coarse_stage1.py` | `forward_coarse`; `CoarseGeomAttributeModel` / `AttributeOnlyModel` |
| Reranker training | `scripts/train_two_stage_rerank.py` | `run_two_stage_training` → `forward_two_stage_rerank` → frozen coarse + trainable `RelationAwareGeomModel` |
| Two-stage eval (natural shortlist) | `src/rag3d/evaluation/two_stage_eval.py` (`eval_two_stage`, `eval_two_stage_bottleneck`) | `target_index=None` at eval ⇒ pure coarse top-K |
| Oracle shortlist at eval | `TwoStageCoarseRerankModel.forward(..., inject_gold_in_shortlist=True)` | Forces gold into top-K while keeping distractors from coarse top-K |
| Shortlist construction | `src/rag3d/relation_reasoner/two_stage_rerank.py` (`_topk_union_target`, `_effective_topk`) | Training: gold injection optional via `shortlist_train_inject_gold` + `inject_gold_in_shortlist` |
| Coarse Recall@K / slices | `src/rag3d/evaluation/coarse_recall.py` (`eval_coarse_stage1_metrics`, `stratified_recall_from_lists`) | Used for retrieval diagnostics |
| Hard negatives (coarse / full) | `src/rag3d/relation_reasoner/losses.py` (`same_class_hinge_loss`, `spatial_nearby_hinge`, `hardest_negative_margin_loss`) | YAML under `loss.*` |
| Extended rerank metrics | `src/rag3d/evaluation/two_stage_rerank_metrics.py` (new) | Acc@1/5, MRR, shortlist recall, conditional rerank Acc |
| Val coarse recall logging | `src/rag3d/training/runner.py` | `TrainingConfig.val_coarse_recall_ks` → logs `val_coarse_recall@K` + clutter/high-load slices when set |
| Prior hybrid / oracle shortlist probe | `scripts/run_hybrid_router_phase.py` | Reference only for bottleneck numbers |

**Not used this phase:** attribute-only line **D**; diagnosis entity/full manifests are absent in default `data/processed/` — regime split uses **candidate_load** on the single val manifest.

## 2. Code reused vs added

**Reused:** `train_coarse_stage1`, `train_two_stage_rerank`, `load_two_stage_model`, `eval_coarse_stage1_metrics`, loss stack, collate/datasets.

**Minimal additions:**

- `TwoStageCoarseRerankModel.forward`: `inject_gold_in_shortlist` override; `forward_two_stage_rerank` respects `shortlist_train_inject_gold`.
- `train_two_stage_rerank.py`: `shortlist_train_inject_gold`, `fine_init_checkpoint` YAML keys.
- `train_coarse_stage1.py`: `--init-checkpoint`, `val_coarse_recall_ks` in YAML.
- `runner.py`: optional val coarse recall metrics each epoch.
- `two_stage_rerank_metrics.py`: eval helpers.
- `scripts/run_shortlist_rerank_upgrade_phase.py`: orchestration, configs, tables, figures.

## 3. Experiment order (mandatory)

1. **Oracle-shortlist reranker** — train fine head with **gold injection on** vs **off** (natural shortlist at train), same frozen baseline coarse; eval each under **natural** and **oracle** inject at test. **K=10** primary (existing stack); other K optional if time.
2. **Shortlist-aware coarse** — finetune coarse from `coarse_geom_recall_last.pt` with recall-oriented losses + **val Recall@{5,10,20,40}** logging.
3. **Rerank on new coarse** — train fine (natural + oracle protocols) with **upgraded** coarse frozen.
4. **Combined eval** — compare baseline pipeline vs upgraded coarse + rerank variants; stratify **candidate_load low vs high** as controlled vs full-scene proxy.

## 4. Expected output directory

`outputs/<timestamp>_shortlist_rerank_upgrade/`

- `generated_configs/*.yaml`
- `checkpoints/*.pt` (references + new saves)
- `logs/train_*.log`
- `oracle_reranker_*`, `shortlist_retrieval_*`, `shortlist_rerank_combined_*`
- `repro_commands.sh`, `report_bundle/README.md`

## 5. Mandatory vs optional

| Mandatory | Optional |
|-----------|----------|
| Plan + dataset protocol note | Curriculum entity→full (document only if skipped) |
| Oracle vs natural **rerank training** + eval grid | K ∈ {5,20,40} extra rerank trains |
| Coarse finetune + recall logging | Mining from saved per-sample B/C dumps |
| Combined table + figure + summary | Extra MRR breakdowns by K |

## 6. Checkpoint naming (this phase)

- `coarse_shortlist_aware_last.pt` — finetuned coarse
- `rerank_train_oracle_last.pt` — fine trained with gold-in-shortlist
- `rerank_train_natural_last.pt` — fine trained without gold injection
- `rerank_newcoarse_oracle_last.pt` / `rerank_newcoarse_natural_last.pt` — fine on upgraded coarse

Baseline references (read-only): `outputs/checkpoints_stage1/coarse_geom_recall_last.pt`, `outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt`.
