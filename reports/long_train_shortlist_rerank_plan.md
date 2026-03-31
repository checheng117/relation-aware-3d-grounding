# Long-train shortlist + rerank phase — audit and execution plan

This document maps **actual repository entrypoints** to the three pillars (longer rerank, correct checkpoint selection, focused coarse hard negatives) and the combined evaluation. **Phase D (geometry modeling)** and **hybrid router** are explicitly out of scope for this phase.

---

## 1. Entrypoints (scripts / modules)

| Capability | Script or module | Notes |
|------------|------------------|--------|
| **Coarse (stage-1) training** | `scripts/train_coarse_stage1.py` | Loads YAML; `forward_coarse`; `run_training_loop` in `src/rag3d/training/runner.py`. |
| **Reranker (two-stage fine) training** | `scripts/train_two_stage_rerank.py` | Frozen coarse + `TwoStageCoarseRerankModel`; `forward_two_stage_rerank`. |
| **Natural two-stage evaluation (full-scene logits)** | `src/rag3d/evaluation/two_stage_eval.py` (`eval_two_stage`) | Used by `scripts/eval_rerank_blueprint.py`. |
| **Oracle vs natural shortlist + MRR + conditional Acc** | `src/rag3d/evaluation/two_stage_rerank_metrics.py` (`eval_two_stage_inject_mode`, `eval_by_candidate_load_bucket`) | `inject_gold_in_shortlist` toggles oracle shortlist. |
| **Coarse-only Recall@K + slices** | `src/rag3d/evaluation/coarse_recall.py` (`eval_coarse_stage1_metrics`) | Called from runner when `val_coarse_recall_ks` set; used in phase eval. |
| **Checkpoint selection (NEW)** | `src/rag3d/training/checkpoint_selection.py` | Coarse: `evaluate_coarse_with_fixed_rerank` + YAML `val_two_stage_selection`. Rerank: best ckpt in `train_two_stage_rerank.py`. |
| **Shortlist construction / gold injection** | `src/rag3d/relation_reasoner/two_stage_rerank.py` | `_topk_union_target`, `shortlist_train_inject_gold`, optional `freeze_coarse` for selection wrapper. |
| **Hard-negative-style losses (coarse)** | `src/rag3d/relation_reasoner/losses.py` | `ranking_margin` (hardest non-gold), `spatial_nearby_hinge`, `hard_negative` (same-class hinge). |
| **Orchestrated longtrain phase** | `scripts/run_longtrain_shortlist_rerank_phase.py` | Timestamped `outputs/<stamp>_longtrain_shortlist_rerank/`, repro script, report bundle. |
| **Prior short upgrade phase (reference)** | `scripts/run_shortlist_rerank_upgrade_phase.py` | Shorter default epochs; same eval ideas. |
| **Legacy coarse→rerank promotion (recall-sorted)** | `scripts/promote_coarse_opt_rerank.py` | Sorts coarse candidates by `recall@20`; **not** the primary selection metric for this phase. |

---

## 2. Reused code paths (no duplicate frameworks)

- **Training loops:** `runner.run_training_loop` (coarse); custom loop in `train_two_stage_rerank.py` (rerank — unchanged structure, extended metrics).
- **Data:** `ReferIt3DManifestDataset` + `make_grounding_collate_fn`; `configs/dataset/referit3d.yaml` → `data/processed/*_manifest.jsonl`.
- **Eval:** `load_two_stage_model`, `eval_two_stage_inject_mode`, `eval_coarse_stage1_metrics` (same as upgrade phase).

---

## 3. Minimal code changes (implemented)

1. **`TwoStageCoarseRerankModel`**: `freeze_coarse: bool = True` so a **training** coarse module can be wrapped for val **without** forcing `requires_grad=False` on the live trainer module.
2. **`runner.py`**: Optional `coarse_pipeline_selection` — each epoch logs **natural two-stage val Acc@1** with a **frozen reference reranker**; saves `{run_name}_best_pipeline_natural.pt`.
3. **`train_two_stage_rerank.py`**: Each epoch logs natural + oracle shortlist val metrics; **primary** selection file `{run_name}_best_natural_two_stage.pt` by **natural** two-stage Acc@1.
4. **`train_coarse_stage1.py`**: Parses `val_two_stage_selection` from YAML into `TrainingConfig`.
5. **New module** `checkpoint_selection.py` for coarse+fixed-rerank val metrics.

---

## 4. Experiment order (mandatory)

1. **Audit** (this file) + `reports/checkpoint_selection_note.md` + `reports/focused_shortlist_hard_negative_note.md`.
2. **Fix / verify checkpoint selection** (code above + notes).
3. **Long reranker training**
   - **Protocol N** (natural shortlist train): generated `rerank_longtrain_natural.yaml`.
   - **Protocol O** (oracle shortlist train): generated `rerank_longtrain_oracle.yaml`.
   - **Protocol M** (mixed): **skipped** in orchestrator (would need per-sample oracle mix probability in forward; documented in JSON).
4. **Focused coarse retrieval** — train with three loss terms only + `val_coarse_recall_ks` + pipeline selection YAML.
5. **Combined eval** — rows A–D on **val** (same convention as upgrade phase).
6. **Summary + README** + `report_bundle/README.md` + `repro_commands.sh`.

---

## 5. Expected output directory layout

All under `outputs/<timestamp>_longtrain_shortlist_rerank/`:

| Path | Content |
|------|---------|
| `generated_configs/` | Rerank N/O + coarse focused YAML actually run |
| `checkpoints/` | Epoch + `last` + `best_*` artifacts |
| `logs/` | Training stdout |
| `metrics_*.jsonl` | Per-epoch metrics |
| `oracle_reranker_*_longtrain*` | Rerank tables + JSON + curves PNG + interpretation |
| `shortlist_retrieval_*_longtrain*` | Retrieval JSON/CSV/MD |
| `shortlist_rerank_combined_*_longtrain*` | Combined JSON/CSV/MD + figure |
| `repro_commands.sh` | One-shot re-run |
| `report_bundle/README.md` | Claim ↔ artifact map |

Repository-level templates (optional hand-runs):  
`configs/train/rerank/rerank_longtrain_{natural,oracle}.yaml`,  
`configs/train/coarse/coarse_focused_hardneg_longtrain.yaml`.

---

## 6. Metrics that drive model selection

| Stage | Primary | Secondary (logged) |
|-------|---------|---------------------|
| **Reranker** | **Natural two-stage val Acc@1** (`val_natural_two_stage_acc@1`) | Oracle val Acc@1, conditional in-K, MRR, train loss |
| **Coarse (when selection enabled)** | **Natural two-stage val Acc@1** (`val_pipeline_natural_acc@1`) | Recall@5/10/20/40, coarse val acc, oracle pipeline Acc@1 |

---

## 7. Mandatory vs optional

| Item | Status |
|------|--------|
| Checkpoint selection aligned to natural two-stage | **Mandatory** |
| Longer rerank N and O | **Mandatory** |
| Focused three-term coarse training | **Mandatory** |
| Combined A–D eval | **Mandatory** |
| Protocol M mixed training | **Optional — skipped** (documented) |
| Entity-controlled regime eval | **Optional** (needs separate eval config/manifest; not added here) |
| Changing `promote_coarse_opt_rerank.py` | **Optional** (legacy recall sort; noted in docs only) |

---

## 8. Prerequisites to execute

- `data/processed/train_manifest.jsonl`, `val_manifest.jsonl`
- Baseline **coarse** and **rerank** checkpoints (e.g. `outputs/checkpoints_stage1/coarse_geom_recall_last.pt`, `outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt`) — phase script errors early if missing.

---

## 9. Command

```bash
cd /path/to/relation-aware-3d-grounding
PYTHONPATH=src python scripts/run_longtrain_shortlist_rerank_phase.py --device cuda --epochs-rerank 12 --epochs-coarse 8
```

Eval-only (after training into the same stamp):

```bash
PYTHONPATH=src python scripts/run_longtrain_shortlist_rerank_phase.py --stamp <timestamp>_longtrain_shortlist_rerank --skip-train
```

(Use the **directory name suffix** after `outputs/` as `--stamp` if you pass only the timestamp token; the script expects `outputs/{stamp}_longtrain_shortlist_rerank` — see script: `out = outputs / f"{stamp}_longtrain_shortlist_rerank"`. So `--stamp` should be the **date part only**, e.g. `20260327_120000`, not the full folder name.)
