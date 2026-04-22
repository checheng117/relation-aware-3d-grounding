# Formal Round-1 Inventory

**Date**: 2026-04-19
**Purpose**: Inventory of all assets needed for Round-1 formal COVER-3D method validation.

---

## 1. Clean Baseline Assets

### 1.1 Checkpoint

| Asset | Path | Status |
| --- | --- | --- |
| Clean baseline checkpoint | `outputs/20260420_clean_sorted_vocab_baseline/formal/best_model.pt` | **READY** |
| Final model checkpoint | `outputs/20260420_clean_sorted_vocab_baseline/formal/final_model.pt` | **READY** |
| Class vocabulary | `outputs/20260420_clean_sorted_vocab_baseline/formal/class_vocabulary.json` | **READY** |
| Training history | `outputs/20260420_clean_sorted_vocab_baseline/formal/training_history.json` | **READY** |

Checkpoint contents:
- `model_state_dict` with keys: class_embedding, point_encoder, lang_encoder, fusion, classifier
- `class_vocabulary` (516 classes, sorted ordering)
- `class_to_idx` mapping
- Best epoch: 28
- Val Acc@1: 33.20%, Test Acc@1: 30.83%

### 1.2 Logits and Margins Export

| Asset | Path | Status |
| --- | --- | --- |
| Test predictions with logits | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_predictions.json` | **READY** |
| Test results | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_results.json` | **READY** |
| Val predictions with logits | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_val_predictions.json` | **READY** |
| Val results | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_val_results.json` | **READY** |
| Test readiness summary | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/readiness_test_summary.json` | **READY**, PASS |

Exported fields per sample:
- `base_logits` (list of floats, length N)
- `base_margin`, `base_entropy`
- `base_top1_logit`, `base_top2_logit`
- `base_top1_probability`, `base_top2_probability`
- `target_logit`, `target_probability`, `target_rank`
- `pred_top1`, `pred_top5`, `correct_at_1`, `correct_at_5`

**GAP IDENTIFIED**: Object embeddings and utterance features NOT exported. Need extraction.

### 1.3 Text Features

| Asset | Path | Status |
| --- | --- | --- |
| Train text features | `data/text_features/full_official_nr3d/train_text_features.npy` | **READY** |
| Val text features | `data/text_features/full_official_nr3d/val_text_features.npy` | **READY** |
| Test text features | `data/text_features/full_official_nr3d/test_text_features.npy` | **READY** |

Dimensions: (N_samples, 768) BERT features

---

## 2. COVER-3D Module Assets

### 2.1 Implemented Modules

| Module | Path | Status |
| --- | --- | --- |
| Cover3DModel | `src/rag3d/models/cover3d_model.py` | **IMPLEMENTED**, smoke-pass |
| DenseRelationModule | `src/rag3d/models/cover3d_dense_relation.py` | **IMPLEMENTED**, smoke-pass |
| SoftAnchorPosteriorModule | `src/rag3d/models/cover3d_anchor_posterior.py` | **IMPLEMENTED**, smoke-pass |
| CalibratedFusionGate | `src/rag3d/models/cover3d_calibration.py` | **IMPLEMENTED**, smoke-pass |
| Cover3DWrapper | `src/rag3d/models/cover3d_wrapper.py` | **IMPLEMENTED**, smoke-pass |

Smoke test result: `reports/cover3d_phase2_smoke_diagnostics.json` - PASS (8/8 tests)

### 2.2 Training Infrastructure

| Asset | Path | Status |
| --- | --- | --- |
| Training script | `scripts/train_cover3d_referit.py` | **PARTIAL** - uses placeholder features |
| Phase 3 short config | `configs/cover3d_phase3_short.yaml` | **MISSING** |
| Wrapper config | `configs/cover3d_referit_wrapper.yaml` | **READY** |

**GAP IDENTIFIED**: Current `train_cover3d_referit.py` uses:
- Hash-based placeholder object features (NOT real embeddings)
- Placeholder text features (NOT real BERT)
- Placeholder base classifier (NOT real ReferIt3DNet logits)

For formal Round-1, need to:
1. Create feature extraction script to export object embeddings
2. Modify training script to use real features and real base logits
3. Or create inference-only reranking approach

---

## 3. Evaluation and Diagnostics Assets

### 3.1 Evaluation Scripts

| Asset | Path | Status |
| --- | --- | --- |
| Coverage diagnostics | `scripts/run_cover3d_coverage_diagnostics.py` | **READY** |
| P3 minimal verification | `scripts/run_cover3d_p3_minimal_verification.py` | **READY** |
| Real logits P3 entry | `scripts/run_cover3d_real_logits_p3_entry.py` | **READY** (proxy) |
| Phase 2 smoke test | `scripts/smoke_test_cover3d_phase2.py` | **READY** |

### 3.2 Diagnostics Reports (Pre-method)

| Report | Path | Status |
| --- | --- | --- |
| Coverage diagnostics | `reports/pre_method_clean_coverage_diagnostics/` | **READY** |
| P3 minimal | `reports/pre_method_clean_p3_minimal/` | **READY** |
| Real logits P3 entry | `reports/pre_method_clean_real_logits_p3/` | **READY** |
| Phase 1 findings | `reports/cover3d_phase1_findings.md` | **READY** |
| Phase 1 hard subsets | `reports/cover3d_phase1_hard_subsets.md` | **READY** |

---

## 4. Data Assets

### 4.1 Scene-Disjoint Manifests

| Asset | Path | Status |
| --- | --- | --- |
| Train manifest | `data/processed/scene_disjoint/official_scene_disjoint/train_manifest.jsonl` | **READY** |
| Val manifest | `data/processed/scene_disjoint/official_scene_disjoint/val_manifest.jsonl` | **READY** |
| Test manifest | `data/processed/scene_disjoint/official_scene_disjoint/test_manifest.jsonl` | **READY** |

### 4.2 Geometry Data

| Asset | Path | Status |
| --- | --- | --- |
| Geometry directory | `data/geometry/*.npz` | **READY** |

Each file contains:
- `object_ids`: object identifiers
- `centers`: [N, 3] center positions
- `sizes`: [N, 3] bounding box sizes

---

## 5. Critical Gaps

| Gap | Severity | Resolution |
| --- | --- | --- |
| **Object embeddings not exported** | HIGH | Create extraction script |
| **Training script uses placeholders** | HIGH | Modify to use real features |
| **No hard-subset evaluation for COVER-3D** | MEDIUM | Extend evaluation scripts |
| **No harmed/recovered tracking for learned model** | MEDIUM | Add to evaluation |

---

## 6. Resolution Plan

### Priority 1: Extract Object Embeddings

Create script to:
1. Load clean baseline checkpoint
2. Run inference on train/val/test splits
3. Export:
   - `object_embeddings` [N, 320] per sample
   - `lang_features` [256] per sample (or use existing text features)
   - Match with existing `base_logits` exports

### Priority 2: Create Formal Training Script

Options:
- **Option A**: Train COVER-3D as post-hoc reranker (uses pre-extracted features)
- **Option B**: Train end-to-end with frozen base (requires full inference each batch)

Recommend **Option A** for Round-1 minimal validation:
- Pre-extract all features once
- Train only DenseRelationModule and CalibratedFusionGate
- Faster iteration, easier debugging

### Priority 3: Create Hard-Subset Evaluation

Extend existing scripts to:
1. Load COVER-3D predictions
2. Compute hard subset metrics (clutter, multi-anchor, etc.)
3. Track harmed/recovered cases
4. Output casebook for analysis

---

## 7. Apr9 Legacy Check

| Asset | Path | Current Usage | Action |
| --- | --- | --- | --- |
| Apr9 checkpoint | outputs/... | NOT in current paths | None needed |
| Apr9 logits | outputs/... | NOT in current paths | None needed |
| Apr14 vocab-fix | outputs/... | NOT in current paths | None needed |

**Confirmed**: Current configs and scripts do NOT reference Apr9/Apr10/Apr14 assets. All use clean 20260420 baseline.

---

## 8. Summary

| Category | Status |
| --- | --- |
| Clean baseline checkpoint | **READY** |
| Clean baseline logits export | **READY** |
| Class vocabulary | **READY** |
| Text features | **READY** |
| COVER-3D modules (implementation) | **READY** |
| COVER-3D modules (trained weights) | **MISSING** |
| Object embeddings export | **MISSING** |
| Formal training script | **PARTIAL** (needs fix) |
| Hard-subset evaluation for COVER-3D | **MISSING** |
| Apr9 legacy contamination | **NONE FOUND** |

**Next Steps**:
1. Create object embedding extraction script
2. Create formal COVER-3D training script using real features
3. Run Base / Dense-no-cal / Dense-calibrated experiments
4. Generate round-1 reports