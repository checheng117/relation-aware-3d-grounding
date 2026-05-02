# Formal Round-1 Repair Log

**Date**: 2026-04-19

## Fixes and Changes Made for Round-1 Formal Method Validation

---

## 1. Object Embedding Extraction Script

**Issue**: COVER-3D modules need real object embeddings from clean baseline, but current infrastructure only exports base logits.

**Fix**: Created `scripts/extract_clean_baseline_embeddings.py`

**Changes**:
- Loads clean baseline checkpoint (`outputs/20260420_clean_sorted_vocab_baseline/formal/best_model.pt`)
- Uses real object features from `data/object_features/*.npz` when available
- Falls back to center+size+class_hash when object features not found
- Uses real BERT text features from `data/text_features/full_official_nr3d/*.npy`
- Exports per-sample:
  - `object_embeddings` [N, 320]
  - `lang_features` [256]
  - `base_logits` [N] (for verification)
  - `class_indices` [N]

**Command**:
```bash
python scripts/extract_clean_baseline_embeddings.py --split test --device cuda
python scripts/extract_clean_baseline_embeddings.py --split train --device cuda
```

**Output**:
- `outputs/20260420_clean_sorted_vocab_baseline/embeddings/test_embeddings.json`
- `outputs/20260420_clean_sorted_vocab_baseline/embeddings/train_embeddings.json`

---

## 2. COVER-3D Round-1 Training Script

**Issue**: Existing `train_cover3d_referit.py` uses placeholder/hash-based features, not real embeddings.

**Fix**: Created `scripts/train_cover3d_round1.py`

**Changes**:
- Loads extracted embeddings from JSON files
- Merges with coverage diagnostics for hard-subset tagging
- Implements three variants:
  - `base`: No COVER-3D, just baseline anchor
  - `dense-no-cal`: DenseRelationModule only, fixed lambda=0.5
  - `dense-calibrated`: DenseRelationModule + CalibratedFusionGate
- Trains only COVER-3D modules (base model frozen conceptually)
- Evaluates on hard subsets
- Tracks harmed/recovered cases
- Outputs full results

**Command**:
```bash
python scripts/train_cover3d_round1.py --variant base --epochs 0
python scripts/train_cover3d_round1.py --variant dense-no-cal --epochs 10
python scripts/train_cover3d_round1.py --variant dense-calibrated --epochs 10
```

---

## 3. Text Features Path Fix

**Issue**: Extraction script expected `{split}_text_features.npy` but actual files are `{split}_bert_embeddings.npy`.

**Fix**: Updated `load_text_features()` to check multiple naming conventions.

---

## 4. Object Features Integration

**Issue**: Original extraction used hash-based synthetic features.

**Fix**: Added `load_object_features()` and `object_feature_dir` parameter to use real per-scene features from `data/object_features/{scene_id}_features.npz`.

---

## 5. Coverage Merge

**Issue**: Embeddings need hard-subset tags for evaluation.

**Fix**: Added `merge_embeddings_with_coverage()` to combine embeddings with coverage diagnostics.

---

## Files Created

| File | Purpose |
| --- | --- |
| `scripts/extract_clean_baseline_embeddings.py` | Extract real embeddings from clean checkpoint |
| `scripts/train_cover3d_round1.py` | Train COVER-3D with real features |
| `reports/formal_round1_inventory.md` | Inventory of Round-1 dependencies |
| `reports/formal_round1_repair_log.md` | This file |

---

## Files Modified

| File | Change |
| --- | --- |
| `.claude/METHOD_PHASE_FREEZE.md` | New: Phase transition to formal method |
| `.claude/CURRENT_STATUS.md` | Updated: Round-1 starting status |
| `.claude/NEXT_TASK.md` | Updated: Round-1 task freeze |

---

## Apr9 Legacy Check

**Status**: None of the new scripts reference Apr9/Apr10/Apr14 assets.

All paths use:
- Clean 20260420 baseline checkpoint
- Clean logits exports
- Clean class vocabulary
- Full official Nr3D text features
- Scene-disjoint manifests

---

## Remaining Gaps

| Gap | Priority | Status |
| --- | --- | --- |
| Train embeddings extraction | HIGH | In progress (background) |
| Full hard-subset evaluation | MEDIUM | Implemented in training script |
| Casebook generation | MEDIUM | Implemented in training script |

---

## Extraction Results (Test)

| Metric | Value |
| --- | --- |
| Total samples | 4255 |
| Errors | 0 |
| Real object features | 2780 (65.3%) |
| Fallback features | 1475 (34.7%) |
| Object embedding dim | 320 |
| Language feature dim | 256 |

Note: 1475 samples use fallback because not all scenes have pre-extracted object features. This may affect coverage reliability for those samples.