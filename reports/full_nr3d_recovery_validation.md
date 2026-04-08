# Full Nr3D Recovery Validation

**Date**: 2026-04-08
**Phase**: Full Nr3D Dataset Recovery - Step 5

---

## Executive Summary

**Status**: VALIDATED ✓

Recovered dataset is valid with zero scene overlap between splits.

---

## 1. Recovery Statistics

| Metric | Value |
|--------|-------|
| Total recovered samples | 23,186 |
| Unique scenes | 269 |
| Train samples | 18,459 |
| Val samples | 2,046 |
| Test samples | 2,681 |

---

## 2. Scene Overlap Verification

| Split Pair | Overlapping Scenes | Status |
|------------|-------------------|--------|
| Train-Val | 0 | ✓ PASS |
| Train-Test | 0 | ✓ PASS |
| Val-Test | 0 | ✓ PASS |

**Result**: Zero scene overlap confirmed. Scene-disjoint splits are valid.

---

## 3. Sample Validity Checks

### Target-Object Validity

| Check | Result |
|-------|--------|
| All targets found in scene objects | ✓ PASS |
| Target index in valid range | ✓ PASS |

### Duplicate Detection

| Check | Result |
|-------|--------|
| No duplicate samples across splits | ✓ PASS |
| No duplicate utterance_ids within splits | ✓ PASS |

---

## 4. Skipped Samples Analysis

| Reason | Count | Percentage |
|--------|-------|------------|
| Scene not in aggregation pool | 18,317 | 44.1% |
| Invalid metadata | 0 | 0% |
| Target not in scene objects | 0 | 0% |

**Total skipped**: 18,317 samples (scenes without aggregation files)

---

## 5. Coverage Analysis

### Dataset Coverage

| Metric | Official | Recovered | Coverage |
|--------|----------|-----------|----------|
| Samples | 41,503 | 23,186 | 55.9% |
| Scenes | 641 | 269 | 42.0% |

### Split Distribution

| Split | Samples | % of Recovered | % of Official |
|-------|---------|----------------|---------------|
| Train | 18,459 | 79.6% | 44.5% |
| Val | 2,046 | 8.8% | 4.9% |
| Test | 2,681 | 11.6% | 6.5% |

---

## 6. Geometry Quality

| Metric | Value |
|--------|-------|
| Geometry source | ScanNet aggregation JSON |
| Geometry quality | aggregation (not point-based) |
| Real point clouds | No (requires geometry regeneration) |

**Note**: Using aggregation geometry (placeholder centers/sizes from ScanNet). Real point-based geometry would require regenerating geometry files with correct object ID mapping.

---

## 7. Files Generated

| File | Purpose |
|------|---------|
| `data/processed/scene_disjoint/official_scene_disjoint/train_manifest.jsonl` | Train samples |
| `data/processed/scene_disjoint/official_scene_disjoint/val_manifest.jsonl` | Val samples |
| `data/processed/scene_disjoint/official_scene_disjoint/test_manifest.jsonl` | Test samples |
| `data/processed/scene_disjoint/official_scene_disjoint/dataset_summary.json` | Summary stats |
| `data/splits/official_scene_disjoint_train.txt` | Train scene list |
| `data/splits/official_scene_disjoint_val.txt` | Val scene list |
| `data/splits/official_scene_disjoint_test.txt` | Test scene list |

---

## 8. Comparison to Old Subset

| Metric | Old Subset | Recovered | Increase |
|--------|------------|-----------|----------|
| Total samples | 1,544 | 23,186 | **15.0x** |
| Scenes | 266 | 269 | +3 |
| Train samples | 1,211 | 18,459 | **15.2x** |
| Val samples | 148 | 2,046 | **13.8x** |
| Test samples | 185 | 2,681 | **14.5x** |

---

## 9. Validation Commands

```bash
# Verify manifest counts
wc -l data/processed/scene_disjoint/official_scene_disjoint/*.jsonl

# Verify scene overlap
python scripts/validate_scene_disjoint_splits.py \
    --manifest-dir data/processed/scene_disjoint/official_scene_disjoint

# Load and check first sample
python -c "
from rag3d.datasets.referit3d import ReferIt3DManifestDataset
ds = ReferIt3DManifestDataset('data/processed/scene_disjoint/official_scene_disjoint/train_manifest.jsonl')
print(f'Loaded {len(ds)} samples')
s = ds[0]
print(f'Sample: scene={s.scene_id}, utterance={s.utterance[:50]}...')
"
```

---

## 10. Conclusion

**Recovered dataset is valid and ready for use.**

- Zero scene overlap between splits
- All samples have valid target mappings
- 15x increase in sample count
- 55.9% coverage of official Nr3D

**Ready for**: Trustworthy baseline rerun with recovered dataset.