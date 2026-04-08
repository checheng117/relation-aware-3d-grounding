# Full Nr3D Recovery Audit

**Date**: 2026-04-07
**Phase**: Full Nr3D Dataset Recovery - Step 0

---

## Executive Summary

**Root Cause Identified**: The Hugging Face dataset `chouss/nr3d` is a **subset** (1,569 samples) of the official Nr3D dataset (41,503 samples). The official full dataset is available via Google Drive from the ReferIt3D website.

---

## 1. Where the Current 1,544 Samples Come From

### Data Pipeline

```
Official Nr3D (41,503 samples)
       ↓
Hugging Face `chouss/nr3d` (1,569 samples - SUBSET)
       ↓
nr3d_annotations.json (1,569 rows)
       ↓
build_records_nr3d_hf_with_scans() → 1,544 records (25 skipped)
       ↓
split_records_by_scene() → scene-disjoint manifests
       ↓
Train: 1,211 / Val: 148 / Test: 185
```

### Key Files

| File | Count | Source |
|------|-------|--------|
| `data/raw/referit3d/annotations/nr3d_annotations.json` | 1,569 rows | Hugging Face `chouss/nr3d` |
| `data/raw/referit3d/annotations/train.csv` | 1,255 rows | Derived from HF |
| `data/raw/referit3d/annotations/val.csv` | 156 rows | Derived from HF |
| `data/raw/referit3d/annotations/test.csv` | 158 rows | Derived from HF |

---

## 2. What Defines the Current Dataset Boundary

### Loader Script

**File**: `src/rag3d/datasets/nr3d_hf.py`

**Function**: `records_from_nr3d_hf_json()` reads from `nr3d_annotations.json`

### Builder Script

**File**: `scripts/prepare_data.py`

**Mode**: `--mode build-nr3d-geom-scene-disjoint`

**Key Function**: `build_records_nr3d_hf_with_scans()` in `src/rag3d/datasets/builder.py`

### Filtering Logic

| Stage | Input | Output | Skipped | Reason |
|-------|-------|--------|---------|--------|
| NR3D HF JSON | 1,569 rows | - | - | - |
| parse_nr3d_row_meta() | 1,569 | 1,569 | 0 | All rows parse |
| load_scene_objects_with_geometry() | 1,569 | 1,544 | 25 | Target not in scene objects |
| split_records_by_scene() | 1,544 | 1,544 | 0 | All records kept |

---

## 3. Where the Official 41,503 Figure Comes From

### Official Source

**Website**: [ReferIt3D](https://referit3d.github.io/)

**Download**: [Nr3D Google Drive](https://drive.google.com/file/d/1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC/view) (10.7 MB)

### Official Dataset Statistics

| Dataset | Samples | Source |
|---------|---------|--------|
| **Nr3D** | 41,503 | Natural human utterances |
| Sr3D | 34,519 | Template-based utterances |
| Sr3D+ | 83,573 | Augmented templates |

### Scene Coverage

| Dataset | Scenes |
|---------|--------|
| Nr3D | 707 ScanNet scenes |
| Our current subset | 269 scenes |

---

## 4. Why We Only Have 1,569 Samples

### Root Cause

The Hugging Face dataset `chouss/nr3d` is **not the full Nr3D dataset**. It appears to be:
- A test set subset
- A filtered sample
- An incomplete upload

### Evidence

1. **Sample count mismatch**: 1,569 vs 41,503
2. **Scene count mismatch**: 269 vs 707
3. **Download size**: HF download is ~900KB vs official 10.7MB

### Why This Happened

The project originally used the Hugging Face dataset for convenience, assuming it was the full dataset. This was a mistaken assumption.

---

## 5. Cleanest Path to Recover More Samples

### Option A: Download Official Nr3D (RECOMMENDED)

**Source**: https://drive.google.com/file/d/1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC/view

**Steps**:
1. Download official Nr3D JSON (10.7 MB)
2. Verify sample count (should be ~41,503)
3. Update loader to use official format
4. Rebuild manifests

**Estimated Recovery**: 41,503 samples (full dataset)

### Option B: Check Sr3D Dataset

**Source**: https://drive.google.com/drive/folders/1DS4uQq7fCmbJHeE-rEbO8G1-XatGEqNV

Sr3D has 34,519 template-based utterances. Could augment training data.

**Estimated Recovery**: +34,519 samples (if compatible)

### Option C: Alternative Hugging Face Datasets

Search for other Hugging Face datasets with full Nr3D coverage.

**Estimated Recovery**: Unknown

---

## 6. Required Changes to Builder

### Current Builder Limitations

The current builder (`build_records_nr3d_hf_with_scans`) expects Hugging Face format:
- Single JSON array with specific fields
- `unique_id`, `scene_id`, `object_id`, `descriptions`

### Official Nr3D Format

Need to verify official format differs from HF format. May require:
- New parsing function
- Updated field mapping
- Scene split verification

---

## 7. Scene Geometry Requirements

### Current Geometry Status

| Requirement | Status |
|-------------|--------|
| ScanNet aggregation files | ✅ 269 scenes available |
| Real geometry (Pointcept) | ✅ Available in `data/geometry/` |
| Object features | ✅ Sparse features in manifests |

### Geometry Bottleneck

The 269 scenes with geometry may limit usable samples from official Nr3D.

**Question**: Do all 707 Nr3D scenes have ScanNet geometry available?

---

## 8. Estimated Recovery Potential

| Scenario | Estimated Samples | Confidence |
|----------|-------------------|------------|
| Full Nr3D + all 707 scenes | 41,503 | High (data exists) |
| Full Nr3D + current 269 scenes | ~15,000 | Medium (scene overlap) |
| Current subset only | 1,544 | Current state |

---

## 9. Recommended Next Steps

### Priority 1: Download Official Nr3D

```bash
# Download from Google Drive
# https://drive.google.com/file/d/1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC/view

# Expected file: nr3d.csv or similar
# Verify: wc -l nr3d.csv should show ~41,503 lines
```

### Priority 2: Inspect Official Format

```python
# Check format and fields
import pandas as pd
df = pd.read_csv('nr3d.csv')
print(f'Samples: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Scenes: {df["scene_id"].nunique()}')
```

### Priority 3: Update Builder

Create new builder mode for official Nr3D format if different from HF.

---

## 10. Conclusion

**The missing ~40,000 samples are due to using a Hugging Face subset instead of the official Nr3D dataset.**

**Solution**: Download the official Nr3D dataset from Google Drive.

**Blocking Issue**: None - the full dataset is publicly available.

---

## References

- [ReferIt3D Website](https://referit3d.github.io/)
- [Nr3D Download](https://drive.google.com/file/d/1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC/view)
- [Sr3D Download](https://drive.google.com/drive/folders/1DS4uQq7fCmbJHeE-rEbO8G1-XatGEqNV)
- [ReferIt3D GitHub](https://github.com/referit3d/referit3d)