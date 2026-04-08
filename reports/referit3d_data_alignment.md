# ReferIt3D Data Alignment Report

**Date**: 2026-04-03
**Objective**: Document data pipeline alignment with official protocol

---

## 1. Files Being Used

### 1.1 Annotation Files

| File | Location | Status |
|---|---|---|
| `nr3d_annotations.json` | `data/raw/referit3d/annotations/` | ✅ Available |
| `train.csv` | `data/raw/referit3d/annotations/` | ✅ Available |
| `val.csv` | `data/raw/referit3d/annotations/` | ✅ Available |
| `test.csv` | `data/raw/referit3d/annotations/` | ✅ Available |

### 1.2 ScanNet Scene Files

| File Type | Location | Status |
|---|---|---|
| Scene directories | `data/raw/referit3d/scans/scene*/` | ✅ Available (269 scenes) |
| Aggregation files | `*.aggregation.json` | ✅ Available |
| ScanNet meshes | `*.ply` or `*.obj` | ❌ Missing |
| Point cloud data | Extracted from meshes | ❌ Missing |

### 1.3 Processed Files

| File | Location | Status |
|---|---|---|
| `train_manifest.jsonl` | `data/processed/` | ✅ Available (1,255 samples) |
| `val_manifest.jsonl` | `data/processed/` | ✅ Available (156 samples) |
| `test_manifest.jsonl` | `data/processed/` | ✅ Available (158 samples) |

---

## 2. Data Statistics

### 2.1 Current vs Official

| Metric | Official Nr3D | Our Data | Gap |
|---|---|---|---|
| Total samples | ~41,503 | 1,569 | 96% less |
| Train samples | ~27,000 | 1,255 | 95% less |
| Val samples | ~5,000 | 156 | 97% less |
| Test samples | ~9,500 | 158 | 98% less |
| Unique scenes | ~500+ | 269 | Unknown |

**Critical finding**: Our data appears to be a **subset** of official Nr3D. Need to verify if this is sufficient for baseline reproduction or if we need the full dataset.

### 2.2 Entity Statistics

| Metric | Value |
|---|---|
| Avg entities per sample | 2.37 |
| Min entities | 0 |
| Max entities | 7 |

---

## 3. Mapping to Official Pipeline

### 3.1 Annotation Format

**Our format**:
```json
{
  "unique_id": 0,
  "scene_id": "scene0525_00",
  "object_id": "9",
  "object_name": "plant",
  "descriptions": ["The plant at the far right..."],
  "answer": 0,
  "entities": ["9_plant", "34_bookshelf", "38_window"],
  "image": "860.jpg"
}
```

**Official ReferIt3D format** (expected):
- Similar structure with scene_id, object_id, utterance
- Should include easy/hard labels
- Should include view-dependence labels

### 3.2 Scene Geometry

**What we have**:
- Aggregation files with segmentation groups
- No actual mesh or point cloud data

**What we need**:
- ScanNet mesh files (`.ply`)
- Per-object point cloud samples (1024 points)

**Status**: ❌ Critical gap - need to download ScanNet meshes

### 3.3 Object Candidates

**Our approach**: Entity-referenced objects only (subset of scene objects)

**Official approach**: Need to verify - may be all scene objects or entity subset

---

## 4. Mismatches from Official Baseline Setup

### 4.1 Data Size Mismatch (CRITICAL)

| Issue | Impact |
|---|---|---|
| 96% fewer samples | May affect model convergence |
| May not match official splits | Difficult to compare to benchmark |

**Mitigation**: Either:
1. Accept subset for initial reproduction
2. Download full Nr3D from official source

### 4.2 Missing Geometry (CRITICAL)

| Issue | Impact |
|---|---|---|
| No real point clouds | Cannot use PointNet++ backbone |
| Placeholder centers/sizes | Not representative of real data |

**Mitigation**: Download ScanNet processed meshes and extract point clouds

### 4.3 Missing Metadata (MEDIUM)

| Issue | Impact |
|---|---|---|
| No easy/hard labels | Cannot evaluate on sub-metrics |
| No view-dependence labels | Cannot split by view-dependence |

**Mitigation**: Derive from utterance patterns or find official metadata

---

## 5. Approximations Made

### 5.1 Candidate Set Approximation

**Current**: Using only entity-referenced objects as candidates

**Official**: May include all scene objects

**Impact**: Smaller candidate sets, potentially easier task

### 5.2 Geometry Approximation

**Current**: Placeholder geometry (centers all [0,0,0], sizes all [0.1,0.1,0.1])

**Official**: Real ScanNet bounding boxes and point clouds

**Impact**: Cannot evaluate true geometric reasoning

### 5.3 Feature Approximation

**Current**: Synthetic features generated during collation

**Official**: PointNet++ features from real point clouds

**Impact**: Features not representative of real data

---

## 6. Data Pipeline Steps

### Step 1: Download ScanNet Meshes

```bash
# Requires ScanNet access credentials
# See: http://www.scan-net.org/
```

### Step 2: Extract Point Clouds

```python
# For each scene:
# 1. Load mesh
# 2. For each segment ID in aggregation.json:
#    - Extract points belonging to segment
#    - Sample 1024 points
#    - Save as .npy or .pt
```

### Step 3: Generate Real Features

```python
# Option A: Pre-compute PointNet++ features
# Option B: Compute on-the-fly during training
```

---

## 7. Verification Checklist

### 7.1 Before Reproduction

- [ ] Verify NR3D annotation count matches expectation
- [ ] Verify scene IDs match between annotations and scans
- [ ] Verify object IDs exist in aggregation files
- [ ] Check if we need full Nr3D or subset is acceptable

### 7.2 After Geometry Download

- [ ] Verify mesh files load correctly
- [ ] Verify point cloud extraction works
- [ ] Verify point cloud quality (non-degenerate)
- [ ] Verify object ID mapping to segments

---

## 8. Recommendations

### 8.1 Short-term (for quick reproduction)

1. Use current subset for initial pipeline validation
2. Accept that results may not match official benchmark
3. Focus on getting end-to-end pipeline working

### 8.2 Long-term (for credible reproduction)

1. Download full Nr3D annotations
2. Download ScanNet meshes
3. Extract real point clouds
4. Reproduce official training exactly

---

## 9. Current Status Summary

| Component | Status | Action Needed |
|---|---|---|
| Annotations | ✅ Available | None |
| Scene directories | ✅ Available | None |
| Aggregation files | ✅ Available | None |
| ScanNet meshes | ❌ Missing | Download |
| Point clouds | ❌ Missing | Extract |
| Easy/hard labels | ❌ Missing | Derive or find |
| View-dependence labels | ❌ Missing | Derive or find |

---

**End of Data Alignment Report**