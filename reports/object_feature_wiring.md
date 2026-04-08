# Object Feature Wiring Report

## Summary

Successfully integrated real object point features into the reproduction pipeline.

**Before**: Class name hash (synthetic) + geometry center/size
**After**: 256-dim point-based features extracted from real ScanNet point clouds

---

## Files Modified

### `scripts/compute_object_features.py` (NEW)

Script to compute per-object point features from geometry files.

```bash
python scripts/compute_object_features.py
```

Output: `data/object_features/<scene_id>_features.npz`

### `src/rag3d/datasets/schemas.py`

Added `point_features_computed` to `FeatureSource` type.

### `src/rag3d/datasets/scannet_objects.py`

Updated `scene_objects_from_geometry_file()` to load pre-computed features:
- Added `feature_dir` parameter
- Loads `*_features.npz` files when available
- Populates `feature_vector` on SceneObject

### `src/rag3d/datasets/builder.py`

Updated `build_records_nr3d_hf_with_scans()` to pass `feature_dir`.

### `scripts/prepare_data.py`

Added feature_dir resolution and passing to builder.

### `repro/referit3d_baseline/scripts/train.py`

Updated `collate_fn()`:
- Uses real `feature_vector` when available
- Tracks `feature_coverage` for diagnostics
- Removed synthetic fallback (now explicit error if missing)

---

## Feature Extraction

### Point Feature Computation

Features extracted from real ScanNet point clouds:

| Feature | Channels | Description |
|---------|----------|-------------|
| Point mean | 0:3 | Mean of points relative to bbox center, normalized |
| Point std | 3:6 | Std of points, normalized by bbox size |
| Point min | 6:9 | Min extent relative to center |
| Point max | 9:12 | Max extent relative to center |
| BBox center | 12:15 | Absolute bbox center |
| BBox size | 15:18 | BBox dimensions |
| Point count | 18 | Log-normalized point count |
| Point density | 21 | Points per unit volume (log) |
| Reserved | 24:256 | Zeros (for PointNet features) |

### Extraction Statistics

```
Scenes processed: 269
Objects: 9,457
Points: 19,366,164
Objects with features: 9,457 (100%)
```

---

## Coverage Verification

### Manifest Verification

```
Sample scene: scene0467_00
N objects: 34
  Object 0: class=lamp, has_feature=True, len=256, source=point_features_computed
  Object 1: class=lamp, has_feature=True, len=256, source=point_features_computed
  Object 2: class=floor, has_feature=True, len=256, source=point_features_computed
```

### Collate Verification

```
Dataset size: 1235
BERT features: (1235, 768)
Collate result keys: [..., 'feature_coverage', 'bert_features']
feature_coverage: {'real': 175, 'fallback': 0}
✓ Real object features: 175/175 (100.0%)
✓ BERT features wired: True
```

---

## Feature Source Distribution

| Source | Count | Description |
|--------|-------|-------------|
| `point_features_computed` | 9,457 | Real features from point extraction |
| `pointcept_extracted` | 0 | Legacy (no longer used) |
| `synthetic_collate` | 0 | Eliminated from reproduction track |

---

## Comparison: Before vs After

| Attribute | Before | After |
|-----------|--------|-------|
| `feature_vector` | None | 256-dim real features |
| `feature_source` | N/A | `point_features_computed` |
| Collate fallback | Class hash + center/size | Error if missing |
| Feature coverage | 0% real | 100% real |

---

## Remaining Work

### Architecture Upgrade (Optional)

Current `SimplePointEncoder` is a simple MLP. For full baseline reproduction, consider:

1. **PointNet++ backbone**: Original ReferIt3D architecture
2. **Pre-computed PointNet features**: Extract with pretrained encoder
3. **Or keep current**: SimpleMLP may be sufficient with better features

---

## Verification Checklist

- [x] Object features computed from point clouds
- [x] Features saved to `data/object_features/*.npz`
- [x] `scene_objects_from_geometry_file` loads features
- [x] Manifests rebuilt with feature vectors
- [x] Collate uses real features (not fallback)
- [x] 100% feature coverage in reproduction track
- [x] Feature dimension matches model input (256)