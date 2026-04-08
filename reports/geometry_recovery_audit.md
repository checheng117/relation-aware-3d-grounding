# Geometry Recovery Audit

## Executive Summary

**Current state**: All geometry in the reproduction pipeline is synthetic/placeholder.
**Root cause**: ScanNet aggregation files lack OBB fields; no mesh/PLY files available.
**Impact**: ~19.5 percentage point gap (16.03% vs official 35.6%) largely attributable to geometry fidelity.

---

## 1. Synthetic Geometry Production Points

### 1.1 `scripts/prepare_scannet_geometry.py`

**Function**: `generate_synthetic_geometry()` (lines 94-171)

```python
def generate_synthetic_geometry(obj_id, label, num_points=1024, seed=None):
    # Generates random center based on obj_id
    center = np.array([
        (obj_id % 10) * 0.5 + np.random.uniform(-0.1, 0.1),
        (obj_id // 10) * 0.3 + np.random.uniform(-0.1, 0.1),
        np.random.uniform(0.3, 1.5),
    ])
    # Generates random size based on class name
    size_scales = {"chair": (0.5, 0.5, 0.8), ...}
    # Generates random points within box
    points = np.random.uniform(-0.5, 0.5, (num_points, 3)) * size + center
```

**Status**: PLACEHOLDER - generates fake point clouds, fake centers, fake sizes.

**Warning at line 408-410**:
```python
print("  ⚠️  WARNING: Using SYNTHETIC geometry (placeholder)")
print("  For actual reproduction, replace with real ScanNet point clouds")
```

### 1.2 `src/rag3d/datasets/scannet_objects.py`

**Function**: `_obb_to_center_size()` (lines 35-41)

```python
def _obb_to_center_size(obb: dict | None):
    if not obb:
        return (0.0, 0.0, 0.0), (0.1, 0.1, 0.1)  # DEFAULT FALLBACK
    c = obb.get("centroid") or [0.0, 0.0, 0.0]
    ax = obb.get("axesLengths") or [0.1, 0.1, 0.1]
    return (float(c[0]), float(c[1]), float(c[2])), (float(ax[0]), float(ax[1]), float(ax[2]))
```

**Status**: When OBB missing → returns default `(0,0,0)` center and `(0.1,0.1,0.1)` size.

**Function**: `scene_objects_from_aggregation()` (lines 44-83)

```python
has_real_obb = bool(obb and isinstance(obb, dict) and "centroid" in obb and "axesLengths" in obb)
geo_q = "obb_aabb" if has_real_obb else "fallback_centroid"
```

**Status**: All objects currently get `geometry_quality="fallback_centroid"`.

### 1.3 `src/rag3d/datasets/nr3d_hf.py`

**Function**: `_placeholder_center_size()` (lines 33-35)

```python
def _placeholder_center_size(oid: str, idx: int):
    h = (hash((oid, idx)) % 10000) / 10000.0
    return (float(idx) * 0.25 + h * 0.01, h * 0.5, (h * 0.3) % 0.2), (0.2, 0.2, 0.3)
```

**Status**: Generates deterministic but fake center/size based on hash.

### 1.4 `src/rag3d/datasets/collate.py`

**Function**: `make_grounding_collate_fn()` (lines 15-45)

```python
def _collate(batch: list[GroundingSample]):
    if attach_features:
        batch = [attach_synthetic_features(s, feat_dim) for s in batch]
```

**Status**: Collate attaches synthetic features when `feature_vector` missing.

### 1.5 `src/rag3d/datasets/transforms.py`

**Function**: `attach_synthetic_features()` (lines 13-28)

```python
def attach_synthetic_features(sample: GroundingSample, dim: int):
    rng = np.random.default_rng(hash(sample.scene_id) % (2**32))
    v = rng.standard_normal(dim).astype(float)
    v = (v / (np.linalg.norm(v) + 1e-8)).tolist()
    # ... sets feature_source="synthetic_collate"
```

**Status**: Generates random unit vectors as object features.

---

## 2. Fallback Injection Points

| File | Line | Fallback Value | Trigger Condition |
|------|------|----------------|-------------------|
| `scannet_objects.py` | 38 | `(0,0,0), (0.1,0.1,0.1)` | `obb is None` |
| `nr3d_hf.py` | 33-35 | hash-based pseudo-coords | no aggregation loaded |
| `transforms.py` | 21-24 | random unit vector | `feature_vector is None` |
| `geom_context.py` | 29-30 | flag=1.0 | `geometry_quality == "fallback_centroid"` |

---

## 3. Geometry Fields Expected by Downstream Model Code

### 3.1 `SceneObject` schema (`schemas.py` lines 32-45)

```python
class SceneObject(BaseModel):
    object_id: str
    class_name: str
    center: tuple[float, float, float]     # REQUIRED
    size: tuple[float, float, float]       # REQUIRED
    bbox: tuple[6 floats] | None           # OPTIONAL (min_x, min_y, min_z, max_x, max_y, max_z)
    point_indices: list[int] | None        # OPTIONAL
    sampled_points: np.ndarray | None      # OPTIONAL [N, 3] or [N, 6]
    visibility_occlusion_proxy: float | None
    feature_vector: list[float] | None     # REQUIRED for model input
    geometry_quality: GeometryQuality      # "obb_aabb" | "fallback_centroid" | "unknown"
    feature_source: FeatureSource          # "synthetic_collate" | "aggregated_file" | ...
```

### 3.2 `geom_context.py` - 8-channel geometry tensor

```python
def object_geom_context_tensor8(sample, device):
    # [N, 8]: center3, size3, fallback_geom_flag, synthetic_feat_flag
    out[j, 0:3] = tanh(center / 5.0)    # center normalized
    out[j, 3:6] = tanh(size / 2.0)      # size normalized
    out[j, 6] = 1.0 if geometry_quality == "fallback_centroid" else 0.0
    out[j, 7] = 1.0 if feature_source == "synthetic_collate" else 0.0
```

**Status**: Model receives 8-channel tensor, but channels 0-5 are fake, channels 6-7 are both 1.0.

### 3.3 `repro/referit3d_baseline/scripts/train.py` - collate usage

```python
# lines 86-94
for j, obj in enumerate(sample.objects):
    if obj.feature_vector is not None and len(obj.feature_vector) == feat_dim:
        object_features[i, j] = torch.tensor(obj.feature_vector)
    else:
        # SYNTHETIC FALLBACK
        feat_hash = hash(obj.class_name) % feat_dim
        object_features[i, j, feat_hash] = 1.0
        if obj.center:
            object_features[i, j, :3] = torch.tensor(obj.center)  # uses fake center!
```

**Status**: If no feature_vector, uses hash-based one-hot AND injects fake center into first 3 channels.

### 3.4 `repro/referit3d_net.py` - Point encoder input

```python
class SimplePointEncoder(nn.Module):
    def forward(self, points: torch.Tensor, mask):
        # points: [B, N, P, C] for raw points OR [B, N, C] for aggregated features
        # Currently receives [B, N, 256] where first 3 channels are fake center
```

**Status**: Encoder expects point features; receives synthetic features with fake geometry baked in.

---

## 4. Cleanest Replacement Point for Real Geometry

### Option A: Replace at `scannet_objects.py` level (RECOMMENDED)

**Why**: This is where `SceneObject` instances are created from aggregation files.

**Change needed**:
1. Read real mesh/PLY files to extract per-object point clouds
2. Compute center/size from actual point bboxes (if OBB missing)
3. Set `geometry_quality` to `"point_bbox"` or `"obb_aabb"` (not `"fallback_centroid"`)

**Integration point**: `scene_objects_from_aggregation()` and `load_scene_objects_for_scene_id()`

### Option B: Replace at `prepare_scannet_geometry.py` level

**Why**: This script is explicitly designed for geometry preparation.

**Change needed**: Replace `generate_synthetic_geometry()` with real mesh extraction.

**Output**: Save per-scene `.npz` files with real point clouds, load in builder.

### Option C: Replace at collate level (NOT RECOMMENDED)

**Why**: Too late - geometry fields already set to fake values upstream.

---

## 5. Required ScanNet Asset Files

### 5.1 Files currently available

```
data/raw/referit3d/scans/<scene_id>/<scene_id>.aggregation.json
```

**Contents**:
- `segGroups`: list of objects with `objectId`, `segments`, `label`
- **MISSING**: `obb` block (centroid, axesLengths)

**Source**: Downloaded from `zahidpichen/scannet-dataset` or `Gen3DF/scannet` HF mirrors.

### 5.2 Files required for real geometry

| File | Purpose | ScanNet Original Name |
|------|---------|----------------------|
| **Mesh PLY** | Per-scene 3D mesh with vertex coordinates | `<scene_id>_vh_clean.ply` |
| **Segmentation PLY** | Per-vertex segment labels | `<scene_id>_vh_clean.segs.json` or embedded in PLY |
| **Aggregation JSON** (with OBB) | Object groups with bounding boxes | `<scene_id>_vh_clean_aggregation.json` |

**Alternative sources**:
1. Official ScanNet website (requires license agreement)
2. Pointcept preprocessing tar: `scripts/extract_scannet_aggregations_pointcept_tar.py`
3. Pre-computed point features from other papers

### 5.3 Asset availability check needed

We need to determine:
- Do we have `*_vh_clean.ply` mesh files anywhere?
- Do we have `*_vh_clean.segs.json` segmentation files?
- Can we obtain them from an accessible source?
- Are there pre-computed per-object point clouds available?

---

## 6. Recommended Implementation Order

### Phase 1: Asset Discovery (Step 1)

1. Create `scripts/index_scannet_assets.py` to inventory available files
2. Check for: `.ply`, `.segs.json`, aggregation variants
3. Report missing assets per scene

### Phase 2: Point Extraction (Step 2)

1. If mesh available: extract per-object points using segment → vertex mapping
2. If only segmentation available: compute point bbox from segment vertices
3. Update `scannet_objects.py` to use real geometry

### Phase 3: Pipeline Integration (Step 4)

1. Ensure `builder.py` uses geometry-backed objects
2. Ensure collate no longer injects synthetic features for reproduction track
3. Set proper `geometry_quality` flags

### Phase 4: Validation (Step 3)

1. Verify non-default centers/sizes
2. Verify point count distribution
3. Verify geometry quality flags

---

## 7. Key Files to Modify

| File | Change |
|------|--------|
| `scripts/index_scannet_assets.py` | NEW - asset discovery |
| `scripts/prepare_scannet_geometry.py` | Replace synthetic with real extraction |
| `src/rag3d/datasets/scannet_objects.py` | Use real geometry, compute from points |
| `src/rag3d/datasets/builder.py` | Ensure geometry-backed path |
| `repro/referit3d_baseline/scripts/train.py` | Remove synthetic fallback in collate |
| `scripts/validate_scannet_geometry.py` | NEW - geometry validation |

---

## 8. Blockers

1. **Missing mesh files**: No `*_vh_clean.ply` found in current data directory
2. **Missing OBB in aggregation**: Current aggregation JSON lacks `obb` fields
3. **Missing segmentation**: No `*_vh_clean.segs.json` for vertex-to-object mapping

**Resolution paths**:
- Check if Pointcept tar has mesh data
- Check if pre-computed object features exist elsewhere
- Request/obtain official ScanNet mesh files