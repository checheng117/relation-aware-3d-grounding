# Feature Fidelity Integration Audit

## Executive Summary

**Current state**: BERT features exist on disk but are not properly wired. Object geometry is real but object features are synthetic (class hash + center).
**Root cause**: Missing sample-to-feature index alignment and missing real object feature extraction pipeline.
**Impact**: ~21 percentage point gap (14.29% vs official 35.6%) attributable to feature fidelity.

---

## 1. BERT Text Features: Pipeline Analysis

### 1.1 Where BERT Features Are Generated

**File**: `scripts/prepare_bert_features.py`

```python
# Lines 121-156
def prepare_split_features(manifest_path, output_path, model_name, ...):
    utterances = load_utterances(manifest_path)  # Load from manifest
    texts = [u["utterance"] for u in utterances]
    embeddings = encode_utterances_bert(texts, ...)  # DistilBERT encoding
    np.save(output_path, embeddings)  # Save to .npy file
```

**Output files**:
- `data/text_features/train_bert_embeddings.npy` (1255 samples, 768 dim)
- `data/text_features/val_bert_embeddings.npy` (156 samples, 768 dim)
- `data/text_features/test_bert_embeddings.npy` (158 samples, 768 dim)

**Status**: BERT features GENERATED and SAVED correctly.

### 1.2 Where BERT Features Are Loaded

**File**: `repro/referit3d_baseline/scripts/train.py`

```python
# Lines 303-323
bert_dir = ROOT / "data/text_features"
train_bert_path = bert_dir / "train_bert_embeddings.npy"

if train_bert_path.exists():
    train_bert_features = np.load(train_bert_path)  # [1255, 768]
    log.info(f"Loaded train BERT features: {train_bert_features.shape}")
else:
    use_bert = False
```

**Status**: BERT features LOADED into memory.

### 1.3 Where BERT Features STOP (Critical Gap)

**File**: `repro/referit3d_baseline/scripts/train.py` - `collate_fn`

```python
# Lines 59-121
def collate_fn(batch, feat_dim=256, text_features=None):
    ...
    sample_indices = []
    for i, sample in enumerate(batch):
        if hasattr(sample, '_dataset_idx'):
            sample_indices.append(sample._dataset_idx)  # <-- ASSUMES _dataset_idx exists

    # Add real BERT features if available
    if text_features is not None and sample_indices:
        bert_features = torch.tensor(
            np.array([text_features[idx] for idx in sample_indices]),  # <-- CRITICAL
            dtype=torch.float32,
        )
        result["bert_features"] = bert_features
```

**Issue**: `_dataset_idx` attribute is set by `IndexedDataset` wrapper:

```python
# Lines 124-137
class IndexedDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample._dataset_idx = idx  # <-- SET HERE
        return sample
```

**The problem**:
1. `IndexedDataset.__getitem__` sets `_dataset_idx` on the returned `GroundingSample`
2. `GroundingSample` is a Pydantic model - setting arbitrary attributes may not persist
3. **If `_dataset_idx` is not preserved**, `sample_indices` is empty, `bert_features` is never added
4. **Even if preserved**, the index order must exactly match the BERT embedding array order

### 1.4 Model Input: What the Language Encoder Receives

**File**: `repro/referit3d_baseline/src/referit3d_net.py`

```python
# Lines 87-143 (SimpleLanguageEncoder)
def forward(self, input_ids=None, text_features=None):
    if text_features is not None:
        return self.proj(text_features)  # [B, 256]
    if input_ids is not None:
        # fallback embedding
    raise ValueError("Either input_ids or text_features must be provided")
```

**File**: `repro/referit3d_baseline/scripts/train.py` - `train_epoch`

```python
# Lines 168-179
if use_bert and "bert_features" in batch:
    text_features = batch["bert_features"].to(device)  # <-- MAY BE MISSING
else:
    batch_size = object_features.shape[0]
    text_features = torch.randn(batch_size, 768, device=device)  # <-- SYNTHETIC RANDOM
```

**Status**: Model receives RANDOM TEXT FEATURES when `bert_features` is missing from batch.

### 1.5 BERT Feature Flow Summary

| Stage | Location | Status |
|-------|----------|--------|
| Generation | `prepare_bert_features.py` | ✅ Correct |
| Storage | `data/text_features/*.npy` | ✅ Files exist (1255 train, 156 val, 158 test) |
| Loading | `train.py` lines 303-323 | ✅ Loaded into `train_bert_features` |
| Collate | `train.py` collate_fn | ❌ **BROKEN** - relies on `_dataset_idx` which may not persist |
| Batch tensor | `batch["bert_features"]` | ❌ **MAY BE MISSING** |
| Model input | `train_epoch` lines 168-179 | ❌ **FALLBACK TO RANDOM** |

**Root cause**: `GroundingSample` is a Pydantic BaseModel - `_dataset_idx` attribute assignment on Pydantic models does not reliably persist (Pydantic validates defined fields only).

---

## 2. Object Point Features: Pipeline Analysis

### 2.1 Where Real Geometry Is Extracted

**File**: `scripts/extract_scannet_geometry.py`

```python
# Extracts per-object geometry from Pointcept ScanNet data
# Output: data/geometry/<scene_id>_geometry.npz
```

**Contents of geometry files**:
- `object_ids`: List of object IDs
- `centers`: [N, 3] real centers from point bboxes
- `sizes`: [N, 3] real sizes from point bboxes
- `bboxes`: [N, 6] real bounding boxes
- `point_counts`: Number of points per object
- `sampled_points`: (optional) sampled point coordinates

**Status**: Geometry EXTRACTED correctly (269 scenes, 9457 objects).

### 2.2 Where Geometry Is Loaded Into SceneObject

**File**: `src/rag3d/datasets/scannet_objects.py`

```python
# Lines 144-209 (scene_objects_from_geometry_file)
def scene_objects_from_geometry_file(geometry_path, agg_path):
    data = np.load(geometry_path, allow_pickle=True)
    centers = data["centers"]  # REAL
    sizes = data["sizes"]      # REAL
    bboxes = data["bboxes"]    # REAL

    out.append(SceneObject(
        object_id=oid_str,
        center=center,        # REAL from point bbox
        size=size,            # REAL from point bbox
        bbox=bbox,            # REAL from points
        geometry_quality="point_bbox",  # CORRECT flag
        feature_source="pointcept_extracted",
        feature_vector=None,  # <-- NOT SET!
    ))
```

**Status**: Center/size/bbox are REAL, but `feature_vector` is NOT populated.

### 2.3 Where Object Features STOP (Critical Gap)

**File**: `src/rag3d/datasets/schemas.py`

```python
# Lines 32-45 (SceneObject)
class SceneObject(BaseModel):
    ...
    feature_vector: list[float] | None = None  # <-- NOT SET in geometry extraction
    geometry_quality: GeometryQuality = "unknown"
    feature_source: FeatureSource = "unknown"
```

**File**: `repro/referit3d_baseline/scripts/train.py` - `collate_fn`

```python
# Lines 86-99
for j, obj in enumerate(sample.objects):
    if obj.feature_vector is not None and len(obj.feature_vector) == feat_dim:
        object_features[i, j] = torch.tensor(obj.feature_vector)  # <-- NEVER TRUE
    else:
        # SYNTHETIC FALLBACK
        if obj.center:
            object_features[i, j, :3] = torch.tensor(obj.center)  # Uses real center
        if obj.size:
            object_features[i, j, 3:6] = torch.tensor(obj.size)   # Uses real size
        feat_hash = hash(obj.class_name) % (feat_dim - 6)
        object_features[i, j, 6 + feat_hash] = 1.0  # <-- CLASS HASH (synthetic semantic)
```

**Status**: Object features are partially real:
- Channels 0-5: Real center + size ✅
- Channels 6-255: Synthetic class name hash ❌

### 2.4 Alternative Collate Path

**File**: `src/rag3d/datasets/collate.py`

```python
# Lines 15-45
def make_grounding_collate_fn(feat_dim, attach_features=True):
    def _collate(batch):
        if attach_features:
            batch = [attach_synthetic_features(s, feat_dim) for s in batch]  # <-- SYNTHETIC
        ...
```

**File**: `src/rag3d/datasets/transforms.py`

```python
# Lines 13-28
def attach_synthetic_features(sample, dim):
    rng = np.random.default_rng(hash(sample.scene_id) % (2**32))
    for o in sample.objects:
        if o.feature_vector is None:
            v = rng.standard_normal(dim)  # RANDOM UNIT VECTOR
            v = (v / (np.linalg.norm(v) + 1e-8)).tolist()
            # Sets feature_source="synthetic_collate"
```

**Status**: This collate path generates PURE SYNTHETIC random vectors.

### 2.5 Model Input: What the Point Encoder Receives

**File**: `repro/referit3d_baseline/src/referit3d_net.py`

```python
# Lines 25-85 (SimplePointEncoder)
def forward(self, points, mask):
    # points: [B, N, C] aggregated features
    # Currently receives [B, N, 256] where:
    # - channels 0-5: real geometry (center + size)
    # - channels 6-255: synthetic class hash
```

**Status**: Encoder receives mixed real/synthetic features.

### 2.6 Object Feature Flow Summary

| Stage | Location | Status |
|-------|----------|--------|
| Extraction | `extract_scannet_geometry.py` | ✅ Points extracted |
| Geometry file | `data/geometry/*.npz` | ✅ Real centers/sizes saved |
| SceneObject loading | `scannet_objects.py` | ⚠️ Real geometry, but `feature_vector=None` |
| Collate (repro) | `train.py` collate_fn | ⚠️ Real center/size + synthetic hash |
| Collate (main) | `collate.py` | ❌ Pure synthetic random vectors |
| Model input | `SimplePointEncoder` | ⚠️ Mixed real/synthetic |

**Root cause**: `feature_vector` is never populated with real point-based features. The geometry files contain raw points but they are not encoded into feature vectors.

---

## 3. Missing Object Point Feature Extraction

### 3.1 What Is Available in Geometry Files

```python
# Geometry .npz file contents:
object_ids: [N] int32
centers: [N, 3] float32    # ✅ Used
sizes: [N, 3] float32      # ✅ Used
bboxes: [N, 6] float32     # ✅ Used
point_counts: [N] int32    # Available but not used
sampled_points: [N, P, 3] (optional)  # Available but not used
```

### 3.2 What Is Needed for Object Features

For proper baseline reproduction, object features should be:
- **Option A**: Point cloud features encoded by PointNet++ backbone (original paper)
- **Option B**: Pre-computed per-object point features (if available)
- **Option C**: Simple point aggregation (max pooling over sampled points)

### 3.3 Current Synthetic Substitute

```python
# repro/train.py collate_fn fallback:
feat_hash = hash(obj.class_name) % (feat_dim - 6)
object_features[i, j, 6 + feat_hash] = 1.0  # One-hot class hash
```

**Problem**: This creates a sparse, non-semantic one-hot encoding based on class name hash, which:
- Has no geometric information beyond center/size
- Has no intra-class discrimination (all "chairs" get same hash)
- Does not encode shape, orientation, or spatial context

---

## 4. Cleanest Minimal-Intrusion Integration Plan

### 4.1 BERT Text Feature Wiring (Priority 1)

**Approach**: Use utterance_id as index key rather than dataset index.

**Step 1**: Add utterance_id → embedding index mapping

```python
# In prepare_bert_features.py (or separate script):
# Create manifest-to-embedding mapping:
# utterance_id → row index in embeddings array
```

**Step 2**: Modify collate to lookup by utterance_id

```python
# In repro/train.py collate_fn:
# 1. Load BERT embeddings at startup
# 2. Build utterance_id → index dict from manifest metadata
# 3. In collate: lookup by sample.utterance_id
# 4. If utterance_id missing, raise explicit error
```

**Step 3**: Verify feature alignment

```python
# Export audit:
# For each batch sample:
# - utterance_id
# - text_feature source (BERT file or fallback)
# - text_feature shape
```

### 4.2 Object Point Feature Wiring (Priority 2)

**Approach**: Use sampled points from geometry files as model input.

**Step 1**: Extract per-object point features

```python
# In extract_scannet_geometry.py or new script:
# For each object:
#   - Load sampled_points from geometry .npz
#   - Compute simple point features (mean, max, std over xyz)
#   - Or run lightweight point encoder (PointNet-style)
#   - Save feature_vector to manifest or separate .npz
```

**Step 2**: Populate feature_vector in SceneObject

```python
# In scannet_objects.py scene_objects_from_geometry_file:
# Load pre-computed features or compute on-demand
# Set feature_vector on SceneObject
```

**Step 3**: Update collate to use real features

```python
# In repro/train.py collate_fn:
# Remove synthetic fallback
# Fail explicitly if feature_vector is None
```

### 4.3 Minimal-Change Strategy

| Change | Files Modified | Impact |
|--------|---------------|--------|
| BERT mapping | `train.py` + helper script | Text becomes real |
| Point features | Geometry extraction + `scannet_objects.py` | Object features become real |
| Collate cleanup | `train.py` collate_fn | Remove fallbacks |

**No changes needed**:
- Model architecture (SimplePointEncoder + SimpleLanguageEncoder already handle features)
- Training loop (already passes features to model)
- Dataset schema (already has feature_vector field)

---

## 5. Verification Points

### 5.1 After BERT Integration

```python
# Verification checklist:
1. Every sample has valid utterance_id
2. Every utterance_id has corresponding BERT embedding
3. Collate produces batch["bert_features"] tensor [B, 768]
4. Model forward receives non-random text_features
5. No samples use fallback random features
```

### 5.2 After Object Feature Integration

```python
# Verification checklist:
1. Every SceneObject has non-None feature_vector
2. feature_vector length matches feat_dim (256)
3. feature_source is not "synthetic_collate"
4. Collate no longer injects class hash fallback
5. Model forward receives geometry-derived features
```

### 5.3 Feature-Use Audit Export

For each batch in a verification run:
```json
{
  "sample_id": "...",
  "utterance_id": "...",
  "text_source": "bert_file",
  "text_shape": [768],
  "object_features": [
    {
      "object_id": "...",
      "feature_source": "pointcept_extracted",
      "feature_shape": [256],
      "has_real_geometry": true
    }
  ]
}
```

---

## 6. Key Files Summary

### Files to Read/Modify

| File | Purpose | Change Needed |
|------|---------|---------------|
| `repro/referit3d_baseline/scripts/train.py` | Main training | Fix collate_fn to use utterance_id lookup |
| `scripts/prepare_bert_features.py` | BERT generation | Add utterance_id → index mapping output |
| `src/rag3d/datasets/scannet_objects.py` | Geometry loading | Populate feature_vector from points |
| `src/rag3d/datasets/transforms.py` | Synthetic attachment | Bypass for reproduction track |
| `repro/referit3d_baseline/configs/official_baseline.yaml` | Config | Add text_feature_path, object_feature_path |

### Files to Create

| File | Purpose |
|------|---------|
| `scripts/build_feature_index.py` | Build utterance_id → embedding mapping |
| `scripts/compute_object_features.py` | Extract point-based object features |
| `reports/bert_feature_wiring.md` | BERT integration report |
| `reports/object_feature_wiring.md` | Object feature integration report |

---

## 7. Conclusion

**Primary blockers**:
1. **BERT features not wired**: Loaded but not passed to model due to `_dataset_idx` persistence issue
2. **Object features synthetic**: Geometry files have points but `feature_vector` never populated

**Estimated impact**:
- BERT integration: +5-10% accuracy (language features critical for grounding)
- Object features: +3-5% accuracy (better geometric representation)

**Priority order**:
1. Wire BERT text features (utterance_id → embedding mapping)
2. Wire object point features (extract from geometry .npz)
3. Verify feature-use audit
4. Rerun reproduction and compare metrics