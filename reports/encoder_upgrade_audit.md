# Encoder Upgrade Audit

## Summary

This audit identifies the cleanest insertion point for replacing SimplePointEncoder with PointNet++ while keeping the reproduction protocol fixed.

---

## 1. Current SimplePointEncoder Usage

### Location
**File**: `repro/referit3d_baseline/src/referit3d_net.py`

### Classes Using SimplePointEncoder

| Class | Line | Usage |
|-------|------|-------|
| `ReferIt3DNet` | 169-174 | Main baseline model, `self.point_encoder` |
| `ReferIt3DNetWithAttention` | 259-265 | Attention variant, `self.point_encoder` |

### Current Implementation

```python
class SimplePointEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=256):
        # 3-layer MLP: input_dim -> hidden_dim -> hidden_dim -> output_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, points, mask):
        # points: [B, N, P, C] OR [B, N, C]
        # Returns: [B, N, output_dim]
```

### Current Input Tensor

| Shape | Description |
|-------|-------------|
| `[B, N, 256]` | B=batch, N=objects, 256=feature dim |

### Current Feature Channels (256-dim)

| Channels | Content | Source |
|----------|---------|--------|
| 0:3 | Center (normalized by /5.0) | Geometry file |
| 3:6 | Size (normalized by /2.0) | Geometry file |
| 6:256 | Class name hash (one-hot) | Synthetic from class_name |

### Parameter Count

| Component | Parameters |
|-----------|------------|
| Linear(256→128) | 32,896 |
| Linear(128→128) | 16,512 |
| Linear(128→256) | 33,024 |
| **Total** | **~82,432** |

---

## 2. Object Input Tensor Analysis

### What Currently Reaches the Encoder

The collate function in `train.py` constructs:

```python
object_features[i, j, 0:3] = torch.tensor(obj.center) / 5.0
object_features[i, j, 3:6] = torch.tensor(obj.size) / 2.0
feat_hash = hash(obj.class_name) % 250
object_features[i, j, 6 + feat_hash] = 1.0
```

### What Could Reach the Encoder (Raw Points Available)

**Geometry files** contain raw point clouds per object:

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `points_{i}` | (4096, 3) | XYZ coordinates for object i |
| `point_counts[i]` | int | Actual point count (varies: 924 to 11593) |

**Key insight**: Each object has up to 4096 sampled points with XYZ coordinates.

### Raw Point Statistics

From `scene0002_00_geometry.npz`:
- Object 0: 11,593 original points, sampled to 4,096
- Object 1: 7,948 original points, sampled to 4,096
- Point coordinates: XYZ in ScanNet coordinate frame

---

## 3. Existing PointNet++ Implementations

### External Package Check

| Package | Available? | PointNet++ Support |
|---------|------------|--------------------|
| pytorch3d | **No** | Yes (if installed) |
| torch_geometric | **No** | Yes (if installed) |
| torch_points3d | **No** | Yes (if installed) |
| mmdet3d | **No** | Yes (if installed) |

### Internal Repo Check

| File | Contents |
|------|----------|
| `src/rag3d/encoders/point_encoder.py` | `PointNetToyEncoder` (minimal, NOT PointNet++) |
| `src/rag3d/encoders/object_encoder.py` | `ObjectMLPEncoder`, `ObjectMLPEncoderWithGeomContext` |

**Conclusion**: No existing PointNet++ implementation in the repo. Must implement or install external package.

---

## 4. Minimal PointNet++ Implementation Options

### Option A: Install External Package

**Pros**:
- Battle-tested implementations
- Pre-trained weights available
- Multi-scale grouping included

**Cons**:
- Adds dependency complexity
- May require CUDA-specific builds
- Environment.yml modifications needed

**Recommended packages**:
- `torch_geometric` (most flexible, pip-installable)
- `pytorch3d` (Meta-supported, but complex install)

### Option B: Minimal In-Repo Implementation

**Pros**:
- No external dependencies
- Full control for controlled experiment
- Easier debugging
- Direct integration with existing collate pattern

**Cons**:
- More implementation effort
- Need to verify correctness

**Minimal implementation needed**:
1. Point sampling (farthest point sampling or random)
2. Local grouping (ball query)
3. Mini-PointNet for local features
4. Hierarchical aggregation (2-3 levels)

---

## 5. What Must Remain Fixed

### For Clean Controlled Comparison

| Component | Must Remain Fixed |
|-----------|-------------------|
| Dataset | NR3D split (train/val/test manifests) |
| Geometry pipeline | Real centers/sizes from geometry files |
| BERT features | DistilBERT embeddings (768-dim) |
| Language encoder | SimpleLanguageEncoder (BERT → 256) |
| Fusion | Concatenation + 2-layer MLP |
| Classifier | Linear(fusion_dim → 1) |
| Loss | CrossEntropyLoss |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR |
| Training epochs | 30 |
| Batch size | 16 |
| Seed | 42 |
| Evaluator | Top-1 and Top-5 accuracy |

### Changed Component (ONLY)

| Component | Before | After |
|-----------|--------|-------|
| Object encoder | SimplePointEncoder (MLP) | PointNet++ (or PointNet++-style) |

### Allowed Shape Adaptations (Minimal)

| Change | Reason |
|--------|--------|
| Collate may need to load raw points | PointNet++ requires point clouds |
| Collate may need to normalize points differently | PointNet++ expects centered/scaled points |
| Input tensor shape change | [B,N,256] → [B,N,P,3] or [B*N,P,3] |

---

## 6. Integration Points

### Current Integration

```
collate_fn() → object_features [B,N,256] → SimplePointEncoder → [B,N,256]
```

### Proposed Integration (Option B: Minimal PointNet++)

```
collate_fn() → raw_points [B,N,P,3] → PointNet++Encoder → [B,N,256]
```

### Key Integration Requirements

1. **Collate modification**: Load raw points from geometry files instead of computed features
2. **Point normalization**: Center points at object bbox center, normalize by bbox size
3. **Output dimension**: Must output 256-dim to match downstream fusion layer
4. **Parameter tracking**: Log encoder type, parameters, input/output shapes

### Files to Modify

| File | Change |
|------|--------|
| `repro/referit3d_baseline/src/referit3d_net.py` | Add PointNet++Encoder class |
| `repro/referit3d_baseline/scripts/train.py` | Modify collate_fn to load raw points |
| `repro/referit3d_baseline/scripts/evaluate.py` | Modify collate_fn to load raw points |
| `repro/referit3d_baseline/configs/official_baseline.yaml` | Add encoder_type config option |

---

## 7. Raw Points Availability Check

### Geometry File Structure

```python
geometry.npz contents:
- object_ids: (N,) int32
- centers: (N, 3) float32
- sizes: (N, 3) float32
- bboxes: (N, 6) float32
- point_counts: (N,) int32
- points_0: (4096, 3) float32  # Object 0 points
- points_1: (4096, 3) float32  # Object 1 points
- ...
- points_{N-1}: (4096, 3) float32  # Object N-1 points
```

### Point Loading Strategy

1. Load geometry file for scene
2. Map object_id to points_{index}
3. Normalize points: center - bbox_center, divide by bbox_size
4. Pad/truncate to fixed length (e.g., 1024 or 2048)

### Coverage

| Attribute | Status |
|-----------|--------|
| Geometry files | 269 scenes available |
| Raw points | 100% objects have points |
| Point count | 924 to 11593 (avg ~5000) |
| Sampled points | Fixed at 4096 per object |

---

## 8. Recommended Implementation Path

### Phase 1: Minimal PointNet++ Implementation

**Why minimal?**
- No external dependency risk
- Faster iteration
- Controlled experiment
- Can upgrade to external package later if needed

**Implementation steps**:
1. Add `PointNetPPEncoder` class to `referit3d_net.py`
2. Implement:
   - Random/farthest point sampling (1024 points)
   - Local grouping with ball query
   - Mini-PointNet layers
   - Max pooling aggregation
3. Modify collate_fn to load raw points from geometry files
4. Add config switch `encoder_type: simple_point | pointnetpp`

### Phase 2: Verify with Smoke Test

1. Forward pass sanity check
2. Backward pass sanity check
3. No NaN/Inf gradients
4. Shape verification at each layer

### Phase 3: Controlled Comparison Run

Run both encoders under identical protocol:
- Same data splits
- Same training hyperparameters
- Same evaluation metrics

---

## 9. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Point loading complexity | Medium | Use existing geometry file format |
| Shape mismatch | Low | Verify input/output shapes in forward |
| Training instability | Medium | Use gradient clipping, check for NaN |
| Memory increase (raw points) | Medium | Use smaller point count (1024) |
| PointNet++ implementation bugs | Medium | Test with synthetic data first |

---

## 10. Conclusion

### Insertion Point Identified

- **Location**: `repro/referit3d_baseline/src/referit3d_net.py`
- **Class**: Replace `SimplePointEncoder` with `PointNetPPEncoder`
- **Collate**: Modify to load raw points from geometry files
- **Config**: Add `encoder_type` switch for controlled comparison

### Data Pipeline Verified

- Raw points available in `data/geometry/*.npz`
- 4096 points per object (XYZ)
- 100% coverage across scenes
- Normalizable with bbox center/size

### Implementation Recommendation

**Option B: Minimal in-repo PointNet++ implementation**

Reasons:
1. No external dependency complexity
2. Full control for debugging
3. Matches reproduction track isolation goal
4. Easier to verify encoder actually used

### Next Step

Proceed to write `reports/encoder_upgrade_protocol.md` defining the controlled comparison protocol.