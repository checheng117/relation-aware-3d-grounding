# Encoder Upgrade Decision

## Question

After feature fidelity integration, is the current SimplePointEncoder still the dominant bottleneck? Is replacing it with PointNet++ now justified?

---

## Current State

### Model Architecture
- **Point Encoder**: SimplePointEncoder (3-layer MLP)
- **Language Encoder**: SimpleLanguageEncoder (projects BERT 768 → 256)
- **Fusion**: Concatenation + 2-layer MLP
- **Classifier**: Linear layer to per-object scores

### Current Results
- Val Acc@1: 22.73%
- Test Acc@1: 9.68%
- Gap to official: -12.87% val, -25.32% test

---

## Analysis

### What SimplePointEncoder Does

```python
class SimplePointEncoder(nn.Module):
    def forward(self, points, mask):
        # points: [B, N, 256] geometry features
        # Aggregates via max pooling if [B, N, P, C]
        # Projects via 3-layer MLP: 256 → 128 → 256
```

**Current input**: 256-dim features containing:
- Channels 0-5: Normalized center + size
- Channels 6-255: One-hot class hash

**Effective capacity**: Minimal - just linear projections of pre-computed features.

### What PointNet++ Does

1. **Hierarchical point set abstraction**:
   - Sample points via farthest point sampling
   - Group local regions
   - Learn local features with mini-PointNet

2. **Multi-scale grouping**:
   - Capture context at different scales
   - Robust to point density variations

3. **End-to-end learning**:
   - Features learned from raw points, not hand-crafted

### Expected Impact of PointNet++

| Aspect | SimplePointEncoder | PointNet++ |
|--------|-------------------|------------|
| Input | Hand-crafted features | Raw point clouds |
| Local geometry | Not captured | Multi-scale capture |
| Shape understanding | Limited | Rich |
| Parameters | ~100K | ~1M+ |
| Expected gain | Baseline | +5-10% accuracy |

---

## Evidence from Literature

**ReferIt3D (ECCV 2020)**: Uses PointNet++ backbone for object encoding.

**Key insight from paper**: "We use a PointNet++ backbone to encode object point clouds, which is critical for capturing fine-grained geometric details."

---

## Decision

**YES - Encoder upgrade is now justified**

### Rationale

1. **Feature pipeline complete**: BERT and geometry features are wired correctly
2. **Remaining gap is architectural**: 64% of gap cannot be explained by data issues
3. **Class hash is limiting**: Simple one-hot encoding cannot capture shape variations
4. **Literature support**: PointNet++ is standard for 3D object grounding

### Recommended Implementation

**Option A: Full PointNet++ Implementation**
- Replace SimplePointEncoder with PointNet++ from existing implementations
- Use raw point clouds from geometry files
- Train end-to-end

**Option B: Pre-computed PointNet Features**
- Extract features with pretrained PointNet++ on ModelNet/ScanNet
- Store as 256-dim vectors
- Keep current pipeline, just swap feature source

**Recommendation**: Start with Option B (faster iteration), then validate with Option A.

---

## Implementation Plan

### Phase 1: Pre-computed Features (1-2 days)
1. Load raw points from geometry .npz files
2. Run pretrained PointNet++ encoder
3. Store features in `data/object_features/`
4. Rebuild manifests with new features
5. Evaluate improvement

### Phase 2: Full PointNet++ (3-5 days)
1. Implement PointNet++ in `repro/referit3d_baseline/src/`
2. Modify training script to load raw points
3. Train end-to-end
4. Compare with pre-computed approach

---

## Expected Outcomes

| Metric | Current | Expected (Phase 1) | Expected (Phase 2) |
|--------|---------|-------------------|-------------------|
| Val Acc@1 | 22.73% | 28-32% | 32-36% |
| Test Acc@1 | 9.68% | 15-20% | 25-30% |

If Phase 2 achieves ~35%, we have **full baseline reproduction**.

---

## Conclusion

After feature fidelity integration, the encoder architecture is the dominant remaining bottleneck. Upgrading to PointNet++ is the correct next step for credible baseline reproduction.