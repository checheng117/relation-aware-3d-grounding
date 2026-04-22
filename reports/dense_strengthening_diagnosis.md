# Dense Scorer Strengthening Diagnosis

**Date**: 2026-04-21
**Status**: DIAGNOSIS COMPLETE

---

## Current State

**Dense-no-cal-v1 Performance**:
- Acc@1: 31.05% (+0.22% over Base)
- Net: +9 (402 recovered, 393 harmed)
- Multi-anchor: 28.34% (+5.3% over Base)
- Relative-position: 30.68% (+0.82% over Base)

**Assessment**: Real but modest gain. Significant room for improvement.

---

## DenseRelationModule Analysis

### Current Architecture (`cover3d_dense_relation.py`)

**Pair Feature Computation**:
```python
pair_features = concat([
    obj_i,           # [B, N, D]
    obj_j,           # [B, N, D]  
    lang_exp,        # [B, N, D]
    # Optional: relative_geom (not used in current config)
])
# pair_input_dim = 2*320 + 256 = 896 (no geometry)
```

**Relation MLP**:
```python
MLP: 896 → 256 → GELU → Dropout → 1 → Tanh
# 2 layers, 256 hidden dim, Tanh output
```

**Aggregation**:
```python
# For each candidate i:
pair_weights = softmax(relation_scores)  # [B, N, N]
relation_context = weighted_sum(j_embeddings)
relation_scores = weighted_agg(pair_scores)  # [B, N]
```

**Residual Enhancement**:
```python
enhanced = obj_embeddings + 0.1 * output_proj(relation_context)
```

---

## Identified Bottlenecks

### 1. Aggregation: Simple Weighted Sum (HIGH PRIORITY)

**Current**: Weighted average of all pair scores.

**Limitation**:
- No distinction between informative vs noisy pairs
- All anchors contribute equally (weighted by score)
- No modeling of anchor-candidate interaction patterns

**Potential Fix**:
- **Top-k pooling**: Only aggregate from highest-scoring pairs
- **Attention pooling**: Learn query vectors to select relevant anchors
- **Max + Mean hybrid**: Combine max relation signal with average context
- **Coverage-aware**: Penalize over-concentration on few anchors

**Expected Impact**: Better multi-anchor discrimination.

---

### 2. Pair Representation: Limited Geometry (HIGH PRIORITY)

**Current**: `use_geometry=False` in Round-1 config.

**Limitation**:
- No explicit spatial features (distance, angle, relative position)
- Embedding difference is implicit proxy for geometry
- May miss geometric cues important for spatial language

**Potential Fix**:
- **Add geometry features**: center_diff (3D), size (3D), distance (scalar)
- **Normalized geometry**: Divide by scene scale for invariance
- **Explicit distance encoding**: RBF kernels or bins
- **Direction encoding**: Spherical coordinates or relative directions

**Expected Impact**: Better relative-position and clutter discrimination.

---

### 3. Score Range: Tanh Bounds (MEDIUM PRIORITY)

**Current**: `Tanh` activation → output ∈ [-1, 1].

**Limitation**:
- Symmetric bounds may not match score distribution needs
- Negative scores: unclear interpretation (is -0.8 "bad" or "low confidence"?)
- May limit expressivity for hard cases

**Potential Fix**:
- **Remove Tanh**: Let MLP produce unbounded scores
- **Learnable temperature**: Scale scores for better separation
- **Margin-aware training**: Encourage score gap between correct/incorrect

**Expected Impact**: Better score separation, improved hard case handling.

---

### 4. Hard-Negative Training: No Explicit Focus (HIGH PRIORITY)

**Current**: Standard cross-entropy, all samples weighted equally.

**Limitation**:
- Easy samples dominate training
- Clutter / multi-anchor failures not emphasized
- Model doesn't learn from hard cases

**Potential Fix**:
- **Focal weighting**: Down-weight easy samples
- **Subset reweighting**: Emphasize clutter / multi-anchor samples
- **Hard-negative mining**: Focus on samples where model fails
- **Margin ranking loss**: Directly optimize score gap

**Expected Impact**: Reduced harmed, better clutter/multi-anchor performance.

---

### 5. Residual Enhancement: Fixed Scale (LOW PRIORITY)

**Current**: `residual_scale = 0.1` (fixed).

**Limitation**:
- Relation context contribution is constant
- May be too weak for relation-heavy cases
- No learning of enhancement strength

**Potential Fix**:
- **Learnable scale**: `enhanced = obj + alpha * relation_context`
- **Gated enhancement**: `enhanced = obj + gate * relation_context`

**Expected Impact**: Better integration of relation context.

---

## Summary: Prioritized Interventions

| Priority | Bottleneck | Fix Direction | Expected Impact |
|----------|-----------|---------------|-----------------|
| HIGH | Aggregation | Top-k / attention pooling | Multi-anchor +5% → +8% |
| HIGH | Geometry | Add spatial features | Relative-position +1% |
| HIGH | Hard-negative | Focal weighting | Harmed -10%, Net +15 |
| MEDIUM | Score range | Remove Tanh / temperature | Score separation |
| LOW | Residual scale | Learnable alpha | Minor refinement |

---

## Proposed Strengthening Variants

### Variant 1: Dense-v2-AttPool (Aggregation Focus)
- Replace weighted sum with attention pooling
- Learn query vector to select relevant anchors
- Expected: Better multi-anchor, higher net gain

### Variant 2: Dense-v3-Geo (Geometry Focus)
- Add explicit geometry features (distance, direction)
- Enable `use_geometry=True`
- Expected: Better relative-position, clutter discrimination

### Variant 3: Dense-v4-HardNeg (Hard-Negative Focus)
- Focal weighting on clutter/multi-anchor samples
- Emphasize hard cases in training
- Expected: Reduced harmed, higher net gain

---

## Next Steps

1. Implement Dense-v2-AttPool (highest priority)
2. Implement Dense-v3-Geo (geometry features)
3. Implement Dense-v4-HardNeg (hard-negative training)
4. Smoke test each variant
5. Run 10-epoch controlled reruns
6. Compare against anchors (Base, Dense-no-cal-v1)

---

## Files to Modify

- `src/rag3d/models/cover3d_dense_relation.py` - Core module changes
- `scripts/train_cover3d_round1.py` - Add variant configs, hard-negative loss
- `configs/cover3d_round1/` - Optional: variant config files
