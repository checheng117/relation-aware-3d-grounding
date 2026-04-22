# COVER-3D Phase 2: Integration Report

**Date**: 2026-04-19
**Status**: Implementation + Smoke Validation Complete

---

## Integration Target: ReferIt3DNet

COVER-3D is designed as a **model-agnostic reranker** that wraps around any base 3D grounding model. The primary integration target is the trusted ReferIt3DNet baseline.

### ReferIt3DNet Baseline

| Metric | Value |
|--------|-------|
| Test Acc@1 | 30.79% |
| Test Acc@5 | 91.75% |
| Checkpoint | `outputs/20260409_learned_class_embedding/formal/best_model.pt` |

---

## Integration Points

### 1. Base Logits

**Source**: ReferIt3DNet classification head output

**Description**: The raw prediction logits from the base model's final layer.

**Shape**: `[B, N]` where B is batch size, N is number of objects

**Access Pattern**:
```python
# From ReferIt3DNet forward pass
base_logits = model.classification_head(fusion_features)
```

---

### 2. Object Embeddings

**Source**: ReferIt3DNet encoder output (before classification)

**Description**: The object-level feature embeddings from the point encoder + language fusion.

**Shape**: `[B, N, D]` where D = 320 (256 point features + 64 class embedding)

**Access Pattern**:
```python
# Object features before classification
object_embeddings = model.fusion_layer(fused_features)  # Before classification head
```

**Construction**:
- Point encoder: SimplePointEncoder → 256-dim
- Class embedding: learned embedding → 64-dim
- Total: 320-dim per object

---

### 3. Utterance Features

**Source**: ReferIt3DNet language encoder output

**Description**: The utterance-level language features from BERT encoding.

**Shape**: `[B, 256]` (language hidden dimension)

**Access Pattern**:
```python
# Language features
lang_features = model.lang_encoder(bert_features)  # [B, 768] → [B, 256]
```

---

### 4. Geometry

**Source**: Scene object centers and sizes

**Status**: **LIMITATION - Geometry may be fallback**

**Description**: Spatial features for each object (center xyz + size xyz).

**Shape**: `[B, N, 6]`

**Expected**: Actual ScanNet geometry with real 3D positions

**Actual**: May be fallback placeholders:
- centers: `[0.0, 0.0, 0.0]` (fallback)
- sizes: `[0.1, 0.1, 0.1]` (fallback)

**Fallback Handling**:
```python
# COVER-3D handles missing geometry
if object_geometry is None:
    # Use embedding similarity as proxy
    dense_relation_scores = compute_from_embeddings_only()
else:
    # Use full geometric features
    dense_relation_scores = compute_with_geometry()
```

**Impact**:
- Geometry-based relation scoring will be limited
- Module uses embedding distance as proxy
- Dense coverage still computed (all N² pairs)

---

### 5. Candidate Mask

**Source**: Scene metadata

**Description**: Boolean mask indicating valid objects.

**Shape**: `[B, N]`

**Default**: All objects valid (no masking)

---

### 6. Class Features

**Source**: Learned class embedding

**Description**: Semantic class features for each object.

**Shape**: `[B, N, 64]`

**Access**:
```python
# Class indices → embeddings
class_features = model.class_embedding(class_indices)
```

---

## Integration Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    ReferIt3DNet + COVER-3D                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    ReferIt3DNet                               │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │ │
│  │  │ Point       │  │ Language    │  │ Fusion + Classifier │   │ │
│  │  │ Encoder     │  │ Encoder     │  │                     │   │ │
│  │  │ [B,N,256]   │  │ [B,256]     │  │ base_logits [B,N]   │   │ │
│  │  └──────┬──────┘  └──────┬──────┘  │ obj_emb [B,N,320]  │   │ │
│  │         │                │         └───────┬─────────────┘   │ │
│  └─────────┴────────────────┴─────────────────┼─────────────────┘ │
│                                               │                    │
│                                               ▼                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    COVER-3D Reranker                          │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │ Input: base_logits, obj_emb, lang_feat, geometry        │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │                              │                                │ │
│  │                              ▼                                │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │ DenseRelationModule → relation_scores [B,N]             │ │ │
│  │  │ (Chunked, memory-safe, all N² pairs)                    │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │                              │                                │ │
│  │                              ▼                                │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │ SoftAnchorPosterior → anchor_posterior [B,N]            │ │ │
│  │  │ anchor_entropy [B], top_anchor_margin [B]               │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │                              │                                │ │
│  │                              ▼                                │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │ CalibratedFusion → fused_logits [B,N]                   │ │ │
│  │  │ gate_values [B]                                         │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │                              │                                │ │
│  └──────────────────────────────┼───────────────────────────────┘ │
│                               ▼                                    │
│                      final_logits [B,N]                            │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Integration Code Pattern

```python
# Integration with ReferIt3DNet
from rag3d.models import Cover3DModel

# Load base model
base_model = ReferIt3DNet.load_from_checkpoint(checkpoint_path)

# Create COVER-3D reranker
cover3d = Cover3DModel(
    object_dim=320,  # Matches ReferIt3DNet output
    language_dim=256,  # Matches ReferIt3DNet lang output
    geometry_dim=6,  # 3 center + 3 size
    class_dim=64,  # Matches class embedding
)

# Forward pass
# Step 1: Get base model outputs
with torch.no_grad():
    base_logits, object_embeddings = base_model.forward_with_embeddings(
        point_features, lang_features, class_features
    )

# Step 2: Apply COVER-3D reranking
result = cover3d.forward(
    base_logits=base_logits,
    object_embeddings=object_embeddings,
    utterance_features=lang_features,
    object_geometry=geometry,  # May be None/fallback
    candidate_mask=candidate_mask,
)

# Step 3: Use reranked predictions
final_logits = result["fused_logits"]
final_predictions = final_logits.argmax(dim=-1)
```

---

## Parts Ready for Phase 3 Training

| Component | Status |
|-----------|--------|
| COVER-3D modules | IMPLEMENTED |
| Smoke validation | PASSED |
| Forward pass | WORKING |
| Module independence | WORKING |
| Config files | CREATED |

---

## Parts NOT Ready (Blocked)

| Blocker | Reason |
|---------|--------|
| Full geometry integration | Geometry may be fallback placeholders |
| GPU training | Need stable hardware (driver instability) |
| Multi-seed experiments | Need formal training infrastructure |
| Hard subset evaluation | Need trained model checkpoints |

---

## Fallback Strategy for Limited Geometry

When geometry is unavailable or fallback:

1. **Dense relation module**: Uses embedding similarity as proxy
   - Pairwise features: `[obj_i, obj_j, language]` (no geometry)
   - Relation scoring: conditioned on language only

2. **Anchor posterior**: Uses class-utterance matching
   - Class features provide semantic anchor candidates
   - Language features match anchor semantics

3. **Fusion gate**: Still calibrated
   - Entropy/margin signals still computed
   - Gate bounds prevent collapse

---

## Recommended Integration Test

Before Phase 3 formal training:

```python
# Integration smoke test with actual ReferIt3DNet outputs
def test_integration():
    # Load a test batch from official_scene_disjoint
    batch = load_test_batch()

    # Get base model outputs
    base_logits = base_model(batch)

    # Apply COVER-3D
    result = cover3d.forward(
        base_logits=base_logits,
        object_embeddings=object_embeddings,
        utterance_features=lang_features,
    )

    # Verify shapes and no NaN
    assert result["fused_logits"].shape == base_logits.shape
    assert not torch.isnan(result["fused_logits"]).any()

    # Compare predictions
    base_pred = base_logits.argmax()
    cover_pred = result["fused_logits"].argmax()
    print(f"Base: {base_pred}, COVER-3D: {cover_pred}")
```

---

## Summary

COVER-3D implementation skeleton is complete and smoke-validated. Integration with ReferIt3DNet requires:

1. Extracting object embeddings from base model (modify forward hook)
2. Handling geometry limitations gracefully
3. Running formal training on stable GPU hardware

**Phase 2 Status**: COMPLETE ✓