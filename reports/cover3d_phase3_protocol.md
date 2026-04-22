# COVER-3D Phase 3: Training Protocol

**Date**: 2026-04-19
**Phase**: Protocol + Short Validation First

---

## Executive Summary

Phase 3 objective: Validate COVER-3D end-to-end training on real data and produce subset-first evaluation.

This phase is NOT full paper-final training. It produces:
- Clean training protocol
- Short end-to-end validation run
- Subset-first evaluation report
- Go / no-go decision for longer training

---

## 1. Exact Backbone Used First

**Backbone**: ReferIt3DNet (trusted baseline only)

| Property | Value |
|----------|-------|
| Test Acc@1 | 30.79% |
| Test Acc@5 | 91.75% |
| Checkpoint | `outputs/20260409_learned_class_embedding/formal/best_model.pt` |
| Object dim | 320 (256 point + 64 class embedding) |
| Language dim | 256 |

**Rationale**: Use only the trusted baseline first. SAT (28.27%) is weaker and not the comparison anchor. No cross-backbone experiments until COVER-3D proves itself on ReferIt3DNet.

---

## 2. Exact Input/Output Path

### Training Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COVER-3D Training Flow                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  [Dataset]                                                        │
│  official_scene_disjoint/test_manifest.jsonl                      │
│           │                                                         │
│           ▼                                                         │
│  [Data Loading]                                                    │
│  ReferIt3DManifestDataset → batch                                  │
│  - object_features [B, N, 256] (placeholder)                       │
│  - texts [B]                                                       │
│  - target_index [B]                                                │
│  - object_mask [B, N]                                              │
│           │                                                         │
│           ▼                                                         │
│  [Feature Extraction]                                              │
│  BERT encoder → text_features [B, 768]                             │
│  Language encoder → lang_features [B, 256]                         │
│  Point encoder + class → object_embeddings [B, N, 320]             │
│           │                                                         │
│           ▼                                                         │
│  [Base Model Forward]                                              │
│  ReferIt3DNet classification → base_logits [B, N]                  │
│           │                                                         │
│           ▼                                                         │
│  [COVER-3D Forward]                                                │
│  DenseRelationModule → relation_scores [B, N]                      │
│  SoftAnchorPosterior → anchor_posterior [B, N], entropy [B]        │
│  CalibratedFusion → fused_logits [B, N]                            │
│           │                                                         │
│           ▼                                                         │
│  [Loss Computation]                                                │
│  CrossEntropy(fused_logits, target_index)                          │
│           │                                                         │
│           ▼                                                         │
│  [Backprop]                                                        │
│  Update COVER-3D parameters only (base frozen)                     │
│                                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Input Extraction Details

| Input | Source | Shape | Notes |
|-------|--------|-------|-------|
| `base_logits` | ReferIt3DNet forward (frozen) | [B, N] | No gradient through base |
| `object_embeddings` | ReferIt3DNet encoder output | [B, N, 320] | Before classification |
| `utterance_features` | Language encoder output | [B, 256] | Projected BERT features |
| `object_geometry` | Dataset (may be fallback) | [B, N, 6] | Limited/placeholder |
| `candidate_mask` | `object_mask` | [B, N] | From dataset |
| `class_features` | Learned class embedding | [B, N, 64] | From base model |

### Output Path

| Output | Shape | Use |
|--------|-------|-----|
| `fused_logits` | [B, N] | Training loss, evaluation |
| `gate_values` | [B] | Diagnostics |
| `anchor_entropy` | [B] | Calibration analysis |
| `diagnostics` | dict | Logging, debugging |

---

## 3. Training Stages

### Stage A: Integration Sanity Check (CPU or short GPU)

**Objective**: Verify end-to-end pipeline runs without crash

**Duration**: < 10 minutes

**Steps**:
1. Load ReferIt3DNet checkpoint
2. Attach COVER-3D reranker
3. Run 1-2 forward passes on test batch
4. Verify shapes, no NaN, no inf

**Success Criteria**:
- No crash
- Output shapes match expectations
- No NaN/inf in fused_logits
- Gate values in [0.1, 0.9]

**If Fail**: Debug integration before proceeding

---

### Stage B: Short Training Validation (5 epochs)

**Objective**: Validate training stability, produce first checkpoint

**Duration**: 2-3 hours (GPU) or ~30 min (CPU with tiny subset)

**Parameters**:
| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch size | 8 (GPU) or 4 (CPU) |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Weight decay | 1e-5 |
| Frozen base | Yes (only COVER-3D trainable) |
| Seed | 42 |

**Logging**:
- Every 50 batches: loss, gate_mean, anchor_entropy_mean
- Every epoch: val Acc@1, train loss mean
- End: save checkpoint

**Success Criteria**:
- Training completes without crash
- Loss decreases (not exploding)
- Gate values not collapsed (not all 0.1 or all 0.9)
- Overall Acc@1 ≥ baseline - 2% (≥ 28.79%)

**If Fail**: Analyze diagnostics, adjust calibration

---

### Stage C: Longer Training (Only if justified)

**Objective**: Achieve target accuracy on hard subsets

**Duration**: 8-10 hours (20 epochs)

**Trigger**: Only run if Stage B shows:
- Overall Acc@1 ≥ 28.79% (within 2% of baseline)
- Hard subsets show improvement trend
- Training stable (no NaN, no collapse)

**Parameters**:
| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Batch size | 16 |
| Learning rate | 1e-4 (same) |
| Seeds | 3 (for final validation) |

---

## 4. Evaluation Metrics

### Primary Metrics (Required Every Run)

| Metric | Description | Baseline | Target |
|--------|-------------|----------|--------|
| Overall Acc@1 | Test set accuracy | 30.79% | ≥ 32% |
| Overall Acc@5 | Top-5 accuracy | 91.75% | Near baseline |
| Same-Class Clutter Acc@1 | Clutter ≥ 3 | 21.96% | ≥ 25% |
| High Clutter Acc@1 | Clutter ≥ 5 | 16.07% | ≥ 20% |
| Multi-Anchor Acc@1 | 2+ anchors | 11.90% | ≥ 20% |

### Diagnostic Metrics (Logged During Training)

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| Gate mean | Average fusion gate | 0.2 - 0.5 |
| Gate std | Gate variation | > 0.01 (not collapsed) |
| Anchor entropy mean | Anchor uncertainty | 0.5 - 2.0 |
| Top-anchor margin | Anchor concentration | 0.1 - 0.5 |
| Base margin | Base confidence | Varies |
| Relation margin | Relation concentration | Varies |
| Loss | Training loss | Decreasing, not exploding |

---

## 5. Subset Definitions (Exact Logic)

### Same-Class Clutter (≥3)

```python
# From manifest objects
target_class = objects[target_index].class_name
same_class_count = sum(1 for obj in objects if obj.class_name == target_class)
is_clutter = same_class_count >= 3
```

### High Clutter (≥5)

```python
is_high_clutter = same_class_count >= 5
```

### Multi-Anchor

```python
# From annotation entities (if available)
# entities field: ["target_id_class", "anchor1_id_class", "anchor2_id_class", ...]
anchor_count = len(entities) - 1  # Exclude first (target)
is_multi_anchor = anchor_count >= 2
```

**Note**: Entity annotations only cover ~11% of samples. Multi-anchor subset is small but extremely hard.

---

## 6. Decision Criteria

### Proceed to Longer Training (Decision A)

| Criterion | Threshold |
|-----------|-----------|
| Training stable | No crash, no NaN |
| Overall Acc@1 | ≥ 28.79% (within 2% of baseline) |
| Gate not collapsed | std > 0.01, mean ∈ [0.2, 0.5] |
| Hard subset trend | Any improvement on clutter/multi-anchor |

### Revise Training (Decision B)

| Criterion | Threshold |
|-----------|-----------|
| Overall Acc@1 | < 28.79% but hard subsets improve |
| Gate issues | Collapsed or saturating |
| Calibration problems | Entropy/margin not varying |

**Action**: Adjust gate bounds, learning rate, or entropy regularization

### Stop and Debug (Decision C)

| Criterion | Threshold |
|-----------|-----------|
| Training crash | GPU instability or software error |
| NaN/inf | Persistent in outputs |
| Broken integration | Shapes mismatch, wrong data flow |

---

## 7. Data Split

**Split**: `official_scene_disjoint`

| Split | Samples | Scenes |
|-------|---------|--------|
| Train | 33,829 | 512 |
| Val | 3,419 | 64 |
| Test | 4,255 | 65 |

**Evaluation**: Test split only (scene-disjoint ensures unseen scenes)

---

## 8. Hardware Considerations

### Known Issue

Previous machine showed GPU driver instability:
- Implicit v3 crashed at epoch 17
- v1 crashed at epoch 17 (memory spike)

### Mitigation

1. **Short runs first**: 5 epochs to validate stability
2. **Chunked processing**: COVER-3D uses chunk_size=16
3. **CPU fallback**: If GPU unstable, run short validation on CPU
4. **Checkpointing**: Save every epoch to recover from crashes

---

## 9. Output Files

| File | Content |
|------|---------|
| `reports/cover3d_phase3_protocol.md` | This document |
| `scripts/train_cover3d_referit.py` | Training script |
| `scripts/evaluate_cover3d_subsets.py` | Subset evaluation script |
| `configs/cover3d_phase3_short.yaml` | Short training config |
| `reports/cover3d_phase3_short_validation.md` | Validation results |
| `reports/cover3d_phase3_diagnostics.md` | Training diagnostics |
| `reports/cover3d_phase3_decision.md` | Go/no-go decision |

---

## 10. Execution Order

| Step | Task | Priority |
|------|------|----------|
| 0 | Freeze protocol | DONE (this document) |
| 1 | Create training script | HIGH |
| 2 | Create subset evaluation script | HIGH |
| 3 | Create short config | MEDIUM |
| 4 | Run Stage A sanity check | HIGH |
| 5 | Run Stage B short training | HIGH |
| 6 | Report diagnostics | MEDIUM |
| 7 | Make decision | HIGH |

---

## Final Statement

**Protocol frozen. Implementation may proceed.**

**First step**: Create `scripts/train_cover3d_referit.py`

**Key constraint**: Base model frozen, only COVER-3D trainable. Subset metrics required.