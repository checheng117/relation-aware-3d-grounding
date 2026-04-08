# Encoder Upgrade Protocol

## Objective

Perform a controlled comparison between SimplePointEncoder and PointNet++ encoder to determine whether encoder upgrade materially reduces the reproduction gap to the 35.6% official baseline.

---

## Fixed Components

These components MUST remain identical between baseline and upgraded runs.

### Dataset

| Component | Value | Source |
|-----------|-------|--------|
| Split | NR3D train/val/test | `data/processed/*_manifest.jsonl` |
| Train samples | 1235 | Manifest count |
| Val samples | 154 | Manifest count |
| Test samples | 155 | Manifest count |
| Candidate set | full_scene | All objects per scene |

### Geometry Pipeline

| Component | Value |
|-----------|-------|
| Geometry source | `data/geometry/*.npz` |
| Centers | Real bbox centers |
| Sizes | Real bbox sizes |
| Normalization | center / 5.0, size / 2.0 (for SimplePointEncoder) |

### BERT Features

| Component | Value |
|-----------|-------|
| Model | `distilbert-base-uncased` |
| Dimension | 768 |
| Source | `data/text_features/*_bert_embeddings.npy` |
| Coverage | 100% |

### Language Encoder

| Component | Value |
|-----------|-------|
| Class | `SimpleLanguageEncoder` |
| Input dim | 768 (BERT) |
| Hidden dim | 256 |
| Output dim | 256 |

### Fusion Layer

| Component | Value |
|-----------|-------|
| Type | Concatenation + 2-layer MLP |
| Input | object_features + lang_features (concatenated) |
| Hidden | 512 |
| Output | 512 |
| Dropout | 0.1 |

### Classifier

| Component | Value |
|-----------|-------|
| Type | Linear |
| Input | 512 (fusion_dim) |
| Output | 1 (per-object score) |

### Loss

| Component | Value |
|-----------|-------|
| Type | CrossEntropyLoss |
| Reduction | mean |
| Target | target_index per sample |

### Optimizer

| Component | Value |
|-----------|-------|
| Type | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Gradient clipping | 1.0 |

### Scheduler

| Component | Value |
|-----------|-------|
| Type | CosineAnnealingLR |
| T_max | 30 epochs |

### Training

| Component | Value |
|-----------|-------|
| Epochs | 30 |
| Batch size | 16 |
| Seed | 42 |
| Device | cuda (if available) |

### Evaluation

| Component | Value |
|-----------|-------|
| Metrics | Acc@1, Acc@5 |
| Batch size | 32 |
| Splits | val, test |

---

## Changed Component

### ONLY Variable: Object Encoder

| Run | Encoder | Input | Output |
|-----|---------|-------|--------|
| Baseline | SimplePointEncoder | [B, N, 256] (center+size+hash) | [B, N, 256] |
| Upgraded | PointNetPPEncoder | [B, N, P, 3] (raw points) | [B, N, 256] |

### Encoder Specification

**SimplePointEncoder (Baseline)**:
- 3-layer MLP
- Input: 256-dim hand-crafted features
- Hidden: 128
- Output: 256
- Parameters: ~82K

**PointNetPPEncoder (Upgraded)**:
- Hierarchical point set abstraction (2-3 levels)
- Input: Raw XYZ points (1024 sampled per object)
- Local grouping with ball query
- Mini-PointNet per group
- Max pooling aggregation
- Output: 256 (same as baseline)
- Parameters: ~300-500K (estimated)

---

## Primary Metrics

| Metric | Description | Target Threshold |
|--------|-------------|------------------|
| Val Acc@1 | Validation accuracy at top-1 | Primary comparison |
| Test Acc@1 | Test accuracy at top-1 | Primary comparison |
| Test Acc@5 | Test accuracy at top-5 | Primary comparison |

---

## Secondary Metrics

| Metric | Description |
|--------|-------------|
| Training stability | Loss curve, no NaN/Inf |
| Convergence behavior | Epoch where best val acc achieved |
| Per-subset accuracy | Easy/hard if available |
| Runtime | Training time per epoch |
| Memory | GPU memory usage |

---

## Success Thresholds

### Threshold Definitions

| Threshold | Val Acc@1 | Interpretation |
|-----------|-----------|----------------|
| Meaningful gain | ≥ +3.0 absolute points | Encoder upgrade provides detectable benefit |
| Strong gain | ≥ +5.0 absolute points | Encoder upgrade is significant |
| Near-credible reproduction | ≥ 30.0% | Approaching target zone |
| Credible target zone | 32-35%+ | Full baseline reproduction achieved |

### Decision Matrix

| Result | Decision |
|--------|----------|
| +5 or more on Val Acc@1 | **A**: Encoder upgrade closes gap → baseline anchor trustworthy |
| +3 to +5 on Val Acc@1 | **B**: Partial benefit → investigate training/protocol refinement |
| Less than +3 on Val Acc@1 | **C**: Limited benefit → bottleneck shifted elsewhere |

---

## Experimental Runs

### Run 1: Baseline (SimplePointEncoder)

**Purpose**: Confirm baseline results before encoder change

**Configuration**:
- Encoder: SimplePointEncoder
- All fixed components as specified above
- Output: `outputs/encoder_upgrade/baseline_simple_encoder/`

**Expected metrics** (from previous run):
- Val Acc@1: ~22.73%
- Test Acc@1: ~9.68%
- Test Acc@5: ~40.00%

### Run 2: Smoke Test (PointNetPPEncoder)

**Purpose**: Verify PointNet++ implementation works without shape errors

**Configuration**:
- Encoder: PointNetPPEncoder
- Debug mode: max_batches=5
- Single epoch
- Check: forward pass, backward pass, no NaN, no shape errors

**Output**: `outputs/encoder_upgrade/smoke_pointnetpp/`

### Run 3: Controlled Verification (PointNetPPEncoder)

**Purpose**: Intermediate run to assess improvement

**Configuration**:
- Encoder: PointNetPPEncoder
- Full epochs: 30
- All fixed components unchanged

**Output**: `outputs/encoder_upgrade/verification_pointnetpp/`

### Run 4: Formal Controlled Rerun (PointNetPPEncoder)

**Purpose**: Final controlled comparison

**Configuration**:
- Encoder: PointNetPPEncoder
- Full epochs: 30
- All fixed components unchanged
- Fresh checkpoint directory

**Output**: `outputs/encoder_upgrade/formal_pointnetpp_rerun/`

---

## Comparison Protocol

### Metrics to Compare

| Metric | Baseline | Upgraded | Delta |
|--------|----------|----------|-------|
| Val Acc@1 | recorded | recorded | Δ_val |
| Test Acc@1 | recorded | recorded | Δ_test |
| Test Acc@5 | recorded | recorded | Δ_test5 |
| Best epoch | recorded | recorded | Δ_epoch |
| Training time | recorded | recorded | Δ_time |

### Comparison Output

Create:
- `reproduction_stage_comparison_with_encoder.json`
- `reproduction_stage_comparison_with_encoder.md`

Format:
```json
{
  "stages": {
    "placeholder": {"val_acc@1": 11.03, "test_acc@1": 1.94},
    "geometry_recovered": {"val_acc@1": 14.29, "test_acc@1": 1.94},
    "feature_integrated": {"val_acc@1": 22.73, "test_acc@1": 9.68},
    "encoder_upgraded": {"val_acc@1": X, "test_acc@1": Y}
  },
  "improvement_from_encoder": {
    "val_acc@1_delta": X - 22.73,
    "test_acc@1_delta": Y - 9.68
  },
  "gap_to_official": {
    "val": 35.6 - X,
    "test": 35.0 - Y
  }
}
```

---

## Batching Logic

### Baseline Collate (SimplePointEncoder)

```python
# Current collate constructs:
object_features[i, j, 0:3] = center / 5.0
object_features[i, j, 3:6] = size / 2.0
object_features[i, j, 6 + feat_hash] = 1.0  # class hash
```

### Upgraded Collate (PointNetPPEncoder)

```python
# Modified collate must:
# 1. Load raw points from geometry file
# 2. Normalize points: points - center, / size
# 3. Sample fixed number (1024) via FPS or random
# 4. Stack into [B, N, P, 3] tensor
```

### Shape Adaptations Allowed

| Change | Required | Reason |
|--------|----------|--------|
| Load raw points | Yes | PointNet++ requires point clouds |
| Normalize differently | Yes | PointNet++ expects centered/scaled points |
| Sample fixed count | Yes | Variable point counts need batching |
| Pad/truncate | Yes | Batch requires uniform shape |

---

## Encoder Use Verification

### Required Audit Content

Create `outputs/encoder_upgrade/.../encoder_use_audit.md`:

| Check | Evidence |
|-------|----------|
| Encoder type | Config + model class name |
| Input shape | Tensor shape at forward entry |
| Output shape | Tensor shape at forward exit |
| Parameters | Count from `sum(p.numel())` |
| Pretrained weights | Yes/No, source if used |
| Real points reaching encoder | Collate coverage check |

### Verification Script

```python
# In model forward, log:
log.info(f"Encoder type: {self.point_encoder.__class__.__name__}")
log.info(f"Input shape: {points.shape}")
# After encoding:
log.info(f"Output shape: {obj_features.shape}")
log.info(f"Encoder params: {sum(p.numel() for p in self.point_encoder.parameters())}")
```

---

## Constraints

### Do Not Change

- Dataset splits
- BERT features
- Language encoder
- Fusion architecture
- Classifier
- Loss function
- Optimizer/scheduler hyperparameters
- Training duration
- Seed

### Do Not Do

- Hyperparameter sweeps
- Multi-view features
- Custom structured methods
- MVT reproduction
- Data augmentation experiments
- Attention-based fusion changes

---

## Acceptance Criteria

This phase is complete ONLY if:

| Criterion | Evidence |
|-----------|----------|
| PointNet++ encoder implemented | Code in `referit3d_net.py` |
| Encoder-use audit exported | `encoder_use_audit.md` |
| Smoke run completed | No errors in smoke output |
| Formal rerun completed | Full training results |
| Stage comparison exported | JSON + MD comparison files |
| Bottleneck reassessment written | `post_encoder_bottleneck_reassessment.md` |
| Final decision report written | `encoder_upgrade_results.md` |

---

## Commands Reference

### Baseline Run

```bash
python repro/referit3d_baseline/scripts/train.py \
    --config repro/referit3d_baseline/configs/official_baseline.yaml \
    --device cuda
```

### Upgraded Run

```bash
python repro/referit3d_baseline/scripts/train.py \
    --config repro/referit3d_baseline/configs/pointnetpp_encoder.yaml \
    --device cuda
```

### Evaluation

```bash
python repro/referit3d_baseline/scripts/evaluate.py \
    --checkpoint outputs/encoder_upgrade/formal_pointnetpp_rerun/best_model.pt \
    --split val \
    --device cuda

python repro/referit3d_baseline/scripts/evaluate.py \
    --checkpoint outputs/encoder_upgrade/formal_pointnetpp_rerun/best_model.pt \
    --split test \
    --device cuda
```

---

## Timeline Estimate

| Step | Duration |
|------|----------|
| Implement PointNet++ encoder | 2-3 hours |
| Modify collate for raw points | 1-2 hours |
| Smoke test | 15 min |
| Verification run | 1-2 hours |
| Formal rerun | 2-3 hours |
| Analysis + reports | 1 hour |

**Total**: ~8-10 hours

---

## Summary

This protocol ensures a clean controlled comparison between SimplePointEncoder and PointNet++ while keeping all other experimental variables fixed. The primary outcome is determining whether encoder architecture is the dominant bottleneck remaining in the reproduction gap.