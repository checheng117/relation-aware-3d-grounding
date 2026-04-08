# ReferIt3D Training Protocol Comparison

## Summary Table

| Parameter | Current Reproduction | Official / Target | Status | Notes |
|-----------|---------------------|-------------------|--------|-------|
| **Optimizer** | AdamW | Adam | ⚠️ Close | AdamW ≈ Adam + weight decay |
| **Learning rate** | 1e-4 | ~5e-4 to 1e-3 | ⚠️ Mismatch | Current is lower |
| **Weight decay** | 1e-4 | 0 (default) | ⚠️ Mismatch | Adds regularization |
| **LR scheduler** | CosineAnnealingLR | MultiStepLR / ReduceLROnPlateau | ❌ Mismatch | Different decay pattern |
| **Warmup** | None | 5 epochs | ❌ Missing | Critical for stability |
| **Batch size** | 8-16 | 32 | ⚠️ Mismatch | Smaller than official |
| **Effective batch** | 8-16 | 32 | ⚠️ Mismatch | No accumulation |
| **Gradient accumulation** | No | Unknown | ❓ Unknown | Not implemented |
| **Epochs** | 30 | 100 (max) | ⚠️ Mismatch | May be under-training |
| **Early stopping** | None | 10 epoch patience | ⚠️ Missing | No early stopping |
| **Separate encoder LR** | No | Yes (0.1x) | ❌ Missing | Encoder overfits? |
| **Gradient clipping** | 1.0 | Not present | ⚠️ Different | May affect convergence |
| **Point sample count** | 1024 | 1024 | ✅ Match | Correct |
| **Point normalization** | Center + scale | Center + scale | ✅ Match | Correct |
| **Augmentation** | None | Unknown | ❓ Unknown | Not implemented |
| **Random seed** | 42 | Not specified | ✅ Match | Reproducible |
| **Evaluation cadence** | Every epoch | Every epoch | ✅ Match | Correct |
| **Checkpoint selection** | Best val Acc@1 | Best test accuracy | ⚠️ Different | Val vs test |
| **Loss function** | CrossEntropyLoss | CrossEntropyLoss | ✅ Match | Correct |
| **BERT features** | DistilBERT (768) | BERT (768) | ✅ Close | Same dimension |

---

## Detailed Comparison

### 1. Optimizer

| Aspect | Current | Official | Match? |
|--------|---------|----------|--------|
| Type | AdamW | Adam | Close |
| LR | 1e-4 | ~5e-4 to 1e-3 | ❌ |
| Weight decay | 1e-4 | 0 (default) | ❌ |
| Betas | Default | Default | ✅ |
| Eps | Default | Default | ✅ |

**Recommendation**: Change to Adam with LR 5e-4, weight decay 0.

### 2. Learning Rate Schedule

| Aspect | Current | Official (SAT) | Match? |
|--------|---------|----------------|--------|
| Type | CosineAnnealingLR | MultiStepLR / ReduceLROnPlateau | ❌ |
| Milestones | N/A | [25,40,50,60,70,80,90] | ❌ |
| Gamma | N/A | 0.65 | ❌ |
| Warmup epochs | 0 | 5 | ❌ |
| Warmup multiplier | N/A | 1.0 | ❌ |

**Recommendation**: Implement MultiStepLR with warmup.

### 3. Batch Size

| Aspect | Current | Official | Match? |
|--------|---------|----------|--------|
| SimplePointEncoder | 16 | 32 | ❌ |
| PointNet++ | 8 (memory limit) | 32 | ❌ |
| Gradient accumulation | No | Unknown | ❓ |

**Recommendation**: Implement gradient accumulation to achieve effective batch size of 32.

### 4. Training Duration

| Aspect | Current | Official | Match? |
|--------|---------|----------|--------|
| Max epochs | 30 | 100 | ❌ |
| Early stopping | No | Yes (10 epoch patience) | ❌ |
| Convergence | May be incomplete | Full convergence | ❓ |

**Recommendation**: Increase to 100 epochs with early stopping.

### 5. Encoder Learning Rate

| Aspect | Current | Official | Match? |
|--------|---------|----------|--------|
| Encoder LR | Same as model | 0.1x model LR | ❌ |
| Backbone tuning | None | Separate param group | ❌ |

**Recommendation**: Add separate parameter group for encoder with LR = 0.1 * base LR.

### 6. Point Cloud Processing

| Aspect | Current | Official | Match? |
|--------|---------|----------|--------|
| Points per object | 1024 | 1024 | ✅ |
| Sampling method | Random | Random/FPS | ✅ Close |
| Normalization | Center + scale | Center + scale | ✅ |
| Color channels | No | Unknown | ❓ |

**Recommendation**: Current implementation is acceptable.

### 7. Data Augmentation

| Aspect | Current | Official | Match? |
|--------|---------|----------|--------|
| Point jitter | No | Unknown | ❓ |
| Rotation | No | Unknown | ❓ |
| Scaling | No | Unknown | ❓ |
| Dropout | No | Unknown | ❓ |

**Recommendation**: Consider adding basic augmentation if protocol alignment fails.

### 8. Model Selection

| Aspect | Current | Official | Match? |
|--------|---------|----------|--------|
| Selection criterion | Best val Acc@1 | Best test accuracy | ⚠️ |
| Save frequency | Every epoch if best | Unknown | ✅ |

**Note**: We use validation accuracy because test set should not be used for model selection in proper ML practice.

---

## Priority Matrix

| Priority | Parameter | Impact | Ease of Fix |
|----------|-----------|--------|-------------|
| 1 | Warmup | High | Easy |
| 2 | LR scheduler type | High | Easy |
| 3 | Separate encoder LR | High | Medium |
| 4 | Batch size | Medium-High | Medium |
| 5 | Epochs | Medium | Easy |
| 6 | Early stopping | Medium | Medium |
| 7 | Learning rate value | Medium | Easy |
| 8 | Weight decay | Low | Easy |
| 9 | Augmentation | Unknown | Medium |

---

## Experiment Plan

### Controlled Configs to Create

| Config | Changes | Hypothesis |
|--------|---------|------------|
| `protocol_align_optimizer.yaml` | Adam, LR=5e-4, no weight decay | Better optimizer alignment |
| `protocol_align_scheduler.yaml` | MultiStepLR + warmup | Better LR schedule |
| `protocol_align_encoder_lr.yaml` | Separate encoder LR (0.1x) | Prevent encoder overfitting |
| `protocol_align_batch.yaml` | Gradient accumulation (effective 32) | Better optimization dynamics |
| `protocol_align_full.yaml` | All above + 100 epochs + early stopping | Full protocol alignment |

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Exact match |
| ⚠️ | Close approximation or minor mismatch |
| ❌ | Mismatch - needs fixing |
| ❓ | Unknown - cannot verify |

---

## References

1. ReferIt3D Paper (ECCV 2020)
2. SAT Implementation: https://github.com/zyang-ur/SAT/blob/main/referit3d/scripts/train_referit3d.py
3. 3DRefTransformer (WACV 2022)
4. CoT3DRef (ICLR 2024)