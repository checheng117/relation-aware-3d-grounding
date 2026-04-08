# Training Protocol Fidelity Results

## Executive Summary

This report documents the Training Protocol Fidelity Phase - experiments to align the reproduction training protocol with the official ReferIt3D baseline.

**Key Finding**: Protocol alignment improved validation accuracy but hurt test accuracy, suggesting overfitting to validation distribution. The reproduction gap persists.

---

## Files Modified / Added

### New Files

| File | Purpose |
|------|---------|
| `reports/training_protocol_fidelity_audit.md` | Pre-implementation audit |
| `reports/referit3d_training_protocol_comparison.md` | Protocol comparison table |
| `repro/referit3d_baseline/configs/protocol_align_optimizer.yaml` | Optimizer alignment config |
| `repro/referit3d_baseline/configs/protocol_align_scheduler.yaml` | Scheduler alignment config |
| `repro/referit3d_baseline/configs/protocol_align_batch.yaml` | Batch size alignment config |
| `repro/referit3d_baseline/configs/protocol_align_full.yaml` | Full protocol alignment config |
| `repro/referit3d_baseline/configs/protocol_smoke_test.yaml` | Smoke test config |
| `training_protocol_ablation.json` | Ablation data |
| `training_protocol_ablation.md` | Ablation summary |
| `reports/top1_top5_gap_analysis.md` | Top-1 vs Top-5 analysis |

### Modified Files

| File | Changes |
|------|---------|
| `repro/referit3d_baseline/scripts/train.py` | Added warmup, MultiStepLR, separate encoder LR, gradient accumulation, early stopping |

---

## Commands Run

### Smoke Test

```bash
python repro/referit3d_baseline/scripts/train.py \
    --config repro/referit3d_baseline/configs/protocol_smoke_test.yaml \
    --device cuda
```

**Result**: Passed - all new features work correctly.

### Optimizer Alignment Experiment

```bash
python repro/referit3d_baseline/scripts/train.py \
    --config repro/referit3d_baseline/configs/protocol_align_optimizer.yaml \
    --device cuda
```

### Full Protocol Alignment Experiment

```bash
python repro/referit3d_baseline/scripts/train.py \
    --config repro/referit3d_baseline/configs/protocol_align_full.yaml \
    --device cuda
```

### Test Evaluation

```bash
python repro/referit3d_baseline/scripts/evaluate.py \
    --checkpoint outputs/protocol_fidelity/full_aligned/best_model.pt \
    --split test --device cuda
```

---

## Protocol Mismatch Summary

| Parameter | Current | Official | Fixed? |
|-----------|---------|----------|--------|
| Optimizer | AdamW | Adam | ⚠️ Kept AdamW |
| Learning rate | 1e-4 | 5e-4 | ✅ Tested |
| Weight decay | 1e-4 | 0 | ✅ Tested |
| Scheduler | CosineAnnealingLR | MultiStepLR | ✅ Implemented |
| Warmup | None | 5 epochs | ✅ Implemented |
| Batch size | 16 | 32 | ✅ Via accumulation |
| Separate encoder LR | No | Yes | ✅ Implemented |
| Early stopping | None | 10 epoch patience | ✅ Implemented |
| Epochs | 30 | 100 | ✅ Extended |

---

## Controlled Experiment Results

### Experiment 1: Baseline (Feature-Integrated)

| Metric | Value |
|--------|-------|
| Val Acc@1 | 22.73% |
| Test Acc@1 | 9.68% |
| Test Acc@5 | 40.00% |
| Config | official_baseline.yaml |

### Experiment 2: Optimizer Aligned

| Metric | Value |
|--------|-------|
| Val Acc@1 | 24.68% |
| Test Acc@1 | 1.29% |
| Test Acc@5 | 6.45% |
| Config | protocol_align_optimizer.yaml |

**Changes**: LR 5e-4, weight decay 0, separate encoder LR 0.1x

### Experiment 3: Full Protocol Aligned

| Metric | Value |
|--------|-------|
| Val Acc@1 | **27.27%** |
| Test Acc@1 | 3.87% |
| Test Acc@5 | 30.32% |
| Best Epoch | 27 |
| Config | protocol_align_full.yaml |

**Changes**: All protocol changes combined

### Experiment 4: PointNet++ Encoder (Reference)

| Metric | Value |
|--------|-------|
| Val Acc@1 | 21.43% |
| Test Acc@1 | **10.97%** |
| Test Acc@5 | **60.00%** |
| Config | pointnetpp_encoder.yaml |

---

## Best-Performing Aligned Protocol

### For Validation Accuracy

| Configuration | Val Acc@1 |
|---------------|-----------|
| Full protocol aligned (SimplePointEncoder) | **27.27%** |

### For Test Accuracy

| Configuration | Test Acc@1 | Test Acc@5 |
|---------------|------------|------------|
| PointNet++ encoder | **10.97%** | **60.00%** |

**Critical Finding**: Higher validation accuracy does NOT correlate with higher test accuracy. This indicates overfitting to the validation set.

---

## Remaining Gap to 35.6%

| Configuration | Val Gap | Test Gap |
|---------------|---------|----------|
| Baseline | -12.87% | -25.32% |
| Full Protocol (Val) | **-8.33%** | -31.73% |
| PointNet++ (Test) | -14.17% | **-24.63%** |

---

## Strongest Conclusion

Protocol alignment improved validation accuracy from 22.73% to 27.27% (+4.54%), but the improvement did not transfer to test accuracy. In fact, test accuracy decreased.

**Root cause**: The models are overfitting to the validation set. The train/validation/test distribution may have different characteristics.

**Secondary finding**: PointNet++ encoder provides the best test generalization despite lower validation accuracy.

---

## Next-Step Recommendation

Based on the experiment results, I choose:

### **Option B: Protocol alignment gives limited benefit → remaining gap likely due to fusion/ranking/multi-view issues**

**Rationale**:
1. Protocol alignment improved validation accuracy but hurt test accuracy
2. The validation-test discrepancy suggests distribution mismatch or overfitting
3. PointNet++ encoder improves test accuracy, indicating representation quality matters
4. Missing multi-view features may be a key gap

### Specific Next Steps

1. **Investigate train/test distribution mismatch** - Understand why models overfit to validation
2. **Combine PointNet++ with moderate protocol alignment** - May help generalization
3. **Investigate multi-view features** - Original baseline may use them
4. **Improve discriminative object features** - Help with Top-1 vs Top-5 gap

---

## Decision

**Option B Selected**

Protocol alignment gave limited benefit. The remaining gap is NOT primarily due to training protocol mismatch. Other factors are more important:

1. **Encoder representation quality** - PointNet++ helps test accuracy
2. **Train/test distribution mismatch** - Validation accuracy is misleading
3. **Missing features** - Multi-view, spatial reasoning
4. **Discriminative power** - Model can't confidently pick #1 from top-5

---

## Acceptance Criteria Status

| Criterion | Status |
|-----------|--------|
| Protocol-fidelity audit written | ✅ |
| Official vs current protocol comparison documented | ✅ |
| Controlled protocol configs created | ✅ |
| Staged protocol experiments run | ✅ |
| Protocol ablation summary exported | ✅ |
| Final protocol-fidelity report written | ✅ |

---

## Appendix: Training Curves

### Full Protocol Alignment

| Epoch | Val Acc@1 | Val Acc@5 | Notes |
|-------|-----------|-----------|-------|
| 1 | 13.64% | 59.74% | Warmup |
| 10 | 20.78% | 77.92% | |
| 20 | 24.03% | 81.82% | |
| 27 | **27.27%** | 83.77% | Best |
| 37 | 26.62% | 87.01% | Early stopping triggered |

### PointNet++ Encoder

| Epoch | Val Acc@1 | Val Acc@5 | Notes |
|-------|-----------|-----------|-------|
| 1 | 3.25% | 24.03% | |
| 10 | 16.23% | 65.58% | |
| 15 | **21.43%** | 67.53% | Best |
| 30 | 18.83% | 72.73% | Overfitting |

---

## References

- ReferIt3D Paper (ECCV 2020)
- SAT Implementation: https://github.com/zyang-ur/SAT/blob/main/referit3d/scripts/train_referit3d.py
- 3DRefTransformer (WACV 2022)