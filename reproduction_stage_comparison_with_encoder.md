# Reproduction Stage Comparison with Encoder Upgrade

## Summary

The PointNet++ encoder upgrade did **NOT** close the gap to the official baseline. In fact, validation accuracy slightly decreased.

| Stage | Val Acc@1 | Test Acc@1 | Test Acc@5 | Description |
|-------|-----------|------------|------------|-------------|
| Placeholder | 11.03% | 1.94% | 15.48% | Synthetic geometry + random text |
| Geometry Recovered | 14.29% | 1.94% | 15.48% | Real geometry + random text |
| **Feature Integrated** | **22.73%** | 9.68% | 40.00% | Real geometry + BERT (SimplePointEncoder) |
| Encoder Upgraded | 21.43% | **10.32%** | **63.23%** | Real geometry + BERT (PointNet++) |
| **Target (Official)** | **35.6%** | **~35%** | - | Official baseline |

---

## Key Findings

### 1. Encoder Upgrade Did Not Improve Top-1 Accuracy

| Metric | SimplePointEncoder | PointNet++ | Delta |
|--------|-------------------|------------|-------|
| Val Acc@1 | 22.73% | 21.43% | **-1.30%** |
| Test Acc@1 | 9.68% | 10.32% | +0.64% |
| Test Acc@5 | 40.00% | 63.23% | **+23.23%** |

### 2. Unexpected Result

The PointNet++ encoder (211K parameters) performed **worse** on validation accuracy than the SimplePointEncoder (82K parameters).

This is surprising because:
- PointNet++ processes raw XYZ points (richer information)
- SimplePointEncoder uses hand-crafted features (center + size + class hash)
- PointNet++ has 2.57x more parameters

### 3. Top-5 Accuracy Improved Dramatically

Test Acc@5 improved from 40.00% to 63.23% (+23.23 percentage points).

This suggests:
- PointNet++ is learning meaningful object representations
- The ranking is improving, but top-1 confidence is not

---

## Improvement Analysis

### Stage-by-Stage Improvements

| Transition | Val Acc@1 Δ | % Improvement |
|------------|-------------|---------------|
| Placeholder → Geometry | +3.26% | 29.5% |
| Geometry → Feature | +8.44% | 59.1% |
| Feature → Encoder | **-1.30%** | **-5.7%** |
| **Cumulative** | **+10.40%** | **94.2%** |

### Cumulative Improvement from Placeholder

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Val Acc@1 | 11.03% | 21.43% | +10.40% (94.2%) |
| Test Acc@1 | 1.94% | 10.32% | +8.38% (432%) |
| Test Acc@5 | 15.48% | 63.23% | +47.75% (308%) |

---

## Gap to Official Baseline

| Stage | Val Gap | Test Gap |
|-------|---------|----------|
| Feature Integrated | -12.87% | -25.32% |
| Encoder Upgraded | -14.17% | -24.68% |

**The gap WIDENED on validation after encoder upgrade.**

---

## Encoder Comparison

| Attribute | SimplePointEncoder | PointNet++ |
|-----------|-------------------|------------|
| Parameters | 82,432 | 211,904 |
| Input | Hand-crafted [B,N,256] | Raw points [B,N,P,3] |
| Architecture | 3-layer MLP | 2-level PointNet-style |
| Val Acc@1 | 22.73% | 21.43% |
| Test Acc@1 | 9.68% | 10.32% |
| Test Acc@5 | 40.00% | 63.23% |
| Best Epoch | Unknown | 15 |

---

## Training Stability

| Epoch | Val Acc@1 | Val Acc@5 | Best |
|-------|-----------|-----------|------|
| 1 | 3.25% | 24.03% | |
| 10 | 16.23% | 65.58% | |
| **15** | **21.43%** | 67.53% | **Best** |
| 30 | 18.83% | 72.73% | |

Training showed overfitting after epoch 15.

---

## Conclusions

### Primary Finding

**Encoder architecture is NOT the dominant bottleneck.**

Replacing SimplePointEncoder with PointNet++ did not improve top-1 accuracy. The remaining gap to 35.6% is NOT primarily due to the encoder being too simple.

### Secondary Findings

1. **PointNet++ learned useful features**: Test Acc@5 improved dramatically
2. **Overfitting**: Best epoch was 15, suggesting regularization needed
3. **Val/Test discrepancy**: Test Acc@1 improved slightly while Val Acc@1 dropped

### Remaining Bottleneck Candidates

1. **Training protocol mismatch**: Learning rate, batch size, augmentation
2. **Data coverage**: Missing scenes, objects, or utterance types
3. **Model capacity**: May need more capacity elsewhere (fusion, classifier)
4. **Multi-view features**: Original baseline may use multi-view images
5. **Class imbalance**: May need weighted loss or sampling

---

## Recommendation

**Do NOT proceed with more encoder upgrades.**

The encoder upgrade did not close the gap. Next steps should focus on:
1. Training protocol refinement (learning rate schedule, augmentation)
2. Data coverage analysis
3. Multi-view feature investigation
4. Fusion architecture improvements