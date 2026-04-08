# Top-1 vs Top-5 Gap Analysis

## Executive Summary

A significant gap exists between Top-1 and Top-5 accuracy across all experiments, suggesting the model learns meaningful rankings but lacks confidence in distinguishing the correct object.

---

## Data Overview

| Model | Val Acc@1 | Val Acc@5 | Val Gap | Test Acc@1 | Test Acc@5 | Test Gap |
|-------|-----------|-----------|---------|------------|------------|----------|
| Baseline | 22.73% | 80.52% | 57.79% | 9.68% | 40.00% | 30.32% |
| Optimizer Aligned | 24.68% | 77.92% | 53.24% | 1.29% | 6.45% | 5.16% |
| Full Protocol | 27.27% | 87.01% | 59.74% | 3.87% | 30.32% | 26.45% |
| **PointNet++** | 21.43% | 87.01% | 65.58% | **10.97%** | **60.00%** | 49.03% |

---

## Analysis

### 1. Gap Magnitude

The Top-5 accuracy is consistently 3-6x higher than Top-1 accuracy.

| Model | Top-5 / Top-1 Ratio (Test) |
|-------|---------------------------|
| Baseline | 4.13x |
| Optimizer Aligned | 5.00x |
| Full Protocol | 7.83x |
| **PointNet++** | **5.47x** |

**Interpretation**: The correct object is frequently in the top-5 predictions, but the model struggles to rank it as #1.

### 2. PointNet++ Behavior

PointNet++ shows the most dramatic improvement in Top-5 accuracy:

| Metric | Baseline | PointNet++ | Improvement |
|--------|----------|------------|-------------|
| Test Acc@1 | 9.68% | 10.97% | +1.29% |
| Test Acc@5 | 40.00% | 60.00% | **+20.00%** |
| Target in top-5 (additional) | - | - | +20% of samples |

**Key Insight**: PointNet++ encoder significantly improves ranking quality - the target object now appears in the top 5 for 60% of test samples (vs 40% baseline).

### 3. Protocol Alignment Effect

Full protocol alignment improved both Top-1 and Top-5 on validation:

| Metric | Baseline | Full Protocol | Delta |
|--------|----------|---------------|-------|
| Val Acc@1 | 22.73% | 27.27% | +4.54% |
| Val Acc@5 | 80.52% | 87.01% | +6.49% |

But hurt test accuracy:

| Metric | Baseline | Full Protocol | Delta |
|--------|----------|---------------|-------|
| Test Acc@1 | 9.68% | 3.87% | -5.81% |
| Test Acc@5 | 40.00% | 30.32% | -9.68% |

**Interpretation**: Protocol alignment causes overfitting to validation distribution.

### 4. Confidence Distribution Analysis

The large Top-5 / Top-1 gap suggests:

1. **Soft decision boundary**: Model has difficulty distinguishing the best candidate from runners-up
2. **Feature confusion**: Multiple objects may have similar representations
3. **Language ambiguity**: Utterances may refer to multiple similar objects

### 5. Ranking Sharpness

| Model | Test Acc@1 | Test Acc@5 | Ranking Sharpness (Acc@1/Acc@5) |
|-------|------------|------------|--------------------------------|
| Baseline | 9.68% | 40.00% | 0.242 |
| Optimizer Aligned | 1.29% | 6.45% | 0.200 |
| Full Protocol | 3.87% | 30.32% | 0.128 |
| **PointNet++** | **10.97%** | **60.00%** | **0.183** |

Lower sharpness = more uncertain ranking. Full Protocol has the lowest sharpness, indicating it produces more uncertain predictions despite better recall.

---

## Hypotheses

### Hypothesis 1: Representation Overlap

Objects of the same class or similar spatial positions may have similar representations, making it hard to distinguish the correct one.

**Evidence**: SimplePointEncoder uses class hash, so all chairs share the same semantic encoding.

### Hypothesis 2: Insufficient Discriminative Features

The model lacks features that uniquely identify objects within a scene.

**Evidence**: PointNet++ improves ranking quality, suggesting raw point features are more discriminative than hand-crafted features.

### Hypothesis 3: Language-Vision Misalignment

The utterance may not provide enough discriminative information to uniquely identify the target.

**Evidence**: Large gap persists even with real BERT features.

### Hypothesis 4: Training-Test Distribution Mismatch

The validation set may have different characteristics than the test set.

**Evidence**: Validation accuracy much higher than test accuracy for protocol-aligned models.

---

## Recommendations

### Immediate

1. **Use PointNet++ encoder** - Improves ranking quality significantly
2. **Focus on test accuracy** - Validation accuracy is misleading
3. **Add discriminative features** - Beyond class hash

### Medium-term

1. **Investigate train/test distribution** - Understand why models overfit to validation
2. **Add spatial reasoning** - May help distinguish similar objects
3. **Multi-view features** - May provide additional discriminative power

### Long-term

1. **Attention mechanisms** - Help model focus on relevant objects
2. **Contrastive learning** - Learn more discriminative object representations
3. **Data augmentation** - Improve generalization

---

## Conclusion

The large Top-1 vs Top-5 gap indicates:

1. **Models learn meaningful rankings** - Target is often in top-5
2. **Confidence is the issue** - Model struggles to pick #1
3. **PointNet++ helps** - Improves ranking quality
4. **Protocol alignment overfits** - Helps val, hurts test

The main bottleneck is **not** that the model can't identify relevant objects, but that it lacks the discriminative power to confidently select the correct one.

**Next focus**: Improve discriminative object representations and investigate train/test distribution mismatch.