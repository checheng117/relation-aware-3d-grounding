# Overlap vs Scene-Disjoint Results Comparison

**Date**: 2026-04-07
**Phase**: Trustworthy Baseline Rerun - Step 4

---

## Executive Summary

The old split had 53 overlapping scenes between val and test, inflating validation accuracy and corrupting test evaluation. Scene-disjoint evaluation shows test accuracy dropped from ~10% to ~3%.

---

## 1. Why Old Results Are Invalid

### Scene Overlap Analysis

| Metric | Old Split | New Split |
|--------|-----------|-----------|
| Val-Test overlapping scenes | 53 | **0** |
| Val samples in overlap | 89 (57.8%) | 0 |
| Test samples in overlap | 82 (52.9%) | 0 |

**Impact**: Overlapping scenes meant val and test were not independent. A model that memorized val scenes would also perform well on test, inflating both metrics.

### Specific Issues

1. **Val accuracy inflated**: Models learned scene-specific patterns that applied to both val and test
2. **Test evaluation corrupted**: Test set was not truly held out
3. **Generalization illusion**: The val-test gap appeared reasonable but was unreliable
4. **Protocol alignment paradox**: Old results showed protocol alignment improved val but hurt test - this may have been an artifact of overlap

---

## 2. Metrics Comparison

### Baseline Model

| Metric | Old Split | Scene-Disjoint | Delta |
|--------|-----------|----------------|-------|
| Val Acc@1 | 22.73% | 4.7% (rag3d) / 11.49% (repro) | -11% to -18% |
| Test Acc@1 | 9.68% | 0.54% (rag3d) / 2.70% (repro) | **-7% to -9%** |
| Test Acc@5 | 40.00% | 20.54% (rag3d) / 10.81% (repro) | -19% to -29% |

### PointNet++ Encoder (Old Split Only)

| Metric | Old Split |
|--------|-----------|
| Val Acc@1 | 21.43% |
| Test Acc@1 | 10.97% |
| Test Acc@5 | 60.00% |

**Note**: PointNet++ not yet tested on scene-disjoint split.

---

## 3. Key Observations

### 3.1 Val Accuracy Inflation

Old baseline had 22.73% val accuracy. Scene-disjoint repro baseline has 11.49% val accuracy.

**Interpretation**: ~50% of val accuracy came from overlapping scenes (scene memorization, not true generalization).

### 3.2 Test Accuracy Drop

Old baseline had 9.68% test accuracy. Scene-disjoint repro baseline has 2.70% test accuracy.

**Interpretation**: ~72% of test accuracy came from overlapping scenes (test leakage).

### 3.3 Val-Test Gap Change

| Split | Val Acc@1 | Test Acc@1 | Val-Test Gap |
|-------|-----------|------------|--------------|
| Old | 22.73% | 9.68% | 13.05% |
| Scene-Disjoint | 11.49% | 2.70% | **8.79%** |

**Interpretation**: The absolute gap is smaller on scene-disjoint, but the ratio (test/val) is similar (~25%).

### 3.4 Protocol Alignment Re-evaluation

Old results showed protocol alignment improved val (27.27%) but hurt test (3.87%). This was interpreted as overfitting to val.

**Re-interpretation**: The improvement in val was likely due to better fitting the overlapping scenes, not true generalization improvement. The test drop may have been exaggerated by overlap artifacts.

---

## 4. Conclusions That Still Hold

| Conclusion | Status | Evidence |
|------------|--------|----------|
| Dataset size is limiting | **HOLDS** | 1,544 vs 41,503 |
| Encoder architecture not dominant | **UNCERTAIN** | Need PointNet++ rerun |
| Sparse features limit performance | **HOLDS** | 97% zeros in feature vectors |
| Novel scene generalization hard | **CONFIRMED** | Large val-test gap on clean split |

---

## 5. Conclusions That Must Be Revised

| Old Conclusion | Revision Needed |
|----------------|-----------------|
| Protocol alignment hurts test | May have been overlap artifact; need rerun |
| PointNet++ improves test | Need scene-disjoint rerun to confirm |
| Val accuracy is 20%+ | Actual val accuracy is ~11% |
| Test accuracy is ~10% | Actual test accuracy is ~3% |

---

## 6. Impact on Project Goals

| Goal | Old Assessment | New Assessment |
|------|----------------|----------------|
| Baseline credibility | Weak (overlap) | **Trustworthy** |
| Gap to 35.6% | ~25% | **~33%** |
| Main bottleneck | Uncertain | **Dataset scale + generalization** |
| Priority | Encoder/protocol | **Dataset recovery** |

---

## Conclusion

The scene-disjoint rerun reveals that previous results significantly overstated model performance. The trustworthy baseline is Test Acc@1 = 2.70%, with a remaining gap of 32.9 percentage points to the target. The primary bottleneck is dataset scale combined with novel scene generalization challenges.