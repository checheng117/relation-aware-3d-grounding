# Subset vs Recovered 23K Results

**Date**: 2026-04-08

---

## Executive Summary

The recovered 23K dataset achieved **Test Acc@1 = 26.26%**, compared to **2.70%** on the old 1.5K subset—a **9.7x improvement**.

---

## Dataset Comparison

| Metric | Old Subset | Recovered 23K | Ratio |
|--------|------------|---------------|-------|
| Total samples | 1,544 | 23,186 | 15x |
| Train samples | 1,211 | 18,459 | 15.2x |
| Val samples | 148 | 2,046 | 13.8x |
| Test samples | 185 | 2,681 | 14.5x |
| Scenes | 266 | 269 | ~same |

---

## Results Comparison

### Test Accuracy

| Metric | Old (1.5K) | Recovered (23K) | Improvement |
|--------|------------|-----------------|-------------|
| Test Acc@1 | 2.70% | **26.26%** | +23.56% |
| Test Acc@5 | 10.81% | **85.49%** | +74.68% |

### Val Accuracy

| Metric | Old (1.5K) | Recovered (23K) | Improvement |
|--------|------------|-----------------|-------------|
| Val Acc@1 | 11.49% | **28.98%** | +17.49% |
| Val Acc@5 | - | 87.59% | - |

### Val-Test Gap

| Metric | Old (1.5K) | Recovered (23K) |
|--------|------------|-----------------|
| Val-Test Gap | 8.79% | **2.72%** |
| Interpretation | Overfitting to val | Good generalization |

---

## Overlap-Contaminated Results (Invalidated)

Previous results on overlap-contaminated splits:

| Config | Val Acc@1 | Test Acc@1 | Status |
|--------|-----------|------------|--------|
| SimplePointEncoder (old split) | 22.73% | 9.68% | INVALIDATED |
| PointNet++ (old split) | 21.43% | 10.32% | INVALIDATED |
| Protocol Aligned (old split) | 27.27% | 3.87% | INVALIDATED |

**Reason**: Scene overlap between val and test contaminated evaluation.

---

## Prior Conclusions - Still Hold?

### Dataset Scale is Critical

| Conclusion | Old | Recovered | Status |
|------------|-----|-----------|--------|
| Dataset scale is dominant bottleneck | ✓ Confirmed | ✓ Confirmed | **STILL HOLDS** |

**Evidence**: 15x more data → 9.7x better test accuracy.

### Encoder Quality

| Conclusion | Old | Recovered | Status |
|------------|-----|-----------|--------|
| SimplePointEncoder sufficient | ✓ Marginal | ✓ Better | **REINFORCED** |

**Evidence**: SimplePointEncoder achieves 26.26% with recovered data.

### Protocol Mismatch

| Conclusion | Old | Recovered | Status |
|------------|-----|-----------|--------|
| Protocol alignment hurts test | Uncertain | Not tested | **NEEDS RE-TEST** |

**Note**: Previous results showed protocol alignment hurting test, but were contaminated by overlap.

---

## Key Insights

### 1. Data Scale Effect

- Old: 1,544 samples → Test Acc@1 = 2.70%
- Recovered: 23,186 samples → Test Acc@1 = 26.26%
- **Conclusion**: Each 1,000 samples adds ~1% test accuracy

### 2. Val-Test Gap

- Old: 8.79% gap (overfitting)
- Recovered: 2.72% gap (good generalization)
- **Conclusion**: More data improves generalization

### 3. Progress to Target

- Old: 2.70% / 35.6% = 7.6%
- Recovered: 26.26% / 35.6% = 73.8%
- **Conclusion**: Recovered dataset closes most of the gap

---

## Remaining Bottlenecks

| Bottleneck | Impact | Evidence |
|------------|--------|----------|
| Missing 44% of data | High | 18K samples without geometry |
| Placeholder geometry | Medium | Aggregation-based centers/sizes |
| Synthetic features | Medium | Hash-based, not visual embeddings |

---

## Conclusion

The recovered 23K dataset provides a **9.7x improvement** in test accuracy and validates that:
1. Dataset scale is the dominant bottleneck
2. Scene-disjoint evaluation is essential for trustworthy results
3. The remaining 9.34% gap to target is likely due to missing data and placeholder geometry