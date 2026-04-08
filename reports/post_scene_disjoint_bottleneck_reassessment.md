# Post Scene-Disjoint Bottleneck Reassessment

**Date**: 2026-04-07
**Phase**: Trustworthy Baseline Rerun - Step 5

---

## Executive Summary

After clean scene-disjoint evaluation, the dominant bottlenecks are **dataset scale** and **novel scene generalization**. Encoder and protocol choices are secondary to data limitations.

---

## 1. Best Trustworthy Baseline

| Config | Model | Test Acc@1 | Test Acc@5 |
|--------|-------|------------|------------|
| repro ReferIt3DNet | SimplePointEncoder | **2.70%** | 10.81% |
| rag3d AttributeOnly | ObjectMLPEncoder | 0.54% | 20.54% |

**Best Baseline**: repro ReferIt3DNet with Test Acc@1 = 2.70%

---

## 2. Does PointNet++ Still Help?

**Status**: Not yet tested on scene-disjoint split.

| Encoder | Old Split Test Acc@1 | Scene-Disjoint Test Acc@1 |
|---------|---------------------|---------------------------|
| SimplePointEncoder | 9.68% | 2.70% |
| PointNet++ | 10.97% | **Unknown** |

**Recommendation**: Rerun PointNet++ on scene-disjoint to confirm if the ~1% improvement holds under clean evaluation.

**Hypothesis**: PointNet++ may still help, but the absolute improvement will be smaller (estimated 3-4% vs 2.7%).

---

## 3. Does Protocol Alignment Still Hurt Test?

**Status**: Not yet tested on scene-disjoint split.

| Protocol | Old Split Val Acc@1 | Old Split Test Acc@1 | Scene-Disjoint |
|----------|---------------------|----------------------|----------------|
| Baseline | 22.73% | 9.68% | Val: 11.49%, Test: 2.70% |
| Protocol Aligned | 27.27% | 3.87% | **Unknown** |

**Old Interpretation**: Protocol alignment overfitted to val, hurting test.

**Revised Interpretation**: The improvement in val (27.27% vs 22.73%) may have been driven by better fitting overlapping scenes. The test drop (3.87% vs 9.68%) may have been exaggerated by overlap artifacts.

**Recommendation**: Rerun protocol alignment on scene-disjoint to verify whether it genuinely hurts generalization.

---

## 4. Dominant Remaining Bottleneck

### Bottleneck Ranking

| Rank | Bottleneck | Impact | Evidence |
|------|------------|--------|----------|
| 1 | **Dataset scale** | Critical | 1,544 vs 41,503 samples (~3.7%) |
| 2 | **Novel scene generalization** | High | 8.79% val-test gap |
| 3 | **Feature fidelity** | Medium | 97% zero features |
| 4 | **Encoder quality** | Low | SimplePointEncoder vs PointNet++ similar |
| 5 | **Protocol mismatch** | Uncertain | Need clean rerun |

### Analysis

#### Dataset Scale (Critical)

- Current: 1,544 samples
- Official: 41,503 samples
- Ratio: 3.7%

**Impact**: With 3.7% of data, achieving 35.6% target is unrealistic. The model has limited training signal to learn 3D grounding.

#### Novel Scene Generalization (High)

- Val Acc@1: 11.49%
- Test Acc@1: 2.70%
- Gap: 8.79 percentage points

**Impact**: Model struggles to generalize to unseen scenes. This is expected with limited scene diversity (266 scenes total).

#### Feature Fidelity (Medium)

- Feature vectors: 256 dimensions
- Zero fraction: 97.66%
- Non-zero features per object: ~6

**Impact**: Sparse features limit discriminative power. Features are likely derived from class labels and basic geometry, not rich visual or semantic embeddings.

#### Encoder Quality (Low)

Both SimplePointEncoder and ObjectMLPEncoder are simple MLPs. PointNet++ showed marginal improvement (~1%) on old split. Encoder is not the primary bottleneck.

#### Protocol Mismatch (Uncertain)

Old results suggested protocol alignment hurt test. Need clean rerun to confirm. Given the overlap issues, previous conclusions are unreliable.

---

## 5. What Should Change?

### Priority Changes

| Old Priority | New Priority |
|--------------|--------------|
| Encoder upgrade | **Dataset recovery / Nr3D expansion** |
| Protocol alignment | **Feature fidelity improvement** |
| MVT attention | **Not a priority (baseline weak)** |
| Structured parser | **Not a priority (baseline weak)** |

### Recommended Focus

1. **Recover more Nr3D data**: Target full 41,503 samples
2. **Improve feature quality**: Add visual embeddings, reduce sparsity
3. **Test generalization**: Design cross-validation with scene splits
4. **Skip advanced methods**: MVT, parser-aware, etc. are premature

---

## 6. Expected Improvement from Each Fix

| Fix | Expected Test Acc@1 | Confidence |
|-----|---------------------|------------|
| Full Nr3D recovery (41,503 samples) | 15-25% | High |
| Better features (visual embeddings) | 5-10% | Medium |
| PointNet++ encoder | 3-4% | Medium |
| Protocol optimization | 0-3% | Low |
| MVT attention | 2-5% | Low |

---

## 7. Conclusion

The dominant bottleneck is **dataset scale**. With only 3.7% of the official Nr3D dataset, achieving the 35.6% target is unrealistic. Secondary bottlenecks are novel scene generalization and feature fidelity. Encoder and protocol choices are tertiary concerns.

**Recommendation**: Prioritize dataset recovery before investing in model improvements.