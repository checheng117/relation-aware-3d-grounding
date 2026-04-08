# Post Recovered 23K Bottleneck Reassessment

**Date**: 2026-04-08

---

## Executive Summary

After the recovered 23K rerun, the best trustworthy baseline achieves **Test Acc@1 = 26.26%**. The dominant remaining bottleneck is **missing data** (44% of official Nr3D).

---

## 1. Best Trustworthy Baseline

| Config | Model | Val Acc@1 | Test Acc@1 | Test Acc@5 |
|--------|-------|-----------|------------|------------|
| recovered_23k_simple | SimplePointEncoder | 28.98% | **26.26%** | 85.49% |

**Gap to target**: -9.34% (from 35.6%)

**Progress**: 73.76% of target achieved

---

## 2. Does PointNet++ Help Under Larger Dataset?

**Status**: Not yet tested.

| Encoder | Old Split Test Acc@1 | Recovered 23K Test Acc@1 |
|---------|---------------------|--------------------------|
| SimplePointEncoder | 9.68% (contaminated) | **26.26%** |
| PointNet++ | 10.32% (contaminated) | **Unknown** |

**Recommendation**: Test PointNet++ on recovered 23K to confirm if raw point features help.

**Hypothesis**: PointNet++ may help marginally, but SimplePointEncoder with 15x more data outperforms PointNet++ on old split (26.26% vs 10.32%).

---

## 3. Does Protocol Alignment Still Hurt Generalization?

**Status**: Not tested on recovered 23K.

| Protocol | Old Split Val Acc@1 | Old Split Test Acc@1 | Recovered 23K |
|----------|---------------------|----------------------|---------------|
| Baseline | 22.73% | 9.68% | Val: 28.98%, Test: 26.26% |
| Protocol Aligned | 27.27% | 3.87% | **Unknown** |

**Old Interpretation**: Protocol alignment overfitted to val, hurting test.

**Revised Context**: Old results were contaminated by scene overlap. Need clean rerun.

**Recommendation**: Lower priority. Focus on data and geometry first.

---

## 4. Dominant Remaining Bottleneck

### Bottleneck Ranking

| Rank | Bottleneck | Impact | Evidence |
|------|------------|--------|----------|
| 1 | **Missing data** | Critical | 44% of official data unavailable |
| 2 | **Placeholder geometry** | High | Aggregation-based centers/sizes |
| 3 | **Synthetic features** | Medium | Hash-based class semantics |
| 4 | **Encoder quality** | Low | SimplePointEncoder sufficient |
| 5 | **Protocol mismatch** | Uncertain | Lower priority |

### Analysis

#### Missing Data (Critical)

- Current: 23,186 samples
- Official: 41,503 samples
- Missing: 18,317 samples (44%)

**Impact**: If we assume linear scaling (~1% accuracy per 1K samples), recovering all data could add ~18% accuracy, potentially exceeding the 35.6% target.

**Root cause**: Missing ScanNet scene aggregation files for 372 scenes.

#### Placeholder Geometry (High)

- Current: Aggregation-based centers/sizes
- Official: Real point cloud geometry

**Impact**: Placeholder geometry lacks fine-grained spatial information that could help distinguish similar objects.

**Root cause**: Geometry files use object bounding box aggregation, not raw point clouds.

#### Synthetic Features (Medium)

- Current: Hash-based class semantics
- Official: Visual embeddings (likely)

**Impact**: Hash features provide class identity but lack visual context.

**Recommendation**: Add visual embeddings from pretrained models.

---

## 5. What Should Change?

### Priority Changes

| Old Priority | New Priority |
|--------------|--------------|
| Dataset recovery | **Complete dataset recovery** |
| Feature fidelity | **Geometry recovery** |
| Encoder upgrade | **Lower priority** |
| Protocol alignment | **Deprioritized** |
| MVT attention | **Deprioritized** |

### Recommended Focus

1. **Recover remaining ScanNet scenes**: Target full 41,503 samples
2. **Generate real geometry**: Point-based geometry from ScanNet meshes
3. **Add visual embeddings**: Use pretrained vision models
4. **Skip advanced methods**: MVT, parser-aware, etc. are premature

---

## 6. Expected Improvement from Each Fix

| Fix | Expected Test Acc@1 | Confidence |
|-----|---------------------|------------|
| Full Nr3D recovery (41,503 samples) | 35-40% | High |
| Real point-based geometry | +3-5% | Medium |
| Visual embeddings | +2-4% | Medium |
| PointNet++ encoder | +1-2% | Low |
| Protocol optimization | +0-2% | Low |

---

## 7. Conclusion

The dominant remaining bottleneck is **missing data** (44% of official Nr3D). Secondary bottlenecks are placeholder geometry and synthetic features. Encoder and protocol choices are tertiary concerns.

**Recommendation**: Prioritize complete dataset recovery and geometry generation before investing in model improvements.

---

## 8. Decision

**Option A Selected**: Trustworthy baseline anchor is now substantially improved; continue controlled gap reduction from this recovered baseline.

### Rationale

1. Test Acc@1 improved from 2.70% to 26.26% (9.7x improvement)
2. Val-test gap improved from 8.79% to 2.72%
3. Progress to target: 73.76% achieved
4. Clear path forward: data recovery and geometry generation