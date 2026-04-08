# Generalization Gap Analysis Report

**Date**: 2026-04-06
**Phase**: Distribution Mismatch Investigation - Step 3

---

## Executive Summary

Three fundamental issues contribute to generalization failure:

1. **Small dataset size**: 1,569 samples vs official 41,503 (~3.8% of full)
2. **Scene overlap**: Splits are NOT scene-disjoint
3. **Protocol mismatch**: Training protocol may not match official ReferIt3DNet

---

## 1. Dataset Size Issue (CRITICAL)

| Metric | Current | Official | Ratio |
|--------|---------|----------|-------|
| Total samples | 1,569 | ~41,503 | 3.8% |
| Train samples | 1,235 | ~33,000 | 3.7% |
| Val samples | 154 | ~4,000 | 3.9% |
| Test samples | 155 | ~4,500 | 3.4% |
| Scenes | 269 | ~800+ | ~33% |

**Implications**:
1. **Insufficient training data**: 1,235 train samples cannot adequately learn 269 scenes with diverse objects
2. **High variance**: Small val/test (154/155) leads to unstable evaluation metrics
3. **Scene diversity**: Even with 269 scenes, sample diversity is limited
4. **Relation coverage**: Cannot cover all relation types and object combinations

**Evidence**: Target 35.6% accuracy requires adequate data coverage. 3.8% of data makes credible reproduction impossible.

---

## 2. Configuration Comparison

| Configuration | Val Acc@1 | Test Acc@1 | Val-Test Gap | Train Data |
|--------------|-----------|------------|--------------|------------|
| Baseline (feature fidelity) | 22.73% | 9.68% | 13.05% | 1,235 |
| PointNet++ | 21.43% | 10.97% | 10.46% | 1,235 |
| Full protocol alignment | 27.27% | 3.87% | 23.40% | 1,235 |

**Observation**: Protocol alignment shows largest val-test gap (23.40%), suggesting overfitting to val.

---

## 3. Protocol Differences

### 3.1 Current Baseline Config
```yaml
epochs: 6
batch_size: 24
lr: 0.0001
weight_decay: 0.01
seed: 42
model: attribute_only
```

### 3.2 Official ReferIt3DNet Protocol (from paper)
- BERT-based language encoder
- PointNet++ geometry encoder
- Multi-view transformer (MVT) for attention
- 41,503 training samples
- Scene-disjoint splits

**Mismatch summary**:
| Aspect | Current | Official |
|--------|---------|----------|
| Language encoder | Hash-based | BERT |
| Geometry encoder | Simple MLP / PointNet++ | PointNet++ |
| Attention | None / basic | MVT |
| Data size | 1,569 | 41,503 |
| Splits | Random shuffle | Scene-disjoint |

---

## 4. Scene Overlap Impact on Generalization

With scene overlap:
- Val and test share 53 scenes (57.8% of val, 52.9% of test)
- Train shares scenes with both val and test
- Model can learn scene-specific patterns

**Why val improves but test degrades**:
1. Protocol tuning optimizes for val's specific queries
2. Val queries may match learned scene patterns
3. Test queries in overlapping scenes target different objects
4. Scene-specific patterns don't transfer → test degradation

---

## 5. Overfitting Analysis

### 5.1 PointNet++ vs Baseline
- PointNet++ has **smaller** val-test gap (10.46% vs 13.05%)
- Better generalization despite similar data
- Suggests encoder quality matters more than protocol

### 5.2 Protocol Alignment Overfitting
- Largest val-test gap (23.40%)
- Val improves (22.73% → 27.27%) but test collapses (9.68% → 3.87%)
- **Strong evidence of val-specific overfitting**
- Protocol changes (LR, batch size, etc.) may exploit val quirks

---

## 6. Root Cause Hierarchy

| Priority | Issue | Evidence | Impact |
|----------|-------|----------|--------|
| 1 | **Small dataset** | 1,569 vs 41,503 | Fundamental capacity limit |
| 2 | **Scene overlap** | 53 overlapping scenes | Invalidates evaluation |
| 3 | **Protocol mismatch** | Hash vs BERT, no MVT | Feature encoding weakness |
| 4 | **Val overfitting** | 27.27% val, 3.87% test | Protocol tuning exacerbates |

---

## 7. Implications for Reproduction

A **credible reproduction** of 35.6% accuracy requires:
1. Full Nr3D dataset (41,503 samples)
2. Scene-disjoint splits
3. BERT language encoder
4. PointNet++ geometry encoder
5. MVT attention mechanism

Current setup:
- Uses subset (3.8% of data)
- Random shuffle splits (not scene-disjoint)
- Simple language encoder
- PointNet++ available but underutilized

**Conclusion**: Current reproduction cannot achieve credible results due to fundamental data and protocol limitations.

---

## 8. Files Inspected

- `configs/train/nr3d_geom_first_baseline.yaml`
- `configs/train/baseline.yaml`
- `repro/referit3d_baseline/README.md`
- `.claude/CURRENT_STATUS.md`
- `data/processed/dataset_summary.json`

---

## 9. Recommendations

### Immediate Actions
1. **Obtain full Nr3D dataset** or acknowledge subset limitations
2. **Implement scene-disjoint splits** before further experiments
3. **Document subset limitation** clearly in all reports

### Before Any New Training Run
- Scene overlap must be resolved
- Evaluation on scene-disjoint splits only
- Otherwise all metrics are potentially invalid

### Long-term Reproduction Path
1. Full dataset acquisition
2. BERT integration
3. MVT implementation
4. Official protocol alignment

---

## 10. Next Steps

1. Decide: Continue with subset or acquire full dataset?
2. If subset: Create scene-disjoint splits and re-evaluate
3. If full: Plan data acquisition and pipeline update
4. After splits: Single controlled PointNet++ experiment on clean splits