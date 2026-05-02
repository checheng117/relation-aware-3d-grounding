# Formal Round-1 Results

**Date**: 2026-04-20
**Status**: Infrastructure established, numerical stability issues blocking learned experiments

---

## Executive Summary

Round-1 formal COVER-3D validation infrastructure has been successfully created:

- Object embeddings extracted (33829 train, 4255 test)
- Training scripts created
- Evaluation pipeline implemented

However, two critical issues prevent completion of learned experiments:

1. **Base mismatch**: Extracted embeddings produce 26.04% Acc@1 vs expected 30.83% from clean baseline
2. **NaN collapse**: DenseRelationModule produces NaN values during training

These issues require fixes before learned COVER-3D experiments can produce valid results.

---

## Infrastructure Status

| Component | Status | Notes |
| --- | --- | --- |
| Clean baseline checkpoint | READY | 30.83% test Acc@1 |
| Clean logits exports | READY | Consistent with evaluation |
| Object embeddings extraction | READY | 33829 train, 4255 test |
| Text features | READY | BERT embeddings, 768 dims |
| COVER-3D modules | IMPLEMENTED | DenseRelation, Calibration, SoftAnchor |
| Training script | IMPLEMENTED | train_cover3d_round1.py |
| Evaluation script | IMPLEMENTED | Hard subset metrics, harmed/recovered |

---

## Experiment Results

### Base (Clean Baseline Anchor)

| Metric | Result | Expected | Gap |
| --- | ---: | ---: | ---: |
| Test Acc@1 | 26.04% | 30.83% | -4.79% |
| Test Acc@5 | 77.41% | 91.87% | -14.46% |

**Issue**: Base experiment does not match clean baseline predictions.

**Root cause hypothesis**:
1. Object features extracted may not match training-time features
2. Text features ordering may mismatch with samples
3. Feature normalization differences

### Dense-no-cal

**Status**: FAILED - NaN collapse during training

**Error**: DenseRelationModule produces NaN in relation_scores immediately upon forward pass.

**Root cause hypothesis**:
1. Large input values (embeddings not normalized)
2. MLP initialization produces extreme values
3. Missing gradient clipping

### Dense-calibrated

**Status**: NOT RUN - blocked by Dense-no-cal failure

---

## Proxy P3 Reference Results (Not Learned)

For reference, the proxy P3 entry (using geometry-based relation scores) showed:

| Variant | Acc@1 | Acc@5 | Recovered | Harmed | Net |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 30.83% | 91.87% | - | - | - |
| Sparse no-cal | 33.44% | 92.20% | 121 | 10 | +111 |
| Dense no-cal | 34.29% | 92.22% | 169 | 22 | +147 |
| Dense calibrated | 34.24% | 92.17% | 167 | 22 | +145 |

**Note**: These are proxy/readiness results using oracle-anchor geometry relation scores, NOT learned dense relation scorer. They show that the comparison pipeline runs, but do not constitute method evidence.

---

## Blockers Identified

### Blocker 1: Base Prediction Mismatch

The extracted embeddings when passed through the model produce predictions that don't match the original clean baseline exports.

**Evidence**:
- Clean baseline test predictions: 30.83% Acc@1
- Extracted embeddings evaluation: 26.04% Acc@1
- Gap: 4.79% absolute

**Impact**: Cannot trust that COVER-3D training uses real baseline features.

**Resolution needed**:
- Verify object feature extraction matches training-time features
- Check sample ordering consistency between manifest and embeddings
- Debug model inference path

### Blocker 2: DenseRelationModule NaN Collapse

The DenseRelationModule produces NaN values in relation scores during the first forward pass.

**Evidence**:
- Immediate NaN warnings upon first training batch
- 1000+ NaN warnings in first few seconds

**Impact**: Cannot train learned dense relation scorer.

**Resolution needed**:
- Add input normalization (L2 norm or scale)
- Better MLP initialization (smaller weights)
- Add gradient clipping
- Add NaN detection and recovery

---

## Route Decision

Based on Round-1 status, the project is in **Route B** territory:

- Learned experiments did not complete successfully
- Infrastructure exists but numerical issues block progress
- Need to fix method implementation before scaling

**Next priority**: Fix numerical stability and embedding consistency, NOT expand to multi-seed.

---

## Files Generated

| File | Path |
| --- | --- |
| Base results | `outputs/cover3d_round1/base_results.json` |
| Base predictions | `outputs/cover3d_round1/base_predictions.json` |
| Test embeddings | `outputs/20260420_clean_sorted_vocab_baseline/embeddings/test_embeddings.json` |
| Train embeddings | `outputs/20260420_clean_sorted_vocab_baseline/embeddings/train_embeddings.json` |

---

## Conclusion

Round-1 learned COVER-3D experiments are blocked by numerical stability issues.

**Q1 (learned dense scorer converts recoverable conditions to gain)**: Not answered - experiments failed.

**Q2 (calibration reduces harm without suppressing recovery)**: Not answered - experiments failed.

**Required fixes before retry**:
1. Normalize object embeddings before DenseRelationModule
2. Fix MLP initialization
3. Add gradient clipping
4. Verify embedding extraction matches baseline predictions

The proxy P3 results suggest potential (34.29% Acc@1 potential gain), but learned validation cannot confirm this yet.
