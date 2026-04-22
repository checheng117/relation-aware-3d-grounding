# Round-1 Dense-no-cal Formal Run Report

**Date**: 2026-04-21
**Status**: COMPLETE

---

## Executive Summary

**Finding**: Dense-no-cal shows modest improvement over Base.

**Result**: 
- Acc@1: 30.83% → 31.05% (+0.22%)
- Acc@5: 91.87% → 92.01% (+0.14%)
- Net recovered: +9 (402 recovered, 393 harmed)

**Assessment**: Positive but modest gain. Not as strong as P2 proxy results (+147 net).

---

## 1. Training Configuration

| Parameter | Value |
| --- | --- |
| Variant | dense-no-cal |
| Epochs | 10 |
| Batch size | 4 |
| Learning rate | 1e-4 |
| Seed | 42 |
| Trainable params | 398,401 |

---

## 2. Training History

| Epoch | Train Loss | Train Acc | Gate Mean |
| --- | ---: | ---: | ---: |
| 1 | 3.6837 | 28.09% | 0.50 |
| 2 | 3.6742 | 28.31% | 0.50 |
| 3 | 3.6742 | 28.43% | 0.50 |
| 4 | 3.6740 | 28.42% | 0.50 |
| 5 | 3.6740 | 28.36% | 0.50 |
| 6 | 3.6740 | 28.51% | 0.50 |
| 7 | 3.6740 | 28.43% | 0.50 |
| 8 | 3.6739 | 28.37% | 0.50 |
| 9 | 3.6739 | 28.47% | 0.50 |
| 10 | 3.6739 | 28.11% | 0.50 |

**Loss stability**: Stable around 3.67, no NaN issues.

---

## 3. Test Set Results

### Overall Metrics

| Metric | Value |
| --- | --- |
| Acc@1 | 31.05% |
| Acc@5 | 92.01% |
| Total samples | 4255 |

### Comparison with Base

| Metric | Base | Dense-no-cal | Delta |
| --- | ---: | ---: | ---: |
| Acc@1 | 30.83% | 31.05% | +0.22% |
| Acc@5 | 91.87% | 92.01% | +0.14% |

### Harmed/Recovered Analysis

| Metric | Value |
| --- | --- |
| Recovered | 402 |
| Harmed | 393 |
| Net | +9 |

**Interpretation**: Dense-no-cal recovers 402 cases that Base got wrong, but harms 393 cases that Base got right. Net gain is modest (+9).

---

## 4. Hard Subset Metrics

| Subset | Acc@1 | Correct | Total | vs Base |
| --- | ---: | ---: | ---: | ---: |
| Overall | 31.05% | 1321 | 4255 | +0.22% |
| Easy | 31.35% | 1198 | 3821 | -0.37% |
| Multi-anchor | 28.34% | 123 | 434 | +5.30% |
| Relative position | 30.68% | 297 | 968 | +0.82% |

**Key finding**: Multi-anchor subset shows largest improvement (+5.3%), suggesting Dense relation scorer helps with multi-anchor cases.

---

## 5. Loss Curve Analysis

```
Epoch 1: 3.6837
Epoch 2: 3.6742 (-0.0095)
Epoch 3: 3.6742 (0.0000)
...
Epoch 10: 3.6739 (-0.0003 from epoch 6)
```

**Observation**: Loss converges quickly (by epoch 2-3) and plateaus. Minimal improvement after epoch 6.

**Hypothesis**: Relation scorer may need:
- Higher capacity (hidden dim)
- Better initialization
- Longer training with lower LR

---

## 6. Comparison with P2 Proxy Results

| Metric | P2 Proxy | Round-1 Learned | Gap |
| --- | ---: | ---: | ---: |
| Acc@1 | 34.29% | 31.05% | -3.24% |
| Net | +147 | +9 | -138 |

**Interpretation**: P2 proxy (geometry-based relation scores) significantly outperforms learned dense scorer. This suggests:
1. Learned scorer has not yet captured full relation signal
2. Geometry features may be more discriminative than language-conditioned embeddings
3. Training approach needs refinement

---

## 7. Files Generated

| File | Path |
| --- | --- |
| Results JSON | `outputs/cover3d_round1/dense-no-cal_results.json` |
| Predictions JSON | `outputs/cover3d_round1/dense-no-cal_predictions.json` |
| Training log | `outputs/cover3d_round1/dense-no-cal_training2.log` |
| Checkpoint | `outputs/cover3d_round1/best_model.pt` |

---

## 8. Conclusion

**Status**: COMPLETE ✓

**Result**: Dense-no-cal shows modest improvement over Base (+0.22% Acc@1, +9 net).

**Assessment**: 
- Positive direction (gain, not loss)
- Magnitude smaller than P2 proxy suggested
- Multi-anchor subset shows largest gain (+5.3%)

**Next Step**: Compare with Dense-calibrated to assess calibration effect.
