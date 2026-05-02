# Round-1 Base Re-check Report

**Date**: 2026-04-21
**Status**: COMPLETE

---

## Executive Summary

**Finding**: Base parity RESTORED.

**Result**: Round-1 Base-only shows **30.83%** Acc@1, **91.87%** Acc@5.

**Comparison**: Matches clean baseline (30.83% / 91.87%) exactly.

**Status**: PASS within ±0.2% tolerance.

---

## 1. Clean Baseline Anchor

| Metric | Value |
| --- | --- |
| Checkpoint | `outputs/20260420_clean_sorted_vocab_baseline/formal/best_model.pt` |
| Test Acc@1 | 30.83% |
| Test Acc@5 | 91.87% |
| Export path | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_predictions.json` |
| Samples | 4255 |

---

## 2. Round-1 Base-only Measurement

| Metric | Value |
| --- | --- |
| Script | `scripts/train_cover3d_round1.py --variant base --epochs 0` |
| Test Acc@1 | 30.83% |
| Test Acc@5 | 91.87% |
| Output path | `outputs/cover3d_round1/base_results.json` |
| Samples | 4255 |

**Gap**: 0.00% Acc@1, 0.00% Acc@5

---

## 3. Hard Subset Metrics

| Subset | Acc@1 | Correct | Total |
| --- | ---: | ---: | ---: |
| Overall | 30.83% | 1312 | 4255 |
| Easy | 31.72% | 1212 | 3821 |
| Multi-anchor | 23.04% | 100 | 434 |
| Relative position | 29.86% | 289 | 968 |

Note: same_class_clutter_ge3 and same_class_clutter_ge5 not populated in this run.

---

## 4. Harmed/Recovered Analysis

| Metric | Value |
| --- | --- |
| Recovered | 0 |
| Harmed | 0 |
| Net | 0 |

Expected for Base variant (no COVER-3D modules).

---

## 5. Verification Checklist

| Check | Status |
| --- | ---: |
| Acc@1 within ±0.2% | PASS |
| Acc@5 within ±0.5% | PASS |
| No NaN errors | PASS |
| Output files saved | PASS |

---

## 6. Conclusion

**Base Parity Status**: RESTORED ✓

**Root Cause Fixed**: Padding mask applied before argmax in evaluate function.

**Next Step**: Proceed to Dense-no-cal formal training.

---

## Appendix: Command and Output

```bash
python scripts/train_cover3d_round1.py --variant base --epochs 0
```

Key output:
```
Test Acc@1: 30.83%
Test Acc@5: 91.87%
Recovered: 0, Harmed: 0, Net: 0
```
