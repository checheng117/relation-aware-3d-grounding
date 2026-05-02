# Formal Round-1 Results Summary (Updated)

**Date**: 2026-04-21
**Status**: COMPLETE - ROUTE C CONFIRMED

---

## Executive Summary

**Completed Experiments**:
- Base: 30.83% Acc@1, 91.87% Acc@5
- Dense-no-cal: 31.05% Acc@1, 92.01% Acc@5, Net +9
- Dense-calibrated (v1): 30.60% Acc@1, 91.89% Acc@5, Net -10 (gate collapse)
- Dense-calibrated-v2: 30.55% Acc@1, 91.80% Acc@5, Net -12 (gate fixed, perf bad)

**Key Finding**: Dense-no-cal provides modest improvement. Calibration approach is NOT salvageable without new signal sources.

---

## 1. Main Results Table

| Variant | Acc@1 | Acc@5 | Harmed | Recovered | Net |
|---------|-------|-------|--------|-----------|-----|
| Base | 30.83% | 91.87% | - | - | - |
| Dense-no-cal | 31.05% | 92.01% | 393 | 402 | +9 |
| Dense-calibrated (v1) | 30.60% | 91.89% | 444 | 434 | -10 |
| Dense-calibrated-v2 | 30.55% | 91.80% | 471 | 459 | -12 |

---

## 2. Hard Subset Metrics

### Base (Reference)

| Subset | Acc@1 | Correct | Total |
|--------|-------|---------|-------|
| Overall | 30.83% | 1312 | 4255 |
| Easy | 31.72% | 1212 | 3821 |
| Multi-anchor | 23.04% | 100 | 434 |
| Relative position | 29.86% | 289 | 968 |

### Dense-no-cal (Best)

| Subset | Acc@1 | Correct | Total |
|--------|-------|---------|-------|
| Overall | 31.05% | 1321 | 4255 |
| Easy | 31.35% | 1198 | 3821 |
| Multi-anchor | 28.34% | 123 | 434 |
| Relative position | 30.68% | 297 | 968 |

### Dense-calibrated-v2 (Redesigned)

| Subset | Acc@1 | Correct | Total |
|--------|-------|---------|-------|
| Overall | 30.55% | 1300 | 4255 |
| Easy | 30.49% | 1165 | 3821 |
| Multi-anchor | 31.11% | 135 | 434 |
| Relative position | 29.86% | 289 | 968 |

---

## 3. Training Stability

| Variant | Epochs | Loss Stability | Gate Issue | Gate Mean | Gate Std |
|---------|--------|----------------|------------|-----------|----------|
| Base | 0 (eval only) | N/A | None | 1.0 (fixed) | 0 |
| Dense-no-cal | 10 | Stable (~3.67) | None | 0.5 (fixed) | 0 |
| Dense-calibrated (v1) | 10 | Stable (~3.52) | Collapsed | 0.9 | ~0 |
| Dense-calibrated-v2 | 10 | Stable (~3.62) | Fixed | 0.687 | 0.066 |

---

## 4. Research Questions

### Q1: Does Dense-no-cal convert P2's recoverable conditions into gains?

**Answer**: YES, but marginally.

- Dense-no-cal shows +0.22% Acc@1 gain (30.83% → 31.05%)
- Net recovered: +9 (402 recovered, 393 harmed)
- Hard subsets show improvement:
  - Multi-anchor: 23.04% → 28.34% (+5.3%)
  - Relative position: 29.86% → 30.68% (+0.82%)

**Assessment**: Modest gain observed, but not as strong as P2 proxy results suggested (+147 net).

---

### Q2: Does calibration reduce harm without suppressing recovery?

**Answer**: NO.

**Evidence**:
- Dense-calibrated-v2 Acc@1 (30.55%) < Base (30.83%) by -0.28%
- Dense-calibrated-v2 Net (-12) < Dense-no-cal (+9) by -21
- Dense-calibrated-v2 Net (-12) < Base (0) by -12

**Root Cause**: Calibration signals (base_margin, anchor_entropy, relation_margin) are uninformative. Even with proper gate regularization, the MLP cannot learn when to trust relation scores.

**Gate Behavior**:
- v1: Gate collapsed to 0.9 (max bound) - fixed with prior regularization
- v2: Gate stabilized at 0.687 with std=0.066 - but decisions still wrong

**Conclusion**: Gate collapse was a symptom, not the root cause. The fundamental issue is signal quality.

---

## 5. Route Decision

**Route C-confirmed: CALIBRATION-NOT-WORTH-CONTINUING**

Rationale:
- Dense-no-cal shows positive but modest gain (+0.22% Acc@1, +9 net)
- Calibration (even redesigned) performs worse than baselines
- Gate collapse was fixed but performance unchanged
- Signal redesign would require new features, not just tuning

**Recommendation**:
1. **Adopt Dense-no-cal** as lightweight improvement
2. **Pause calibration line** - not worth further investment
3. **Consider dense scorer improvements** if more gains needed

---

## 6. Calibration Redesign Summary

### What Was Fixed
- init_bias: 0.3 → 0.5 (balanced initialization)
- gate_prior_weight: 0.0 → 1.0 (L2 regularization toward 0.5)
- Gate behavior: 0.9 collapsed → 0.687 stable with variance

### What Wasn't Fixed
- Signal quality: Still uninformative for predicting benefit/harm
- Performance: Still worse than Base and Dense-no-cal
- Gate bias: Still relation-heavy (0.687 vs ideal 0.5)

### Files Generated
- `reports/calibration_failure_analysis.md` - Root cause analysis
- `reports/calibration_redesign_log.md` - Design iterations
- `reports/calibration_controlled_rerun.md` - Full results
- `reports/calibration_comparison_table.csv` - Comparison data

---

## 7. Files Generated

| File | Path |
|------|------|
| Base results | `outputs/cover3d_round1/base_results.json` |
| Dense-no-cal results | `outputs/cover3d_round1/dense-no-cal_results.json` |
| Dense-calibrated results | `outputs/cover3d_round1/dense-calibrated_results.json` |
| Dense-calibrated-v2 results | `outputs/cover3d_round1/dense-calibrated-v2_results.json` |
| Main table | `reports/formal_round1_table.csv` |
| This summary | `reports/formal_round1_summary.md` |

---

## 8. Conclusion

**Round-1 + Redesign Status**: COMPLETE

**Summary**:
- Dense-no-cal: modest improvement over Base (+0.22% Acc@1, +9 net)
- Dense-calibrated (v1): gate collapse to 0.9
- Dense-calibrated-v2: gate fixed (0.687 stable), but performance still worse
- Calibration approach NOT salvageable without new signal sources

**Final Recommendation**: Adopt Dense-no-cal as the COVER-3D method contribution.
