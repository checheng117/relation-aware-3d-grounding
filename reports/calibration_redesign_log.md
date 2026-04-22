# Calibration Redesign Log (Final)

**Date**: 2026-04-21
**Status**: COMPLETE

---

## Summary

Successfully redesigned `CalibratedFusionGate` to prevent gate collapse. Gate stabilized at ~0.687 with meaningful variance (std=0.066), but overall performance still lags behind Dense-no-cal.

---

## Iteration History

### Iteration 1: Prior Weight 0.1 (FAILED)
- init_bias: 0.5, gate_prior_weight: 0.1
- Result: Gate collapsed to 0.9

### Iteration 2: Prior Weight 1.0 (SUCCESS - Gate Stabilized)
- init_bias: 0.5, gate_prior_weight: 1.0
- Smoke test (2 epoch): Gate mean=0.69, Net=0
- Full run (10 epoch): Gate mean=0.687, Gate std=0.066

---

## Final Results (10 Epochs)

| Metric | Dense-calibrated-v2 |
|--------|---------------------|
| Acc@1 | 30.55% |
| Acc@5 | 91.80% |
| Gate Mean | 0.687 |
| Gate Std | 0.066 |
| Recovered | 459 |
| Harmed | 471 |
| Net | -12 |

### Training History

| Epoch | Loss | CE Loss | Gate Loss | Acc | Gate Mean |
|-------|------|---------|-----------|-----|-----------|
| 1 | 3.6486 | 3.6180 | 0.0306 | 28.35% | 0.668 |
| 2 | 3.6350 | 3.5983 | 0.0366 | 28.31% | 0.690 |
| 3 | 3.6337 | 3.5974 | 0.0363 | 28.48% | 0.688 |
| 4 | 3.6319 | 3.5957 | 0.0362 | 28.59% | 0.686 |
| 5 | 3.6297 | 3.5942 | 0.0355 | 28.51% | 0.681 |
| 6 | 3.6272 | 3.5915 | 0.0358 | 28.69% | 0.678 |
| 7 | 3.6255 | 3.5884 | 0.0371 | 28.59% | 0.679 |
| 8 | 3.6243 | 3.5867 | 0.0375 | 28.38% | 0.677 |
| 9 | 3.6236 | 3.5858 | 0.0379 | 28.60% | 0.677 |
| 10 | 3.6235 | 3.5861 | 0.0375 | 28.57% | 0.675 |

**Observation**: Gate stable throughout training (~0.67-0.69), no collapse.

---

## Full Comparison

| Variant | Acc@1 | Acc@5 | Gate Mean | Gate Std | Net | Status |
|---------|-------|-------|-----------|----------|-----|--------|
| Base | 30.83% | 91.87% | 1.0 (fixed) | 0 | 0 | Reference |
| Dense-no-cal | 31.05% | 92.01% | 0.5 (fixed) | 0 | +9 | Best |
| Dense-calibrated | 30.60% | 91.89% | 0.9 (collapsed) | ~0 | -10 | Failed |
| Dense-calibrated-v2 | 30.55% | 91.80% | 0.687 | 0.066 | -12 | Gate OK, Perf Bad |

---

## Key Findings

### Success: Gate No Longer Collapses
- Gate mean: 0.9 → 0.687 (no longer at max bound)
- Gate std: ~0 → 0.066 (meaningful variance across samples)
- Gate is now modulating based on calibration signals

### Failure: Performance Still Worse Than Baseline
- Acc@1: 30.55% < Base 30.83% (by -0.28%)
- Acc@1: 30.55% < Dense-no-cal 31.05% (by -0.50%)
- Net: -12 < Base 0 (more harmed than recovered)
- Net: -12 < Dense-no-cal +9 (19 worse)

### Root Cause: Calibration Signals Still Uninformative
Gate is working (not collapsed), but calibration is still making wrong decisions:
- Gate ~0.687 means model still relation-heavy (ideally ~0.5 for balanced)
- Calibration signals (margin, entropy) don't correlate with when relation helps vs harms
- Gate variance (0.066) shows modulation, but modulation is not aligned with benefit

---

## Conclusion

**CALIBRATION-NOT-WORTH-CONTINUING** (current approach)

The gate collapse issue is FIXED, but the fundamental problem remains: **calibration signals are not informative enough** to determine when relation scores should be trusted.

**What Worked**:
- Gate prior regularization (weight=1.0) successfully prevents collapse
- init_bias=0.5 provides balanced initialization
- Gate now has meaningful variance (std=0.066)

**What Didn't Work**:
- Calibration still harms more than it helps (Net -12)
- Performance worse than both Base and Dense-no-cal
- Gate modulation is not aligned with actual benefit/harm

**Recommendation**: 
- Keep Dense-no-cal as lightweight improvement (+0.22% Acc@1, +9 net)
- Abandon current calibration approach (signals need complete redesign)
- If calibration is essential, need new signals that actually predict when relation helps

---

## Files Generated

- `reports/calibration_failure_analysis.md` - Root cause analysis
- `reports/calibration_redesign_log.md` - Design iterations
- `reports/calibration_controlled_rerun.md` - Full results
- `reports/calibration_comparison_table.csv` - Comparison data
