# Calibration Controlled Rerun Results

**Date**: 2026-04-21
**Status**: COMPLETE

---

## Experiment Configuration

**Variant**: dense-calibrated-v2
**Epochs**: 10
**Batch Size**: 4
**Seed**: 42

**Redesign Changes**:
- init_bias: 0.3 → 0.5 (balanced initialization)
- gate_prior_weight: 0.0 → 1.0 (L2 regularization toward 0.5)
- gate_prior_target: 0.5 (balanced gate prior)

---

## Test Set Results

### Overall Metrics

| Metric | Value |
|--------|-------|
| Acc@1 | 30.55% |
| Acc@5 | 91.80% |
| Gate Mean | 0.687 |
| Gate Std | 0.066 |
| Total Samples | 4255 |

### Harmed/Recovered Analysis

| Category | Count |
|----------|-------|
| Recovered | 459 |
| Harmed | 471 |
| Net | -12 |

### Hard Subset Performance

| Subset | Acc@1 | Correct | Total |
|--------|-------|---------|-------|
| Overall | 30.55% | 1300 | 4255 |
| Easy | 30.49% | 1165 | 3821 |
| Multi-anchor | 31.11% | 135 | 434 |
| Relative position | 29.86% | 289 | 968 |

---

## Comparison with All Variants

### Main Results Table

| Variant | Acc@1 | Acc@5 | Harmed | Recovered | Net | Gate Mean | Gate Std |
|---------|-------|-------|--------|-----------|-----|-----------|----------|
| Base | 30.83% | 91.87% | - | - | 0 | 1.0 (fixed) | 0 |
| Dense-no-cal | 31.05% | 92.01% | 393 | 402 | +9 | 0.5 (fixed) | 0 |
| Dense-calibrated (v1) | 30.60% | 91.89% | 444 | 434 | -10 | 0.9 (collapsed) | ~0 |
| **Dense-calibrated-v2** | **30.55%** | **91.80%** | **471** | **459** | **-12** | **0.687** | **0.066** |

### Delta Analysis

| Variant | Δ Acc@1 vs Base | Δ Net vs Base | Δ Net vs Dense-no-cal |
|---------|-----------------|---------------|----------------------|
| Base | - | - | - |
| Dense-no-cal | +0.22% | +9 | - |
| Dense-calibrated (v1) | -0.23% | -10 | -19 |
| Dense-calibrated-v2 | -0.28% | -12 | -21 |

---

## Gate Behavior Analysis

### Training Evolution

| Epoch | Gate Mean | Gate Std | Gate Loss |
|-------|-----------|----------|-----------|
| 1 | 0.668 | - | 0.0306 |
| 2 | 0.690 | - | 0.0366 |
| 3 | 0.688 | - | 0.0363 |
| 4 | 0.686 | - | 0.0362 |
| 5 | 0.681 | - | 0.0355 |
| 6 | 0.678 | - | 0.0358 |
| 7 | 0.679 | - | 0.0371 |
| 8 | 0.677 | - | 0.0375 |
| 9 | 0.677 | - | 0.0379 |
| 10 | 0.675 | - | 0.0375 |
| Test | 0.687 | 0.066 | N/A |

**Observation**: Gate stable at ~0.67-0.69 throughout training. No collapse to bounds.

### Gate Distribution (Test Set)

- Mean: 0.687
- Std: 0.066
- Min/Max: Not reported (but no collapse warning triggered)

**Interpretation**: 
- Gate std=0.066 indicates meaningful modulation across samples
- Gate mean=0.687 indicates relation-heavy bias (but not collapsed)
- Calibration is working mechanically, but decisions are suboptimal

---

## Research Questions

### Q2-redesign: Does redesigned calibration improve over collapsed calibration?

**Answer**: PARTIAL

- Gate behavior: IMPROVED (0.9 collapsed → 0.687 with variance)
- Acc@1: SLIGHTLY WORSE (30.60% → 30.55%)
- Net: WORSE (-10 → -12)

The redesign fixed gate collapse but did not improve overall performance.

### Q2-final: Does calibration reduce harm without suppressing recovery?

**Answer**: NO

- Dense-calibrated-v2 Net (-12) < Base (0)
- Dense-calibrated-v2 Net (-12) < Dense-no-cal (+9)
- Dense-calibrated-v2 Acc@1 (30.55%) < Base (30.83%)

Calibration still harms more than it helps, even with proper gate regularization.

---

## Root Cause Analysis

### What Was Fixed
1. **Gate collapse prevented**: Prior regularization successfully anchors gate near 0.5-0.7
2. **Gate variance restored**: std=0.066 shows gate is modulating based on signals
3. **Balanced initialization**: init_bias=0.5 provides neutral starting point

### What Remains Broken
1. **Signal quality**: Calibration signals (margin, entropy) don't predict when relation helps
2. **Gate bias**: Mean=0.687 still relation-heavy (should be ~0.5 for true balance)
3. **Wrong modulation**: Gate changes are not aligned with actual benefit/harm per sample

### Fundamental Issue
The calibration MLP receives uninformative signals:
- `base_margin`: Doesn't distinguish recoverable vs unrecoverable cases
- `anchor_entropy`: Doesn't correlate with when relation evidence is trustworthy
- `relation_margin`: Doesn't predict if relation will help or harm

Gate modulation based on these signals produces noise, not calibration.

---

## Route Decision

**Route C-confirmed: CALIBRATION-NOT-WORTH-CONTINUING**

**Rationale**:
1. Gate collapse is FIXED but performance is still worse than baselines
2. Signal redesign would require fundamental changes (new features, not just tuning)
3. Dense-no-cal already provides modest gain (+0.22% Acc@1, +9 net) with no complexity
4. Further calibration tuning has diminishing returns

**Recommendation**:
- **Keep Dense-no-cal** as lightweight improvement over Base
- **Pause calibration line** - not worth further investment without new signal sources
- **Focus on dense scorer improvements** if further gains are needed

---

## Files Generated

| File | Description |
|------|-------------|
| `outputs/cover3d_round1/dense-calibrated-v2_results.json` | Full results |
| `outputs/cover3d_round1/dense-calibrated-v2_predictions.json` | Per-sample predictions |
| `outputs/cover3d_round1/dense-calibrated-v2_training_full.log` | Training log |
| `reports/calibration_failure_analysis.md` | Root cause analysis |
| `reports/calibration_redesign_log.md` | Design iterations |
| `reports/calibration_controlled_rerun.md` | This report |
| `reports/calibration_comparison_table.csv` | Comparison data |

---

## Conclusion

**Calibration redesign succeeded mechanically but failed functionally**:

- Mechanically: Gate no longer collapses, has meaningful variance
- Functionally: Gate decisions still harm more than help

**Final verdict**: Current calibration approach is NOT salvageable without new signal sources. Dense-no-cal is the recommended lightweight improvement.
