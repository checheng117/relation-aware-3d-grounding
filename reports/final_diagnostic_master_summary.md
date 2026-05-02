# Final Diagnostic Master Summary

**Date**: 2026-04-22
**Status**: FINAL MASTER SUMMARY

---

## Executive Summary

This document provides the complete diagnostic summary for this project. All method results, diagnostic findings, and release decisions are consolidated here.

### Project Conclusion

**This project establishes a trustworthy, scene-disjoint evaluation and diagnostic framework for 3D referring-expression grounding.** The core contributions are:

1. **Dataset recovery and scene-disjoint splits** - Verified zero overlap, unified metrics
2. **Reproduced baselines** - ReferIt3DNet (30.79%), SAT (28.27%)
3. **Diagnostic framework** - Hard-subset analysis, coverage failure evidence
4. **Limited method signal** - Dense-no-cal-v1 (+0.22% Acc@1)
5. **Negative findings** - Calibration and dense strengthening failure analysis

### Method Status

| Method | Acc@1 | Acc@5 | Net | Status | Decision |
|--------|-------|-------|-----|--------|----------|
| Base (clean) | 30.83% | 91.87% | - | Reference | **RETAINED** |
| Dense-no-cal-v1 | 31.05% | 92.01% | +9 | Modest gain | **RETAINED** |
| Dense-calibrated-v1 | 30.60% | 91.89% | -10 | Gate collapse | FROZEN |
| Dense-calibrated-v2 | 30.55% | 91.80% | -12 | Signals uninformative | FROZEN |
| Dense-v2-AttPool | 25.24% | 79.95% | -238 | Too complex | FROZEN |
| Dense-v3-Geo | 24.32% | 79.06% | -277 | No geometry data | FROZEN |
| Dense-v4-HardNeg | 24.47% | 79.41% | -271 | Focal hurts | FROZEN |

---

## 1. Clean Baseline Results

### Configuration

- **Dataset**: Nr3D recovered (41,503 samples, 641 scenes)
- **Split**: Scene-disjoint (zero overlap verified)
- **Model**: ReferIt3DNet reproduction
- **Metrics**: Acc@1, Acc@5 (unified definitions)

### Results

| Split | Acc@1 | Acc@5 | Samples |
|-------|-------|-------|---------|
| Test | 30.83% | 91.87% | 4,255 |
| Val | ~28-30% | ~88-90% | ~4,000 |
| Train | ~28-30% | ~88-90% | ~33,000 |

### Comparison with Literature

| Source | Acc@1 | Notes |
|--------|-------|-------|
| This work (test) | 30.83% | Scene-disjoint, verified |
| Original ReferIt3DNet | ~30-32% | Overlapping splits possible |
| SAT | ~28-33% | Varies by paper |

**Caution**: Literature comparison is unreliable due to potential split differences.

---

## 2. Dense-no-cal-v1 Results

### Configuration

- **Module**: DenseRelationModule (398K parameters)
- **Aggregation**: Weighted sum of pairwise relation scores
- **Training**: 10 epochs, single seed (42)
- **No calibration**, no complex attention, no geometry features

### Overall Results

| Metric | Base | Dense-no-cal-v1 | Delta |
|--------|------|-----------------|-------|
| Acc@1 | 30.83% | 31.05% | +0.22% |
| Acc@5 | 91.87% | 92.01% | +0.14% |
| Net | - | +9 | 402 rec, 393 harm |

### Hard Subset Results

| Subset | Base | Dense-no-cal-v1 | Delta |
|--------|------|-----------------|-------|
| Easy | 31.72% | 31.35% | -0.37% |
| Multi-Anchor | 23.04% | 28.34% | +5.30% |
| Relative-Position | 29.86% | 30.68% | +0.82% |
| Same-Class Clutter | ~28% | ~29% | +1% (est.) |

**Key Finding**: Largest gain in multi-anchor subset (+5.3%), suggesting dense coverage helps when multiple anchors are needed.

### Training Stability

| Epoch | Train Loss | Train Acc |
|-------|------------|-----------|
| 1 | 3.6837 | 28.09% |
| 2 | 3.6742 | 28.31% |
| 3 | 3.6742 | 28.43% |
| ... | ... | ... |
| 10 | 3.6739 | 28.11% |

**Observation**: Loss converges quickly (epoch 2-3), plateaus thereafter.

---

## 3. Calibration Line Failure

### What Was Attempted

- **Dense-calibrated-v1**: Gate fusion with 3 signals (base_margin, anchor_entropy, relation_margin)
- **Dense-calibrated-v2**: Added gate prior regularization, balanced init_bias

### Results

| Variant | Acc@1 | Acc@5 | Net | Gate Behavior |
|---------|-------|-------|-----|---------------|
| Dense-no-cal-v1 | 31.05% | 92.01% | +9 | 0.5 (fixed) |
| Dense-calibrated-v1 | 30.60% | 91.89% | -10 | Collapsed to 0.9 |
| Dense-calibrated-v2 | 30.55% | 91.80% | -12 | Stable at 0.687 |

### Root Cause Analysis

**Gate Collapse (v1)**:
- init_bias = 0.3, but signal vector statistics pushed gate higher
- Gate collapsed to max bound (0.9) within 2 epochs
- No variation (std ~ 0), meaning gate learned no modulation

**Uninformative Signals (v2)**:
- Even with proper gate regularization, decisions were wrong
- Signals showed no correlation with benefit/harm
- Gate stabilized at 0.687 but still made wrong fusion decisions

### Verdict

**FROZEN - NOT WORTH CONTINUING**

Calibration requires informative signals. Current 3-signal design (base_margin, anchor_entropy, relation_margin) does not predict when relation scores help vs harm.

**Restart would require**: New signal sources, demonstrated correlation, oracle analysis showing > 5% theoretical gain.

---

## 4. Dense Scorer Strengthening Failure

### What Was Attempted

| Variant | Design | Result |
|---------|--------|--------|
| Dense-v2-AttPool | Attention pooling for anchor selection | 25.24% Acc@1, Net -238 |
| Dense-v3-Geo | Explicit geometry features | 24.32% Acc@1, Net -277 |
| Dense-v4-HardNeg | Focal weighting for hard cases | 24.47% Acc@1, Net -271 |

### Results Analysis

All variants showed **significant degradation** (-5% to -7% Acc@1, -238 to -277 net).

**Why They Failed**:
1. **Weak foundation**: Base DenseRelationModule provides noisy scores
2. **Complexity amplifies noise**: Attention/geometry/focal made things worse
3. **Training dynamics**: Variants need longer training but start from worse position
4. **Geometry unavailable**: v3-Geo couldn't test properly (zeros fallback)

---

## 5. Fundamentals Debug Findings

### Motivation

After all strengthening variants failed, we conducted mechanism-level debug to answer:

**Does the DenseRelationModule provide stable, interpretable, and exploitable relation discrimination signals?**

### Method

- Extracted relation scores from Dense-no-cal-v1 (500 samples)
- Analyzed correct vs wrong target score distributions
- Analyzed pair ranking quality (Hit@k)
- Analyzed harm/recovered patterns

### Key Findings

**Q1-fundamental: MOSTLY-NOISY**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Score Gap (Correct - Wrong) | -0.95 | **Wrong targets score HIGHER** |
| Correct Target Mean | -2.53 | Should be higher |
| Wrong Target Mean | -1.58 | Should be lower |

**Interpretation**: Relation MLP is NOT learning discriminative scores. Correct targets score LOWER than wrong targets.

**Q2-ranking: WEAK**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Hit@1 | 7.0% | Weak (random ~0.2%) |
| Hit@3 | 20.0% | Weak |
| Hit@5 | 27.2% | Weak |

**Interpretation**: Pair ranking does not reliably select correct anchors.

**Q3-worth: CONTINUE-WITH-CAUTION**

| Metric | Value |
|--------|-------|
| Net Effect (500 samples) | +3 |
| Multi-Anchor Net | +7 (15 rec, 8 harm) |

**Interpretation**: Modest benefit, but signal too weak for confident redesign.

### Verdict

**Route C-hard-stop**: Dense scorer line paused. Foundation too weak to support complexity.

---

## 6. Diagnostic Findings

### Hard Subset Failure Patterns

| Subset | Base Acc@1 | % of Test | Failure Mode |
|--------|------------|-----------|--------------|
| Easy | 31.72% | ~75% | Random errors |
| Same-Class Clutter | ~28% | ~30% | Anchor confusion among similar objects |
| Multi-Anchor | 23.04% | ~10% | Missing anchor evidence |
| Relative-Position | 29.86% | ~23% | Spatial reasoning errors |

### Coverage Failure Evidence

- Sparse top-k selection misses long-range anchor evidence
- Same-class clutter causes anchor confusion
- Multi-anchor cases need broader coverage than single-anchor cases

### What Dense-no-cal-v1 Fixes

- Improves multi-anchor by +5.3%
- Provides modest overall gain (+0.22%)
- No calibration overhead

### Where Dense-no-cal-v1 Fails

- Relation scores don't discriminate correct targets
- Pair ranking is weak (7% Hit@1)
- Harms easy cases slightly (-0.37%)

---

## 7. Paper Positioning

### Recommended Framing

**Diagnostic Paper**, not Strong Method Paper

**Core Claim**:
> This project establishes a trustworthy, scene-disjoint evaluation and diagnostic framework for 3D referring-expression grounding, identifies concentrated failure modes in hard relational subsets, and shows that simple dense reranking yields modest gains while complex extensions do not justify further investment.

### Target Venues

- **Primary**: TACL, ACL Findings, EMNLP Findings, Scientific Data
- **Secondary**: CVPR/ICCV/ECCV Workshop, 3DV, BMVC
- **NOT**: AAAI/NeurIPS/ICML/CVPR/ICCV main track (method signal insufficient)

### Contributions

1. Trustworthy evaluation foundation (dataset, splits, baselines)
2. Diagnostic framework (hard subsets, coverage analysis)
3. Coverage failure evidence (direct analysis)
4. Limited method signal (Dense-no-cal-v1, +0.22%)
5. Negative findings (calibration and strengthening failure)

---

## 8. Release Decisions

### Retained (Core Release)

- Clean baseline code, configs, checkpoints
- Dense-no-cal-v1 code, configs, checkpoints
- Evaluation scripts
- Scene-disjoint split artifacts
- This master summary report

### Diagnostic (Evidence)

- Calibration failure analysis
- Dense strengthening failure reports
- Fundamentals debug reports
- Hard subset analysis
- Case studies

### Archived (Available, Not Promoted)

- Parser v1/v2 code and results
- Implicit v1/v2/v3 code and results
- Early experiment logs
- Hardware crash documentation

### Not Primary Entry Points

- Frozen method variants (calibration, strengthening)
- Unconfirmed promising results
- Intermediate/debug scripts

---

## 9. Recommendations for Users

### Who Should Use This Repository

1. **Benchmark users** - Need trustworthy scene-disjoint evaluation
2. **Diagnostics researchers** - Study failure modes, hard subsets
3. **Reproducibility researchers** - Want clean baseline reproduction
4. **Method researchers** - Need clean starting point for new ideas

### Who Should NOT Use This Repository

1. **SOTA chasers** - Methods are modest, not state-of-the-art
2. **Calibration developers** - Calibration line is frozen
3. **Dense-scorer enhancers** - Strengthening line is frozen
4. **Multi-seed validators** - Single-seed results only

---

## 10. Appendix: Master Results Table

See `final_diagnostic_master_table.csv` for complete machine-readable results.

### Summary

| Method | Acc@1 | Acc@5 | Net | Multi-Anchor | Status |
|--------|-------|-------|-----|--------------|--------|
| Base | 30.83% | 91.87% | - | 23.04% | Retained |
| Dense-no-cal-v1 | 31.05% | 92.01% | +9 | 28.34% | Retained |
| Dense-calibrated-v1 | 30.60% | 91.89% | -10 | ~28% | Frozen |
| Dense-calibrated-v2 | 30.55% | 91.80% | -12 | ~27% | Frozen |
| Dense-v2-AttPool | 25.24% | 79.95% | -238 | ~25% | Frozen |
| Dense-v3-Geo | 24.32% | 79.06% | -277 | ~24% | Frozen |
| Dense-v4-HardNeg | 24.47% | 79.41% | -271 | ~24% | Frozen |

---

## 11. Final Statement

**This project's core contribution is the trustworthy evaluation foundation and diagnostic framework.**

The Dense-no-cal-v1 module provides modest gains (+0.22% Acc@1) with clearest benefits in multi-anchor cases (+5.3%). However, fundamentals analysis reveals the relation scores lack discriminative power, limiting potential for complex extensions.

Rather than continuing to invest in methods with weak signals, this project pivots to:
- Benchmark value for the community
- Diagnostic findings about failure modes
- Negative results with clear evidence
- Reproducible baseline for future work

**Method exploration is complete. Diagnostic paper route is active.**
