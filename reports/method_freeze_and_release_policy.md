# Method Freeze and Release Policy

**Date**: 2026-04-22
**Status**: PERMANENT FREEZE

---

## Executive Summary

All method exploration lines are now **permanently frozen**. The project has transitioned to **diagnostic paper + open-source release** mode.

This document defines:
1. What is frozen and why
2. What is retained as the method contribution
3. Restart prerequisites (if method line to be revisited)
4. Release policy for open-source distribution

---

## Frozen Method Lines

### 1. Calibration Line (PERMANENT FREEZE)

**Variants**:
- Dense-calibrated-v1 (Acc@1: 30.60%, Net: -10)
- Dense-calibrated-v2 (Acc@1: 30.55%, Net: -12)

**Reason for Freeze**:
- Gate collapse to max bound (0.9) within 2 epochs
- Calibration signals (base_margin, anchor_entropy, relation_margin) are uninformative
- Even with gate prior regularization, decisions remain wrong
- Signal quality too weak to support learning

**Restart Prerequisites** (ALL must be met):
1. New signal sources beyond current 3-signal design
2. Demonstrated correlation between signals and benefit/harm
3. Oracle signal analysis showing theoretical upper bound > 5% gain
4. Not just hyperparameter tuning - requires new features

**Verdict**: **NOT RECOMMENDED** - Signal redesign is effectively a new method, not calibration fix.

---

### 2. Dense Scorer Strengthening Line (PERMANENT FREEZE)

**Variants**:
- Dense-v2-AttPool (Acc@1: 25.24%, Net: -238)
- Dense-v3-Geo (Acc@1: 24.32%, Net: -277)
- Dense-v4-HardNeg (Acc@1: 24.47%, Net: -271)

**Reason for Freeze**:
- Fundamentals debug revealed relation scores are MOSTLY-NOISY
- Score gap = -0.95 (correct targets score LOWER than wrong targets)
- Pair ranking Hit@1 = 7.0% (weak anchor selection)
- Adding complexity (attention, geometry, focal) amplified noise, not signal
- No clear redesign path from weak foundation

**Restart Prerequisites** (ALL must be met):
1. New pair representation (not current obj_i + obj_j + lang)
2. Demonstrated discriminative signal in pair scores (gap > 0.5)
3. Pair ranking Hit@1 > 30% on validation
4. New architecture (not MLP on current features)
5. External supervision or stronger geometry features

**Verdict**: **NOT RECOMMENDED** - Current dense scorer foundation cannot support complexity.

---

### 3. Multi-Seed Method Validation (FROZEN)

**Reason for Freeze**:
- Method signal is modest (+0.22% Acc@1)
- Variance would likely swamp signal
- Not worth computational cost for marginal confirmation

**Restart Prerequisites**:
1. Method improvement > 1.0% Acc@1
2. Clear theoretical justification for multi-seed stability
3. Hardware resources for 3-seed runs without crashes

**Verdict**: **NOT RECOMMENDED** - Diagnostic paper value exceeds method confirmation value.

---

### 4. Architecture Redesign (FROZEN)

**Reason for Freeze**:
- Would be "throwing good money after bad"
- Current evaluation foundation is the core contribution
- Method gains are diminishing returns

**Restart Prerequisites**:
1. New data modality (multi-view, temporal, etc.)
2. New supervision signal (part-level, attention maps, etc.)
3. Cross-backbone validation framework in place

**Verdict**: **NOT RECOMMENDED** - Better to start fresh method paper with new foundation.

---

## Retained Method Contribution

### Dense-no-cal-v1 (RETAINED)

**Results**:
| Metric | Value | Delta |
|--------|-------|-------|
| Acc@1 | 31.05% | +0.22% |
| Acc@5 | 92.01% | +0.14% |
| Net | +9 | 402 recovered, 393 harmed |
| Multi-Anchor | 28.34% | +5.3% |
| Parameters | 398K | - |

**What It Is**:
- A lightweight dense reranking module
- Simple weighted aggregation of pairwise relation scores
- No calibration, no complex attention, no geometry
- Stable training, reproducible results

**What It Is NOT**:
- A strong SOTA-challenging method
- A scalable foundation for complex extensions
- A calibration-ready base
- Near-SOTA improvement

**Framing for Paper**:
> "A simple dense relation reranker provides modest gains (+0.22% Acc@1) with largest benefits in multi-anchor cases (+5.3%). However, fundamentals analysis reveals the relation scores lack discriminative power, limiting potential for more complex extensions."

---

## Release Policy

### What to Release

**Core (Always Include)**:
- Clean baseline reproduction
- Scene-disjoint split artifacts
- Dense-no-cal-v1 method code
- Evaluation scripts
- Master summary report
- Diagnostic analysis reports

**Diagnostic (Include as Evidence)**:
- Calibration failure analysis
- Dense strengthening failure reports
- Fundamentals debug reports
- Hard-subset analysis
- Case studies

**Archived (Available but Not Promoted)**:
- Early failed variants (Parser v1/v2, Implicit v1/v2)
- Crash logs and hardware issues
- Intermediate experiment logs

**Do Not Release as Primary**:
- Unconfirmed promising results (Implicit v3)
- Hardware-dependent runs
- Multi-seed attempts (not done)

### Release Framing

**DO Say**:
- "Trustworthy evaluation foundation"
- "Diagnostic framework for failure analysis"
- "Modest method gain with clear boundaries"
- "Negative results with evidence"

**DON'T Say**:
- "Strong method contribution"
- "SOTA-challenging results"
- "Scalable framework"
- "Foundation for future method work"

---

## Final Statement

**This project's core contribution is the trustworthy evaluation foundation and diagnostic framework, not method innovation.**

The Dense-no-cal-v1 module provides evidence that dense relation coverage can yield modest gains, but the fundamentals analysis reveals significant limitations in the current approach.

Rather than continuing to invest in methods with weak signals, this project pivots to:
- Benchmark value for the community
- Diagnostic findings about failure modes
- Negative results with clear evidence
- Reproducible baseline for future work

**Method line is permanently frozen. Diagnostic paper route is active.**
