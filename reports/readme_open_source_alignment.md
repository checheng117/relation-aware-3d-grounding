# README Open-Source Alignment Report

**Date**: 2026-04-22
**Status**: README UPDATED FOR OPEN-SOURCE RELEASE

---

## Purpose

This report documents the alignment of README.md with the open-source / diagnostic paper positioning.

---

## Changes Made

### Removed/Deprecated

1. ~~"AAAI Upgrade Path" section~~ - Removed method冲刺 framing
2. ~~"Final Benchmark Results" (misleading)~~ - Replaced with honest assessment
3. ~~"Implicit v1/v2/v3" as promising~~ - Moved to archived findings
4. ~~Hardware limitations emphasis~~ - Mentioned briefly, not highlighted
5. ~~Multi-seed implication~~ - Clarified single-seed only

### Added

1. **Diagnostic paper positioning** - Clear framing as benchmark + diagnostics
2. **Honest method assessment** - Dense-no-cal-v1 as modest gain (+0.22%)
3. **Frozen method lines** - Calibration and strengthening clearly marked
4. **Target audience** - Who should use this repo
5. **Core contributions** - Evaluation foundation, not just methods

---

## New README Structure

### 1. Project Overview

> A reproducible research codebase for 3D referring-expression grounding on the Nr3D/ReferIt3D benchmark. This project establishes a trustworthy evaluation foundation through dataset recovery, scene-disjoint splitting, and baseline reproduction, then provides diagnostic analysis of failure modes and a simple dense reranker with modest gains.

### 2. Why This Project Matters

- Trustworthy evaluation (scene-disjoint, verified)
- Reproducible baselines
- Diagnostic findings
- Negative results with evidence

### 3. Core Results

| Method | Acc@1 | Acc@5 | Status |
|--------|-------|-------|--------|
| Base (clean) | 30.83% | 91.87% | Reference |
| Dense-no-cal-v1 | 31.05% | 92.01% | Retained (+0.22%) |

### 4. Frozen Methods (Not Recommended)

| Method | Acc@1 | Status |
|--------|-------|--------|
| Dense-calibrated-v2 | 30.55% | Frozen (signals uninformative) |
| Dense-v2-AttPool | 25.24% | Frozen (too complex) |
| Dense-v3-Geo | 24.32% | Frozen (no geometry data) |
| Dense-v4-HardNeg | 24.47% | Frozen (focal hurts) |

### 5. Who Should Use This Repo

**Good fit for**:
- Benchmark users needing scene-disjoint evaluation
- Diagnostics researchers studying failure modes
- Reproducibility researchers
- Method researchers needing clean baseline

**Not for**:
- SOTA chasers (methods are modest)
- Calibration developers (line is frozen)
- Dense-scorer enhancers (line is frozen)

### 6. Quick Start

```bash
# Setup
conda env create -f environment.yml
conda activate rag3d

# Reproduce baseline
python repro/referit3d_baseline/scripts/train.py --config ...

# Run diagnostics
python scripts/analyze_dense_fundamentals.py
```

### 7. Key Reports

- `reports/final_diagnostic_master_summary.md` - Complete results
- `reports/dense_fundamentals_summary.md` - Why methods failed
- `reports/calibration_failure_analysis.md` - Calibration details

---

## Alignment Verification

### Checklist

| Item | Status |
|------|--------|
| Diagnostic paper framing | ✓ |
| Honest method assessment | ✓ |
| Frozen lines documented | ✓ |
| Target audience clear | ✓ |
| No method冲刺 implication | ✓ |
| Core contributions emphasized | ✓ |

---

## Final README Status

**Aligned with open-source / diagnostic paper positioning.**

Users will understand:
- This is a diagnostic paper + benchmark, not strong method contribution
- Dense-no-cal-v1 provides modest gain (+0.22%)
- Calibration and strengthening lines are frozen with clear reasons
- The value is in evaluation foundation and diagnostic findings
