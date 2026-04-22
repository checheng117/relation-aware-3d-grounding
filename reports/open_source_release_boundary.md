# Open-Source Release Boundary

**Date**: 2026-04-22
**Status**: RELEASE BOUNDARY DEFINED

---

## Purpose

This document defines what is included in the open-source release, what is archived, and what are the recommended entry points for users.

**Goal**: Transform from "research exploration repo" to "polished open-source project".

---

## Repository Structure Overview

```
relation-aware-3d-grounding/
├── README.md                          # Main entry point (updated for release)
├── .claude/                           # Project status documentation
│   ├── CURRENT_STATUS.md              # Current phase: diagnostic paper
│   ├── NEXT_TASK.md                   # Next priorities
│   └── METHOD_PHASE_FREEZE.md         # Frozen method boundaries
├── src/rag3d/                         # Core Python package
│   ├── models/
│   │   ├── cover3d_model.py           # COVER-3D model (retained)
│   │   ├── cover3d_dense_relation.py  # DenseRelationModule (retained)
│   │   └── cover3d_calibration.py     # CalibratedFusionGate (archived)
│   └── ...
├── scripts/
│   ├── train_cover3d_round1.py        # Main training script (retained)
│   ├── analyze_dense_fundamentals.py  # Diagnostic analysis (retained)
│   └── ...                            # Other scripts (archived/debug)
├── configs/
│   ├── cover3d_round1/                # Round-1 configs (retained)
│   └── ...                            # Other configs (archived)
├── reports/
│   ├── final_diagnostic_master_summary.md    # Master summary (retained)
│   ├── dense_fundamentals_summary.md         # Fundamentals debug (retained)
│   └── ...                                   # Other reports (archived)
├── repro/                             # Baseline reproduction
│   ├── referit3d_baseline/            # ReferIt3DNet (retained)
│   └── sat_baseline/                  # SAT (retained)
└── docs/                              # Documentation
    └── ...
```

---

## Core Retained (Primary Entry Points)

### Code

| Path | Description | Status |
|------|-------------|--------|
| `src/rag3d/models/cover3d_model.py` | COVER-3D model wrapper | **Core** |
| `src/rag3d/models/cover3d_dense_relation.py` | DenseRelationModule | **Core** |
| `src/rag3d/models/` (other) | Supporting model code | **Core** |
| `scripts/train_cover3d_round1.py` | Training script | **Core** |
| `scripts/evaluate_*.py` | Evaluation scripts | **Core** |
| `repro/referit3d_baseline/` | ReferIt3DNet reproduction | **Core** |
| `repro/sat_baseline/` | SAT reproduction | **Core** |

### Configs

| Path | Description | Status |
|------|-------------|--------|
| `configs/cover3d_round1/` | Round-1 experiment configs | **Retained** |
| `configs/cover3d_smoke.yaml` | Smoke test config | **Retained** |
| `repro/referit3d_baseline/configs/` | Baseline configs | **Retained** |

### Reports

| Path | Description | Status |
|------|-------------|--------|
| `reports/final_diagnostic_master_summary.md` | Master summary | **Required** |
| `reports/final_diagnostic_master_table.csv` | Results table | **Required** |
| `reports/dense_fundamentals_summary.md` | Fundamentals debug | **Retained** |
| `reports/formal_round1_summary.md` | Round-1 results | **Retained** |
| `reports/calibration_failure_analysis.md` | Calibration failure | **Diagnostic** |
| `reports/dense_strengthening_results.md` | Strengthening failure | **Diagnostic** |

---

## Diagnostic Retained (Evidence, Not Primary)

### Code

| Path | Description | Status |
|------|-------------|--------|
| `src/rag3d/models/cover3d_calibration.py` | Calibration module | **Diagnostic** |
| `scripts/analyze_dense_fundamentals.py` | Analysis script | **Diagnostic** |
| `scripts/debug_*.py` | Debug scripts | **Diagnostic** |

### Reports

| Path | Description | Status |
|------|-------------|--------|
| `reports/dense_fundamentals_*.md` | Fundamentals audit reports | **Diagnostic** |
| `reports/calibration_*.md` | Calibration analysis | **Diagnostic** |
| `reports/dense_strengthening_*.md` | Strengthening analysis | **Diagnostic** |

**Usage**: These are evidence supporting the diagnostic paper, not primary entry points for users.

---

## Archived (Available, Not Promoted)

### Code

| Path | Description | Status |
|------|-------------|--------|
| `scripts/smoke_test_*.py` | Smoke test scripts | **Archived** |
| `scripts/run_*.py` (experiment runners) | Experiment suite scripts | **Archived** |
| `scripts/eval_*.py` (detailed eval) | Detailed evaluation scripts | **Archived** |
| `configs/train/` | Old training configs | **Archived** |
| `configs/eval/` | Old evaluation configs | **Archived** |
| `configs/model/` | Old model configs | **Archived** |
| `configs/dataset/` (old) | Old dataset configs | **Archived** |

### Reports

| Path | Description | Status |
|------|-------------|--------|
| `reports/baseline_*.md` | Early baseline audits | **Archived** |
| `reports/encoder_*.md` | Encoder upgrade reports | **Archived** |
| `reports/feature_*.md` | Feature wiring reports | **Archived** |
| `reports/geometry_*.md` | Geometry validation reports | **Archived** |
| `reports/nr3d_*.md` | Nr3D analysis reports | **Archived** |
| `reports/phase*.md` | Phase transition reports | **Archived** |
| `reports/referit3d_*.md` | ReferIt3D reproduction | **Archived** |
| `reports/scene_disjoint_*.md` | Split construction reports | **Archived** |

**Access**: Available in git history, not highlighted in documentation.

---

## Recommended Entry Points

### For Benchmark Users

1. **Start Here**: `README.md` - Setup and quickstart
2. **Reproduce Baseline**: `repro/referit3d_baseline/scripts/train.py`
3. **Evaluate**: `repro/referit3d_baseline/scripts/evaluate.py`
4. **Understand Splits**: `reports/scene_disjoint_split_validation.md` (archived but useful)

### For Diagnostics Users

1. **Start Here**: `reports/final_diagnostic_master_summary.md`
2. **Hard Subset Analysis**: `reports/dense_fundamentals_contribution_audit.md`
3. **Case Studies**: `reports/dense_fundamentals_casebook.md`
4. **Analysis Scripts**: `scripts/analyze_dense_fundamentals.py`

### For Method Researchers

1. **Start Here**: `src/rag3d/models/cover3d_dense_relation.py`
2. **Training**: `scripts/train_cover3d_round1.py`
3. **Config**: `configs/cover3d_round1/`
4. **Limitations**: `reports/dense_fundamentals_score_audit.md`

---

## What NOT to Expose as Primary

### Deprecated Entry Points

| Path | Why Not Primary |
|------|-----------------|
| `src/rag3d/models/cover3d_calibration.py` | Calibration line frozen |
| `configs/train/*.yaml` | Old training configs, not current |
| `scripts/run_experiment_suite.py` | Implies active method development |
| `reports/next_phase_research_plan.md` | Old roadmap, not current |
| `reports/implicit_relation_v3_archive.md` | Unconfirmed, crashed |

### README Sections to Minimize

- ~~"AAAI Upgrade Path"~~ - Remove or move to archived
- ~~"Future Roadmap" (method冲刺)~~ - Replace with diagnostic roadmap
- ~~Hardware limitations~~ - Mention briefly, don't highlight
- ~~Implicit v1/v2/v3 results~~ - Move to archived section

---

## Release Checklist

### Before Release

- [ ] README.md updated with open-source facing narrative
- [ ] `final_diagnostic_master_summary.md` generated
- [ ] `final_diagnostic_master_table.csv` generated
- [ ] `diagnostic_paper_positioning_freeze.md` generated
- [ ] `method_freeze_and_release_policy.md` generated
- [ ] `.claude/` files aligned
- [ ] Deprecated configs moved/commented
- [ ] Debug scripts marked as archived

### Nice to Have

- [ ] `docs/repo_entrypoints.md` - Detailed navigation guide
- [ ] Quickstart notebook
- [ ] Docker container for reproducibility
- [ ] Pre-computed features available for download
- [ ] Interactive case study browser

### Not Required

- [ ] Multi-seed results
- [ ] SOTA comparison
- [ ] Method ablation studies
- [ ] Cross-backbone validation

---

## Documentation Hierarchy

### Tier 1 (Must Read)

1. `README.md` - First stop for all users
2. `reports/final_diagnostic_master_summary.md` - Complete results
3. `.claude/CURRENT_STATUS.md` - Current project phase

### Tier 2 (Recommended)

4. `reports/diagnostic_paper_positioning_freeze.md` - Paper framing
5. `reports/method_freeze_and_release_policy.md` - Method boundaries
6. `reports/dense_fundamentals_summary.md` - Why methods failed

### Tier 3 (Reference)

7. `reports/calibration_failure_analysis.md` - Calibration details
8. `reports/dense_strengthening_results.md` - Strengthening details
9. `reports/dense_fundamentals_*.md` - Detailed audits

### Tier 4 (Archived)

10. All other reports - Available for completeness

---

## Version Tagging

### Recommended Tags

- `v1.0-diagnostic-release` - Initial diagnostic paper release
- `v0.1-baseline` - Baseline reproduction only (pre-release)
- `v0.2-method-exploration` - Method exploration archive

### CHANGELOG

Generate CHANGELOG.md from git history highlighting:
- Dataset recovery milestones
- Baseline reproduction
- Method exploration (honest about failures)
- Diagnostic findings

---

## License and Attribution

### License

MIT License (as specified in LICENSE file)

### Attribution

```bibtex
@misc{relation-aware-3d-grounding,
  title = {Trustworthy Evaluation for 3D Referring Expression Grounding: Benchmark, Diagnostics, and Reproducibility},
  author = {...},
  year = {2026},
  note = {Diagnostic paper + open-source release}
}
```

### Data Licenses

- Nr3D: Original license applies
- ScanNet: Original license applies
- Generated splits: MIT License

---

## Final Statement

**This repository is released as a diagnostic paper companion, not a strong method contribution.**

The核心价值 is:
- Trustworthy evaluation foundation
- Diagnostic findings with evidence
- Reproducible baseline
- Negative results documented

Users should know what they're getting:
- Clean baseline reproduction ✓
- Modest method gain (+0.22%) ✓
- Clear failure analysis ✓
- Not SOTA, not method冲刺 ✓
