# Relation-Aware 3D Grounding: Benchmark and Diagnostics

A reproducible research codebase for 3D referring-expression grounding on the Nr3D/ReferIt3D benchmark. This project establishes a **trustworthy evaluation foundation** through dataset recovery, scene-disjoint splitting, and baseline reproduction, then provides **diagnostic analysis** of failure modes and a simple dense reranker with modest gains.

**Project Status**: Diagnostic paper + open-source release. Method exploration is complete.

---

## Why This Project Matters

Most public 3D grounding repositories suffer from reproducibility problems:

- **Partial datasets**: Released code assumes proprietary or incomplete data
- **Split leakage**: Train/val/test scenes overlap, inflating metrics
- **Unclear protocols**: Metric definitions vary across papers
- **Weak traceability**: No run artifacts, no ablations, no failure analysis

This makes apples-to-apples comparison impossible and published numbers unreliable. This project fixes that by building a fully transparent evaluation pipeline and honestly reporting what works and what doesn't.

---

## Quick Results Summary

| Method | Test Acc@1 | Test Acc@5 | Net | Status |
|--------|------------|------------|-----|--------|
| **Base (clean)** | **30.83%** | **91.87%** | - | **Reference** |
| **Dense-no-cal-v1** | **31.05%** | **92.01%** | +9 | **Retained** |

**Other methods explored** (calibration, dense strengthening) **did not improve results** and are frozen. See [reports/final_diagnostic_master_summary.md](reports/final_diagnostic_master_summary.md) for complete analysis.

---

## Core Contributions

### 1. Trustworthy Evaluation Foundation

- **Recovered full Nr3D dataset**: 41,503 samples across 641 ScanNet scenes
- **Scene-disjoint splits**: Train/val/test share zero scenes (verified)
- **Zero duplicates**: All samples validated as unique
- **Unified metrics**: Acc@1 and Acc@5 computed consistently

### 2. Reproduced Baselines

| Baseline | Test Acc@1 | Test Acc@5 | Notes |
|----------|------------|------------|-------|
| ReferIt3DNet (reproduced) | 30.79% | 91.75% | Primary trusted baseline |
| SAT (reproduced) | 28.27% | 87.64% | Secondary baseline |

Both baselines reproduced from original papers with scene-disjoint evaluation.

### 3. Diagnostic Framework

- **Hard-subset tagging**: same-class clutter, multi-anchor, relative-position
- **Coverage analysis**: anchor reachability, coverage@k, long-range evidence
- **Harm/recovered taxonomy**: identifies where methods help vs hurt
- **Case study export**: qualitative analysis with relation score diagnostics

### 4. Limited Method Signal

**Dense-no-cal-v1** - A simple dense relation reranker:

| Metric | Value | Delta vs Base |
|--------|-------|---------------|
| Acc@1 | 31.05% | +0.22% |
| Acc@5 | 92.01% | +0.14% |
| Net | +9 | 402 recovered, 393 harmed |
| Multi-Anchor | 28.34% | +5.3% |
| Parameters | 398K | Lightweight |

**What it is**: Modest but real gain, simple weighted aggregation, no calibration overhead.

**What it is NOT**: SOTA-challenging, scalable foundation, or strong method contribution.

### 5. Negative Findings (With Evidence)

| Method | Acc@1 | Net | Why It Failed |
|--------|-------|-----|---------------|
| Dense-calibrated-v2 | 30.55% | -12 | Calibration signals uninformative |
| Dense-v2-AttPool | 25.24% | -238 | Attention too complex for weak foundation |
| Dense-v3-Geo | 24.32% | -277 | No geometry data available |
| Dense-v4-HardNeg | 24.47% | -271 | Focal weighting hurts initial training |

**Fundamentals debug revealed**:
- Relation scores are mostly noisy (score gap = -0.95, correct targets score LOWER)
- Pair ranking is weak (Hit@1 = 7%)
- Adding complexity amplifies noise, not signal

See [reports/dense_fundamentals_summary.md](reports/dense_fundamentals_summary.md) for detailed analysis.

---

## Who Should Use This Repository

### Good Fit For

- **Benchmark users** - Need trustworthy scene-disjoint evaluation
- **Diagnostics researchers** - Study failure modes, hard subsets, coverage
- **Reproducibility researchers** - Want clean baseline reproduction
- **Method researchers** - Need clean starting point for new ideas

### Not For

- **SOTA chasers** - Methods are modest (+0.22%), not state-of-the-art
- **Calibration developers** - Calibration line is frozen (signals uninformative)
- **Dense-scorer enhancers** - Strengthening line is frozen (foundation weak)
- **Multi-seed validators** - Single-seed results only

---

## Repository Structure

```
relation-aware-3d-grounding/
├── README.md                          # This file
├── .claude/                           # Project status documentation
│   ├── CURRENT_STATUS.md              # Current phase: diagnostic paper
│   ├── NEXT_TASK.md                   # Next priorities
│   └── METHOD_PHASE_FREEZE.md         # Frozen method boundaries
├── src/rag3d/                         # Core Python package
│   ├── models/
│   │   ├── cover3d_model.py           # COVER-3D wrapper (retained)
│   │   ├── cover3d_dense_relation.py  # DenseRelationModule (retained)
│   │   └── cover3d_calibration.py     # Calibration module (archived)
│   └── ...
├── scripts/
│   ├── train_cover3d_round1.py        # Training script (retained)
│   ├── analyze_dense_fundamentals.py  # Diagnostic analysis (retained)
│   └── ...                            # Other scripts (archived/debug)
├── configs/
│   ├── cover3d_round1/                # Round-1 configs (retained)
│   └── ...                            # Other configs (archived)
├── reports/
│   ├── final_diagnostic_master_summary.md    # Master summary
│   ├── dense_fundamentals_summary.md         # Fundamentals debug
│   └── ...                                   # Other reports
├── repro/
│   ├── referit3d_baseline/            # ReferIt3DNet reproduction
│   └── sat_baseline/                  # SAT reproduction
└── docs/                              # Additional documentation
```

---

## How To Reproduce

### Setup

```bash
conda env create -f environment.yml
conda activate rag3d
pip install -e ".[dev,viz]"

make check-env
make test
```

### Baseline Reproduction

ReferIt3DNet:

```bash
python repro/referit3d_baseline/scripts/train.py \
  --config repro/referit3d_baseline/configs/learned_class_embedding.yaml \
  --device cuda

python repro/referit3d_baseline/scripts/evaluate.py \
  --config repro/referit3d_baseline/configs/learned_class_embedding.yaml \
  --device cuda
```

### Run Dense-no-cal-v1

```bash
python scripts/train_cover3d_round1.py \
  --variant dense-no-cal \
  --epochs 10 \
  --device cuda
```

### Run Diagnostics

```bash
python scripts/analyze_dense_fundamentals.py
```

---

## Key Reports

| Report | Description |
|--------|-------------|
| [final_diagnostic_master_summary.md](reports/final_diagnostic_master_summary.md) | **Start here** - Complete results and analysis |
| [final_diagnostic_master_table.csv](reports/final_diagnostic_master_table.csv) | Machine-readable results table |
| [dense_fundamentals_summary.md](reports/dense_fundamentals_summary.md) | Why dense scorer methods failed |
| [calibration_failure_analysis.md](reports/calibration_failure_analysis.md) | Why calibration failed |
| [dense_strengthening_results.md](reports/dense_strengthening_results.md) | Strengthening variant failures |
| [diagnostic_paper_positioning_freeze.md](reports/diagnostic_paper_positioning_freeze.md) | Paper framing and target venues |

---

## Method Status Summary

| Method | Acc@1 | Status | Decision |
|--------|-------|--------|----------|
| Base (clean) | 30.83% | Retained | Reference anchor |
| Dense-no-cal-v1 | 31.05% | Retained | Lightweight method contribution |
| Dense-calibrated-v1 | 30.60% | Frozen | Gate collapse |
| Dense-calibrated-v2 | 30.55% | Frozen | Signals uninformative |
| Dense-v2-AttPool | 25.24% | Frozen | Too complex |
| Dense-v3-Geo | 24.32% | Frozen | No geometry data |
| Dense-v4-HardNeg | 24.47% | Frozen | Focal hurts |

See [reports/method_freeze_and_release_policy.md](reports/method_freeze_and_release_policy.md) for freeze rationale.

---

## Project Positioning

**Paper Type**: Diagnostic / Benchmark / Reproducibility + Limited Method Signal

**NOT**: Strong Method Paper

### Core Claim

> This project establishes a trustworthy, scene-disjoint evaluation and diagnostic framework for 3D referring-expression grounding, identifies concentrated failure modes in hard relational subsets, provides direct evidence of coverage failure under sparse candidate-anchor selection, and shows that a simple dense reranker yields only limited gains while more complex extensions do not justify further investment.

### Target Venues

- **Primary**: TACL, ACL Findings, EMNLP Findings, Scientific Data
- **Secondary**: CVPR/ICCV/ECCV Workshop, 3DV, BMVC
- **NOT**: AAAI/NeurIPS/ICML/CVPR/ICCV main track (method signal insufficient)

---

## Development Principles

- Preserve the trusted evaluation base before making method claims
- Do not report unconfirmed results as final improvements
- Prefer stratified and diagnostic evaluation over single-number accuracy
- Treat parser outputs as noisy weak signals, not oracle structure
- Keep generated data, checkpoints, and feature caches out of git
- Any paper claim requires run artifacts, ablations, and failure analysis
- **Honest framing over overselling** - Reviewers forgive modest claims, not inflated ones

---

## License

MIT License. See [LICENSE](LICENSE).
