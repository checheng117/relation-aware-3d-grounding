# Distribution Mismatch Investigation Audit

**Date**: 2026-04-06
**Phase**: Distribution Mismatch Investigation
**Goal**: Identify why protocol alignment improves validation but degrades test

---

## 1. Current Best Results Summary

| Configuration | Val Acc@1 | Test Acc@1 | Test Acc@5 | Notes |
|--------------|-----------|------------|------------|-------|
| Baseline (feature fidelity) | 22.73% | 9.68% | 40.00% | Starting point |
| PointNet++ | 21.43% | 10.97% | 60.00% | Best test Acc@1, strong Acc@5 |
| Full protocol alignment | 27.27% | 3.87% | 30.32% | Catastrophic test drop |

**Key observation**: Val-test gap ranges from ~13% (baseline) to ~23% (protocol alignment).

---

## 2. Suspected Reasons for Val-Test Discrepancy

### Hypothesis 1: Split Distribution Mismatch
- Train/val/test may have different:
  - Scene distributions (indoor vs outdoor, room types)
  - Object class frequencies
  - Relation-type distributions (spatial, color, size)
  - Utterance complexity
  - Candidate set sizes (distractor counts)

### Hypothesis 2: Sample Difficulty Mismatch
- Val set may be easier (shorter utterances, clearer targets)
- Test set may have more:
  - Hard distractors (same-class objects)
  - Complex relations
  - Ambiguous descriptions

### Hypothesis 3: Ranking/Fusion Instability
- Protocol alignment may overfit to val's score distribution
- Top-5 improvement without top-1 suggests:
  - Correct answer often in top-5 but not top-1
  - Score margin between correct and near-correct may be small
  - Ranking may be unstable

### Hypothesis 4: Overfitting to Validation via Protocol Tuning
- Protocol alignment was tuned against val
- Learning rate, batch size, scheduler settings may exploit val quirks
- Test set has different optimization landscape

---

## 3. Relevant Files and Configs

### Data Processing
- `scripts/prepare_data.py` - Data preparation pipeline
- `src/rag3d/datasets/builder.py` - Dataset construction
- `configs/` - Configuration files

### Training
- `src/rag3d/models/` - Model implementations
- `src/rag3d/training/` - Training loops

### Data Splits (to inspect)
- Nr3D split files (train/val/test)
- Scene metadata
- Object annotations

### Reports (previous findings)
- Look for any previous split analysis

---

## 4. Minimal Controlled Investigation Plan

### Step 1: Split Distribution Audit
**Goal**: Quantify differences between train/val/test

Metrics to compute:
- Sample counts per split
- Scene counts per split
- Utterance length statistics (mean, std, distribution)
- Target class distribution (histogram)
- Relation keyword frequency ("left", "right", "front", "back", etc.)
- Same-class clutter rate (how often distractors are same class as target)
- Candidate set size distribution

**Output**: `reports/split_distribution_comparison.md`, `split_distribution_comparison.json`

### Step 2: Ranking Gap Analysis
**Goal**: Understand top-1 vs top-5 discrepancy

Metrics to compute:
- Top-1 vs top-5 gap per configuration
- Score margin distributions (correct vs top-predicted)
- When wrong, is correct answer in top-5? top-10?
- Confidence calibration (score vs accuracy)

**Output**: `reports/ranking_gap_analysis.md`

### Step 3: Generalization Gap Analysis
**Goal**: Compare configurations for overfitting patterns

Metrics to compute:
- Train/val/test divergence per configuration
- Loss curves if available
- Effect of batch size / LR / scheduler on generalization

**Output**: `reports/generalization_gap_analysis.md`

### Step 4: Single Next Experiment Recommendation
Based on findings, recommend ONE of:
- A: PointNet++ + mild protocol alignment
- B: PointNet++ + ranking/fusion refinement
- C: Further split/data fidelity investigation

**Output**: Updated recommendation in final report

### Step 5: Final Report
Consolidate all findings and update `.claude/CURRENT_STATUS.md`

**Output**: `reports/distribution_mismatch_results.md`

---

## 5. Constraints

- Do NOT read `outputs/`, `logs/`, `artifacts/`, `checkpoints/` unless explicitly required
- Do NOT expand full repository tree
- Do NOT introduce new methods (MVT, structured parsers, etc.)
- Do NOT do broad hyperparameter sweeps
- Every conclusion must be backed by data, not speculation

---

## 6. Expected Outcome

Identify the primary driver of val-test discrepancy and determine the single most impactful next controlled experiment.