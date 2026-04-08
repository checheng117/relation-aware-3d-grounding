# Distribution Mismatch Investigation Results

**Date**: 2026-04-06
**Phase**: Distribution Mismatch Investigation - Final Report

---

## Executive Summary

**Primary finding**: The val-test accuracy discrepancy is caused by three fundamental issues, ranked by priority:

1. **Small dataset size** (CRITICAL): 1,569 samples vs official 41,503 (~3.8%)
2. **Scene overlap** (CRITICAL): Splits are NOT scene-disjoint (53 overlapping val-test scenes)
3. **Protocol mismatch**: Hash-based encoder vs BERT, missing MVT

**Recommendation**: **Option C - Further data/split investigation before any new run**

No new training experiments should be run until scene-disjoint splits are implemented.

---

## 1. Files Inspected

### Configuration Files
- `.claude/PROJECT_CONTEXT.md`
- `.claude/CURRENT_STATUS.md`
- `.claude/WORKING_RULES.md`
- `.claude/NEXT_TASK.md`
- `configs/dataset/referit3d.yaml`
- `configs/train/nr3d_geom_first_baseline.yaml`
- `configs/train/baseline.yaml`
- `configs/eval/nr3d_geom_first.yaml`

### Data Files
- `data/processed/train_manifest.jsonl` (1235 samples)
- `data/processed/val_manifest.jsonl` (154 samples)
- `data/processed/test_manifest.jsonl` (155 samples)
- `data/processed/dataset_summary.json`
- `data/raw/referit3d/annotations/nr3d_annotations.json`

### Source Files
- `scripts/prepare_data.py`
- `src/rag3d/datasets/builder.py` (split_records function - random shuffle)
- `src/rag3d/datasets/nr3d_hf.py`
- `src/rag3d/evaluation/metrics.py`
- `src/rag3d/evaluation/evaluator.py`

### Documentation
- `repro/referit3d_baseline/README.md`

### Generated Reports
- `reports/distribution_mismatch_audit.md`
- `reports/split_distribution_comparison.md`
- `reports/ranking_gap_analysis.md`
- `reports/generalization_gap_analysis.md`
- `split_distribution_comparison.json`

---

## 2. Key Statistics

### Split Distribution
| Split | Samples | Unique Scenes | Samples/Scene |
|-------|---------|---------------|---------------|
| Train | 1,235 | 243 | 5.08 |
| Val | 154 | 100 | 1.54 |
| Test | 155 | 110 | 1.41 |

### Scene Overlap
| Overlap | Scenes | Samples in Overlap |
|---------|--------|-------------------|
| Train-Val | 89 | - |
| Train-Test | 97 | - |
| **Val-Test** | **53** | **89 val (57.8%), 82 test (52.9%)** |

### Utterance Statistics
| Split | Mean Length | Std |
|-------|-------------|-----|
| Val | 11.47 words | 5.29 |
| Test | 10.39 words | 4.54 |

### Candidate Statistics
| Split | Mean Size | Std |
|-------|-----------|-----|
| Val | 38.55 | 13.53 |
| Test | 36.53 | 15.42 |

### Hard Tags Distribution
| Tag | Train | Val | Test |
|-----|-------|-----|------|
| anchor_confusion | 64 (5.2%) | 11 (7.1%) | 5 (3.2%) |

### Current Results
| Configuration | Val Acc@1 | Test Acc@1 | Test Acc@5 | Val-Test Gap |
|--------------|-----------|------------|------------|--------------|
| Baseline | 22.73% | 9.68% | 40.00% | 13.05% |
| PointNet++ | 21.43% | 10.97% | 60.00% | 10.46% |
| Protocol alignment | 27.27% | 3.87% | 30.32% | 23.40% |

---

## 3. Strongest Explanation for Val-Test Mismatch

### Root Cause: Scene Overlap + Protocol Overfitting

**Mechanism**:
1. Random shuffle split creates scene overlap between val and test
2. Protocol alignment tuned on val exploits scene-specific patterns
3. Val samples in overlapping scenes match learned patterns → improved accuracy
4. Test samples in overlapping scenes query different targets → patterns don't transfer
5. Test samples in non-overlapping scenes lack any learned patterns → poor accuracy

**Evidence**:
- Protocol alignment: Val improves (22.73% → 27.27%), test collapses (9.68% → 3.87%)
- Scene overlap: 57.8% of val and 52.9% of test in same 53 scenes
- PointNet++ (better encoder, no protocol tuning): Smaller val-test gap (10.46%)

**Secondary factors**:
- Small dataset (3.8% of official) limits generalization capacity
- Utterance length difference (val longer than test by 9.4%)
- Anchor confusion difference (val 7.1%, test 3.2%)

---

## 4. Ranking Gap Analysis

### Top-1 vs Top-5 Gap
| Configuration | Gap (Acc@5 - Acc@1) | Interpretation |
|--------------|---------------------|----------------|
| Baseline | 30.32% | Poor retrieval + discrimination |
| PointNet++ | 49.03% | Good retrieval, poor discrimination |
| Protocol alignment | 26.45% | Reduced retrieval overall |

**Insight**: PointNet++ finds correct answers (60% in top-5) but cannot rank them correctly (only 10.97% top-1). This suggests:
- Encoder quality improves feature matching
- Relation encoding still weak for discrimination
- Same-class clutter (99.4%) requires relation-aware signals

---

## 5. Single Highest-Priority Next Experiment

**Recommendation: Option C - Further split/data investigation before any new run**

### Rationale
1. Scene overlap invalidates current evaluation metrics
2. Small dataset limits credible reproduction
3. Any new training on current splits will produce unreliable results
4. Protocol tuning experiments are meaningless on overlapping splits

### Required Actions Before New Experiments
1. **Implement scene-disjoint splits**:
   - Create splits where no scene appears in multiple splits
   - Preserve approximate sample ratios (80/10/10)
   
2. **Verify split methodology**:
   - Check if official Nr3D has predefined splits
   - If available, use official splits instead
   
3. **Re-evaluate existing checkpoints**:
   - Run evaluation on scene-disjoint test set
   - Compare results to current overlapping test

4. **Document subset limitation**:
   - Acknowledge 1,569 vs 41,503 in all future reports
   - Target for subset reproduction should be adjusted (not 35.6%)

### After Scene-Disjoint Splits
Then proceed to **Option A: PointNet++ + mild protocol alignment**:
- Use PointNet++ encoder (best generalization so far)
- Apply mild protocol changes only (no aggressive tuning)
- Evaluate on scene-disjoint splits

---

## 6. Update for CURRENT_STATUS.md

```markdown
# Current Status

Completed phases:
- Geometry recovery
- Feature fidelity integration
- Encoder upgrade
- Training protocol fidelity
- Distribution mismatch investigation (NEW)

Current best measured results (on overlapping splits - INVALIDATED):
- Baseline after feature fidelity:
  - Val Acc@1 = 22.73%
  - Test Acc@1 = 9.68%
  - Test Acc@5 = 40.00%

- PointNet++:
  - Val Acc@1 = 21.43%
  - Test Acc@1 = 10.97%
  - Test Acc@5 = 60.00%

- Full protocol alignment:
  - Val Acc@1 = 27.27%
  - Test Acc@1 = 3.87%
  - Test Acc@5 = 30.32%

Critical findings (NEW):
1. Dataset size: 1,569 samples vs official 41,503 (~3.8%)
2. Scene overlap: Val-Test share 53 scenes (57.8% of val, 52.9% of test)
3. Splits NOT scene-disjoint - evaluation metrics potentially invalid
4. Protocol alignment overfits to overlapping val scenes

Interpretation:
- Current evaluation results are unreliable due to scene overlap
- Small dataset fundamentally limits credible reproduction
- PointNet++ shows best generalization tendency (smallest val-test gap)
- Must implement scene-disjoint splits before further experiments

Next step: Implement scene-disjoint splits, then re-evaluate
```

---

## 7. Summary of Investigation

| Step | Key Finding | Evidence |
|------|-------------|----------|
| Step 1: Split audit | Scene overlap (53 scenes) | Scene lists analysis |
| Step 2: Ranking gap | PointNet++: 49% top-1/top-5 gap | Accuracy comparison |
| Step 3: Generalization | Small dataset (3.8%) + val overfitting | Dataset summary, config comparison |

---

## 8. Investigation Constraints Met

- Did not read `outputs/`, `logs/`, `artifacts/`, `checkpoints/`
- Did not expand full repository tree
- Kept changes isolated to reproduction track
- No new methods introduced (MVT, structured parsers)
- No hyperparameter sweeps
- Every conclusion backed by data analysis

---

## 9. Files Generated

- `reports/distribution_mismatch_audit.md`
- `reports/split_distribution_comparison.md`
- `reports/ranking_gap_analysis.md`
- `reports/generalization_gap_analysis.md`
- `reports/distribution_mismatch_results.md` (this file)
- `split_distribution_comparison.json`
- `scripts/analyze_split_distribution.py`