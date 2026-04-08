# Scene-Disjoint Rerun Protocol

**Date**: 2026-04-06
**Phase**: Trustworthy Baseline Rerun - Step 1

---

## Executive Summary

**Configs to rerun**: 1 mandatory + 1 optional
**Output directory**: `outputs/<timestamp>_scene_disjoint_rerun/`
**Success criteria**: Establish trustworthy baseline anchor on zero-overlap split

---

## 1. Exact Configs to Rerun

### Mandatory Config

| Config | Model | Encoder | Split |
|--------|-------|---------|-------|
| `configs/train/scene_disjoint_baseline.yaml` | attribute_only | ObjectMLPEncoder | scene_disjoint |

**Command**:
```bash
python scripts/train.py --config configs/train/scene_disjoint_baseline.yaml
```

### Optional Config (If Compute Permits)

| Config | Model | Encoder | Split |
|--------|-------|---------|-------|
| TBD (PointNet++ scene_disjoint) | referit3dnet | PointNet++ | scene_disjoint |

**Decision**: Will determine after mandatory baseline completes. PointNet++ previous phase showed best test generalization; rerun on clean split may confirm or refute this.

---

## 2. Fixed Split

### Dataset Config

**File**: `configs/dataset/referit3d_scene_disjoint.yaml`

### Split Statistics (Verified)

| Split | Scenes | Samples | Overlap with Val | Overlap with Test |
|-------|--------|---------|------------------|-------------------|
| Train | 212 | 1,211 | 0 scenes | 0 scenes |
| Val | 26 | 148 | - | **0 scenes** |
| Test | 28 | 185 | **0 scenes** | - |

**Manifests**:
- `data/processed/scene_disjoint/train_manifest.jsonl`
- `data/processed/scene_disjoint/val_manifest.jsonl`
- `data/processed/scene_disjoint/test_manifest.jsonl`

---

## 3. Fixed Evaluator

### Evaluation Config

**File**: `configs/eval/scene_disjoint_eval.yaml`

### Evaluation Settings

| Parameter | Value |
|-----------|-------|
| Split | test |
| Batch size | 24 |
| Margin threshold | 0.15 |
| Models | attribute_only |

### Metrics Export

- Val Acc@1 (from training checkpoints)
- Test Acc@1 (from evaluation)
- Test Acc@5 (from evaluation)
- Per-epoch training metrics

---

## 4. Fixed Comparison Metrics

### Primary Metrics

| Metric | Definition |
|--------|------------|
| Val Acc@1 | Accuracy on validation set (top-1) |
| Test Acc@1 | Accuracy on test set (top-1) |
| Test Acc@5 | Accuracy on test set (top-5) |

### Secondary Metrics (If Available)

| Metric | Definition |
|--------|------------|
| Training loss trajectory | Per-epoch loss |
| Best epoch | Epoch with highest Val Acc@1 |
| Convergence stability | Loss variance in final epochs |

### Comparison Against

| Reference | Source | Status |
|-----------|--------|--------|
| Old baseline Val Acc@1 | 22.73% | INVALIDATED |
| Old baseline Test Acc@1 | 9.68% | INVALIDATED |
| Old PointNet++ Test Acc@1 | 10.97% | INVALIDATED |
| Target baseline | 35.6% | Official target |

**Note**: Old results are for reference only. They cannot be used for conclusions.

---

## 5. Success Criteria

### Criterion 1: Pipeline Execution

| Check | Expected |
|-------|----------|
| Dataset loads | 1,211 train, 148 val, 185 test |
| Training runs | No errors, stable loss |
| Evaluation runs | Metrics exported |
| Output files created | Checkpoint + metrics file |

**Status**: PASS if all checks succeed.

### Criterion 2: Trustworthy Anchor Established

| Metric | Minimum Threshold | Target |
|--------|-------------------|--------|
| Test Acc@1 | > 0% (valid evaluation) | 35.6% |
| Val-Test relationship | Val ≥ Test (expected) | - |

**Status**: PASS if Test Acc@1 > 0 and metrics are exported.

### Criterion 3: Select Best Baseline

| Criterion | Definition |
|-----------|------------|
| Best baseline | Config with highest Test Acc@1 on clean split |
| Anchor selection | Use best baseline for future comparisons |

**Note**: Test Acc@1 is primary because it reflects generalization on unseen scenes.

---

## 6. Output Directory Structure

### Proposed Structure

```
outputs/<timestamp>_scene_disjoint_rerun/
├── checkpoints/
│   └── scene_disjoint_baseline/
│       ├── best_model.pt
│       └── epoch_*.pt
├── metrics/
│   ├── train_scene_disjoint_baseline_metrics.jsonl
│   └── eval_scene_disjoint_baseline_metrics.json
├── logs/
│   └── training.log
└── config_snapshot/
    └── scene_disjoint_baseline.yaml
```

**Timestamp format**: `YYYYMMDD_HHMMSS`

---

## 7. Staged Execution Plan

### Stage A: Smoke Rerun

**Purpose**: Verify pipeline works with scene_disjoint split.

**Steps**:
1. Run training for 1-2 epochs
2. Check train/val counts
3. Verify no NaN/Inf
4. Check metrics export

**Success**: Pipeline runs without errors.

### Stage B: Controlled Verification Rerun

**Purpose**: Full training run with monitoring.

**Steps**:
1. Run full training (6 epochs per config)
2. Monitor convergence
3. Run evaluation on test
4. Export metrics

**Success**: Metrics comparable to or better than old baseline.

### Stage C: Formal Rerun (Optional Extension)

**Purpose**: Extended training if needed for comparison.

**Steps**:
1. Extend training if underfitting
2. Export final comparison bundle

**Success**: Best baseline selected.

---

## 8. Commands to Run

### Stage A: Smoke Test

```bash
# Quick smoke test (1 epoch)
python scripts/train.py --config configs/train/scene_disjoint_baseline.yaml --epochs 1
```

### Stage B: Controlled Rerun

```bash
# Full training
python scripts/train.py --config configs/train/scene_disjoint_baseline.yaml

# Evaluation
python scripts/evaluate.py --config configs/eval/scene_disjoint_eval.yaml
```

---

## 9. Comparison Export Format

### JSON Export

**File**: `scene_disjoint_rerun_comparison.json`

```json
{
  "timestamp": "2026-04-06T...",
  "split": "scene_disjoint",
  "configs": [
    {
      "name": "scene_disjoint_baseline",
      "config_file": "configs/train/scene_disjoint_baseline.yaml",
      "model": "attribute_only",
      "val_acc1": ...,
      "test_acc1": ...,
      "test_acc5": ...,
      "best_epoch": ...,
      "training_stability": "stable|unstable"
    }
  ],
  "best_baseline": {
    "name": "...",
    "test_acc1": ...
  },
  "old_results_invalidated": {
    "baseline": {"val_acc1": 22.73, "test_acc1": 9.68},
    "pointnetpp": {"val_acc1": 21.43, "test_acc1": 10.97},
    "protocol_aligned": {"val_acc1": 27.27, "test_acc1": 3.87}
  }
}
```

### Markdown Export

**File**: `scene_disjoint_rerun_comparison.md`

- Summary table of new results
- Comparison to old (invalidated) results
- Selected trustworthy baseline
- Gap to target 35.6%

---

## 10. What to Document After Rerun

### Required Documentation

1. `scene_disjoint_rerun_comparison.json` - Metrics export
2. `scene_disjoint_rerun_comparison.md` - Summary report
3. `reports/overlap_vs_scene_disjoint_results.md` - Old vs new comparison
4. `reports/post_scene_disjoint_bottleneck_reassessment.md` - Bottleneck analysis
5. `reports/scene_disjoint_rerun_results.md` - Final report

---

## 11. Decision Framework

After rerun, choose one:

| Option | Condition | Action |
|--------|-----------|--------|
| A | Baseline trustworthy, reasonable gap | Continue gap reduction from anchor |
| B | Results too weak (e.g., Test Acc@1 < 5%) | Prioritize dataset scale / Nr3D recovery |
| C | Encoder/protocol conclusions changed materially | One more controlled refinement |

---

## 12. Limitations Acknowledged

| Limitation | Impact |
|------------|--------|
| Dataset size (1,544 vs 41,503) | May limit achievable accuracy |
| Hash-based text encoder | Not BERT |
| Simple MLP object encoder | Not PointNet++ |
| Missing multi-view features | May affect results |

**Interpretation**: Results should be evaluated relative to these constraints, not absolute target of 35.6%.

---

## Conclusion

Protocol defined. Proceed to Stage A smoke test to verify pipeline execution with scene-disjoint split.