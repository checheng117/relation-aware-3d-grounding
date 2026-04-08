# Recovered 23K Rerun Protocol

**Date**: 2026-04-08
**Phase**: Trustworthy Baseline Rerun on Recovered Dataset - Step 1

---

## Executive Summary

This protocol defines exact configs, dataset paths, and evaluation parameters for the controlled rerun on the recovered 23,186-sample scene-disjoint dataset.

---

## 1. Exact Configs to Rerun

### Config A: SimplePointEncoder Baseline

| File | Path |
|------|------|
| Original | `repro/referit3d_baseline/configs/scene_disjoint_baseline.yaml` |
| Adapted | `repro/referit3d_baseline/configs/recovered_23k_simple.yaml` |

**Key parameters**:
```yaml
encoder_type: simple_point
point_input_dim: 256
epochs: 30
batch_size: 16
lr: 0.0001
```

### Config B: PointNet++ Encoder

| File | Path |
|------|------|
| Original | `repro/referit3d_baseline/configs/pointnetpp_encoder.yaml` |
| Adapted | `repro/referit3d_baseline/configs/recovered_23k_pointnetpp.yaml` |

**Key parameters**:
```yaml
encoder_type: pointnetpp
pointnetpp_num_points: 1024
epochs: 30
batch_size: 8
lr: 0.0001
```

---

## 2. Fixed Dataset Configuration

| Parameter | Value |
|-----------|-------|
| Manifest directory | `data/processed/scene_disjoint/official_scene_disjoint/` |
| Train manifest | `train_manifest.jsonl` |
| Val manifest | `val_manifest.jsonl` |
| Test manifest | `test_manifest.jsonl` |
| Raw data root | `data/raw/referit3d` |

**Expected counts**:
- Train: 18,459 samples, 215 scenes
- Val: 2,046 samples, 26 scenes
- Test: 2,681 samples, 28 scenes

---

## 3. Fixed Scene-Disjoint Split

| Split Pair | Expected Overlap | Verification |
|------------|------------------|--------------|
| Val-Test | 0 scenes | Required before run |
| Train-Val | 0 scenes | Required before run |
| Train-Test | 0 scenes | Required before run |

**Verification command**:
```bash
python scripts/validate_scene_disjoint_splits.py \
    --manifest-dir data/processed/scene_disjoint/official_scene_disjoint
```

---

## 4. Fixed Evaluator

| Parameter | Value |
|-----------|-------|
| Evaluator script | `repro/referit3d_baseline/scripts/evaluate.py` |
| Evaluation batch size | 32 |
| Metrics computed | Acc@1, Acc@5 |

---

## 5. Fixed Metrics

### Primary Metrics (for baseline selection)

| Metric | Description | Priority |
|--------|-------------|----------|
| Val Acc@1 | Top-1 accuracy on validation | Monitor during training |
| Test Acc@1 | Top-1 accuracy on test | **PRIMARY** |
| Test Acc@5 | Top-5 accuracy on test | **PRIMARY** |

### Secondary Metrics

| Metric | Description |
|--------|-------------|
| Training stability | No NaN/Inf, loss convergence |
| Runtime | Training time per epoch |
| Memory usage | GPU memory consumption |
| Config snapshot | Save config with results |

---

## 6. Success Criteria for Baseline Selection

| Criterion | Threshold | Weight |
|-----------|-----------|--------|
| Test Acc@1 | Highest among configs | **Primary** |
| Test Acc@5 | Highest among configs | Secondary |
| Val-test gap | < 10% | Tiebreaker |
| Training stability | No crashes | Required |

**Selection rule**: Config with highest Test Acc@1 is the best trustworthy baseline.

---

## 7. Output Directory Structure

```
outputs/20260408_recovered_23k_rerun/
├── simple_encoder/
│   ├── best_model.pt
│   ├── config.yaml
│   ├── logs/
│   │   └── training.log
│   └── metrics/
│   │   ├── val_metrics.json
│   │   └── test_metrics.json
├── pointnetpp_encoder/
│   ├── best_model.pt
│   ├── config.yaml
│   ├── logs/
│   │   └── training.log
│   └── metrics/
│   │   ├── val_metrics.json
│   │   └── test_metrics.json
├── comparison/
│   ├── recovered_23k_rerun_comparison.json
│   └── recovered_23k_rerun_comparison.md
```

---

## 8. Exact Commands

### Stage A: Smoke Rerun

```bash
# Verify scene overlap
python scripts/validate_scene_disjoint_splits.py \
    --manifest-dir data/processed/scene_disjoint/official_scene_disjoint

# Quick smoke test (2 epochs)
python repro/referit3d_baseline/scripts/train.py \
    --config repro/referit3d_baseline/configs/recovered_23k_simple.yaml \
    --epochs 2 \
    --device cuda
```

### Stage B: Controlled Verification Rerun

```bash
# SimplePointEncoder (10 epochs)
python repro/referit3d_baseline/scripts/train.py \
    --config repro/referit3d_baseline/configs/recovered_23k_simple.yaml \
    --epochs 10 \
    --device cuda

# PointNet++ (10 epochs)
python repro/referit3d_baseline/scripts/train.py \
    --config repro/referit3d_baseline/configs/recovered_23k_pointnetpp.yaml \
    --epochs 10 \
    --device cuda
```

### Stage C: Formal Rerun

```bash
# SimplePointEncoder (30 epochs)
python repro/referit3d_baseline/scripts/train.py \
    --config repro/referit3d_baseline/configs/recovered_23k_simple.yaml \
    --device cuda

# PointNet++ (30 epochs)
python repro/referit3d_baseline/scripts/train.py \
    --config repro/referit3d_baseline/configs/recovered_23k_pointnetpp.yaml \
    --device cuda

# Evaluate both on test
python repro/referit3d_baseline/scripts/evaluate.py \
    --checkpoint outputs/20260408_recovered_23k_rerun/simple_encoder/best_model.pt \
    --split test \
    --device cuda

python repro/referit3d_baseline/scripts/evaluate.py \
    --checkpoint outputs/20260408_recovered_23k_rerun/pointnetpp_encoder/best_model.pt \
    --split test \
    --device cuda
```

---

## 9. Config Snapshots

Each run must save:
1. Full config YAML (copy to output dir)
2. Dataset manifest paths
3. Model architecture parameters
4. Training hyperparameters
5. Seed value

---

## 10. Dataset Summary

| Attribute | Value |
|-----------|-------|
| Source | Official Nr3D CSV |
| Total samples | 23,186 |
| Coverage | 55.9% of 41,503 |
| Scenes | 269 |
| Split method | Scene-disjoint |
| Geometry | ScanNet aggregation (placeholder) |
| Features | Hash-based (synthetic) |

---

## 11. Comparison Format

### JSON Format

```json
{
  "date": "2026-04-08",
  "dataset": {
    "source": "official_nr3d",
    "total_samples": 23186,
    "train_samples": 18459,
    "val_samples": 2046,
    "test_samples": 2681
  },
  "results": [
    {
      "config": "recovered_23k_simple",
      "encoder": "simple_point",
      "val_acc1": "<value>",
      "test_acc1": "<value>",
      "test_acc5": "<value>",
      "epochs": 30,
      "training_time": "<value>"
    },
    {
      "config": "recovered_23k_pointnetpp",
      "encoder": "pointnetpp",
      "val_acc1": "<value>",
      "test_acc1": "<value>",
      "test_acc5": "<value>",
      "epochs": 30,
      "training_time": "<value>"
    }
  ],
  "best_baseline": "<config_name>",
  "gap_to_target": "<value>"
}
```

---

## 12. Execution Checklist

| Step | Action | Status |
|------|--------|--------|
| 1 | Create adapted configs | Pending |
| 2 | Verify scene overlap | Pending |
| 3 | Smoke rerun (2 epochs) | Pending |
| 4 | Verification rerun (10 epochs) | Pending |
| 5 | Formal rerun (30 epochs) | Pending |
| 6 | Export comparison | Pending |
| 7 | Write results report | Pending |

---

## Conclusion

This protocol establishes the controlled rerun framework. Next step is to create the adapted configs and begin Stage A smoke rerun.