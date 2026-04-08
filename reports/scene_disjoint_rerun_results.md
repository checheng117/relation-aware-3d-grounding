# Scene-Disjoint Rerun Results

**Date**: 2026-04-07
**Phase**: Trustworthy Baseline Rerun - Step 6 (Final Report)

---

## Executive Summary

**Decision**: Option B - Clean rerun shows current setup is too weak; prioritize dataset scale / full Nr3D recovery next.

Trustworthy baseline established at Test Acc@1 = 2.70%. Gap to 35.6% target is 32.9 percentage points. Dataset scale is the dominant bottleneck.

---

## 1. Files and Configs Used

### Dataset Config

| File | Purpose |
|------|---------|
| `configs/dataset/referit3d_scene_disjoint.yaml` | Scene-disjoint split configuration |
| `data/processed/scene_disjoint/train_manifest.jsonl` | Train manifest (1,211 samples) |
| `data/processed/scene_disjoint/val_manifest.jsonl` | Val manifest (148 samples) |
| `data/processed/scene_disjoint/test_manifest.jsonl` | Test manifest (185 samples) |

### Training Configs

| File | Model |
|------|-------|
| `configs/train/scene_disjoint_baseline.yaml` | rag3d AttributeOnlyModel |
| `repro/referit3d_baseline/configs/scene_disjoint_baseline.yaml` | repro ReferIt3DNet |

### Evaluation Config

| File | Purpose |
|------|---------|
| `configs/eval/scene_disjoint_eval.yaml` | Evaluation configuration |

---

## 2. Exact Commands Run

### Data Preparation (Already Done)

```bash
# Build scene-disjoint splits (previous phase)
python scripts/prepare_data.py --mode build-nr3d-geom-scene-disjoint

# Validate split integrity
python scripts/validate_scene_disjoint_splits.py
```

### Training

```bash
# rag3d baseline
python scripts/train_baseline.py --config configs/train/scene_disjoint_baseline.yaml

# repro baseline
python repro/referit3d_baseline/scripts/train.py \
    --config repro/referit3d_baseline/configs/scene_disjoint_baseline.yaml \
    --device cuda
```

### Evaluation

```bash
# rag3d evaluation (via Python script)
python -c "..." # See inline evaluation in logs

# repro evaluation
python repro/referit3d_baseline/scripts/evaluate.py \
    --checkpoint outputs/scene_disjoint_rerun/repro_baseline/best_model.pt \
    --split test --device cuda
```

---

## 3. Rerun Metrics

### rag3d AttributeOnlyModel

| Metric | Value |
|--------|-------|
| Val Acc@1 | ~4.7% (epoch 3) |
| Test Acc@1 | 0.54% (1/185) |
| Test Acc@5 | 20.54% (38/185) |
| Training epochs | 6 |
| Training loss (final) | 3.24 |

### repro ReferIt3DNet (SimplePointEncoder)

| Metric | Value |
|--------|-------|
| Val Acc@1 | 11.49% (best, epoch 8) |
| Test Acc@1 | 2.70% (5/185) |
| Test Acc@5 | 10.81% (20/185) |
| Training epochs | 30 |
| Training loss (final) | 2.16 |

---

## 4. Selected Trustworthy Baseline

**Model**: repro ReferIt3DNet (SimplePointEncoder)

| Metric | Value |
|--------|-------|
| Test Acc@1 | **2.70%** |
| Test Acc@5 | 10.81% |
| Val Acc@1 | 11.49% |
| Val-Test Gap | 8.79% |

**Reason**: Higher Test Acc@1 than rag3d baseline (2.70% vs 0.54%).

---

## 5. Remaining Gap to 35.6%

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Test Acc@1 | 2.70% | 35.6% | **-32.9%** |
| Test Acc@5 | 10.81% | ~70% (estimated) | ~-59% |

**Progress**: 7.6% of target achieved (2.70 / 35.6)

---

## 6. Strongest Current Conclusion

The trustworthy baseline on scene-disjoint split achieves **Test Acc@1 = 2.70%**, far below the 35.6% target. The dominant bottleneck is **dataset scale** (1,544 vs 41,503 samples). Secondary bottlenecks are novel scene generalization and feature fidelity. Previous results showing ~10% test accuracy were inflated by scene overlap between val and test.

---

## 7. Next-Step Recommendation

**Primary**: Recover full Nr3D dataset (41,503 samples) to enable meaningful baseline reproduction.

**Secondary**: Improve feature fidelity (reduce 97% zero features) and test PointNet++ on scene-disjoint split.

**Deprioritized**: Protocol alignment, MVT attention, structured parser methods are premature until baseline is stronger.

---

## 8. CURRENT_STATUS.md Update

```markdown
# Current Status

Completed phases:
- Geometry recovery
- Feature fidelity integration
- Encoder upgrade
- Training protocol fidelity
- Distribution mismatch investigation
- Scene-disjoint split recovery
- Trustworthy baseline rerun (NEW)

Trustworthy baseline established:
- repro ReferIt3DNet on scene-disjoint split
- Test Acc@1: 2.70%
- Test Acc@5: 10.81%
- Val Acc@1: 11.49%

Old results (INVALIDATED):
- Baseline: Val 22.73%, Test 9.68%
- PointNet++: Val 21.43%, Test 10.97%
- Protocol Aligned: Val 27.27%, Test 3.87%

Current bottleneck:
- Dataset scale (1,544 vs 41,503 samples)
- Novel scene generalization (8.79% val-test gap)
- Feature fidelity (97% zero features)

Next step: Dataset recovery / full Nr3D expansion
```

---

## 9. Decision

**Option B Selected**: Clean rerun shows current setup is too weak; prioritize dataset scale / full Nr3D recovery next.

### Rationale

1. Test Acc@1 = 2.70% is far below 35.6% target
2. Dataset has only 3.7% of official samples
3. Previous results were inflated by scene overlap
4. Model improvements are premature without adequate data
5. Feature fidelity issues compound data scarcity

### Next Actions

1. Investigate recovering full Nr3D dataset from Hugging Face
2. If not possible, focus on feature quality improvement
3. Only after data issues resolved, revisit PointNet++ and protocol alignment