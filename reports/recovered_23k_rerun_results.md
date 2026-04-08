# Recovered 23K Rerun Results

**Date**: 2026-04-08
**Phase**: Trustworthy Baseline Rerun on Recovered Dataset - Step 6 (Final Report)

---

## Executive Summary

**Decision**: Option A - Trustworthy baseline anchor is now substantially improved; continue controlled gap reduction from this recovered baseline.

The recovered 23,186-sample dataset achieved **Test Acc@1 = 26.26%**, a **9.7x improvement** over the old 1,544-sample subset.

---

## 1. Files/Configs Used

### Training Config

| File | Path |
|------|------|
| Config | `repro/referit3d_baseline/configs/recovered_23k_simple.yaml` |

### Dataset

| File | Path |
|------|------|
| Train manifest | `data/processed/scene_disjoint/official_scene_disjoint/train_manifest.jsonl` |
| Val manifest | `data/processed/scene_disjoint/official_scene_disjoint/val_manifest.jsonl` |
| Test manifest | `data/processed/scene_disjoint/official_scene_disjoint/test_manifest.jsonl` |
| BERT features | `data/text_features/recovered_23k/` |

### Outputs

| File | Path |
|------|------|
| Best model | `outputs/20260408_recovered_23k_rerun/simple_encoder/best_model.pt` |
| Training history | `outputs/20260408_recovered_23k_rerun/simple_encoder/training_history.json` |
| Test results | `outputs/20260408_recovered_23k_rerun/simple_encoder/eval_test_results.json` |
| Val results | `outputs/20260408_recovered_23k_rerun/simple_encoder/eval_val_results.json` |
| Comparison | `outputs/20260408_recovered_23k_rerun/recovered_23k_rerun_comparison.json` |

---

## 2. Exact Commands Run

### BERT Feature Generation

```bash
python scripts/prepare_bert_features.py \
    --manifest-dir data/processed/scene_disjoint/official_scene_disjoint \
    --output-dir data/text_features/recovered_23k \
    --batch-size 64 \
    --device cuda
```

### Training

```bash
python repro/referit3d_baseline/scripts/train.py \
    --config repro/referit3d_baseline/configs/recovered_23k_simple.yaml \
    --device cuda
```

### Evaluation

```bash
python repro/referit3d_baseline/scripts/evaluate.py \
    --checkpoint outputs/20260408_recovered_23k_rerun/simple_encoder/best_model.pt \
    --split test \
    --manifest-dir data/processed/scene_disjoint/official_scene_disjoint \
    --bert-dir data/text_features/recovered_23k \
    --device cuda
```

---

## 3. Rerun Metrics

### SimplePointEncoder (Best Baseline)

| Metric | Value |
|--------|-------|
| Val Acc@1 | 28.98% |
| Val Acc@5 | 87.59% |
| Test Acc@1 | **26.26%** |
| Test Acc@5 | 85.49% |
| Val-Test Gap | 2.72% |
| Best Epoch | 27 |
| Training Time | ~11 min |

### Training Progress

| Epoch | Train Loss | Train Acc | Val Acc@1 | Val Acc@5 |
|-------|------------|-----------|-----------|-----------|
| 1 | 3.19 | 6.38% | 8.06% | 45.85% |
| 5 | 2.39 | 16.95% | 19.40% | 72.63% |
| 10 | 1.97 | 24.30% | 24.73% | 82.55% |
| 15 | 1.71 | 28.06% | 26.64% | 85.83% |
| 20 | 1.58 | 29.64% | 28.79% | 87.10% |
| 27 | 1.49 | 30.86% | **28.98%** | 87.34% |
| 30 | 1.48 | 31.90% | 28.84% | 87.63% |

---

## 4. Selected Trustworthy Baseline

**Model**: repro ReferIt3DNet (SimplePointEncoder)

| Metric | Value |
|--------|-------|
| Test Acc@1 | **26.26%** |
| Test Acc@5 | 85.49% |
| Val Acc@1 | 28.98% |
| Val-Test Gap | 2.72% |
| Parameters | 3,463,681 |

**Reason**: Best performing config on scene-disjoint test set.

---

## 5. Remaining Gap to 35.6%

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Test Acc@1 | 26.26% | 35.6% | **-9.34%** |
| Test Acc@5 | 85.49% | ~70% | +15.49% |

**Progress**: 73.76% of target achieved (was 7.6%)

---

## 6. Strongest Current Conclusion

The recovered 23K dataset achieves **Test Acc@1 = 26.26%**, a **9.7x improvement** over the old 1.5K subset. The dominant remaining bottleneck is **missing data** (44% of official Nr3D). Secondary bottlenecks are placeholder geometry and synthetic features.

---

## 7. Next-Step Recommendation

### Primary

1. **Recover remaining ScanNet scenes**: Target full 41,503 samples
   - Expected improvement: +10-15% Test Acc@1

2. **Generate real geometry**: Point-based geometry from ScanNet meshes
   - Expected improvement: +3-5% Test Acc@1

### Secondary

3. **Add visual embeddings**: Use pretrained vision models
   - Expected improvement: +2-4% Test Acc@1

4. **Test PointNet++**: Confirm if raw point features help
   - Expected improvement: +1-2% Test Acc@1

### Deprioritized

- Protocol alignment
- MVT attention
- Structured parser methods

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
- Trustworthy baseline rerun
- Full Nr3D dataset recovery
- Recovered 23K trustworthy baseline rerun (NEW)

---

## Trustworthy Baseline (Recovered 23K)

| Metric | Value |
|--------|-------|
| Test Acc@1 | **26.26%** |
| Test Acc@5 | 85.49% |
| Val Acc@1 | 28.98% |
| Val-Test Gap | 2.72% |

**Improvement**: 9.7x from old 1.5K subset (Test Acc@1 = 2.70%)

**Progress**: 73.76% of 35.6% target

---

## Dataset Statistics

| Split | Samples | Scenes |
|-------|---------|--------|
| Train | 18,459 | 215 |
| Val | 2,046 | 26 |
| Test | 2,681 | 28 |
| **Total** | **23,186** | **269** |

**Coverage**: 55.9% of official Nr3D

---

## Remaining Bottleneck

- Missing 44% of official data (18,317 samples)
- Placeholder geometry (aggregation-based)
- Synthetic features (hash-based)

---

## Next Step

Complete dataset recovery and geometry generation.
```

---

## 9. Decision

**Option A Selected**: Trustworthy baseline anchor is now substantially improved; continue controlled gap reduction from this recovered baseline.

### Rationale

1. Test Acc@1 improved from 2.70% to 26.26% (9.7x improvement)
2. Val-test gap improved from 8.79% to 2.72%
3. Progress to target: 73.76% achieved
4. Clear path forward: data recovery and geometry generation

---

## 10. Key Reports

- `reports/recovered_23k_rerun_audit.md` - Pre-rerun audit
- `reports/recovered_23k_rerun_protocol.md` - Rerun protocol
- `reports/subset_vs_recovered23k_results.md` - Comparison analysis
- `reports/post_recovered23k_bottleneck_reassessment.md` - Bottleneck analysis