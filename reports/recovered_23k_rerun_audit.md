# Recovered 23K Rerun Audit

**Date**: 2026-04-08
**Phase**: Trustworthy Baseline Rerun on Recovered Dataset - Step 0

---

## Executive Summary

The recovered 23,186-sample dataset provides a **15x increase** over the previous 1,544-sample subset. This audit identifies the baseline configs to rerun and establishes a controlled rerun plan.

---

## 1. Dataset Recovery Summary

| Metric | Old (1,544) | Recovered (23,186) | Ratio |
|--------|-------------|--------------------|-------|
| Total samples | 1,544 | 23,186 | 15x |
| Train samples | 1,211 | 18,459 | 15.2x |
| Val samples | 148 | 2,046 | 13.8x |
| Test samples | 185 | 2,681 | 14.5x |
| Scenes | 266 | 269 | ~same |

**Source**: Official Nr3D CSV (41,503 samples), recovered 55.9%

---

## 2. Prior Results - Now Superseded

### Scene-Disjoint Rerun (on 1,544 samples)

| Config | Model | Val Acc@1 | Test Acc@1 | Test Acc@5 |
|--------|-------|-----------|------------|------------|
| repro scene_disjoint_baseline.yaml | SimplePointEncoder | 11.49% | **2.70%** | 10.81% |
| configs/train/scene_disjoint_baseline.yaml | AttributeOnly | ~4.7% | 0.54% | 20.54% |

**Best**: repro ReferIt3DNet with SimplePointEncoder (Test Acc@1 = 2.70%)

### Overlap-Contaminated Results (INVALIDATED)

| Config | Val Acc@1 | Test Acc@1 | Status |
|--------|-----------|------------|--------|
| SimplePointEncoder (old split) | 22.73% | 9.68% | INVALIDATED |
| PointNet++ (old split) | 21.43% | 10.32% | INVALIDATED |
| Protocol Aligned (old split) | 27.27% | 3.87% | INVALIDATED |

**Reason for invalidation**: Scene overlap between val/test contaminated evaluation.

---

## 3. Strongest Baseline Candidates for Rerun

### Primary Candidate: repro ReferIt3DNet (SimplePointEncoder)

| Attribute | Value |
|-----------|-------|
| Config | `repro/referit3d_baseline/configs/scene_disjoint_baseline.yaml` |
| Encoder | SimplePointEncoder |
| Previous Test Acc@1 | 2.70% (1,544 samples) |
| Expected improvement | 10-20% (with 23K samples) |
| Priority | **HIGH - rerun first** |

**Reason**: Already proven best on scene-disjoint evaluation. Simple, stable, reproducible.

### Secondary Candidate: PointNet++ Encoder

| Attribute | Value |
|-----------|-------|
| Config | `repro/referit3d_baseline/configs/pointnetpp_encoder.yaml` |
| Encoder | PointNetPPEncoder |
| Previous Test Acc@1 | Unknown on scene-disjoint |
| Previous Acc@5 gain | +23% (on contaminated split) |
| Priority | **MEDIUM - rerun after SimplePointEncoder** |

**Reason**: Previous experiments showed Acc@5 improvement but marginal Acc@1 gain. Worth testing on clean scene-disjoint split with larger data.

**Action Required**: Adapt config to use official scene-disjoint manifest paths.

### Not Recommended for This Rerun

| Config | Reason |
|--------|--------|
| Protocol-aligned variants | Prior results showed val overfitting; need clean baseline first |
| rag3d AttributeOnly | Underperformed SimplePointEncoder (0.54% vs 2.70%) |
| MVT / Structured parser methods | Premature until baseline trustworthy |

---

## 4. Config Adaptations Required

### SimplePointEncoder Config

Current `repro/referit3d_baseline/configs/scene_disjoint_baseline.yaml` uses:
- `manifest_dir: data/processed/scene_disjoint`
- `train_manifest: train_manifest.jsonl`

**Change Required**: Update to recovered dataset path:
- `manifest_dir: data/processed/scene_disjoint/official_scene_disjoint`
- Same manifest filenames (train_manifest.jsonl, val_manifest.jsonl, test_manifest.jsonl)

### PointNet++ Config

Current `repro/referit3d_baseline/configs/pointnetpp_encoder.yaml` uses:
- `manifest_dir: data/processed`

**Change Required**: Update to recovered dataset path:
- `manifest_dir: data/processed/scene_disjoint/official_scene_disjoint`

---

## 5. Minimal Controlled Rerun Plan

### Stage A: Smoke Rerun (verify infrastructure)

| Step | Command | Purpose |
|------|---------|---------|
| 1 | Load train manifest | Verify 18,459 samples loaded |
| 2 | Load val manifest | Verify 2,046 samples loaded |
| 3 | Load test manifest | Verify 2,681 samples loaded |
| 4 | Check scene overlap | Verify zero overlap |
| 5 | Run 2 epochs | Verify training executes |
| 6 | Run eval | Verify evaluation executes |

**Estimated time**: 5-10 minutes

### Stage B: Controlled Verification Rerun (moderate training)

| Config | Epochs | Expected Time | Purpose |
|--------|--------|---------------|---------|
| SimplePointEncoder | 10 | 15-30 min | Quick baseline estimate |
| PointNet++ | 10 | 20-40 min | Quick encoder comparison |

**Goal**: Identify promising candidate(s) for formal rerun.

### Stage C: Formal Rerun (full training)

| Config | Epochs | Expected Time | Purpose |
|--------|--------|---------------|---------|
| Best from Stage B | 30 | 30-60 min | Final trustworthy baseline |

**Output directory**: `outputs/<timestamp>_recovered_23k_rerun/`

---

## 6. Fixed Parameters for All Reruns

| Parameter | Value | Reason |
|-----------|-------|--------|
| Dataset path | `data/processed/scene_disjoint/official_scene_disjoint/` | Recovered manifests |
| Manifest names | train/val/test_manifest.jsonl | Standard names |
| Split method | Scene-disjoint | Zero overlap, trustworthy |
| Evaluator | repro evaluate.py | Consistent evaluation |
| Primary metrics | Val Acc@1, Test Acc@1, Test Acc@5 | Standard metrics |
| Seed | 42 | Reproducibility |

---

## 7. Success Criteria

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Test Acc@1 improvement | > 5% | Meaningful gain from data increase |
| Training stability | No NaN/Inf | Clean execution |
| Val-test gap | < 10% | Reasonable generalization |
| Baseline establishment | Clear best config | Trustworthy anchor |

---

## 8. Expected Outcomes

| Metric | Old (1.5K) | Expected (23K) | Target |
|--------|------------|----------------|--------|
| Test Acc@1 | 2.70% | 10-20% | 35.6% |
| Test Acc@5 | 10.81% | 30-50% | ~70% |
| Training stability | Unstable | Stable | - |
| Val-test gap | 8.79% | < 10% | - |

---

## 9. Rerun Priority Order

1. **SimplePointEncoder** on recovered 23K - establish baseline anchor
2. **PointNet++** on recovered 23K - test encoder value with more data
3. **Compare** results to determine best trustworthy baseline

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Training instability | Use conservative LR, gradient clipping |
| Overfitting to val | Monitor val-test gap, use early stopping |
| Memory issues with PointNet++ | Reduce batch size if needed |
| Slow convergence | Use pretrained features if available |

---

## Conclusion

The recovered 23K dataset enables a credible baseline rerun. The primary candidate is repro ReferIt3DNet with SimplePointEncoder, followed by PointNet++ encoder comparison. Previous results (2.70% Test Acc@1) are superseded and must be rerun with the larger dataset.

**Next Step**: Write `reports/recovered_23k_rerun_protocol.md` with exact configs and commands.