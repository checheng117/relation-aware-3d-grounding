# Scene-Disjoint Rerun Audit

**Date**: 2026-04-06
**Phase**: Trustworthy Baseline Rerun - Step 0

---

## Executive Summary

**Status**: Previous baseline configs use old split with scene overlap. Scene-disjoint configs are partially ready. Minimal controlled rerun set identified.

---

## 1. Previous Results Invalidated by Scene Overlap

### Old Split Configuration

All previous experiments used:
- `data/processed/train_manifest.jsonl`
- `data/processed/val_manifest.jsonl`
- `data/processed/test_manifest.jsonl`

**Overlap**: 53 scenes between val and test (57.8% of val, 52.9% of test)

### Invalidated Results (DO NOT USE)

| Configuration | Val Acc@1 | Test Acc@1 | Status |
|---------------|-----------|------------|--------|
| Baseline (SimplePointEncoder) | 22.73% | 9.68% | **INVALIDATED** |
| PointNet++ Encoder | 21.43% | 10.97% | **INVALIDATED** |
| Full Protocol Aligned | 27.27% | 3.87% | **INVALIDATED** |

**Reason**: Val-test scene overlap artificially inflated val accuracy and corrupted test evaluation. These numbers cannot be trusted for any conclusion.

---

## 2. Current Baseline Config Candidates

### A. rag3d Package Configs

| Config | Model | Dataset Split | Ready? |
|--------|-------|---------------|--------|
| `configs/train/scene_disjoint_baseline.yaml` | attribute_only | scene_disjoint | **YES** |
| `configs/train/baseline.yaml` | attribute_only | old (overlap) | NO (deprecated) |

### B. repro/referit3d_baseline Configs

| Config | Model | Dataset Split | Ready? |
|--------|-------|---------------|--------|
| `repro/referit3d_baseline/configs/official_baseline.yaml` | SimplePointEncoder | old (overlap) | NO (deprecated) |
| `repro/referit3d_baseline/configs/pointnetpp_encoder.yaml` | PointNet++ | old (overlap) | NO (deprecated) |
| `repro/referit3d_baseline/configs/protocol_align_full.yaml` | SimplePointEncoder + protocol | old (overlap) | NO (deprecated) |

---

## 3. Configs Required for Scene-Disjoint Rerun

### Priority 1: Feature-Fidelity Baseline on Scene-Disjoint

**Config**: `configs/train/scene_disjoint_baseline.yaml` (ALREADY EXISTS)

- Model: attribute_only (ObjectMLPEncoder)
- Dataset: `configs/dataset/referit3d_scene_disjoint.yaml`
- Split: scene_disjoint (zero overlap)
- Status: Ready to run

### Priority 2: PointNet++ Encoder on Scene-Disjoint

**Config**: Needs creation (`repro/referit3d_baseline/configs/pointnetpp_scene_disjoint.yaml`)

- Model: PointNet++ encoder
- Dataset: scene_disjoint manifests
- Status: Requires config update

### Priority 3: Protocol-Aligned Variant (Optional)

**Decision**: Skip for minimal rerun.

**Rationale**: Previous phase showed protocol alignment hurt test accuracy (Val 27.27% → Test 3.87%). Given limited compute and small dataset (1,544 samples), focus on establishing baseline anchor first. Protocol alignment is a secondary investigation.

---

## 4. Minimal Controlled Rerun Plan

### Recommended Set

| Priority | Config | Purpose |
|----------|--------|---------|
| 1 | `configs/train/scene_disjoint_baseline.yaml` | Feature-fidelity baseline anchor |
| 2 | PointNet++ scene-disjoint (new config) | Encoder comparison under clean evaluation |

**Total configs**: 2 (minimal set)

### Why This Set

1. **Baseline** - Must establish trustworthy anchor
2. **PointNet++** - Previous phase showed PointNet++ had best test generalization; need to verify this holds under clean split
3. **Protocol alignment** - Skip because previous results showed it hurt test; not worth rerun until baseline is trustworthy

---

## 5. Config Files to Create/Update

### Required New Config

**File**: `repro/referit3d_baseline/configs/pointnetpp_scene_disjoint.yaml`

**Changes from old config**:
- `manifest_dir`: `data/processed` → `data/processed/scene_disjoint`
- `checkpoint.dir`: Update to new output path

### Alternative: Use rag3d Package for Both

The rag3d package has:
- `configs/train/scene_disjoint_baseline.yaml` - already ready
- Simpler model architecture (ObjectMLPEncoder)
- Direct integration with scene_disjoint dataset config

**Recommendation**: Run rag3d baseline first. If PointNet++ comparison is needed, either:
- Option A: Create scene-disjoint config in repro directory
- Option B: Skip PointNet++ for this phase, focus on establishing rag3d baseline anchor

---

## 6. Previous Phase Conclusions to Re-verify

| Previous Conclusion | Need Re-verification? |
|---------------------|----------------------|
| Encoder NOT dominant bottleneck | YES - verify on clean split |
| Protocol alignment hurt test | YES - but skip for minimal rerun |
| PointNet++ best test generalization | YES - verify on clean split |
| Dataset size is limiting factor | YES - unchanged (1,544 vs 41,503) |

---

## 7. Risks and Unknowns

| Risk | Mitigation |
|------|------------|
| Scene-disjoint val is smaller (148 vs 155 samples) | Expected; acceptable |
| Test set smaller (185 vs 157 samples) | Expected; acceptable |
| Training may show different convergence pattern | Monitor closely |
| Previous conclusions may change materially | Document in final report |

---

## 8. Recommended Next Steps

### Step 1: Create Protocol Report

Write `reports/scene_disjoint_rerun_protocol.md` defining:
- Exact configs to rerun
- Output directory structure
- Evaluation metrics
- Success criteria

### Step 2: Smoke Test

Run minimal smoke test to verify:
- Dataset loads correctly
- Train/val/test counts match expected
- Training pipeline runs
- Metrics export correctly

### Step 3: Controlled Rerun

Run selected configs:
1. rag3d scene_disjoint_baseline (mandatory)
2. PointNet++ scene_disjoint (optional, if time permits)

### Step 4: Compare Old vs New Results

Document how metrics changed and whether conclusions hold.

---

## 9. Decision

**Minimal rerun set**: 1-2 configs

1. **Mandatory**: `configs/train/scene_disjoint_baseline.yaml` (rag3d baseline)
2. **Optional**: PointNet++ on scene_disjoint (if compute permits and PointNet++ comparison is valuable)

**Skip**: Protocol alignment rerun (previous phase showed negative effect on test)

---

## 10. Files Inspected

- `.claude/PROJECT_CONTEXT.md`
- `.claude/CURRENT_STATUS.md`
- `.claude/WORKING_RULES.md`
- `.claude/NEXT_TASK.md`
- `configs/dataset/referit3d_scene_disjoint.yaml`
- `configs/train/scene_disjoint_baseline.yaml`
- `configs/eval/scene_disjoint_eval.yaml`
- `configs/train/baseline.yaml`
- `repro/referit3d_baseline/configs/official_baseline.yaml`
- `repro/referit3d_baseline/configs/pointnetpp_encoder.yaml`
- `reports/scene_disjoint_split_recovery_results.md`
- `reports/scene_disjoint_split_validation.md`
- `reports/encoder_upgrade_results.md`
- `reports/training_protocol_fidelity_results.md`

---

## Conclusion

The audit identifies one ready config (`scene_disjoint_baseline.yaml`) and one optional config (PointNet++ scene-disjoint) for the minimal controlled rerun. Previous protocol alignment experiments should be skipped. Proceed to write the protocol report.