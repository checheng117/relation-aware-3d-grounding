# Scene-Disjoint Split Recovery Results

**Date**: 2026-04-06
**Phase**: Scene-Disjoint Split Recovery - Final Report

---

## Executive Summary

**Result: Scene-disjoint split is ready. Rerun baseline reproduction next (Option A).**

The scene overlap issue has been fixed. Val-Test overlap reduced from 53 scenes (57.8% of val, 52.9% of test) to **zero**.

---

## 1. Files Modified / Added

### Modified Files
| File | Change |
|------|--------|
| `src/rag3d/datasets/builder.py` | Added `split_records_by_scene()` function |
| `scripts/prepare_data.py` | Added `--mode build-nr3d-geom-scene-disjoint` and `cmd_build_nr3d_geom_scene_disjoint()` |

### Added Files

**Scripts**:
- `scripts/validate_scene_disjoint_splits.py` - Validation script

**Configs**:
- `configs/dataset/referit3d_scene_disjoint.yaml` - Scene-disjoint dataset config
- `configs/train/scene_disjoint_baseline.yaml` - Training config for scene-disjoint
- `configs/eval/scene_disjoint_eval.yaml` - Evaluation config for scene-disjoint

**Data**:
- `data/splits/scene_disjoint_train.txt` - Train scene list (212 scenes)
- `data/splits/scene_disjoint_val.txt` - Val scene list (26 scenes)
- `data/splits/scene_disjoint_test.txt` - Test scene list (28 scenes)
- `data/processed/scene_disjoint/train_manifest.jsonl` - Train manifest (1,211 samples)
- `data/processed/scene_disjoint/val_manifest.jsonl` - Val manifest (148 samples)
- `data/processed/scene_disjoint/test_manifest.jsonl` - Test manifest (185 samples)
- `data/processed/scene_disjoint/dataset_summary.json` - Full statistics

**Reports**:
- `reports/scene_disjoint_split_audit.md` - Initial audit
- `reports/scene_disjoint_split_validation.md` - Validation results
- `reports/old_vs_scene_disjoint_split_comparison.md` - Old vs new comparison
- `reports/scene_disjoint_split_recovery_results.md` - This report

**Validation Data**:
- `scene_disjoint_split_validation.json` - Validation JSON

---

## 2. Exact Commands Run

```bash
# Build scene-disjoint splits
python scripts/prepare_data.py --mode build-nr3d-geom-scene-disjoint

# Validate scene-disjoint split integrity
python scripts/validate_scene_disjoint_splits.py

# Smoke verification (inline Python)
python -c "..."  # Verified dataset loading and scene overlap
```

---

## 3. Old Split Integrity Findings

| Metric | Old Split | Status |
|--------|-----------|--------|
| Val-Test overlap | 53 scenes | INVALID |
| Val samples in overlap | 89 (57.8%) | INVALID |
| Test samples in overlap | 82 (52.9%) | INVALID |
| Train-Val overlap | 89 scenes | Acceptable |
| Train-Test overlap | 97 scenes | Acceptable |

**Conclusion**: Old split results are unreliable and should not be used for conclusions.

---

## 4. New Split Construction Logic

### Algorithm: Scene-Level Splitting

```python
def split_records_by_scene(records, train_ratio=0.8, val_ratio=0.1, seed=42):
    # 1. Group records by scene_id
    # 2. Shuffle scene IDs (not samples)
    # 3. Assign scenes to splits: 80% train, 10% val, 10% test
    # 4. Assign all samples of each scene to its split
    # 5. Return train_records, val_records, test_records, scene_info
```

### Key Differences from Old Split

| Aspect | Old Split | New Split |
|--------|-----------|-----------|
| Shuffle level | Sample | Scene |
| Scene overlap | Yes | No |
| Sample overlap | No | No |
| Scene assignment | Random per sample | All samples of scene go to same split |

---

## 5. Validation Results Proving Zero Overlap

| Verification | Result |
|--------------|--------|
| Val-Test scene overlap | **0 scenes** (PASS) |
| Val-Test sample overlap | **0 samples** (PASS) |
| Train-Val scene overlap | **0 scenes** (PASS) |
| Train-Test scene overlap | **0 scenes** (PASS) |
| Sample counts correct | Train 1,211, Val 148, Test 185 (PASS) |
| Manifests load correctly | PASS |
| Object features present | PASS (256-dim) |

**All validations passed.**

---

## 6. Experiments to Rerun Next

### Priority 1: Baseline on Scene-Disjoint Split

**Configuration**: `configs/train/scene_disjoint_baseline.yaml`

```bash
# Recommended command (not run in this phase)
python scripts/train.py --config configs/train/scene_disjoint_baseline.yaml
```

### Priority 2: PointNet++ on Scene-Disjoint Split

If PointNet++ encoder is available, rerun with scene-disjoint config to verify generalization.

### Priority 3: Compare Old vs New Results

After rerun, compare:
- Old baseline results (invalid, for reference only)
- New scene-disjoint baseline results (trustworthy)

---

## 7. Updated Recommendation for CURRENT_STATUS.md

```markdown
# Current Status

Completed phases:
- Geometry recovery
- Feature fidelity integration
- Encoder upgrade
- Training protocol fidelity
- Distribution mismatch investigation
- Scene-disjoint split recovery (NEW)

Critical fix applied:
- Scene-disjoint splits implemented
- Val-Test overlap: 0 scenes (was 53 scenes)
- Evaluation now trustworthy

Current best measured results (OLD - INVALIDATED, use only for reference):
- Baseline: Val Acc@1 = 22.73%, Test Acc@1 = 9.68%
- PointNet++: Val Acc@1 = 21.43%, Test Acc@1 = 10.97%
- Protocol alignment: Val Acc@1 = 27.27%, Test Acc@1 = 3.87%

NEW trustworthy results needed:
- Must rerun baseline on scene-disjoint split
- Previous results unreliable due to scene overlap

New split statistics:
- Train: 212 scenes, 1,211 samples
- Val: 26 scenes, 148 samples
- Test: 28 scenes, 185 samples
- Val-Test overlap: 0 scenes (verified)

Blocking issue resolved:
- Scene-disjoint splits ready for use

Next step: Rerun baseline reproduction on scene-disjoint split
```

---

## 8. Final Decision

**Option A: Scene-disjoint split is ready, rerun baseline reproduction next.**

The scene overlap issue is fixed. The split is validated with zero overlap. Dataset loading works. Configs are ready.

**Recommendation**: Proceed to rerun baseline reproduction training on scene-disjoint split.

---

## 9. Limitations Still Present

| Limitation | Status | Note |
|------------|--------|------|
| Dataset size | 1,544 vs 41,503 | Fundamental limit remains |
| Language encoder | Hash-based | Not BERT |
| Geometry encoder | Simple MLP | Not PointNet++ (optionally) |
| MVT attention | Missing | Not implemented |

**Note**: Scene-disjoint split fixes evaluation validity, but dataset size remains a fundamental limitation for achieving 35.6% target. Results should be interpreted within this constraint.

---

## 10. Files Inspected/Generated

### Inspected
- `.claude/PROJECT_CONTEXT.md`
- `.claude/CURRENT_STATUS.md`
- `.claude/WORKING_RULES.md`
- `.claude/NEXT_TASK.md`
- `src/rag3d/datasets/builder.py`
- `scripts/prepare_data.py`
- `data/processed/*.jsonl` (old split)
- Previous reports

### Generated
- All files listed in Section 1

---

## 11. Conclusion

Scene-disjoint split implementation complete. The evaluation pipeline is now trustworthy. Proceed to rerun baseline experiments on the new split to establish valid results.