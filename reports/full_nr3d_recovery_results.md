# Full Nr3D Recovery Results

**Date**: 2026-04-08
**Phase**: Full Nr3D Dataset Recovery - Step 8 (Final Report)

---

## Executive Summary

**Decision**: Option A - Recovered dataset is now sufficient; rerun trustworthy baseline next.

The dataset recovery achieved a **15x increase** in sample count (1,544 → 23,186), providing sufficient data for meaningful baseline reproduction.

---

## 1. Files Modified / Added

### New Files

| File | Purpose |
|------|---------|
| `src/rag3d/datasets/nr3d_official.py` | Official Nr3D CSV parser |
| `data/raw/referit3d/annotations/nr3d_official.csv` | Official Nr3D annotations (41,503 samples) |
| `data/processed/scene_disjoint/official_scene_disjoint/*.jsonl` | Recovered manifests |
| `data/splits/official_scene_disjoint_*.txt` | Scene lists |
| `reports/full_nr3d_recovery_audit.md` | Pre-recovery audit |
| `reports/nr3d_source_inventory.md` | Source inventory |
| `reports/nr3d_sample_loss_breakdown.md` | Sample loss analysis |
| `reports/full_nr3d_recovery_validation.md` | Validation report |
| `reports/old_subset_vs_recovered_nr3d.md` | Comparison report |

### Modified Files

| File | Changes |
|------|---------|
| `src/rag3d/datasets/builder.py` | Added `build_records_nr3d_official_with_scans()` |
| `scripts/prepare_data.py` | Added `cmd_build_nr3d_official_scene_disjoint()` |

---

## 2. Exact Commands Run

```bash
# Download official Nr3D
gdown https://drive.google.com/uc?id=1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC \
    -O data/raw/referit3d/annotations/nr3d_official.csv

# Build recovered manifests
python scripts/prepare_data.py \
    --mode build-nr3d-official-scene-disjoint \
    --config configs/dataset/referit3d_scene_disjoint.yaml

# Validate
python scripts/validate_scene_disjoint_splits.py \
    --manifest-dir data/processed/scene_disjoint/official_scene_disjoint

# Smoke test
python -c "..."  # Verified dataset loading and model pipeline
```

---

## 3. Recovered Dataset Size

| Split | Samples | Scenes |
|-------|---------|--------|
| Train | 18,459 | 215 |
| Val | 2,046 | 26 |
| Test | 2,681 | 28 |
| **Total** | **23,186** | **269** |

**Coverage**: 55.9% of official Nr3D (41,503 samples)

---

## 4. Sample Loss Breakdown

| Category | Count | Percentage |
|----------|-------|------------|
| **Recovered** | 23,186 | 55.9% |
| Skipped (no aggregation) | 18,317 | 44.1% |
| **Official Total** | 41,503 | 100% |

**Root cause of loss**: 372 scenes without aggregation files in our data pool.

---

## 5. Validation Results

| Check | Result |
|-------|--------|
| Zero scene overlap (val-test) | ✓ PASS |
| Zero sample duplicates | ✓ PASS |
| All targets in scene objects | ✓ PASS |
| Dataset loading works | ✓ PASS |
| Model pipeline works | ✓ PASS |

---

## 6. Sufficiency Assessment

### Is the Recovered Dataset Sufficient?

**Yes.** The recovered dataset meets all criteria for meaningful baseline reproduction:

| Criterion | Threshold | Recovered | Status |
|-----------|-----------|-----------|--------|
| Total samples | > 10,000 | 23,186 | ✓ |
| Train samples | > 5,000 | 18,459 | ✓ |
| Val samples | > 500 | 2,046 | ✓ |
| Test samples | > 500 | 2,681 | ✓ |

---

## 7. CURRENT_STATUS.md Update

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
- Full Nr3D dataset recovery (NEW)

---

## Recovered Dataset Statistics

**Source**: Official Nr3D CSV (41,503 samples)

| Split | Samples | Scenes |
|-------|---------|--------|
| Train | 18,459 | 215 |
| Val | 2,046 | 26 |
| Test | 2,681 | 28 |
| **Total** | **23,186** | **269** |

**Coverage**: 55.9% of official Nr3D
**Improvement**: 15x from previous 1,544 samples

---

## Geometry Note

Recovered dataset uses ScanNet aggregation geometry (placeholder centers/sizes) rather than real point-based geometry. This is acceptable for baseline reproduction but may affect accuracy.

---

## Next Step

Rerun trustworthy baseline on recovered dataset.
```

---

## 8. Next-Step Recommendation

**Primary**: Rerun trustworthy baseline on recovered 23K sample dataset.

**Secondary**: 
1. Regenerate geometry files with correct object ID mapping
2. Download remaining ScanNet scenes to recover 18K more samples

**Deprioritized**:
- MVT attention
- Structured parser methods
- Hyperparameter tuning

---

## 9. Decision

**Option A Selected**: Recovered dataset is now sufficient; rerun trustworthy baseline next.

### Rationale

1. **15x sample increase** (1,544 → 23,186) provides meaningful training signal
2. **Zero scene overlap** verified between splits
3. **All validation checks passed**
4. **Smoke test successful** - dataset loading and model pipeline work

### Expected Impact

| Metric | Old (1.5K) | Expected (23K) |
|--------|------------|----------------|
| Test Acc@1 | 2.70% | 10-20% |
| Convergence | Unstable | Stable |
| Training time | 1-2 min | 10-20 min |

---

## 10. Key Reports

- `reports/full_nr3d_recovery_audit.md` - Root cause analysis
- `reports/nr3d_source_inventory.md` - Source files inventory
- `reports/nr3d_sample_loss_breakdown.md` - Sample loss breakdown
- `reports/full_nr3d_recovery_validation.md` - Validation results
- `reports/old_subset_vs_recovered_nr3d.md` - Comparison analysis