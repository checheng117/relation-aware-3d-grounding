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

**Coverage**: 55.9% of official Nr3D (41,503 samples)

---

## Remaining Bottleneck

| Rank | Bottleneck | Impact |
|------|------------|--------|
| 1 | Missing 44% of official data | Critical |
| 2 | Placeholder geometry | High |
| 3 | Synthetic features | Medium |

---

## Geometry Note

Recovered dataset uses ScanNet aggregation geometry (placeholder centers/sizes) rather than real point-based geometry. This is acceptable for baseline reproduction but may affect accuracy.

---

## Scene Overlap Verification

| Split Pair | Overlap | Status |
|------------|---------|--------|
| Val-Test | 0 scenes | ✓ Valid |
| Train-Val | 0 scenes | ✓ Valid |
| Train-Test | 0 scenes | ✓ Valid |

---

## Key Fixes Applied

1. **Deterministic hash for class features**: Fixed Python hash randomization issue that caused train-eval mismatch
2. **BERT feature alignment**: Regenerated for recovered 23K dataset

---

## Next Step

**Primary**: Complete dataset recovery (target full 41,503 samples)

**Secondary**:
1. Generate real point-based geometry
2. Add visual embeddings
3. Test PointNet++ encoder

**Manifests**: `data/processed/scene_disjoint/official_scene_disjoint/`

---

## Key Reports

- `reports/recovered_23k_rerun_audit.md` - Pre-rerun audit
- `reports/recovered_23k_rerun_protocol.md` - Rerun protocol
- `reports/recovered_23k_rerun_results.md` - Final results
- `reports/subset_vs_recovered23k_results.md` - Comparison analysis
- `reports/post_recovered23k_bottleneck_reassessment.md` - Bottleneck analysis