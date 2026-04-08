# Old vs Scene-Disjoint Split Comparison Report

**Date**: 2026-04-06
**Phase**: Scene-Disjoint Split Recovery - Step 3

---

## Executive Summary

| Metric | Old Split | Scene-Disjoint Split | Change |
|--------|-----------|---------------------|--------|
| Val-Test overlap | 53 scenes | **0 scenes** | FIXED |
| Val samples in overlap | 89 (57.8%) | 0 | FIXED |
| Test samples in overlap | 82 (52.9%) | 0 | FIXED |
| Train scenes | 243 | 212 | -31 |
| Val scenes | 100 | 26 | -74 |
| Test scenes | 110 | 28 | -82 |
| Train samples | 1,235 | 1,211 | -24 |
| Val samples | 154 | 148 | -6 |
| Test samples | 155 | 185 | +30 |
| Total scenes | 269 | 266 | -3 |

**Key improvement**: Val-Test overlap eliminated. Split is now valid for evaluation.

---

## 1. Scene Overlap Statistics

### Old Split (Random Sample Shuffle)

| Overlap Type | Scenes | Val Samples | Test Samples |
|--------------|--------|-------------|---------------|
| Val-Test | **53** | 89 (57.8%) | 82 (52.9%) |
| Train-Val | 89 | N/A | N/A |
| Train-Test | 97 | N/A | N/A |

**Problem**: More than half of val and test samples were from scenes appearing in both splits.

### Scene-Disjoint Split (Scene-Level Split)

| Overlap Type | Scenes | Val Samples | Test Samples |
|--------------|--------|-------------|---------------|
| Val-Test | **0** | 0 | 0 |
| Train-Val | 0 | N/A | N/A |
| Train-Test | 0 | N/A | N/A |

**Fixed**: All splits are completely scene-disjoint.

---

## 2. Sample Count Changes

| Split | Old | New | Change | % Change |
|-------|-----|-----|--------|----------|
| Train | 1,235 | 1,211 | -24 | -2.0% |
| Val | 154 | 148 | -6 | -3.9% |
| Test | 155 | 185 | +30 | +19.4% |

**Note**: Test set increased because high-density scenes happened to be assigned to test.

---

## 3. Scene Count Changes

| Split | Old | New | Change | % Change |
|-------|-----|-----|--------|----------|
| Train | 243 | 212 | -31 | -12.8% |
| Val | 100 | 26 | -74 | -74.0% |
| Test | 110 | 28 | -82 | -74.5% |

**Reason**: Scene-level splitting assigns all samples of a scene to one split, reducing scene diversity in val/test.

---

## 4. Samples Per Scene Distribution

### Old Split
- Samples were distributed across splits from same scene
- Multiple samples from same scene could appear in different splits

### Scene-Disjoint Split
| Split | Mean | Min | Max |
|-------|------|-----|-----|
| Train | 5.72 | 1 | 26 |
| Val | 5.69 | 1 | 26 |
| Test | 6.61 | 1 | 24 |

**Observation**: Test has slightly higher density per scene.

---

## 5. Split Ratio Comparison

### Scene-Level Ratios

| Split | Old (scenes) | New (scenes) | Target |
|-------|--------------|--------------|--------|
| Train | 243 (90.3%) | 212 (79.7%) | 80% |
| Val | 100 (37.3%) | 26 (9.8%) | 10% |
| Test | 110 (40.8%) | 28 (10.5%) | 10% |

**Note**: Old split had inflated scene counts in val/test due to scene overlap. New split properly assigns scenes.

### Sample-Level Ratios

| Split | Old | New | Target |
|-------|-----|-----|--------|
| Train | 1,235 (79.9%) | 1,211 (78.3%) | 80% |
| Val | 154 (10.0%) | 148 (9.6%) | 10% |
| Test | 155 (10.0%) | 185 (12.0%) | 10% |

**Note**: New split ratios close to target, test slightly larger due to scene density.

---

## 6. Impact on Previous Experiments

### Previous Results (Invalidated)

| Configuration | Val Acc@1 | Test Acc@1 | Note |
|--------------|-----------|------------|------|
| Baseline | 22.73% | 9.68% | Test unreliable |
| PointNet++ | 21.43% | 10.97% | Test unreliable |
| Protocol alignment | 27.27% | 3.87% | Test unreliable |

**Why unreliable**:
- 57.8% of val samples from scenes also in test
- 52.9% of test samples from scenes also in val
- Model could learn scene-specific patterns that don't generalize

### Required Action

All baseline experiments must be **rerun on scene-disjoint splits** to obtain trustworthy results.

---

## 7. Usability for Reproduction

### Is the new split usable?

| Aspect | Status | Note |
|--------|--------|------|
| Scene disjoint | PASS | Zero overlap |
| Sample counts | PASS | Sufficient for evaluation |
| Ratio balance | PASS | Close to 80/10/10 |
| Same-class clutter | PASS | 99.4% across all splits |
| Anchor confusion | PASS | 80 samples distributed |

**Conclusion**: The new split is usable for reproduction experiments.

### Limitation Acknowledgment

The dataset is still small (1,544 samples vs official 41,503). This fundamental limitation remains, but evaluation is now valid.

---

## 8. Files Compared

- Old: `data/processed/*.jsonl`
- New: `data/processed/scene_disjoint/*.jsonl`
- Validation: `scene_disjoint_split_validation.json`
- Summary: `data/processed/dataset_summary.json` vs `data/processed/scene_disjoint/dataset_summary.json`

---

## 9. Recommendation

**Use scene-disjoint split for all future experiments.**

Old split results should be:
1. Documented as unreliable
2. Not used for any conclusions
3. Superseded by new split results after rerun

---

## 10. Next Steps

1. Wire scene-disjoint split into reproduction track configs
2. Rerun PointNet++ baseline on new split
3. Compare new results to old (documented as unreliable)
4. Establish valid baseline for future experiments