# Scene-Disjoint Split Validation Report

**Date**: 2026-04-06
**Phase**: Scene-Disjoint Split Recovery - Step 2

---

## Validation Result: **PASS**

The scene-disjoint split has **zero scene overlap** between val and test.

---

## 1. Scene Overlap Verification

### Val-Test Overlap
| Metric | Result |
|--------|--------|
| Overlapping scenes | **0** (PASS) |
| Val samples in overlap | **0** |
| Test samples in overlap | **0** |

### Train-Val Overlap
| Metric | Result |
|--------|--------|
| Overlapping scenes | 0 (acceptable for scene-level splits) |

### Train-Test Overlap
| Metric | Result |
|--------|--------|
| Overlapping scenes | 0 (acceptable for scene-level splits) |

**Note**: In scene-level splitting, it's acceptable for train to share scenes with val/test, as long as val and test are disjoint. However, in our implementation, all three splits are completely disjoint.

---

## 2. Sample Duplication Verification

### Duplicate Samples Across Splits
| Overlap | Result |
|---------|--------|
| Train-Val duplicates | **0** |
| Train-Test duplicates | **0** |
| Val-Test duplicates | **0** |

**PASS**: No sample appears in multiple splits.

---

## 3. Scene Counts per Split

| Split | Scenes | Samples | Samples/Scene (mean) |
|-------|--------|---------|---------------------|
| Train | 212 | 1,211 | 5.72 |
| Val | 26 | 148 | 5.69 |
| Test | 28 | 185 | 6.61 |
| **Total** | **266** | **1,544** | 5.79 |

---

## 4. Scene List Verification

Scene lists stored in `data/splits/`:
- `scene_disjoint_train.txt` - 212 scenes
- `scene_disjoint_val.txt` - 26 scenes
- `scene_disjoint_test.txt` - 28 scenes

Manifests stored in `data/processed/scene_disjoint/`:
- `train_manifest.jsonl` - 1,211 samples
- `val_manifest.jsonl` - 148 samples
- `test_manifest.jsonl` - 185 samples
- `dataset_summary.json` - Full statistics

---

## 5. Class/Difficulty Distribution

From dataset summary:
- Same-class clutter: 1,534 samples (99.4%)
- Anchor confusion: 80 samples (5.2%)
- Relation type: 1,544 "none" (no explicit relation type annotation in this dataset)

**Note**: Distribution is similar across splits due to random scene-level assignment.

---

## 6. Balance Analysis

### Scene Distribution
- Train: 212 scenes (79.7% of total scenes)
- Val: 26 scenes (9.8% of total scenes)
- Test: 28 scenes (10.5% of total scenes)

**Close to target 80/10/10 ratio at scene level.**

### Sample Distribution
- Train: 1,211 samples (78.3% of total samples)
- Val: 148 samples (9.6% of total samples)
- Test: 185 samples (12.0% of total samples)

**Close to target ratio at sample level, slight test skew due to scene sample density.**

### Per-Scene Sample Density
- Train: mean 5.72 samples/scene
- Val: mean 5.69 samples/scene
- Test: mean 6.61 samples/scene

**Test has slightly denser scenes, but variance is acceptable.**

---

## 7. High-Scene-Density Samples

### Train (top 5)
| Scene | Samples |
|-------|---------|
| scene0128_00 | 26 |
| scene0590_00 | 25 |
| scene0668_00 | 22 |
| scene0203_00 | 21 |
| scene0505_00 | 21 |

### Val (top 5)
| Scene | Samples |
|-------|---------|
| scene0211_00 | 26 |
| scene0522_00 | 19 |
| scene0265_00 | 18 |
| scene0419_00 | 18 |
| scene0142_00 | 8 |

### Test (top 5)
| Scene | Samples |
|-------|---------|
| scene0652_00 | 24 |
| scene0246_00 | 22 |
| scene0093_00 | 10 |
| scene0547_00 | 10 |
| scene0565_00 | 10 |

---

## 8. Files Inspected

- `data/processed/scene_disjoint/train_manifest.jsonl`
- `data/processed/scene_disjoint/val_manifest.jsonl`
- `data/processed/scene_disjoint/test_manifest.jsonl`
- `data/processed/scene_disjoint/dataset_summary.json`
- `data/splits/scene_disjoint_train.txt`
- `data/splits/scene_disjoint_val.txt`
- `data/splits/scene_disjoint_test.txt`
- `scene_disjoint_split_validation.json`

---

## 9. Conclusion

**The scene-disjoint split is valid and ready for use.**

- Zero scene overlap between val and test
- Zero sample duplication
- Scene and sample ratios approximately 80/10/10
- Distribution balanced enough for valid evaluation

**Recommendation**: Use this split for all future trustworthy experiments.