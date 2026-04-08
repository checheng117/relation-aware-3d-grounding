# Split Distribution Comparison Report

**Date**: 2026-04-06
**Phase**: Distribution Mismatch Investigation - Step 1

---

## Executive Summary

**Critical Finding**: The train/val/test splits are **NOT scene-disjoint**. This violates standard practice for 3D visual grounding and is likely a major contributor to the val-test accuracy discrepancy.

- 53 scenes appear in both val and test (57.8% of val samples, 52.9% of test samples)
- 89 scenes appear in both train and val
- 97 scenes appear in both train and test

---

## 1. Basic Split Statistics

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Samples | 1235 | 154 | 155 |
| Unique scenes | 243 | 100 | 110 |
| Samples per scene (mean) | 5.08 | 1.54 | 1.41 |

---

## 2. Utterance Statistics

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Utterance length (words) mean | 11.08 | 11.47 | 10.39 |
| Utterance length std | 4.95 | 5.29 | 4.54 |

**Observation**: Test utterances are shorter (10.39 vs 11.47 words). 9.4% shorter utterances in test.

---

## 3. Candidate Set Statistics

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Candidate size (objects) mean | 36.67 | 38.55 | 36.53 |
| Candidate size std | 14.31 | 13.53 | 15.42 |

**Observation**: Similar candidate sizes across splits (~36-38 objects). Val has slightly larger candidate sets (5.2% higher than test).

---

## 4. Same-Class Clutter Analysis

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Same-class clutter rate | 99.4% | 99.4% | 99.4% |
| Same-class clutter samples | 1227 | 153 | 154 |

**Observation**: Nearly all samples (99.4%) have at least one distractor of the same class as the target. This indicates high difficulty across all splits.

---

## 5. Hard Case Tags Distribution

| Tag | Train | Val | Test |
|-----|-------|-----|------|
| anchor_confusion | 64 (5.2%) | 11 (7.1%) | 5 (3.2%) |

**Critical Observation**: Val has higher anchor_confusion rate (7.1%) than test (3.2%). This suggests val contains more samples with difficult anchor objects.

---

## 6. Scene Overlap Analysis (CRITICAL)

| Overlap | Scenes | Val Samples in Overlap | Test Samples in Overlap |
|---------|--------|------------------------|------------------------|
| Train-Val | 89 | N/A | N/A |
| Train-Test | 97 | N/A | N/A |
| **Val-Test** | **53** | **89 (57.8%)** | **82 (52.9%)** |

**Implications**:
1. **Scene-specific memorization**: Model may learn scene-specific patterns (object layouts, common descriptions) that help on val scenes but fail on test scenes' different target objects.
2. **Target distribution mismatch**: Even within the same scene, val and test may query different target objects with different descriptions.
3. **Val optimization leakage**: Protocol alignment tuned on val may exploit scene-specific patterns that don't transfer to test's different queries in overlapping scenes.

---

## 7. Split Methodology Issue

The current split uses **random shuffle** (80/10/10 with seed=42) applied to samples, not scenes. This means:

- Same scene can appear in multiple splits
- No guarantee of scene-disjoint splits
- Violates standard practice in 3D visual grounding (ReferIt3D, ScanRefer use scene-disjoint splits)

**Recommendation**: Investigate whether the official Nr3D dataset has scene-disjoint splits, and if so, adopt them.

---

## 8. Relation Keyword Analysis

Top relation keywords across splits (not significant differences observed):

| Keyword | Train | Val | Test |
|---------|-------|-----|------|
| near | ~30% | ~30% | ~30% |
| left/right | ~25% | ~25% | ~25% |
| front/behind | ~20% | ~20% | ~20% |

---

## 9. Root Cause Hypothesis

**Primary cause of val-test discrepancy**: Scene overlap between splits.

When protocol alignment is tuned on val:
- Model may overfit to scene-specific patterns that happen to work for val's specific queries
- Test's queries for overlapping scenes may target different objects with different descriptions
- Scene-specific patterns don't transfer, causing test degradation

**Secondary causes**:
- Val utterances are slightly longer (more context to learn from)
- Val has higher anchor_confusion rate (model learns to handle these, but test has fewer)
- Candidate size slightly higher in val

---

## 10. Files Inspected

- `data/processed/train_manifest.jsonl`
- `data/processed/val_manifest.jsonl`
- `data/processed/test_manifest.jsonl`
- `data/processed/dataset_summary.json`
- `data/raw/referit3d/annotations/nr3d_annotations.json`
- `src/rag3d/datasets/builder.py` (split_records function)

---

## 11. Next Steps

1. **Verify official Nr3D split methodology**: Check if ReferIt3D/Nr3D uses scene-disjoint splits
2. **Create scene-disjoint splits**: If needed, rebuild data with scene-disjoint splits
3. **Analyze ranking behavior**: Investigate top-1 vs top-5 gap (Step 2)
4. **Test protocol alignment on scene-disjoint splits**: Compare results to current setup

---

## 12. Output Files

- `split_distribution_comparison.json` - Full statistics JSON