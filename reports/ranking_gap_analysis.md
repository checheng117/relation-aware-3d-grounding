# Ranking Gap Analysis Report

**Date**: 2026-04-06
**Phase**: Distribution Mismatch Investigation - Step 2

---

## Executive Summary

The **top-1 vs top-5 accuracy gap** reveals a fundamental discrimination problem:
- Correct answers are often retrieved (in top-5) but not ranked correctly (not top-1)
- PointNet++ shows the largest gap (49%), suggesting improved recall but poor discrimination
- Protocol alignment reduces gap (26.45%) but also reduces overall accuracy

---

## 1. Top-1 vs Top-5 Gap by Configuration

| Configuration | Test Acc@1 | Test Acc@5 | Gap (Acc@5 - Acc@1) | Gap % |
|--------------|------------|------------|---------------------|-------|
| Baseline | 9.68% | 40.00% | 30.32% | 312% |
| PointNet++ | 10.97% | 60.00% | **49.03%** | **447%** |
| Protocol alignment | 3.87% | 30.32% | 26.45% | 686% |

**Key observations**:
1. **PointNet++** has the largest absolute gap (49%) and the best top-5 (60%)
   - It finds the correct answer in top-5 for 60% of test samples
   - But only ranks it as top-1 for 10.97%
   - **Interpretation**: Good recall, poor discrimination

2. **Protocol alignment** has smallest gap (26.45%) but worst top-5 (30.32%)
   - Reduced retrieval capability overall
   - Catastrophic test degradation

3. **Baseline** has moderate gap (30.32%) with poor top-5 (40%)
   - Both retrieval and discrimination are weak

---

## 2. Hypothesis: Discrimination Failure

The massive top-1/top-5 gap suggests:

### 2.1 Score Margin Problem
When correct answer is in top-5:
- The score margin between correct and top-predicted is likely small
- Model cannot distinguish the true target from near-correct distractors
- Similar objects (same class) get similar scores

### 2.2 Same-Class Clutter Impact
With 99.4% same-class clutter rate:
- Nearly every sample has distractors of the same class as target
- Distractors may have similar geometric/visual features
- Language grounding must discriminate via spatial relations or subtle attributes
- Current model may not adequately use relation signals

### 2.3 Relation Encoding Weakness
If the model primarily relies on:
- Object class similarity
- Geometric proximity
- Visual similarity

It will fail to distinguish:
- "the chair left of the table" vs "the chair right of the table"
- Same-class chairs with similar geometry

---

## 3. Why PointNet++ Improves Top-5 but Not Top-1

PointNet++ provides:
- Better point-level geometric encoding
- More discriminative object features

**Effect on top-5**:
- Better object features → better similarity matching
- More likely to include correct answer in candidate pool

**Effect on top-1**:
- Same-class objects still have similar geometry
- Relation encoding unchanged
- Discrimination between similar objects still weak

**Conclusion**: PointNet++ improves feature quality but doesn't address relation-aware discrimination.

---

## 4. Why Protocol Alignment Hurts Test

Protocol alignment (presumably matching ReferIt3D training protocol):
- May include different learning rate, batch size, scheduler
- May change loss function (ranking vs classification)

**Val improvement (22.73% → 27.27%)**:
- Val set may have easier relation patterns
- Or val samples from scenes seen in train, enabling scene-specific learning

**Test degradation (9.68% → 3.87%)**:
- Test samples from different scenes or different target queries in overlapping scenes
- Protocol optimization may overfit to val's specific score distribution
- Ranking loss may cause unstable generalization

---

## 5. Implications for Next Experiment

Based on ranking gap analysis:

### Option A: PointNet++ + Mild Protocol Alignment
- Combine best encoder with mild protocol changes
- Risk: Protocol alignment may still cause test degradation

### Option B: PointNet++ + Ranking/Fusion Refinement
- Address discrimination problem directly
- Focus on score margins, relation encoding, or ensemble methods
- Could reduce top-1/top-5 gap

### Option C: Further Data/Split Investigation
- Scene overlap is still the primary concern
- Ranking gap may be secondary to split methodology issue

---

## 6. Recommended Focus

**Primary**: Fix scene overlap issue (Step 3)
- Scene-disjoint splits are essential for valid evaluation
- Current scene overlap may invalidate all ranking conclusions

**Secondary**: After scene-disjoint splits:
- Re-evaluate ranking gap on clean splits
- Investigate score margin distributions
- Consider relation-aware discrimination improvements

---

## 7. Files Inspected

- `src/rag3d/evaluation/metrics.py` (logit_top12_margin, accuracy_at_k)
- `src/rag3d/evaluation/evaluator.py`
- `.claude/CURRENT_STATUS.md` (result values)

---

## 8. Limitations

- No actual prediction scores analyzed (evaluation outputs not inspected)
- Analysis is theoretical based on reported accuracy values
- Score margin distributions need actual prediction data for full analysis

---

## 9. Next Steps

1. Complete Step 3: Generalization gap analysis
2. Finalize recommendation for single next experiment
3. Consider running evaluation with score logging for future analysis