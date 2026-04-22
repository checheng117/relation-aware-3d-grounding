# Formal Round-1 Case Analysis

**Date**: 2026-04-20

Since learned experiments failed, case analysis is limited to Base experiment and comparison with proxy P3.

---

## Base Experiment Cases

Base experiment produced 1108 correct predictions out of 4255 (26.04%).

Expected from clean baseline: 1312 correct (30.83%).

**Gap analysis**:
- 204 fewer correct predictions than expected
- Suggests extracted embeddings don't match baseline inference

---

## Proxy P3 Cases (For Reference)

From proxy P3 entry, notable cases:

### Recovery Examples

| Scene | Target | Base Pred | Dense Pred | Gate | Utterance |
| --- | --- | --- | --- | --- | --- |
| scene0340_00 | 31 | 28 | 31 | 0.03 | lamp on floor next to curtains |
| scene0058_00 | 28 | 10 | 28 | 0.03 | desk far from blue door |
| scene0640_00 | 2 | 1 | 2 | 0.03 | lamp over the desk |

### Harmed Examples

| Scene | Target | Base Pred | Dense Pred | Gate | Utterance |
| --- | --- | --- | --- | --- | --- |
| (from proxy: 22 harmed cases) | | | | | |

**Note**: Proxy P3 uses oracle-anchor geometry relation scores, not learned dense scorer. These cases show potential recovery patterns but do not prove learned method effectiveness.

---

## Issues Preventing Learned Case Analysis

1. **NaN collapse**: DenseRelationModule produces NaN scores immediately
2. **Base mismatch**: Cannot compare "recovered from base-wrong" when baseline predictions differ

---

## Recommended Next Steps for Case Analysis

1. Fix embedding extraction to match baseline predictions
2. Add input normalization to DenseRelationModule
3. Retry learned experiments
4. Then perform detailed case-by-case analysis of:
   - Recovered cases (base wrong → learned correct)
   - Harmed cases (base correct → learned wrong)
   - Hard subset breakdown
   - Gate behavior analysis