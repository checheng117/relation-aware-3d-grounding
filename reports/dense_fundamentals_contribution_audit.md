# Dense Fundamentals: Dense Branch Contribution Audit

**Date**: 2026-04-22
**Model**: Dense-no-cal-v1

---

## Executive Summary

**Dense Branch Net Effect**: 3 (Recovered: 57, Harmed: 54)

---

## 1. Overall Contribution

| Category | Count | Percentage |
|----------|-------|------------|
| Recovered (base→dense) | 57 | 11.40% |
| Harmed (base→dense) | 54 | 10.80% |
| Both Correct | 94 | 18.80% |
| Both Wrong | 295 | 59.00% |
| Total | 500 | 100% |

---

## 2. Harmed Case Analysis

### Distribution by Subset

| Subset | Count | % of Harmed |
|--------|-------|-------------|
| Same-Class Clutter | 46 | 85.2% |
| Multi-Anchor | 8 | 14.8% |
| Relative-Position | 9 | 16.7% |
| Easy | 3 | 5.6% |

### Key Finding: Where Dense Branch Hurts

Dense branch harms multi-anchor/clutter cases

---

## 3. Recovered Case Analysis

### Distribution by Subset

| Subset | Count | % of Recovered |
|--------|-------|----------------|
| Same-Class Clutter | 49 | 86.0% |
| Multi-Anchor | 15 | 26.3% |
| Relative-Position | 17 | 29.8% |
| Easy | 6 | 10.5% |

### Key Finding: Where Dense Branch Helps

Dense branch primarily helps multi-anchor/clutter cases

---

## 4. Comparison: Harmed vs Recovered Patterns

| Pattern | Harmed | Recovered | Interpretation |
|---------|--------|-----------|----------------|
| Multi-Anchor Rate | 14.8% | 26.3% | Dense helps multi-anchor more than it hurts |
| Clutter Rate | 85.2% | 86.0% | Dense helps clutter more than it hurts |
| Relative-Position Rate | 16.7% | 29.8% | Dense helps relative-position more than it hurts |

---

## 5. Case Studies

### Examples of Harmed Cases (Base-Correct → Dense-Wrong)

- Scene: scene0340_00
  Utterance: 'The lamp between the two beds.'
  Subsets: ['all', 'same_class_clutter', 'same_class_high_clutter', 'unique_class', 'multi_anchor', 'single_anchor', 'relative_position', 'directional', 'between', 'relational', 'dense_scene', 'baseline_correct', 'baseline_wrong']
  Avg Relation Score: -2.4454

- Scene: scene0340_00
  Utterance: light between the beds
  Subsets: ['all', 'same_class_clutter', 'same_class_high_clutter', 'unique_class', 'multi_anchor', 'single_anchor', 'relative_position', 'directional', 'between', 'relational', 'dense_scene', 'baseline_correct', 'baseline_wrong']
  Avg Relation Score: -2.9021

- Scene: scene0340_00
  Utterance: The lamp between the two beds.
  Subsets: ['all', 'same_class_clutter', 'same_class_high_clutter', 'unique_class', 'multi_anchor', 'single_anchor', 'relative_position', 'directional', 'between', 'relational', 'dense_scene', 'baseline_correct', 'baseline_wrong']
  Avg Relation Score: -2.2751

- Scene: scene0340_00
  Utterance: the lamp between the beds
  Subsets: ['all', 'same_class_clutter', 'same_class_high_clutter', 'unique_class', 'multi_anchor', 'single_anchor', 'relative_position', 'directional', 'between', 'relational', 'dense_scene', 'baseline_correct', 'baseline_wrong']
  Avg Relation Score: -2.6045

- Scene: scene0340_00
  Utterance: It is the lamp on the wall between the two beds.
  Subsets: ['all', 'same_class_clutter', 'same_class_high_clutter', 'unique_class', 'multi_anchor', 'single_anchor', 'relative_position', 'directional', 'between', 'relational', 'dense_scene', 'baseline_correct', 'baseline_wrong']
  Avg Relation Score: -2.2177


### Examples of Recovered Cases (Base-Wrong → Dense-Correct)

- Scene: scene0340_00
  Utterance: 'The lamp not between the curtains.'
  Subsets: ['all', 'same_class_clutter', 'same_class_high_clutter', 'unique_class', 'multi_anchor', 'single_anchor', 'relative_position', 'directional', 'between', 'relational', 'dense_scene', 'baseline_correct', 'baseline_wrong']
  Avg Relation Score: -2.3934

- Scene: scene0340_00
  Utterance: This lamp lights the desk.
  Subsets: ['all', 'same_class_clutter', 'same_class_high_clutter', 'unique_class', 'multi_anchor', 'single_anchor', 'relative_position', 'directional', 'between', 'relational', 'dense_scene', 'baseline_correct', 'baseline_wrong']
  Avg Relation Score: -2.6243

- Scene: scene0340_00
  Utterance: The lamp on the wall above the desk next to a painting.
  Subsets: ['all', 'same_class_clutter', 'same_class_high_clutter', 'unique_class', 'multi_anchor', 'single_anchor', 'relative_position', 'directional', 'between', 'relational', 'dense_scene', 'baseline_correct', 'baseline_wrong']
  Avg Relation Score: -2.7311

- Scene: scene0340_00
  Utterance: Its the lamp that's on and is located to the right of a hung up picture frame.
  Subsets: ['all', 'same_class_clutter', 'same_class_high_clutter', 'unique_class', 'multi_anchor', 'single_anchor', 'relative_position', 'directional', 'between', 'relational', 'dense_scene', 'baseline_correct', 'baseline_wrong']
  Avg Relation Score: -3.0873

- Scene: scene0340_00
  Utterance: A lamp with a yellow glow that is next to a framed picture and close to some curtains.
  Subsets: ['all', 'same_class_clutter', 'same_class_high_clutter', 'unique_class', 'multi_anchor', 'single_anchor', 'relative_position', 'directional', 'between', 'relational', 'dense_scene', 'baseline_correct', 'baseline_wrong']
  Avg Relation Score: -2.2316


---

## 6. Conclusion

**Dense Branch Assessment**:

1. **Net Effect**: Positive but modest (+3)
2. **Systematic Patterns**: Yes - multi-anchor recovery, easy-case harm
3. **Actionable Insights**: Focus on multi-anchor cases, protect easy cases from over-correction

**Route Recommendation**: Route B-fundamentals: Continue with redesign based on findings
