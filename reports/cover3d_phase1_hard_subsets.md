# COVER-3D Phase 1: Hard Relational Subsets Definition

**Date**: 2026-04-19
**Split**: official_scene_disjoint (test: 4,255 samples)

---

## Summary Table

| Subset | Definition | Count | Acc@1 | Delta vs Overall |
|--------|------------|-------|-------|------------------|
| **Overall** | All test samples | 4,255 | 31.05% | - |
| Same-Class Clutter | same_class_count ≥ 3 | 2,373 | 21.96% | **-9.09%** |
| High Clutter | same_class_count ≥ 5 | 697 | 16.07% | **-14.98%** |
| Multi-Anchor | anchor_count ≥ 2 | 168 | 11.90% | **-19.15%** |
| Single-Anchor | anchor_count = 1 | 320 | 26.56% | -4.49% |
| Relative Position | has_relative keyword | 1,305 | 28.20% | -2.85% |
| Dense Scene | n_objects ≥ 50 | 752 | 29.39% | -1.66% |
| Sparse Scene | n_objects < 20 | 506 | 27.87% | -3.18% |
| Color Attribute | has_color keyword | 1,047 | 29.04% | -2.01% |
| Unique Class | same_class_count = 1 | - | - | (see below) |

---

## Hard Subset Definitions

### 1. Same-Class Clutter (Primary Hard Subset)

**Definition**: Target object class appears ≥ 3 times in scene.

**Operational**:
```python
same_class_count = count(objects where class_name == target_class)
is_same_class_clutter = same_class_count >= 3
```

**Statistics**:
- Count: 2,373 samples (55.77% of test set)
- Acc@1: 21.96% vs Overall 31.05%
- **Delta**: -9.09%

**Interpretation**: This is the **strongest evidence** for the COVER-3D clutter thesis. Same-class duplicates create ambiguity that the baseline struggles with.

---

### 2. High Same-Class Clutter (Extreme Hard Subset)

**Definition**: Target class appears ≥ 5 times.

**Statistics**:
- Count: 697 samples (16.38%)
- Acc@1: 16.07%
- **Delta**: -14.98%

**Interpretation**: Very strong difficulty signal. At 5+ duplicates, baseline accuracy drops by nearly 15 percentage points.

---

### 3. Multi-Anchor (Relational Hard Subset)

**Definition**: Utterance references ≥ 2 anchor objects (from entities field).

**Statistics**:
- Count: 168 samples (3.95%)
- Acc@1: 11.90%
- **Delta**: -19.15%

**Interpretation**: Multi-anchor relations are extremely hard. The baseline's sparse candidate interactions cannot handle multiple relational references simultaneously.

**Note**: Only 11.47% of samples have entity annotations available. Multi-anchor subset is a subset of those with entities.

---

### 4. Single-Anchor (Moderate Hard Subset)

**Definition**: Exactly 1 anchor object in entities.

**Statistics**:
- Count: 320 samples (7.52%)
- Acc@1: 26.56%
- **Delta**: -4.49%

**Interpretation**: Single-anchor cases also show difficulty, though less extreme than multi-anchor.

---

### 5. Relative Position (Spatial Hard Subset)

**Definition**: Utterance contains relative position keywords ("next to", "near", "by", "beside").

**Statistics**:
- Count: 1,305 samples (30.67%)
- Acc@1: 28.20%
- **Delta**: -2.85%

**Interpretation**: Relative spatial relations are harder than directional ("left/right") which showed no significant drop.

---

### 6. Dense Scene (Scene Complexity Subset)

**Definition**: Scene has ≥ 50 objects.

**Statistics**:
- Count: 752 samples (17.67%)
- Acc@1: 29.39%
- **Delta**: -1.66%

**Interpretation**: Dense scenes show modest difficulty. The candidate pool size affects grounding.

---

## Easy Subset Definitions

### 1. Unique Class (Easy Subset)

**Definition**: Target class appears exactly once (same_class_count = 1).

**Note**: The clutter distribution shows "2" and "3-4" categories but no explicit "1" category. This indicates most samples have some degree of class ambiguity.

From aggregate stats:
- Clutter distribution:
  - 2 (pairs): 1,882 samples
  - 3-4: 1,676 samples
  - 5-6: 697 samples

**Observation**: The dataset is inherently ambiguous - even "easy" cases have at least 2 same-class objects.

---

### 2. Between Relation (Unexpected Easy)

**Definition**: Utterance contains "between".

**Statistics**:
- Count: 129 samples (3.03%)
- Acc@1: 45.74%
- **Delta**: +14.69%

**Interpretation**: "Between" cases are **easier than expected**. This may be because:
1. Two-anchor constraints make the solution unique
2. Spatial between relation has clear geometric interpretation
3. Small subset (129 samples) may have selection bias

---

## Combined Hard Subset

**Definition**: Union of clutter, multi-anchor, dense, or relational flags.

```python
is_hard = (
    same_class_count >= 3 or
    anchor_count >= 2 or
    n_objects >= 50 or
    is_relational  # between, directional, relative, support, container
)
```

**Statistics**:
- Count: 3,579 samples (84.11%)
- Acc@1: 29.20%
- **Delta**: -1.85%

**Note**: Combined subset covers most test samples, showing the dataset is inherently hard.

---

## Subset Categories for Paper Tables

### Recommended Subset Structure

| Category | Subset Name | Count | Acc@1 | Delta |
|----------|-------------|-------|-------|-------|
| **Hard - Clutter** | Same-Class Clutter (≥3) | 2,373 | 21.96% | -9.09% |
| **Hard - Clutter** | High Clutter (≥5) | 697 | 16.07% | -14.98% |
| **Hard - Anchor** | Multi-Anchor (≥2) | 168 | 11.90% | -19.15% |
| **Hard - Anchor** | Single-Anchor (=1) | 320 | 26.56% | -4.49% |
| **Hard - Spatial** | Relative Position | 1,305 | 28.20% | -2.85% |
| **Hard - Scene** | Dense Scene (≥50 obj) | 752 | 29.39% | -1.66% |
| **Easy - Relation** | Between | 129 | 45.74% | +14.69% |
| **Easy - Relation** | Support | 197 | 35.53% | +4.48% |

---

## Key Findings

1. **Clutter Thesis**: STRONGLY SUPPORTED
   - Same-class duplicates cause 9-15% accuracy drop
   - This is the most robust hard-subset signal

2. **Multi-Anchor Thesis**: STRONGLY SUPPORTED
   - Multi-anchor relations have 19% drop
   - Single-anchor also shows 4.5% drop

3. **Coverage Thesis**: PARTIALLY SUPPORTED
   - Entity annotation coverage limited (11.47% samples)
   - Anchor data is incomplete
   - Cannot fully validate anchor-in-top-k hypothesis

4. **Scene Density Thesis**: WEAKLY SUPPORTED
   - Dense scenes show modest 1.66% drop
   - Effect is smaller than clutter/anchor effects

---

## Limitations

1. **Entity annotation coverage**: Only 1,569 annotations matched 4,255 samples
   - Anchor analysis limited to subset with entities
   - May not represent full test set distribution

2. **No geometry**: Cannot compute spatial distance-based metrics
   - Anchor rank by distance unavailable
   - Long-range anchor analysis impossible

3. **Keyword heuristics**: Utterance-based relation detection is imperfect
   - May miss implicit relations
   - May over-classify some cases

---

## Subset Reproducibility

All subsets are defined by deterministic criteria from:
- Manifest `objects` list (class counts, object counts)
- Annotation `entities` field (anchor counts)
- Utterance keyword detection (relation types)

Subset membership is reproducible without manual labeling.