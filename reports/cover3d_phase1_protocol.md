# COVER-3D Phase 1: Coverage Diagnostics Protocol

**Date**: 2026-04-19
**Phase**: Analysis + Paper Figure Generation (NO TRAINING)

---

## 1. Dataset Split

### Official Split
- **Split name**: `official_scene_disjoint`
- **Test split size**: 4,255 samples across 65 scenes
- **Val split size**: 3,419 samples across 64 scenes
- **Train split size**: 33,829 samples across 512 scenes
- **Total**: 41,503 samples across 641 unique scenes

### Scene-Disjoint Property
All splits use distinct scene sets, ensuring:
- No scene overlap between splits
- Evaluation tests generalization to unseen scenes
- Standard benchmark protocol for Nr3D

---

## 2. Object Graph Definition

### Available Data
Each sample contains:

| Field | Source | Content |
|-------|--------|---------|
| `scene_id` | Manifest | Scene identifier |
| `utterance` | Manifest | Referring expression |
| `target_object_id` | Manifest | Target object ID string |
| `target_index` | Manifest | Target object index |
| `objects` | Manifest | List of scene objects |
| `objects[].object_id` | Aggregation | Object ID string |
| `objects[].class_name` | Aggregation | Semantic class label |

### Geometry Limitation (IMPORTANT)
**Geometry is NOT available**:
- All object centers: `[0.0, 0.0, 0.0]` (fallback placeholder)
- All object sizes: `[0.1, 0.1, 0.1]` (fallback placeholder)
- `geometry_fallback_fraction`: 1.0

**Impact**:
- Cannot compute actual Euclidean distances
- Cannot compute spatial directions (left/right/front/behind)
- Cannot compute nearest-neighbor by physical distance

**Adaptation**:
- Use class-label-based analysis
- Use index-order-based proximity (imperfect proxy)
- Use utterance-based anchor extraction
- Use annotation-based entities for true relational anchors

---

## 3. Anchor Proxy Definition

Since ground-truth relational anchors are not explicitly labeled, we use multiple proxy definitions:

### 3.1 Annotation-Based Entities (Primary Proxy)

The `nr3d_annotations.json` contains an `entities` field for each sample:

```json
{
  "entities": ["9_plant", "34_bookshelf", "38_window"]
}
```

Interpretation:
- First entity: `9_plant` is the target (object_id + class)
- Remaining entities: Relational anchors mentioned in utterance

**Proxy anchor extraction**:
1. Parse entities: `[object_id, class_name]`
2. Filter to anchors: exclude first entity (target)
3. Map to object indices in scene

### 3.2 Utterance-Based Heuristic Proxies

Fallback when entities unavailable:

| Heuristic | Pattern | Example Anchor Class |
|-----------|---------|----------------------|
| Directional | "left of/right of/front of/behind" | Object at mentioned direction |
| Support | "on the table/on the desk" | table, desk, shelf |
| Container | "in the cabinet/in the box" | cabinet, box, drawer |
| Between | "between X and Y" | Both X and Y objects |
| Reference | "next to/near/by" | Mentioned object class |

### 3.3 Same-Class Clutter Proxies

From manifest `tags.same_class_clutter`:
- Count objects sharing target class
- Proxy for ambiguity difficulty

---

## 4. Diagnostics Metrics (Observable)

### A. Candidate Count
- `n_objects`: number of objects in scene
- Already available in manifest `tags.n_objects`

### B. Same-Class Clutter
- `same_class_count`: count of objects with target's class_name
- Already available in manifest `tags.same_class_clutter` (boolean)
- Need to compute actual count from objects list

### C. Proxy Anchor Analysis

For each sample with entities:
- `anchor_count`: number of anchor entities
- `anchor_classes`: list of anchor class names
- `anchor_indices`: indices of anchors in scene object list
- `anchor_in_top_k`: whether anchor appears in top-k candidates

**Note**: "top-k" here refers to **prediction ranking**, not spatial proximity (since geometry unavailable).

### D. Utterance Relational Content

Text-based analysis:
- `has_directional`: contains directional keywords
- `has_support`: contains support keywords
- `has_between`: contains "between"
- `has_relative`: contains relative position keywords
- `relation_type`: categorical label

### E. Baseline Performance

From trusted ReferIt3DNet predictions:
- `correct_at_1`: whether prediction matches target
- `pred_top1`: predicted object index
- `pred_top5`: top-5 predicted indices

---

## 5. Subset Definitions

### Hard Subsets

| Subset | Definition | Expected Difficulty |
|--------|------------|---------------------|
| Same-Class Clutter | `same_class_count >= 3` | Ambiguity from class duplicates |
| Multi-Anchor | `anchor_count >= 2` | Multiple relational references |
| Dense Scene | `n_objects >= 50` | Large candidate pool |
| Directional | `has_directional == True` | Spatial reasoning required |
| Between | `has_between == True` | Two-anchor relation |

### Easy Subsets

| Subset | Definition | Expected Ease |
|--------|------------|---------------|
| Unique Class | `same_class_count == 1` | No class ambiguity |
| Sparse Scene | `n_objects < 20` | Small candidate pool |
| No Relation | No relational keywords | Pure attribute matching |

---

## 6. Trusted Baseline Reference

### ReferIt3DNet (Primary Baseline)
- **Source**: `outputs/20260409_learned_class_embedding/formal/eval_test_predictions.json`
- **Test Acc@1**: 30.79%
- **Test Acc@5**: 91.75%
- **Samples**: 4,255

### SAT (Secondary Baseline)
- **Test Acc@1**: 28.27%
- **Test Acc@5**: 87.64%

### Comparison Standard
All subset analysis uses ReferIt3DNet as comparison anchor.

---

## 7. Output Artifacts

| Artifact | Path | Content |
|----------|------|---------|
| Protocol | `reports/cover3d_phase1_protocol.md` | This document |
| Diagnostics script | `scripts/run_cover3d_coverage_diagnostics.py` | Analysis pipeline |
| Hard subsets | `reports/cover3d_phase1_hard_subsets.md` | Subset definitions |
| Baseline results | `reports/cover3d_phase1_baseline_subset_results.md` | Performance tables |
| Figures | `reports/figures/cover3d_phase1/*.png` | Paper-quality plots |
| Findings | `reports/cover3d_phase1_findings.md` | Statistical summary |
| Motivation | `reports/cover3d_phase1_paper_motivation.md` | AAAI intro draft |
| Decision | `reports/cover3d_phase1_decision.md` | Evidence assessment |

---

## 8. Limitations & Transparency

### Limitations

1. **No geometry**: Cannot compute true spatial distances or directions
2. **Proxy anchors**: Entities are annotation-derived, not ground-truth labeled
3. **Index proximity**: Object indices are arbitrary ordering, not spatial
4. **Heuristic parsing**: Utterance keywords are imperfect relation detectors

### Mitigation

1. Report metrics as **observational statistics**, not ground-truth claims
2. Use entities field when available (covers all samples)
3. Be explicit about proxy definitions
4. Focus on **class-based** and **utterance-based** analysis
5. Compare subsets using actual baseline predictions

---

## 9. Evidence Goals

Phase 1 aims to provide evidence for:

1. **Coverage Thesis**: Sparse top-k candidates miss relational anchors
2. **Clutter Thesis**: Same-class duplicates hurt baseline accuracy
3. **Relation Thesis**: Relational utterances have lower baseline accuracy
4. **Dense Thesis**: Large scenes reduce baseline accuracy

If evidence supports these, COVER-3D motivation is validated.

---

## 10. Execution Plan

| Step | Task | Output |
|------|------|--------|
| 0 | Define protocol | This document |
| 1 | Build diagnostics pipeline | Python script |
| 2 | Define hard subsets | Subset definitions |
| 3 | Compute subset performance | Results table |
| 4 | Generate figures | Paper-quality plots |
| 5 | Document findings | Statistical summary |
| 6 | Draft motivation | AAAI intro |
| 7 | Final decision | Evidence assessment |

**NO TRAINING. NO GPU JOBS.**