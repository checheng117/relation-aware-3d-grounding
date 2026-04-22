# COVER-3D Phase 3: Subset Evaluation Specification

**Date**: 2026-04-19
**Purpose**: Define exact evaluation logic for hard subsets

---

## 1. Input Requirements

### Primary Input

| Input | Format | Required |
|-------|--------|----------|
| Predictions | JSONL file or checkpoint | Either |
| Manifest | `test_manifest.jsonl` | Required |

### Prediction File Format

```json
{
  "sample_id": "scene0001_00_obj1",
  "predicted_index": 5,
  "target_index": 7,
  "correct": false,
  "base_predicted_index": 3,
  "gate_value": 0.35
}
```

### Checkpoint Format

If checkpoint provided, run inference on test split first:
1. Load model
2. Iterate test manifest
3. Generate predictions
4. Save to prediction file

---

## 2. Subset Definitions (Exact Logic)

### Same-Class Clutter (≥3)

**Definition**: Target object has ≥3 same-class objects in scene

**Logic**:
```python
def is_same_class_clutter_3(sample):
    """Check if target has ≥3 same-class objects."""
    target_class = sample["objects"][sample["target_index"]]["class_name"]
    same_class_count = sum(
        1 for obj in sample["objects"]
        if obj["class_name"] == target_class
    )
    return same_class_count >= 3
```

**Baseline**: 21.96% Acc@1
**Target**: ≥25% Acc@1 (+3 points)

---

### High Clutter (≥5)

**Definition**: Target object has ≥5 same-class objects in scene

**Logic**:
```python
def is_high_clutter_5(sample):
    """Check if target has ≥5 same-class objects."""
    target_class = sample["objects"][sample["target_index"]]["class_name"]
    same_class_count = sum(
        1 for obj in sample["objects"]
        if obj["class_name"] == target_class
    )
    return same_class_count >= 5
```

**Baseline**: 16.07% Acc@1
**Target**: ≥20% Acc@1 (+4 points)

---

### Multi-Anchor

**Definition**: Annotation mentions ≥2 anchor objects

**Logic**:
```python
def is_multi_anchor(sample):
    """Check if annotation has ≥2 anchors."""
    # entities field: ["target_id_class", "anchor1_id_class", ...]
    entities = sample.get("entities", [])
    if not entities:
        return False  # No entity annotation
    anchor_count = len(entities) - 1  # Exclude target
    return anchor_count >= 2
```

**Baseline**: 11.90% Acc@1
**Target**: ≥20% Acc@1 (+8 points)

**Note**: Entity annotations cover ~11% of samples (473/4255)

---

## 3. Evaluation Metrics

### Per-Subset Metrics

| Metric | Definition |
|--------|------------|
| Acc@1 | Correct predictions / subset size |
| Acc@5 | Target in top-5 predictions / subset size |
| Size | Number of samples in subset |
| Coverage | Subset size / total test size |

### Overall Metrics

| Metric | Definition |
|--------|------------|
| Overall Acc@1 | Correct / total |
| Overall Acc@5 | Top-5 correct / total |

### Delta Metrics

| Metric | Definition |
|--------|------------|
| Δ Overall | Cover-3D Acc@1 - Baseline Acc@1 |
| Δ Clutter | Cover-3D clutter Acc@1 - Baseline clutter Acc@1 |
| Δ High Clutter | Cover-3D high clutter Acc@1 - Baseline high clutter Acc@1 |
| Δ Multi-Anchor | Cover-3D multi-anchor Acc@1 - Baseline multi-anchor Acc@1 |

---

## 4. Output Format

### JSON Output

```json
{
  "overall": {
    "acc1": 32.5,
    "acc5": 92.1,
    "total": 4255,
    "correct": 1383
  },
  "same_class_clutter_3": {
    "acc1": 26.2,
    "acc5": 88.5,
    "size": 1847,
    "coverage": 0.434,
    "correct": 484,
    "baseline_acc1": 21.96,
    "delta": 4.24
  },
  "high_clutter_5": {
    "acc1": 21.3,
    "acc5": 82.1,
    "size": 523,
    "coverage": 0.123,
    "correct": 111,
    "baseline_acc1": 16.07,
    "delta": 5.23
  },
  "multi_anchor": {
    "acc1": 22.5,
    "acc5": 78.4,
    "size": 473,
    "coverage": 0.111,
    "correct": 106,
    "baseline_acc1": 11.90,
    "delta": 10.6
  }
}
```

### Markdown Report

Include:
- Summary table with all metrics
- Delta comparison table
- Statistical significance (if applicable)
- Subset size breakdown

---

## 5. Implementation Requirements

### Script: `scripts/evaluate_cover3d_subsets.py`

**Command-line interface**:
```bash
python scripts/evaluate_cover3d_subsets.py \
    --predictions <predictions.jsonl> \
    --manifest <test_manifest.jsonl> \
    --output <results.json> \
    --report <report.md>
```

**Alternative checkpoint mode**:
```bash
python scripts/evaluate_cover3d_subsets.py \
    --checkpoint <model.pt> \
    --manifest <test_manifest.jsonl> \
    --output <results.json>
```

**Features**:
1. Load predictions or run inference from checkpoint
2. Compute overall Acc@1, Acc@5
3. Compute per-subset Acc@1, Acc@5
4. Compare to baseline numbers
5. Output JSON and markdown report

---

## 6. Baseline Numbers (Reference)

| Subset | Baseline Acc@1 | Baseline Acc@5 | Target |
|--------|---------------|---------------|--------|
| Overall | 30.79% | 91.75% | ≥32% |
| Same-Class Clutter ≥3 | 21.96% | — | ≥25% |
| High Clutter ≥5 | 16.07% | — | ≥20% |
| Multi-Anchor | 11.90% | — | ≥20% |

---

## 7. Decision Thresholds

| Criterion | Threshold | Action |
|-----------|-----------|--------|
| Overall Acc@1 ≥ baseline - 2% | ≥28.79% | Proceed to Stage C |
| Hard subset improvement | Any positive delta | Proceed to Stage C |
| Overall Acc@1 < 28.79% | Failed | Revise training |
| All deltas negative | Failed | Debug calibration |

---

## Final Statement

**Specification frozen. Implementation may proceed.**

**Script**: `scripts/evaluate_cover3d_subsets.py`

**Key constraint**: Exact subset logic, baseline comparison, JSON + markdown output.