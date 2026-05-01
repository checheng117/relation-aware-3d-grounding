# Qualitative Case Studies Plan

This document plans qualitative case studies for the course report and presentation.

---

## Why Qualitative Cases Matter

For a 95+ course project:
1. **Shows deep understanding** — You can analyze not just aggregate metrics
2. **Makes presentation memorable** — Concrete examples stick better than tables
3. **Demonstrates scientific honesty** — Failure cases show you understand limitations
4. **Fills space professionally** — Better than padding with weak content

---

## Case Study Requirements

| Requirement | Success Case | Failure Case |
|-------------|--------------|--------------|
| **Count** | 1 | 1 |
| **Source** | Baseline wrong → Method correct | Baseline wrong → Method still wrong |
| **Elements** | Utterance, scene view, anchors, target, prediction | Same elements |
| **Analysis** | Why method succeeded | Why method failed |
| **Placement** | Report Section 4.4, Presentation backup | Report Section 5.1, Presentation backup |

---

## Case 1: Success Case (Baseline Wrong → Method Correct)

### Recommended Source

**Source File:** `reports/cover3d_phase1/` diagnostic cases or `outputs/phase4_ablation/` per-sample predictions

**Search Strategy:**
1. Load Phase 4 E0 and E1 predictions
2. Filter samples where E0 wrong, E1 correct
3. Pick one with clear visual structure (easy to draw)

### Case Elements to Show

| Element | Content |
|---------|---------|
| **Utterance** | Full text (e.g., "The chair to the left of the table") |
| **Scene** | Top-down view or perspective snapshot |
| **Anchors** | Highlight all mentioned objects (e.g., table) |
| **Target** | Highlight correct target (e.g., chair) |
| **Baseline Prediction** | Show what baseline picked (wrong) |
| **Method Prediction** | Show what method picked (correct) |
| **Relation Type** | Label (e.g., "directional: left") |
| **Subset** | Label (e.g., "same-class clutter") |

### Analysis Paragraph Template

> **Figure X: Success case.** The utterance describes [TARGET] in relation to [ANCHOR] using [RELATION_TYPE]. The baseline incorrectly selects [BASELINE_PRED], likely due to [REASON: same-class confusion / clutter / etc.]. Our model correctly identifies [TARGET], leveraging [viewpoint conditioning / dense coverage / etc.]. This case illustrates [GENERAL_PRINCIPLE].

### What This Case Demonstrates

- [ ] Method works as intended
- [ ] Specific failure mode addressed (e.g., same-class clutter)
- [ ] Architecture component validated (e.g., viewpoint conditioning helps)

---

## Case 2: Failure Case (Baseline Wrong → Method Still Wrong)

### Recommended Source

**Source File:** Same as above — filter E0 wrong AND E1 wrong

**Search Strategy:**
1. Load Phase 4 E0 and E1 predictions
2. Filter samples where both wrong
3. Pick one with clear reason for failure

### Case Elements to Show

Same as success case, plus:

| Element | Content |
|---------|---------|
| **Hypothesized Reason** | Why both methods fail |
| **What Would Help** | What future method would need |

### Analysis Paragraph Template

> **Figure Y: Failure case.** The utterance describes [TARGET] with [COMPLEXITY]. Both baseline and our model incorrectly select [PRED]. We hypothesize this failure is due to [REASON: e.g., multi-anchor reasoning required / occlusion / ambiguous utterance / missing geometry]. Addressing this would require [FUTURE_WORK: e.g., explicit chain reasoning / better object features / etc.].

### What This Case Demonstrates

- [ ] Honest acknowledgment of limitations
- [ ] Understanding of WHY method fails
- [ ] Concrete direction for future work

---

## Existing Visual Assets

### Available in `reports/cover3d_phase1/`

| File | Content | Reusable? |
|------|---------|-----------|
| `fig1_subset_accuracy.png` | Bar chart of subset accuracy | Yes — already in report |
| `fig2_clutter_impact.png` | Clutter vs accuracy | Yes — appendix |
| `fig5_failure_taxonomy.png` | Failure mode pie chart | Yes — Section 5.1 |
| `fig6_hard_vs_easy.png` | Hard vs easy sample visualization | Maybe — adapt for case |

### Need to Create

| Asset | Purpose | Tool |
|-------|---------|------|
| Success case figure | Report Section 4.4 | matplotlib + 3D visualization or simple diagram |
| Failure case figure | Report Section 5.1 | Same |

---

## Recommended Approach: Simple Diagrams

For course report, **simple is better than complex**:

### Option A: Schematic Diagram (Recommended)

```
Scene: [Scene_784_0]
Utterance: "The blue chair next to the desk"

     ┌─────────┐
     │  DESK   │  ← Anchor (mentioned object)
     │  (red)  │
     └────┬────┘
          │
    ┌─────┴─────┐
    │           │
┌───▼───┐  ┌───▼───┐
│Chair A│  │Chair B│  ← Same-class clutter
│(green)│  │(blue) │     Target is blue
└───────┘  └───────┘

Baseline: Chair A (wrong — same class, ignores color)
Ours:     Chair B (correct — uses color + relation)
```

**Why this works:**
- Clear even without 3D rendering
- Shows the key challenge (same-class clutter)
- Explains why baseline fails, method succeeds

### Option B: Top-Down View (If Available)

Use existing geometry from `data/geometry/` to render simple top-down view with:
- Colored rectangles for objects
- Labels for anchor/target
- Arrow for relation

---

## Case Selection Criteria

| Criterion | Weight | Why |
|-----------|--------|-----|
| **Clarity** | High | Ambiguous cases confuse readers |
| **Typicality** | High | Case should represent common failure/success mode |
| **Visual simplicity** | Medium | Complex scenes hard to draw |
| **Subset representation** | Medium | Pick from important subsets (same-class, multi-anchor) |

---

## Placement Summary

| Case | Report Placement | Presentation Placement |
|------|------------------|------------------------|
| Success | Section 4.4 "Qualitative Analysis" | Backup slide 1 |
| Failure | Section 5.1 "Failure Modes" | Backup slide 2 |

**Note:** If presentation time is tight (10 min), skip case studies in main presentation and keep as backup for Q&A.

---

## Action Items

### P0: Find Cases

- [ ] Extract per-sample predictions from Phase 4 E0/E1
- [ ] Filter success cases (E0 wrong, E1 correct)
- [ ] Filter failure cases (E0 wrong, E1 wrong)
- [ ] Select one of each based on criteria above

### P1: Create Figures

- [ ] Create schematic diagram for success case
- [ ] Create schematic diagram for failure case
- [ ] Save as `writing/course-line/assets/case_success.pdf` and `case_failure.pdf`

### P2: Write Analysis

- [ ] Write 1 paragraph per case
- [ ] Link to method section (why it worked/failed)
- [ ] Add figure captions

---

## Example Case (Template)

### Success Case Example

**Utterance:** "The chair to the left of the desk"

**Scene:** `scene_784_0` (ScanNet scene 784)

**Subset:** Same-class clutter

**Baseline prediction:** Chair #1 (right of desk) — WRONG

**Method prediction:** Chair #2 (left of desk) — CORRECT

**Why baseline failed:** Baseline ranks anchors by distance, ignoring directional relation "left of"

**Why method succeeded:** Viewpoint-conditioned scorer encodes directional relation, correctly ranks Chair #2 higher

**Figure description:** Top-down view showing desk (anchor) in center, two chairs (same class) on opposite sides. Baseline picks closer chair; method picks left chair as specified.

---

## Files Referenced

- `outputs/phase4_ablation/viewpoint-conditioned_results.json`
- `outputs/phase4_ablation/dense-v4-hardneg_results.json`
- `reports/cover3d_phase1/`
- `data/geometry/`
- `writing/course-line/assets/`
