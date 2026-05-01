# Presentation Plan: CSC6133 Final Project (10-15 minutes)

This document outlines the slide-by-slide structure for the final presentation.

---

## Presentation Constraints

| Constraint | Value |
|------------|-------|
| **Time** | 10-15 minutes |
| **Slides** | ~12-18 slides (1 min per slide average) |
| **Audience** | Course instructors, TAs, classmates |
| **Goal** | Clear, memorable, defensible presentation |

---

## Slide Overview

| Section | Slides | Time |
|---------|--------|------|
| 1. Title + Hook | 1 | 0.5 min |
| 2. Problem + Motivation | 2-3 | 2 min |
| 3. Method | 2-3 | 3 min |
| 4. Results | 3-4 | 4 min |
| 5. Analysis | 2 | 2 min |
| 6. Conclusion | 1-2 | 1.5 min |
| Backup | 3-5 | Q&A |

---

## Slide-by-Slide Plan

### Slide 1: Title Slide

**Title:** Latent Conditioned Relation Scoring for 3D Visual Grounding

**Subtitle:** CSC6133 Final Project Presentation

**Content:**
- Your name, email
- Course name, date
- Optional: Simple architecture teaser diagram

**Speaker Notes:**
- "Today I'll present my final project on improving 3D visual grounding through latent conditioned relation scoring."

---

### Slide 2: Problem Statement

**Title:** The Challenge of 3D Visual Grounding

**Content:**
- Definition: "Given an utterance (e.g., 'the chair left of the table'), find the target object in a 3D scene"
- Key challenge: Ambiguous relational expressions
- Example: "the chair" — which chair?

**Visual:**
- Left: Sample utterance
- Right: 3D scene with multiple chairs

**Core Message:** 3D grounding requires reasoning about ambiguous spatial relations

**Speaker Notes:**
- "3D visual grounding is the task of locating objects described by natural language. The core challenge is ambiguity — when there are multiple objects of the same class, how do we know which one?"

---

### Slide 3: Baseline Failure Modes (Motivation)

**Title:** Where Do Baselines Fail?

**Content:**
- Table: Baseline accuracy by subset
  - Overall: 30.83%
  - Same-class clutter: 21.96%
  - High clutter: 16.07%
  - Multi-anchor: 11.90%

**Visual:** Bar chart (from `fig1_subset_accuracy.png`)

**Core Message:** Baselines fail systematically on hard cases

**Speaker Notes:**
- "I analyzed where the standard ReferIt3DNet baseline fails. On easy cases it does fine, but on hard cases — same-class clutter, high clutter, multi-anchor — accuracy drops to 12-22%. This tells us exactly where to improve."

---

### Slide 4: Coverage Failure (Motivation)

**Title:** Why Baselines Fail: Coverage Gap

**Content:**
- Coverage@k table:
  - k=5: 67.87% (any anchor), 54.20% (all anchors)
- Key stat: Sparse top-5 misses ALL anchors in 33.95% of baseline-wrong samples

**Visual:** Coverage curve (k on x-axis, coverage% on y-axis)

**Core Message:** Sparse anchor selection misses crucial evidence

**Speaker Notes:**
- "The baseline uses sparse nearest-neighbor anchor selection. But this misses annotated anchors in 34% of error cases. Dense coverage is necessary — though not sufficient — for improvement."

---

### Slide 5: Method Overview

**Title:** Latent Conditioned Relation Scoring

**Content:**
- Core idea: Condition relation scores on latent variables
- Architecture: Viewpoint-conditioned MLP
- Key insight: Architecture helps, explicit supervision doesn't

**Visual:** Architecture diagram (embeddings → scorer → conditioning → output)

**Core Message:** Latent conditioning enables multi-mode relation reasoning

**Speaker Notes:**
- "My method introduces latent conditioned relation scoring. The key is the architecture — conditioning relation scores on latent variables. Interestingly, the architecture itself helps, even without explicit viewpoint supervision."

---

### Slide 6: Method Details

**Title:** Architecture Design

**Content:**
- Input: Object embeddings (320-dim)
- Relation scorer: Pairwise MLP
- Viewpoint conditioning: Concatenate predicted viewpoint embedding
- Output: Score for each candidate anchor

**Visual:** Detailed architecture diagram with dimensions

**Core Message:** Simple but effective modification to relation scoring

**Speaker Notes:**
- "The architecture takes object embeddings, computes pairwise relation features, and conditions on viewpoint. The viewpoint can be predicted or learned — experiments show prediction isn't necessary."

---

### Slide 7: Main Results

**Title:** Main Results

**Content:**
| Method | Acc@1 | Acc@5 |
|--------|-------|-------|
| Baseline | 30.83% | 91.87% |
| Dense-no-cal-v1 | 31.05% | 92.01% |
| Phase 4 E0 | 28.60% | — |
| Phase 4 E1 | 30.74% | — |

**Visual:** Bar chart comparison

**Core Message:** Method improves over baseline (+2.1% in controlled setting)

**Speaker Notes:**
- "Here are the main results. The dense diagnostic shows a small gain. In the controlled Phase 4 setting, the viewpoint-conditioned architecture gives +2.1% over the matched baseline."

---

### Slide 8: Ablation Study

**Title:** Where Does the Gain Come From?

**Content:**
| Experiment | Acc@1 | Gain |
|------------|-------|------|
| E0 Baseline | 28.60% | — |
| E1 + Viewpoint Supervision | 30.74% | +2.14% |
| E2 Parameter-Matched | 28.74% | +0.14% |
| E3 No Supervision | 30.65% | +2.04% |

**Core Message:** Gain is from architecture, NOT viewpoint supervision

**Speaker Notes:**
- "This ablation is crucial. E1 shows the full method. E3 removes viewpoint supervision — and performance is identical. This means the gain comes from the architecture itself, not the supervision signal."

---

### Slide 9: Per-Relation-Type Analysis

**Title:** Does It Help All Relations?

**Content:**
| Relation Type | Gain |
|---------------|------|
| between | +3.88% |
| directional | +1.95% |
| support | +2.30% |
| attribute | +1.67% |

**Core Message:** Gain is general, not relation-specific

**Speaker Notes:**
- "The gain is consistent across relation types — between, directional, support, attribute. This confirms the architecture provides general improvement, not just for viewpoint-dependent cases."

---

### Slide 10: Qualitative Success Case

**Title:** Success Case

**Content:**
- Utterance: "The chair left of the desk"
- Baseline: Wrong chair (right side)
- Method: Correct chair (left side)

**Visual:** Schematic diagram (see `QUAL_CASES_PLAN.md`)

**Core Message:** Method correctly uses directional relation

**Speaker Notes:**
- "Here's a concrete success case. The utterance says 'left of the desk.' Baseline picks the closer chair; method picks the left chair as specified."

---

### Slide 11: Pilot Results (Optional)

**Title:** Ongoing Work: Latent Relation Modes

**Content:**
- K=1 (single mode): 34.10%
- K=4 (MoE): 34.19% (+0.09 pp at 1 epoch)
- Status: Pilot, full training pending

**Core Message:** Early signals promising, validation ongoing

**Speaker Notes:**
- "I'm also exploring latent relation modes — a mixture-of-experts formulation. Early pilots show small gains; full training is ongoing. I'm transparent that this is pilot evidence."

---

### Slide 12: Limitations

**Title:** Limitations

**Content:**
- Multi-anchor: No improvement (24.73% → 24.73%)
- RHN ablation: Inconclusive (metadata format issue)
- Single-seed: Stability not validated
- Protocol differences: Some numbers not directly comparable

**Core Message:** Honest about what's NOT solved

**Speaker Notes:**
- "Important limitations: Multi-anchor cases remain challenging. The RHN control couldn't activate due to data format. All results are single-seed. I'm being transparent about these boundaries."

---

### Slide 13: Conclusion

**Title:** Conclusion

**Content:**
- Problem: Ambiguous relational grounding
- Method: Latent conditioned relation scoring
- Result: +2.1% gain from architecture (not supervision)
- Future: Full latent-mode training, multi-seed validation

**Core Message:** Architecture enables multi-mode reasoning; supervision not required

**Speaker Notes:**
- "In summary: 3D grounding requires relation reasoning. Latent conditioned scoring provides +2.1% gain. The architecture itself is the contribution — explicit supervision isn't necessary. Future work includes full latent-mode validation."

---

### Slide 14: Acknowledgments + Q&A

**Title:** Thank You

**Content:**
- Acknowledge course, instructor, any help
- "Questions?"
- Optional: QR code to repo/reproducibility notes

**Speaker Notes:**
- "Thank you. Happy to take questions!"

---

## Backup Slides

### Backup 1: Coverage Diagnostics Detail

For questions about coverage hypothesis

---

### Backup 2: Counterfactual Pilot

For questions about counterfactual loss

---

### Backup 3: Failure Case

For questions about limitations

---

### Backup 4: Training Curves

For questions about convergence

---

### Backup 5: Extended Subsets

For detailed subset questions

---

## Design Principles

### Visual Style

| Element | Recommendation |
|---------|----------------|
| **Font** | Sans-serif (Arial, Helvetica), 24pt+ body, 36pt+ titles |
| **Colors** | High contrast, colorblind-safe palette |
| **Figures** | Simple, labeled, no clutter |
| **Tables** | Max 5 rows, highlight key numbers |

### Slide Discipline

| Rule | Why |
|------|-----|
| One core message per slide | Audience remembers one thing |
| No paragraphs | Bullets or visuals only |
| No animations | Distraction, time waste |
| No reading slides verbatim | Talk TO audience, not AT slides |

---

## Rehearsal Plan

| Rehearsal | Focus | Target Time |
|-----------|-------|-------------|
| 1st | Content flow | 15 min |
| 2nd | Timing | 12-14 min |
| 3rd | Polish | 11-13 min |

**Buffer:** Aim for 12 minutes to allow 1-2 min transitions/Q&A

---

## Files Referenced

- `writing/course-line/assets/fig1_subset_accuracy.png`
- `writing/course-line/assets/architecture_diagram.pdf` (to create)
- `writing/course-line/assets/case_success.pdf` (to create)
- `reports/cover3d_coverage_diagnostics/coverage_diagnostics_report.md`
- `update/PHASE4_RESULTS_SUMMARY.md`

---

## Deliverables Checklist

- [ ] Slides created (PowerPoint, Keynote, or Beamer)
- [ ] All figures exported at presentation resolution
- [ ] Backup slides prepared
- [ ] Speaker notes written
- [ ] Rehearsed 3+ times
- [ ] Timing verified (10-15 min)
- [ ] Backup plan if tech fails (PDF export, printed notes)
