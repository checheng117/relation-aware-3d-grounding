# Diagnostic Paper Positioning Freeze

**Date**: 2026-04-22
**Status**: FINAL POSITIONING

---

## Executive Summary

This document freezes the diagnostic paper positioning for this project. All documentation, reports, and public-facing materials must align with this positioning.

**Paper Type**: Diagnostic / Benchmark / Reproducibility + Limited Method Signal

**NOT**: Strong Method Paper

---

## Working Title

**Option A (Diagnostic Focus)**:
> "What Makes 3D Referring Expressions Hard? A Scene-Disjoint Diagnostic Study"

**Option B (Benchmark Focus)**:
> "Trustworthy Evaluation for 3D Referring Expression Grounding: Benchmark, Diagnostics, and Reproducibility"

**Option C (Balanced)**:
> "COVER-3D: Coverage-Calibrated Relational Grounding for Scene-Disjoint 3D Referring Expressions"

**Recommended**: Option A or B - honest diagnostic framing, not method framing.

---

## Core Claim (Unified Positioning)

> This project establishes a trustworthy, scene-disjoint evaluation and diagnostic framework for 3D referring-expression grounding, identifies concentrated failure modes in hard relational subsets, provides direct evidence of coverage failure under sparse candidate-anchor selection, and shows that a simple dense reranker yields only limited gains while more complex calibration and dense-scorer extensions do not currently justify further method scaling.

**Short version**:
> We build a reproducible evaluation foundation for 3D grounding, diagnose why hard relational cases fail, and show that simple dense reranking helps modestly while complex extensions do not.

---

## Contributions (What We Actually Claim)

### 1. Trustworthy Evaluation Foundation

- Recovered full Nr3D dataset (41,503 samples, 641 scenes)
- Scene-disjoint splits with verified zero overlap
- Unified metric definitions (Acc@1, Acc@5)
- Reproduced baselines: ReferIt3DNet (30.79%), SAT (28.27%)

**Value**: Enables apples-to-apples comparison. Many prior results may be unreliable due to split leakage.

### 2. Diagnostic Framework

- Hard-subset tagging (same-class clutter, multi-anchor, relative-position)
- Coverage failure analysis (coverage@k, anchor reachability)
- Harm/recovered taxonomy for method analysis
- Case study export for qualitative analysis

**Value**: Explains WHERE and WHY methods fail, not just accuracy numbers.

### 3. Coverage Failure Evidence

- Sparse candidate-anchor selection misses important relational evidence
- Long-range anchors are underweighted in top-k selection
- Same-class clutter cases suffer from anchor confusion
- Multi-anchor cases need broader coverage than single-anchor cases

**Value**: Direct evidence for why sparse relation modeling fails on hard cases.

### 4. Limited Method Signal

- Dense-no-cal-v1: +0.22% Acc@1, +9 net recovered
- Multi-anchor improvement: +5.3%
- No calibration overhead, simple weighted aggregation
- Stable, reproducible, lightweight (398K params)

**Value**: Existence proof that dense coverage can help, but modestly.

### 5. Negative Findings (With Evidence)

- Calibration line failed: signals uninformative, gate learns wrong decisions
- Dense strengthening failed: relation scores mostly noisy, pair ranking weak
- Fundamentals debug: score gap = -0.95, Hit@1 = 7%

**Value**: Saves community from pursuing same dead ends. Clear evidence, not just "didn't work."

---

## What We Do NOT Claim

### Depreciated Claims (No Longer Supported)

| Old Claim | Why Deprecated |
|-----------|----------------|
| "Calibration is a core contribution" | Calibration failed - signals uninformative |
| "Dense scorer strengthening shows promise" | Strengthening variants all failed (-238 to -277 net) |
| "COVER-3D can challenge SOTA" | Best result is +0.22%, not SOTA-level |
| "Multi-seed validation underway" | Method signal too modest for multi-seed investment |
| "Scalable method framework" | Foundation too weak to support complexity |

### Claims We Cannot Make

- **NOT** "State-of-the-art results" - Best is 31.05%, not SOTA
- **NOT** "Strong method contribution" - Modest gain (+0.22%)
- **NOT** "Calibration-ready foundation" - Calibration line failed
- **NOT** "Foundation for future method scaling" - Foundation is weak
- **NOT** "Multi-seed validated" - Single-seed only

---

## Target Venues

### Primary (Recommended)

| Venue | Type | Fit |
|-------|------|-----|
| TACL | Journal | Diagnostic + reproducibility + limited method |
| ACL Findings | Conference | Diagnostic findings, negative results |
| EMNLP Findings | Conference | Same as ACL |
| Scientific Data | Journal | Benchmark + dataset + reproducibility |
| Data Track (various) | Track | Dataset and benchmark focus |

### Secondary (If Method Reframed)

| Venue | Type | Requirements |
|-------|------|--------------|
| CVPR/ICCV/ECCV Workshop | Workshop | Method + diagnostics combo |
| 3DV | Conference | 3D vision + methods |
| BMVC | Conference | More open to diagnostic work |

### NOT Targeted

- **NOT** AAAI/NeurIPS/ICML main track - Method signal insufficient
- **NOT** CVPR/ICCV main track - Not SOTA-level results
- **NOT** ACL/EMNLP main track - Could target Findings instead

---

## Paper Structure (Recommended)

### 1. Introduction

- Motivate trustworthy evaluation problem
- Note dataset/split issues in prior work
- Preview diagnostic findings
- Preview modest method signal with clear boundaries

### 2. Dataset and Splits

- Nr3D recovery process
- Scene-disjoint split construction
- Overlap verification
- Coverage statistics

### 3. Baseline Reproduction

- ReferIt3DNet reproduction
- SAT reproduction
- Comparison with literature (cautionary)
- Split leakage effects

### 4. Diagnostic Framework

- Hard-subset definitions
- Coverage metrics
- Harm/recovered taxonomy
- Case study methodology

### 5. Diagnostic Findings

- Where base model fails (hard subsets)
- Coverage failure evidence
- What Dense-no-cal-v1 fixes
- Where Dense-no-cal-v1 fails

### 6. Method: Dense-no-cal-v1

- Architecture (simple)
- Training (stable)
- Results (modest)
- Limitations (clear)

### 7. Negative Findings

- Calibration failure analysis
- Dense strengthening failure analysis
- Fundamentals debug results
- Why methods didn't work

### 8. Discussion

- Implications for 3D grounding
- Recommendations for practitioners
- Limitations of this work
- Future directions (not method scaling)

### 9. Conclusion

- Summary of contributions
- Honest assessment of method signal
- Value of diagnostic findings

---

## Reviewer Framing

**How to position for honest, favorable review:**

1. **Lead with evaluation problem** - Not method contribution
2. **Be honest about method signal** - Modest gain, clear boundaries
3. **Emphasize diagnostic value** - Negative results with evidence
4. **Highlight reproducibility** - Clean baseline, artifacts available
5. **Don't oversell** - Reviewers forgive modest claims, not inflated ones

**Anticipated Reviewer Concerns + Responses:**

| Concern | Preemptive Response |
|---------|---------------------|
| "Method gain is small" | Acknowledged. Value is in diagnostics + reproducibility. |
| "Why not stronger method?" | Explored calibration and strengthening; both failed with evidence. |
| "Single-seed results" | Acknowledged. Gain modest; variance unlikely to change conclusion. |
| "Relation scores noisy - why?" | Analyzed in fundamentals debug; discussed as limitation. |

---

## Key Figures to Include

1. **Dataset coverage map** - Scenes used vs prior work
2. **Split overlap verification** - Pairwise scene intersection heatmap
3. **Hard subset performance** - Bar chart comparing subsets
4. **Coverage analysis** - Coverage@k curves, anchor rank distribution
5. **Harm/recovered scatter** - Where Dense-no-cal-v1 helps vs hurts
6. **Relation score distributions** - Correct vs wrong (shows noise)
7. **Pair ranking histogram** - Hit@k analysis

---

## Key Tables to Include

1. **Dataset statistics** - Samples, scenes, utterances, anchors
2. **Baseline reproduction** - ReferIt3DNet, SAT vs literature
3. **Hard subset results** - All methods on all subsets
4. **Method comparison** - All variants with status (keep/freeze/discard)
5. **Calibration failure** - Gate behavior, signal analysis
6. **Fundamentals debug** - Score gap, Hit@1, margin distributions

---

## Supplementary Materials

### Must Include

- Train/val/test split manifests
- Evaluation code
- Baseline configs and checkpoints
- Dense-no-cal-v1 code and configs
- All diagnostic reports as appendices

### Nice to Include

- Interactive case study browser
- Failure mode video/visualization
- Coverage analysis dashboard

### Do NOT Include

- Frozen method code as primary entry point
- Unconfirmed promising results
- Hardware crash logs (mention but don't include)

---

## Ethical Considerations

- **Honest reporting**: Modest gains stated clearly, not inflated
- **Negative results**: Failures reported with evidence, not hidden
- **Reproducibility**: Code, configs, checkpoints all available
- **Attribution**: Prior work credited; no "first to do X" overclaims

---

## Success Criteria

Paper is ready for submission when:

1. [ ] All documentation aligned with this positioning
2. [ ] README reflects diagnostic paper framing
3. [ ] Master summary report complete
4. [ ] Method freeze policy documented
5. [ ] All claims supported by evidence in reports
6. [ ] No implication of ongoing method冲刺
7. [ ] Reviewer framing anticipates concerns honestly

---

## Final Statement

**This is a diagnostic paper with modest method signal, not a strong method paper.**

The value is in:
- Trustworthy evaluation foundation
- Diagnostic findings about failure modes
- Reproducible baseline for community
- Negative results with clear evidence

Honest framing will attract favorable reviews. Overselling will backfire.
