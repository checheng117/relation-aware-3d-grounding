# Main Table Plan: Unified Protocol Results

This document designs the "unified protocol main table" for the course report.

---

## Goal

Create one authoritative table that:
1. Shows baseline and all validated methods under the same metric definitions
2. Includes hard subset breakdowns
3. Clearly marks evidence levels (Final/Controlled/Pilot)
4. Does NOT mix incompatible protocols

---

## Table Structure

### Table 1: Main Results (Report Section 4.1)

| Method | Overall Acc@1 | Overall Acc@5 | Same-Class | High Clutter | Multi-Anchor | Evidence Level |
|--------|---------------|---------------|------------|--------------|--------------|----------------|
| ReferIt3DNet (baseline) | 30.83% | 91.87% | 21.96% | 16.07% | 11.90% | Final |
| Dense-no-cal-v1 | 31.05% | 92.01% | — | — | 28.34% | Controlled |
| Phase 4 E0 (baseline) | 28.60% | — | — | — | 24.73% | Controlled |
| Phase 4 E1 (viewpoint-cond) | 30.74% | — | — | — | 24.73% | Controlled |
| Phase 4 E3 (no supervision) | 30.65% | — | — | — | 23.66% | Controlled |

**Footnotes:**
- All accuracies on scene-disjoint test split (N=4,255)
- ReferIt3DNet baseline from `outputs/20260420_clean_sorted_vocab_baseline/`
- Phase 4 methods from `outputs/phase4_ablation/` — note different protocol (logits from trained model, not pre-extracted)
- Dense-no-cal-v1 from `reports/final_diagnostic_master_summary.md` — 10-epoch single seed
- "—" indicates metric not reported in source

**Key observations:**
1. Dense-no-cal-v1 shows +0.22% overall gain with multi-anchor improvement (+5.3%)
2. Phase 4 E1 shows +2.14% gain over E0 in same protocol
3. Phase 4 E3 (no supervision) matches E1 — gain is from architecture, not viewpoint labels

---

### Table 2: Coverage Diagnostics (Report Section 2)

| k | Coverage@k (Any Anchor) | Coverage@k (All Anchors) | Baseline-Wrong Sparse Miss |
|---|------------------------|-------------------------|---------------------------|
| 1 | 29.50% | 18.94% | — |
| 3 | 46.04% | 35.97% | — |
| 5 | 67.87% | 54.20% | 33.95% (110/324) |
| 10 | 83.69% | 73.62% | — |

**Source:** `reports/cover3d_coverage_diagnostics/coverage_diagnostics_report.md`

**Interpretation:**
- Sparse top-5 misses every annotated anchor in 33.95% of baseline-wrong, anchor-evaluable samples
- Dense all-pair coverage recovers 100% of these at candidate-set level
- Does not prove dense reranking improves accuracy — only that coverage is necessary condition

---

### Table 3: Phase 4 Ablation by Relation Type (Report Section 4.3, Appendix)

| Relation Type | E0 Baseline | E1 Viewpoint | E3 No Supervision | E1-E0 | E3-E0 |
|---------------|-------------|--------------|-------------------|-------|-------|
| between | — | — | — | +3.88% | +3.10% |
| directional | — | — | — | +1.95% | +1.88% |
| support | — | — | — | +2.30% | +2.30% |
| attribute | — | — | — | +1.67% | +1.67% |

**Source:** `update/PHASE4_5_MECHANISM_DIAGNOSIS.md`

**Interpretation:** Gain is general across relation types, not concentrated on viewpoint-dependent cases

---

### Table 4: Extended Subset Breakdown (Appendix)

| Subset | Count | Baseline | Dense-no-cal-v1 | Phase 4 E0 | Phase 4 E1 |
|--------|-------|----------|-----------------|------------|------------|
| Easy | — | 31.72% | — | — | — |
| Multi-Anchor | 93 | 23.04% | 28.34% | 24.73% | 24.73% |
| Relative-Position | — | 29.86% | 30.68% | — | — |
| Same-Class Clutter | 2,373 | 21.96% | — | — | — |
| Same-Class High Clutter | 697 | 16.07% | — | — | — |
| viewpoint_sensitive | — | — | — | 26.95% | 29.40% |
| viewpoint_invariant | — | — | — | 28.60% | 30.74% |

**Note:** Not all methods report all subsets. Blank cells indicate data not available.

---

## What Goes in Main Text vs Appendix

### Main Text (Section 4)

| Table | Placement | Reason |
|-------|-----------|--------|
| Table 1 (Main Results) | Main text | Core results |
| Table 2 (Coverage) | Main text Section 2 | Motivation for method |
| Table 3 (Ablation by Type) | Main text Section 4.3 | Supports "architecture not supervision" claim |

### Appendix

| Table | Placement | Reason |
|-------|-----------|--------|
| Table 4 (Extended Subsets) | Appendix A | Too detailed for main text; incomplete cells |

---

## What Does NOT Go in Any Table

| Item | Reason | Alternative Placement |
|------|--------|----------------------|
| Phase 5 Counterfactual (+0.12 pp) | Pilot only (1 epoch) | Text mention in Section 5 (Future Work) |
| Phase 6 Latent-Mode K=4 (+0.09 pp) | Pilot only (1 epoch) | Text mention in Section 5 (Future Work) |
| RHN results | Failed activation (0 coverage) | Limitations Section 5.1 |
| SAT baseline | Number conflicts (28.27% vs 29.17%) | Footnote in Section 3.1 if needed |
| Dense strengthening negatives | Negative results | Appendix B (Negative Results) |

---

## Protocol Incompatibility Notes

### Critical: Do NOT Mix These Protocols

| Protocol | Methods | Why Incompatible |
|----------|---------|------------------|
| **Pre-extracted logits** | ReferIt3DNet baseline (30.83%) | Frozen predictions, no training |
| **Trained logits (Phase 4)** | E0/E1/E3 (28.60%/30.74%/30.65%) | Different training setup, 10 epochs |
| **Embedding-based (Phase 5/6)** | CF pilots, latent-mode pilots | Different data format, validation split only |

**Solution:** Report each protocol separately with clear footnotes. Do NOT create a single "leaderboard" mixing protocols.

---

## Numbers to Cite with Caveats

| Number | Source | Caveat |
|--------|--------|--------|
| 30.83% | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_results.json` | Trusted baseline |
| 31.05% | `reports/final_diagnostic_master_summary.md` | Dense-no-cal-v1, 10 epochs, single seed |
| 28.60%/30.74%/30.65% | `outputs/phase4_ablation/*.json` | Different protocol (trained model, not pre-extracted) |
| 34.30%/34.42% | `outputs/phase5_counterfactual/pilot_*/` | Pilot, 1 epoch, validation split only |
| 34.10%/34.19% | `outputs/phase6_latent_modes/pilot_*/` | Pilot, 1 epoch, validation split only |

---

## Figure Integration

### Figure 1: Subset Accuracy Comparison

**Source:** `reports/cover3d_phase1/fig1_subset_accuracy.png` (already copied to `writing/course-line/assets/`)

**Placement:** Report Section 4.2, after Table 1

**Caption:** "Baseline performance by hard subset. Same-class clutter, high clutter, and multi-anchor cases show significantly lower accuracy than overall test set."

---

### Figure 2: Coverage Curve

**Source:** Need to generate from `reports/cover3d_coverage_diagnostics/coverage_summary.json`

**Placement:** Report Section 2

**Description:** Line chart with x-axis=k (1,3,5,10), y-axis=coverage%, two lines (any-anchor, all-anchors)

---

### Figure 3: Architecture Diagram

**Source:** Need to create (see `COURSE_LINE_TODO.md` P1-3)

**Placement:** Report Section 4.1, before Table 1

**Description:** Flowchart showing object embeddings → relation scorer → viewpoint conditioning → output scores

---

## Checklist Before Finalizing Table

- [ ] All numbers traced to source files
- [ ] Evidence level (Final/Controlled/Pilot) marked for each row
- [ ] Protocol differences noted in footnotes
- [ ] Subset definitions consistent (same-class, high-clutter, multi-anchor)
- [ ] Acc@1 and Acc@5 definitions clear (top-1 vs top-5 accuracy)
- [ ] No mixing of incompatible protocols
- [ ] Pilot results NOT in main table (appendix only)

---

## Files Referenced

- `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_results.json`
- `outputs/phase4_ablation/dense-v4-hardneg_results.json`
- `outputs/phase4_ablation/viewpoint-conditioned_results.json`
- `outputs/phase4_ablation/viewpoint-conditioned-no-supervision_results.json`
- `reports/final_diagnostic_master_summary.md`
- `reports/cover3d_phase1_baseline_subset_results.md`
- `reports/cover3d_coverage_diagnostics/coverage_diagnostics_report.md`
- `reports/cover3d_coverage_diagnostics/coverage_summary.json`
- `update/PHASE4_RESULTS_SUMMARY.md`
- `update/PHASE4_5_MECHANISM_DIAGNOSIS.md`
