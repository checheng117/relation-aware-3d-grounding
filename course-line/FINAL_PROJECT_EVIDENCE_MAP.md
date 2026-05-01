# Final Project Evidence Map

This document catalogs all evidence in the repository by confidence level. Each item includes:
- Evidence level (A/B/C/D)
- Source file path
- Suitable placement (main text / appendix / presentation)

---

## Evidence Level Definitions

| Level | Meaning | Course-line treatment |
|-------|---------|----------------------|
| **A** | Final / trusted evidence | Can appear in main table, main text |
| **B** | Controlled supporting evidence | Can appear in main text with context |
| **C** | Pilot evidence | Appendix or presentation backup only; must be labeled "pilot" |
| **D** | Pending / future work | Future work section only |

---

## Level A: Final / Trusted Evidence

### A1. Scene-Disjoint Split Recovery

| Attribute | Value |
|-----------|-------|
| **Claim** | Dataset split is verified zero-overlap scene-disjoint |
| **Source** | `reports/scene_disjoint_split_recovery_results.md`, `scripts/validate_scene_disjoint_splits.py` |
| **Metric** | Zero scene overlap between train/val/test |
| **Status** | Verified, deterministic |
| **Placement** | Report Section 3.1 (Dataset), appendix for detailed stats |

---

### A2. Clean Baseline: ReferIt3DNet Reproduction

| Attribute | Value |
|-----------|-------|
| **Claim** | Baseline Acc@1 = 30.83% on scene-disjoint test set |
| **Source** | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_results.json`, `reports/final_diagnostic_master_summary.md` |
| **Metric** | Acc@1: 30.83%, Acc@5: 91.87%, N=4,255 test samples |
| **Status** | Final, single-seed but stable |
| **Placement** | Main table (baseline row), Section 4.1 |

---

### A3. Hard Subset Diagnostics

| Attribute | Value |
|-----------|-------|
| **Claim** | Baseline fails systematically on same-class clutter, high clutter, multi-anchor |
| **Source** | `reports/cover3d_phase1_baseline_subset_results.md`, `reports/cover3d_phase1_hard_subsets.md`, `reports/cover3d_phase1/fig1_subset_accuracy.png` |
| **Metrics** | Same-class: 21.96%, High clutter: 16.07%, Multi-anchor: 11.90% |
| **Status** | Final, deterministic subset analysis |
| **Placement** | Main table (subset rows), Section 4.2, Figure 2 |

---

### A4. Scene Size Analysis

| Attribute | Value |
|-----------|-------|
| **Claim** | Baseline accuracy varies by scene object count (clutter level) |
| **Source** | `reports/cover3d_phase1/fig3_scene_size.png` |
| **Metrics** | Accuracy by scene object count bins: <20, 20-29, 30-39, 40-49, 50-59, 60+ |
| **Status** | Final, deterministic diagnostic |
| **Placement** | Appendix Figure A2, Section 4.2 discussion |

---

### A5. Relation Type Analysis

| Attribute | Value |
|-----------|-------|
| **Claim** | Baseline accuracy varies by relation type (directional, relative, between, support, etc.) |
| **Source** | `reports/cover3d_phase1/fig4_relation_type.png` |
| **Metrics** | Per-relation accuracy: Between ~46%, Support ~38%, Container ~25% |
| **Status** | Final, deterministic diagnostic |
| **Placement** | Appendix Figure A3, Section 4.2 discussion |

---

### A6. Coverage Diagnostics

| Attribute | Value |
|-----------|-------|
| **Claim** | Sparse top-5 misses all annotated anchors in 33.95% of baseline-wrong samples |
| **Source** | `reports/cover3d_coverage_diagnostics/coverage_diagnostics_report.md`, `reports/cover3d_coverage_diagnostics/coverage_summary.json` |
| **Metrics** | Coverage@5: 67.87% (any anchor), 54.20% (all anchors); Sparse miss all: 110/324 (33.95%) |
| **Status** | Final, diagnostic (no training) |
| **Placement** | Main text Section 2 (Motivation), Table 2 |

---

### A7. Phase 4 Controlled Architecture Result

| Attribute | Value |
|-----------|-------|
| **Claim** | Viewpoint-conditioned architecture provides +2.1% gain over baseline |
| **Source** | `update/PHASE4_RESULTS_SUMMARY.md`, `outputs/phase4_ablation/*.json` |
| **Metrics** | E0: 28.60%, E1: 30.74%, E3 (no supervision): 30.65% |
| **Status** | Controlled, single-seed, same protocol |
| **Placement** | Main table (E0, E1 rows), Section 4.3 |

---

## Level B: Controlled Supporting Evidence

### B1. Dense-no-cal-v1 Diagnostic

| Attribute | Value |
|-----------|-------|
| **Claim** | Simple dense scorer with weighted sum shows +0.22% gain |
| **Source** | `reports/final_diagnostic_master_summary.md`, `reports/dense_no_cal_results.md` |
| **Metrics** | Acc@1: 31.05% vs 30.83% baseline |
| **Status** | Controlled, 10 epochs, single seed |
| **Placement** | Appendix Table A1 (diagnostic methods), mentioned in Section 2 |

---

### B2. Phase 4.5 Mechanism Diagnosis

| Attribute | Value |
|-----------|-------|
| **Claim** | Gain comes from architecture, NOT viewpoint supervision |
| **Source** | `update/PHASE4_5_MECHANISM_DIAGNOSIS.md`, `update/PAPER_CLAIM_REFRAMING.md` |
| **Metrics** | E1 (30.74%) ≈ E3 (30.65%), both > E0 (28.60%) |
| **Status** | Controlled ablation, same protocol |
| **Placement** | Main text Section 4.3 (Ablation), Table 3 |

---

### B3. Phase 4 Per-Relation-Type Analysis

| Attribute | Value |
|-----------|-------|
| **Claim** | Gain is general across relation types, not viewpoint-specific |
| **Source** | `update/PHASE4_5_MECHANISM_DIAGNOSIS.md` |
| **Metrics** | between: +3.88%, directional: +1.95%, support: +2.30%, attribute: +1.67% |
| **Status** | Controlled subset analysis |
| **Placement** | Appendix Table A2, mentioned in Section 4.3 |

---

## Level C: Pilot Evidence

### C1. Phase 5 Counterfactual Pilot (E0 vs E1-CF)

| Attribute | Value |
|-----------|-------|
| **Claim** | Counterfactual loss shows +0.12 pp pilot gain at 1 epoch |
| **Source** | `update/PHASE5_CONCLUSION_AND_PAPER_NOTES.md`, `update/PHASE5_PILOT_E0_BS8_SAFE_RUN.md`, `update/PHASE5_PILOT_E1_CF_SAFE_RUN.md` |
| **Metrics** | E0: 34.30%, E1-CF: 34.42% (Val Acc@1, 1 epoch, bs=8) |
| **Status** | Pilot, 1 epoch, batch=8, single seed |
| **Caveats** | Small gain (+0.12 pp), needs full training validation |
| **Placement** | Appendix Section B.1 (Counterfactual Pilot), presentation backup slide |

---

### C2. Phase 5 RHN Control (E2-RHN)

| Attribute | Value |
|-----------|-------|
| **Claim** | RHN ablation inconclusive due to missing object metadata |
| **Source** | `update/PHASE5_PILOT_E2_RHN_SAFE_RUN.md`, `update/RHN_ROOT_CAUSE_AUDIT.md` |
| **Metrics** | E2-RHN: 34.30% (=E0), RHN coverage: 0/30,447 (0%) |
| **Status** | Pilot, failed activation |
| **Caveats** | Cannot claim "CF > Random" — RHN never activated |
| **Placement** | Appendix Section B.1 (Limitations), future work |

---

### C3. Phase 6 Latent-Mode K=1 Pilot

| Attribute | Value |
|-----------|-------|
| **Claim** | K=1 (single mode) baseline reaches 34.10% Val Acc@1 at 1 epoch |
| **Source** | `update/PHASE6_PILOT_E0_K1_SAFE_RUN.md`, `outputs/phase6_latent_modes/pilot_E0_K1_safe/` |
| **Metrics** | Val Acc@1: 34.10%, Params: 328,706 |
| **Status** | Pilot, 1 epoch, bs=8, single seed |
| **Caveats** | Not comparable to Phase 4 (different data format: embeddings vs logits) |
| **Placement** | Appendix Section B.2 (Latent Mode Pilots), not in main text |

---

### C4. Phase 6 Latent-Mode K=4 Pilot

| Attribute | Value |
|-----------|-------|
| **Claim** | K=4 (MoE) shows +0.09 pp over K=1 at 1 epoch |
| **Source** | `update/PHASE6_PILOT_E1_K4_SAFE_RUN.md`, `outputs/phase6_latent_modes/pilot_E1_K4_safe/` |
| **Metrics** | Val Acc@1: 34.19%, Params: 1,216,136 |
| **Status** | Pilot, 1 epoch, bs=8, single seed |
| **Caveats** | Small signal (+0.09 pp), needs full 80-epoch training |
| **Placement** | Appendix Section B.2, Figure A3 (training curves), future work |

---

### C5. Dense Strengthening Variants (Negative Results)

| Attribute | Value |
|-----------|-------|
| **Claim** | Dense-v2-AttPool, Dense-v3-Geo, Dense-v4-HardNeg all fail |
| **Source** | `reports/dense_strengthening_results.md`, `reports/dense_fundamentals_score_audit.md` |
| **Metrics** | v2: 25.24%, v3: 24.32%, v4: 24.47% (all << baseline) |
| **Status** | Controlled negative results |
| **Caveats** | Shows what NOT to do; useful for discussion |
| **Placement** | Appendix Section B.3 (Negative Results), discussion Section 5 |

---

## Level D: Pending / Future Work

### D1. Full Latent-Mode Training (K=1 vs K=4)

| Attribute | Value |
|-----------|-------|
| **Claim** | Not yet validated — expected to show larger K=4 advantage at 80 epochs |
| **Source** | Planned in `update/LATENT_MODE_TRAINING_PLAN.md` |
| **Script** | `scripts/train_cover3d_latent_modes.py` |
| **Status** | Pending full remote/headless runs |
| **Placement** | Future work Section 6 |

---

### D2. Multi-Seed Validation

| Attribute | Value |
|-----------|-------|
| **Claim** | Not yet done — would show stability across seeds |
| **Source** | Mentioned in `update/LATENT_MODE_EVAL_PROTOCOL.md` |
| **Status** | Pending compute |
| **Placement** | Future work, limitations |

---

### D3. Metadata-Preserving Pipeline

| Attribute | Value |
|-----------|-------|
| **Claim** | Not yet implemented — would enable RHN and geometry-aware diagnostics |
| **Source** | `update/METADATA_PRESERVING_PIPELINE_PLAN.md`, `update/RHN_ROOT_CAUSE_AUDIT.md` |
| **Script** | `scripts/migrate_embeddings_with_metadata.py` (partial) |
| **Status** | Partial implementation, not validated |
| **Placement** | Future work Section 6 |

---

### D4. Anchor-Chain Reasoning

| Attribute | Value |
|-----------|-------|
| **Claim** | Not validated — multi-anchor subset shows NO gain from current method |
| **Source** | `update/ANCHOR_CHAIN_SCHEMA.md`, `update/PHASE4_5_MECHANISM_DIAGNOSIS.md` |
| **Metrics** | Multi-anchor: E0 24.73%, E1 24.73%, E3 23.66% (no improvement) |
| **Status** | Deferred to future work |
| **Placement** | Future work, NOT in main claims |

---

### D5. Statistical Significance Tests

| Attribute | Value |
|-----------|-------|
| **Claim** | Not performed — would strengthen claims |
| **Source** | Mentioned in `writing/course/SOURCES_USED.md` |
| **Status** | Pending |
| **Placement** | Can be added if time permits; otherwise future work |

---

## Evidence Placement Summary

| Placement | Evidence Items |
|-----------|----------------|
| **Main Text + Main Table** | A2, A3, A4, A5, B2 |
| **Main Text + Appendix Table** | B1, B3 |
| **Appendix Only** | C1, C2, C3, C4, C5 |
| **Presentation Main** | A2, A3, A4, A5 |
| **Presentation Backup** | C1, C4 |
| **Future Work Only** | D1, D2, D3, D4, D5 |

---

## Evidence Hierarchy Figure (for Report)

```
Level A (Final)          Level B (Controlled)       Level C (Pilot)         Level D (Pending)
─────────────────        ──────────────────         ──────────────          ────────────────
• Baseline 30.83%        • Dense-no-cal-v1          • CF +0.12 pp           • Full K=1/K=4
• Hard subsets           • Arch vs supervision      • K=4 +0.09 pp          • Multi-seed
• Coverage@k             • Per-relation gain        • RHN failed            • Metadata pipeline
• Phase 4 E0/E1          • Negative variants        • Latent modes          • Anchor-chain
```

---

## What NOT to Claim (Evidence Boundary)

| Invalid Claim | Why Invalid | Correct Framing |
|---------------|-------------|-----------------|
| "K=4 MoE improves over K=1" | Only 1-epoch pilot (+0.09 pp) | "K=4 shows early pilot signal; full training pending" |
| "Counterfactual > Random Hard Negatives" | RHN never activated (0 coverage) | "RHN ablation inconclusive due to data format" |
| "Viewpoint supervision helps" | E3 (no supervision) = E1 (with) | "Architecture benefit, not supervision benefit" |
| "Method solves multi-anchor" | Multi-anchor subset: 0% gain | "Multi-anchor remains challenging" |
| "SOTA on ReferIt3D" | Only single-seed, scene-disjoint split | "Improves over reproduced baseline on scene-disjoint split" |

---

## Files Referenced

- `reports/final_diagnostic_master_summary.md`
- `reports/cover3d_phase1_baseline_subset_results.md`
- `reports/cover3d_phase1_hard_subsets.md`
- `reports/cover3d_coverage_diagnostics/coverage_diagnostics_report.md`
- `update/PHASE4_RESULTS_SUMMARY.md`
- `update/PHASE4_5_MECHANISM_DIAGNOSIS.md`
- `update/PHASE5_CONCLUSION_AND_PAPER_NOTES.md`
- `update/PHASE5_PILOT_E0_BS8_SAFE_RUN.md`
- `update/PHASE5_PILOT_E1_CF_SAFE_RUN.md`
- `update/PHASE5_PILOT_E2_RHN_SAFE_RUN.md`
- `update/PHASE6_PILOT_E0_K1_SAFE_RUN.md`
- `update/PHASE6_PILOT_E1_K4_SAFE_RUN.md`
- `reports/dense_strengthening_results.md`
- `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_results.json`
- `outputs/phase4_ablation/*.json`
- `outputs/phase5_counterfactual/pilot_*/`
- `outputs/phase6_latent_modes/pilot_*/`
