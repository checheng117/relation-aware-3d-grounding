# CLAIM_LEDGER

| Claim ID | Claim | Status | Evidence Source | Allowed in Main Paper? | What Would Strengthen It |
|---|---|---|---|---|---|
| C-001 | Latent conditioned architecture provides +2.1% gain | Supported | `update/PHASE4_RESULTS_SUMMARY.md`, E0 vs E1/E3 | Yes | Multi-seed validation, full training |
| C-002 | Gain comes from architecture, NOT viewpoint supervision | Supported | `update/PHASE4_5_MECHANISM_DIAGNOSIS.md`, E1 ≈ E3 | Yes | Additional architecture variants |
| C-003 | Baseline 30.83% on scene-disjoint split | Established | `outputs/20260420_clean_sorted_vocab_baseline/` | Yes | Multi-seed validation |
| C-004 | Hard subset failures systematic | Established | `reports/cover3d_phase1_baseline_subset_results.md` | Yes | None needed |
| C-005 | Coverage failure 33.95% | Established | `reports/cover3d_coverage_diagnostics/` | Yes | None needed |
| C-006 | Counterfactual loss provides +0.12 pp | Weak (Pilot) | `update/PHASE5_CONCLUSION_AND_PAPER_NOTES.md` | Appendix only | Full 10-epoch training |
| C-007 | MoE K=4 > K=1 single mode | Weak (Pilot) | `update/PHASE6_PILOT_E1_K4_SAFE_RUN.md` | No (Future work) | Full 80-epoch training |
| C-008 | Counterfactual > Random Hard Negatives | Do Not Claim | `update/PHASE5_PILOT_E2_RHN_SAFE_RUN.md` | No | RHN never activated (0% coverage) |
| C-009 | Method solves multi-anchor grounding | Do Not Claim | `update/PHASE4_RESULTS_SUMMARY.md` | No | Multi-anchor: E0=E1=E3 (24.73%) |
| C-010 | SOTA on ReferIt3D | Do Not Claim | N/A | No | Single-seed, scene-disjoint not comparable |

## Status Definitions

- **Established**: directly supported by completed experiments or verified artifacts.
- **Supported**: supported by partial but credible evidence.
- **Weak**: plausible but not fully supported (pilot results).
- **Speculative**: idea-level claim only.
- **Do Not Claim**: should not appear as a paper claim (evidence contradicts or insufficient).

## Evidence Level Mapping

| Claim Status | Allowed Placement |
|--------------|-------------------|
| Established | Main text, main table |
| Supported | Main text with context, appendix table |
| Weak | Appendix only, labeled as "pilot" |
| Do Not Claim | Limitations section, NOT in claims |