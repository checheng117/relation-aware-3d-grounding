# Reports Index

This directory contains public evidence artifacts for the CSC6133 final project report, **Latent Conditioned Relation Scoring for 3D Visual Grounding**.

## Final Evidence Files

| Artifact | Purpose |
|----------|---------|
| `final_diagnostic_master_summary.md` | Main diagnostic summary for the reproduced baseline and dense diagnostic line. |
| `final_diagnostic_master_table.csv` | Machine-readable summary table for the diagnostic line. |
| `cover3d_phase1_baseline_subset_results.md` | Scene-disjoint hard-subset diagnostics. |
| `cover3d_coverage_diagnostics/` | Anchor coverage reports, summaries, curves, and casebooks. |
| `dense_fundamentals_summary.md` | Dense relation diagnostic failure analysis. |
| `phase4_aggregates/` | Sanitized aggregate JSON summaries for the controlled Phase 4 ablations. |

## Protocol Warning

Frozen-logit diagnostics, dense relation diagnostics, controlled Phase 4 trained ablations, and Phase 5/6 pilots use different protocols. Do not merge them into a single leaderboard.

## Claim Boundary

- Supported: conditioned relation architecture signal over the controlled dense relation baseline.
- Not supported: explicit viewpoint supervision as the causal mechanism.
- Not solved: multi-anchor reasoning.
- Pilot only: Phase 5 counterfactual and Phase 6 latent-mode runs.

Internal release-agent notes and obsolete release checklists are not part of the public evidence surface.
