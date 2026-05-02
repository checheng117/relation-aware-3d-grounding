# Claim Boundary

Project title: **Latent Conditioned Relation Scoring for 3D Visual Grounding**

This repository accompanies a CSC6133 final project report. It should be read as a reproducible diagnostic and controlled-ablation study, not as a state-of-the-art method paper.

## Supported Claims

- The project provides a verified scene-disjoint Nr3D/ReferIt3D-style evaluation pipeline.
- The reproduced baseline fails disproportionately on hard subsets such as same-class clutter, high clutter, and multi-anchor cases.
- Sparse anchor selection creates a measurable coverage bottleneck.
- Controlled Phase 4 ablations support a conditioned relation architecture signal over the controlled dense relation baseline.

## Unsupported Claims

- Do not claim state-of-the-art performance.
- Do not claim that explicit viewpoint supervision is validated as the causal mechanism.
- Do not claim that multi-anchor grounding is solved.
- Do not merge frozen-logit diagnostics, trained Phase 4 ablations, and Phase 5/6 pilots into one leaderboard.

## Evidence Separation

| Evidence level | Public artifacts | Claim status |
|----------------|------------------|--------------|
| Frozen-logit diagnostics | `reports/final_diagnostic_master_summary.md`, `reports/final_diagnostic_master_table.csv` | Final diagnostic evidence |
| Hard-subset diagnostics | `reports/cover3d_phase1_baseline_subset_results.md` | Final diagnostic evidence |
| Coverage diagnostics | `reports/cover3d_coverage_diagnostics/` | Final diagnostic evidence |
| Controlled Phase 4 ablations | `reports/phase4_aggregates/` and report tables | Strongest current method evidence |
| Phase 5/6 pilots | Report discussion only | Pilot evidence, not final claims |

## Preferred Wording

"Conditioned / latent-conditioned relation computation improves over a controlled dense relation baseline, but the gain should not be attributed to explicit semantic viewpoint supervision."
