# Final Project Evidence Map

Project title: **Latent Conditioned Relation Scoring for 3D Visual Grounding**

This map connects the CSC6133 report claims to public repository artifacts. It intentionally separates diagnostic evidence, trained ablations, and pilot evidence.

## Main Evidence

| Topic | Public artifact | Report use |
|-------|-----------------|------------|
| Scene-disjoint reproduced baseline | `reports/final_diagnostic_master_summary.md`, `reports/final_diagnostic_master_table.csv` | Baseline accuracy and dense diagnostic line |
| Hard subsets | `reports/cover3d_phase1_baseline_subset_results.md` | Same-class clutter, high clutter, multi-anchor diagnostics |
| Anchor coverage | `reports/cover3d_coverage_diagnostics/coverage_diagnostics_report.md` and related JSON/CSV files in that directory | Sparse top-k coverage bottleneck |
| Dense relation diagnostic line | `reports/final_diagnostic_master_summary.md`, `reports/dense_fundamentals_summary.md` | Retained and frozen dense relation variants |
| Controlled Phase 4 ablations | `reports/phase4_aggregates/R001_aggregate_results.json`, `reports/phase4_aggregates/R002_aggregate_results.json`, `reports/phase4_aggregates/R004_aggregate_results.json` | Conditioned architecture comparison and negative viewpoint controls |
| Report source | `course-line/report/main.tex`, `course-line/report/sections/` | Final course report |

## Evidence Levels

| Level | Meaning | Examples |
|-------|---------|----------|
| Final diagnostic evidence | Scene-disjoint, reproducible diagnostic outputs used directly in the report | Baseline, hard subsets, coverage diagnostics |
| Controlled method evidence | Trained ablations with separated protocol and seeds | Phase 4 E0/E1/E2/E3 aggregate summaries |
| Pilot evidence | Implementation checks and future-work signals | Phase 5 counterfactual and Phase 6 latent-mode pilots |

## Claim Boundary

The strongest supported method statement is that conditioned / latent-conditioned relation computation improves over a controlled dense relation baseline. Random and shuffled viewpoint controls weaken a semantic viewpoint-supervision explanation. Multi-anchor reasoning remains unresolved.
