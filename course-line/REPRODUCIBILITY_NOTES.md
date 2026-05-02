# Course-Line Reproducibility Notes

Project title: **Latent Conditioned Relation Scoring for 3D Visual Grounding**

These notes identify the public artifacts needed to audit the CSC6133 report. Raw training outputs, checkpoints, generated embeddings, and local pilot run directories are not included in the public release.

## Public Artifact Map

| Result family | Public artifact |
|---------------|-----------------|
| Final report PDF | `course-line/report/report.pdf` |
| Report source | `course-line/report/main.tex`, `course-line/report/sections/` |
| Baseline and dense diagnostic line | `reports/final_diagnostic_master_summary.md`, `reports/final_diagnostic_master_table.csv` |
| Hard-subset diagnostics | `reports/cover3d_phase1_baseline_subset_results.md` |
| Coverage diagnostics | `reports/cover3d_coverage_diagnostics/coverage_diagnostics_report.md` |
| Coverage curves and summaries | `reports/cover3d_coverage_diagnostics/subset_coverage_curves.csv`, `reports/cover3d_coverage_diagnostics/coverage_summary.json` |
| Dense relation failure analysis | `reports/dense_fundamentals_summary.md` |
| Controlled Phase 4 aggregates | `reports/phase4_aggregates/` |
| Claim boundary | `course-line/CLAIM_BOUNDARY.md` |

## Protocol Boundary

Do not merge these evidence types into a single leaderboard:

- frozen-logit baseline and dense relation diagnostics;
- trained Phase 4 ablations;
- Phase 5 counterfactual pilots;
- Phase 6 latent-mode pilots.

The report keeps these protocols separated and labels Phase 5/6 as pilot evidence only.

## Main Reproduction Guidance

Use the report source and public summaries to audit the final numbers. The public repository keeps generated feature caches and raw experiment outputs out of git. Re-running training requires local data setup and should not be inferred from the public summaries alone.
