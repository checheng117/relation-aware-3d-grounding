# CSC6133 Course Report Reproducibility

Report title: **Latent Conditioned Relation Scoring for 3D Visual Grounding**

This note maps the course-report claims to local repository artifacts. It is not a new experiment log and does not merge results from different protocols into one leaderboard.

## Protocol Separation

| Protocol | Role in report | Evidence status |
|----------|----------------|-----------------|
| Frozen-logit diagnostics | Baseline reproduction, hard-subset analysis, anchor coverage, dense reranking diagnostics | Final diagnostic evidence |
| Dense relation diagnostic line | Lightweight reranking variants evaluated on frozen logits | Diagnostic method evidence |
| Controlled Phase 4 trained ablations | E0/E1/E2/E3 trained comparison across seeds 42, 123, 2026 | Strongest current method evidence |
| Phase 5/6 pilots | Counterfactual and latent-mode implementation checks | Pilot evidence only |

Do not mix these protocols into a single leaderboard.

## Main Report Numbers

### Scene-Disjoint Reproduced Baseline

| Metric | Value |
|--------|-------|
| Test samples | 4,255 |
| Acc@1 | 30.83% |
| Acc@5 | 91.87% |

### Hard Subsets

| Subset | Acc@1 |
|--------|-------|
| Same-class clutter | 21.96% |
| High clutter | 16.07% |
| Multi-anchor | 11.90% |

### Anchor Coverage Diagnostic

| Coverage result | Value |
|-----------------|-------|
| Top-5 covers at least one annotated anchor | 67.87% |
| Top-5 covers all annotated anchors | 54.20% |
| Baseline-wrong anchor-evaluable cases with no annotated anchor in top-5 | 33.95% |

### Dense Relation Diagnostic Line

| Method | Test Acc@1 | Test Acc@5 | Net | Status |
|--------|------------|------------|-----|--------|
| Clean base | 30.83% | 91.87% | -- | Reference |
| Dense-no-cal-v1 | 31.05% | 92.01% | +9 | Retained |
| Dense-calibrated-v2 | 30.55% | 91.80% | -12 | Frozen |
| Dense-v2-AttPool | 25.24% | 79.95% | -238 | Frozen |
| Dense-v3-Geo | 24.32% | 79.06% | -277 | Frozen |
| Dense-v4-HardNeg | 24.47% | 79.41% | -271 | Frozen |

### Controlled Phase 4 Trained Ablations

| ID | Configuration | Params | Acc@1 | Acc@5 |
|----|---------------|--------|-------|-------|
| E0 | Dense relation baseline | 398K | 28.61 ± 0.01 | 88.13 ± 0.05 |
| E2 | Near parameter-matched dense control | 622K | 28.87 ± 0.09 | 88.97 ± 0.05 |
| E1 | Conditioned + viewpoint supervision | 623K | 30.50 ± 0.05 | 91.09 ± 0.06 |
| E3 | Conditioned, no viewpoint supervision | 623K | 30.45 ± 0.12 | 91.04 ± 0.05 |

## Local Artifact Paths

| Artifact | Path |
|----------|------|
| Report PDF | `course-line/report/report.pdf` |
| Report source | `course-line/report/main.tex` and `course-line/report/sections/` |
| Report bibliography | `course-line/report/references.bib` |
| Pipeline figure assets | `assets/figures/figure1_pipeline.png`, `assets/figures/figure1_pipeline.svg`, `assets/figures/figure1_pipeline.pdf` |
| Figure generator | `assets/figures/generate_figure1_pipeline.py` |
| Report TikZ figure sources | `course-line/report/figures/` |
| Course evidence map | `course-line/FINAL_PROJECT_EVIDENCE_MAP.md` |
| Claim boundary | `course-line/CLAIM_BOUNDARY.md` |
| Reproducibility notes | `course-line/REPRODUCIBILITY_NOTES.md` |
| Main diagnostic summary | `reports/final_diagnostic_master_summary.md` |
| Baseline subset diagnostics | `reports/cover3d_phase1_baseline_subset_results.md` |
| Coverage diagnostics | `reports/cover3d_coverage_diagnostics/coverage_diagnostics_report.md` |
| Coverage summaries | `reports/cover3d_coverage_diagnostics/coverage_summary.json`, `reports/cover3d_coverage_diagnostics/subset_coverage_curves.csv` |
| Phase 4 aggregates | `update/reports/R001_aggregate_results.json`, `update/reports/R002_aggregate_results.json`, `update/reports/R004_aggregate_results.json` |

## Claim Boundary

- Supported: conditional / latent-conditioned relation computation improves over the controlled dense relation baseline in Phase 4.
- Not supported: explicit viewpoint learning as the causal source of the gain.
- Not solved: multi-anchor reasoning.
- Not final claims: Phase 5 counterfactual and Phase 6 latent-mode pilots.
