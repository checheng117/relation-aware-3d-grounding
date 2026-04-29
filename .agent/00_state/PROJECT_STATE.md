# PROJECT_STATE

## Project Name
Latent Conditioned Relation Scoring for 3D Visual Grounding

## Project Type
Research paper project / 3D visual grounding codebase

## Current Goal
Publish paper demonstrating latent-conditioned relation scoring improves 3D grounding. Method validation complete (pilot); full experiments require headless environment.

## Current Stage
Method pilot validation complete. Paper writing in progress (course-line). Full training validation pending.

## Core Research Question
Can latent-conditioned relation scoring improve compositional relational inference in 3D referring-expression grounding?

## Current Main Claim
**Primary**: Latent conditioned architecture provides +2.1% gain over baseline (Phase 4.5 validated, supervision not needed).

**Secondary (Pilot)**: Counterfactual loss provides +0.12 pp auxiliary improvement (Phase 5 pilot).

**Exploratory (Pending)**: Multi-mode latent relation (K=4) may outperform single-mode (Phase 6 pilot +0.09 pp).

## Method Summary
**Architecture**: ViewpointConditionedRelationScorer - augments pairwise relation MLPs with learnable conditioning variable for multi-mode relation reasoning.

**Key Insight**: Gain comes from conditional computation, not viewpoint supervision (E1 ≈ E3).

**Optional Enhancement**: Counterfactual relation learning via margin ranking loss.

## Existing Evidence
| Level | Evidence | Status |
|-------|----------|--------|
| A | Baseline 30.83%, scene-disjoint verified | Final |
| A | Hard subset diagnostics (same-class 21.96%) | Final |
| A | Coverage failure 33.95% | Final |
| B | Architecture gain +2.14% (E1 vs E0) | Controlled |
| B | Supervision not needed (E1 ≈ E3) | Controlled |
| C | Counterfactual +0.12 pp pilot | Pilot |
| C | MoE K=4 +0.09 pp pilot | Pilot |

## Missing Evidence
- Full training (10 epochs) validation
- Multi-seed statistical significance
- RHN ablation (FAILED: missing metadata)
- Multi-anchor improvement (UNSUPPORTED: 24.73% baseline = method)

## Important Files
**Core Method**: `src/rag3d/relation_reasoner/viewpoint_conditioned_scorer.py`
**Key Reports**: `reports/final_diagnostic_master_summary.md`, `update/PAPER_CLAIM_REFRAMING.md`
**Claim Boundary**: `course-line/CLAIM_BOUNDARY.md`
**Evidence Map**: `course-line/FINAL_PROJECT_EVIDENCE_MAP.md`

## Important Scripts
**Training**: `scripts/train_cover3d_counterfactual.py`, `scripts/train_cover3d_latent_modes.py`
**Evaluation**: `scripts/evaluate_viewpoint_conditioned_scorer.py`
**Diagnosis**: `scripts/phase4_5_mechanism_diagnosis.py`

## Important Results
**Baseline**: `outputs/20260420_clean_sorted_vocab_baseline/` (30.83%)
**Phase 5**: `outputs/phase5_counterfactual/` (pilot)
**Phase 6**: `outputs/phase6_latent_modes/` (pilot)

## Current Risks
- Counterfactual gain is small (+0.12 pp) - may not scale
- RHN ablation impossible - cannot distinguish CF-specific benefit
- Multi-anchor shows NO improvement - core hard subset not addressed
- Single-seed results only - statistical significance unknown
- Local machine cannot run full training - requires headless environment
- Cannot claim "CF > Random", "Multi-anchor solved", or "SOTA"

## Allowed Agent Actions
- Read files.
- Summarize the project.
- Create plans under `.agent/10_plans/`.
- Create reports under `.agent/20_exec/`, `.agent/30_verify/`, `.agent/50_reviews/`.
- Edit Markdown and LaTeX files only when a plan explicitly allows it.
- Edit code only when a plan explicitly allows it.
- Run lightweight checks.

## Forbidden Agent Actions
- Do not delete datasets.
- Do not delete checkpoints.
- Do not overwrite experimental results.
- Do not invent metrics.
- Do not launch long GPU training unless explicitly instructed.
- Do not change the main research claim without updating `CLAIM_LEDGER.md`.