# EXPERIMENT_LEDGER

| Exp ID | Purpose | Status | Script / Config | Result Path | Key Metrics | Claim Supported |
|---|---|---|---|---|---|---|
| E-001 | Clean baseline reproduction | COMPLETE | `scripts/train_cover3d_round1.py` | `outputs/20260420_clean_sorted_vocab_baseline/` | Acc@1: 30.83%, Acc@5: 91.87% | C-003 |
| E-002 | Phase 4 E0 baseline (controlled) | COMPLETE | `scripts/train_cover3d_viewpoint.py` | `outputs/phase4_ablation/` | Acc@1: 28.60% | C-001 (baseline) |
| E-003 | Phase 4 E1 with supervision | COMPLETE | `scripts/train_cover3d_viewpoint.py` | `outputs/phase4_ablation/` | Acc@1: 30.74% (+2.14%) | C-001 |
| E-004 | Phase 4 E3 no supervision | COMPLETE | `scripts/train_cover3d_viewpoint.py` | `outputs/phase4_ablation/` | Acc@1: 30.65% (+2.04%) | C-002 |
| E-005 | Phase 5 E0-matched pilot | COMPLETE (1 epoch) | `scripts/train_cover3d_counterfactual.py` | `outputs/phase5_counterfactual/pilot_E0/` | Val Acc@1: 34.30% | C-006 (baseline) |
| E-006 | Phase 5 E1-CF pilot | COMPLETE (1 epoch) | `scripts/train_cover3d_counterfactual.py` | `outputs/phase5_counterfactual/pilot_E1_CF/` | Val Acc@1: 34.42% (+0.12 pp) | C-006 |
| E-007 | Phase 5 E2-RHN pilot | FAILED | `scripts/train_cover3d_counterfactual.py` | `outputs/phase5_counterfactual/pilot_E2_RHN/` | RHN coverage: 0/30447 | N/A (cannot claim) |
| E-008 | Phase 6 E0-K1 pilot | COMPLETE (1 epoch) | `scripts/train_cover3d_latent_modes.py` | `outputs/phase6_latent_modes/pilot_E0_K1/` | Val Acc@1: 34.10% | C-007 (baseline) |
| E-009 | Phase 6 E1-K4 pilot | COMPLETE (1 epoch) | `scripts/train_cover3d_latent_modes.py` | `outputs/phase6_latent_modes/pilot_E1_K4/` | Val Acc@1: 34.19% (+0.09 pp) | C-007 |
| E-010 | Dense-no-cal-v1 diagnostic | COMPLETE (10 epochs) | `scripts/train_cover3d_round1.py` | `outputs/cover3d_round1/` | Acc@1: 31.05% (+0.22%) | B-001 (auxiliary) |
| E-011 | Full E0-matched (10 epochs) | PENDING | `scripts/train_cover3d_counterfactual.py` | `outputs/phase7_full/` (started) | MISSING_RESULT | C-006 (strengthen) |
| E-012 | Full E1-CF (10 epochs) | PENDING | `scripts/train_cover3d_counterfactual.py` | `outputs/phase7_full/` (started) | MISSING_RESULT | C-006 (strengthen) |
| E-013 | Full K=1 (80 epochs) | PENDING | `scripts/train_cover3d_latent_modes.py` | UNKNOWN | MISSING_RESULT | C-007 (strengthen) |
| E-014 | Full K=4 (80 epochs) | PENDING | `scripts/train_cover3d_latent_modes.py` | UNKNOWN | MISSING_RESULT | C-007 (strengthen) |
| E-015 | Multi-seed validation | PENDING | UNKNOWN | UNKNOWN | MISSING_RESULT | All claims |

## Rules

- Do not write numerical results unless they come from logs, tables, reports, or verified outputs.
- Use UNKNOWN when the result is not yet found.
- Use MISSING_RESULT when an expected result is absent.
- Use FAILED when experiment did not execute correctly.
- Use PENDING when experiment is planned but not started.

## Pilot vs Full Training

| Type | Epochs | Batch Size | Status |
|------|--------|------------|--------|
| Pilot | 1 | 8 | Complete (local) |
| Full | 10 | 64 | Pending (requires headless) |
| MoE Full | 80 | 64 | Pending |