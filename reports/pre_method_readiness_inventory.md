# Pre-Method Readiness Inventory

Date: 2026-04-20

## Executive Status

The repository now has a clean sorted-vocabulary ReferIt3DNet baseline route that can train, evaluate, export logits/margins/entropy/predictions, and feed the pre-method diagnostics.

Apr9 remains the prediction-level benchmark anchor at 30.79% test Acc@1, but it is not used as a fresh-export logits source. Apr10 follows the same learned-class-vocabulary risk pattern. Apr14 vocabulary-fix remains useful historical smoke evidence only; it is not the formal pre-method base.

## Asset Inventory

| Item | Status | Path / Entry | Notes |
| --- | --- | --- | --- |
| Clean sorted-vocabulary train config | Available | `repro/referit3d_baseline/configs/clean_sorted_vocab_baseline.yaml` | Full 30-epoch route, official scene-disjoint manifests, DistilBERT cache, 516 sorted classes. |
| Clean smoke/resume config | Available | `repro/referit3d_baseline/configs/clean_sorted_vocab_smoke.yaml` | Debug max_batches=2 for train/resume/eval/export checks. |
| Baseline train entry | Available, repaired | `repro/referit3d_baseline/scripts/train.py` | Now supports resume and saves checkpoint class vocabulary. |
| Baseline eval/export entry | Available, repaired | `repro/referit3d_baseline/scripts/evaluate.py` | Now exports entropy/probability/tie diagnostics and uses checkpoint vocabulary when present. |
| Logits readiness checker | Available | `scripts/check_clean_baseline_readiness.py` | Verifies prediction/logit/eval consistency and class-vocab metadata. |
| Real-logits P3 entry | Available | `scripts/run_cover3d_real_logits_p3_entry.py` | Minimal Base / Sparse-no-cal / Dense-no-cal / Dense-calibrated readiness entry using real base logits. |
| Scene-disjoint split | Available | `data/processed/scene_disjoint/official_scene_disjoint/*.jsonl` | train=33,829, val=3,419, test=4,255, 641 scenes. |
| Split manifests | Available | `data/processed/splits/official_scene_disjoint_*.txt` | Existing scene-disjoint split manifests. |
| Text features | Rebuilt | `data/text_features/full_official_nr3d/*.npy` | train=(33829,768), val=(3419,768), test=(4255,768), validation issues=0. |
| Object/geometry features | Available with caveats | `data/geometry`, `data/object_features` | Coverage diagnostics can consume geometry; official manifests still record fallback centers/sizes. |
| Current diagnostics | Available | `scripts/run_cover3d_coverage_diagnostics.py`, `scripts/run_cover3d_p3_minimal_verification.py` | Both consumed clean baseline predictions. |
| COVER-3D module smoke | Available | `scripts/smoke_test_cover3d_phase2.py` | 8/8 smoke tests passed. |
| Formal wrapper config | Updated | `configs/cover3d_referit_wrapper.yaml` | Now points to clean checkpoint/logits instead of Apr9 checkpoint. |

## Required Product Paths

| Product | Path |
| --- | --- |
| Clean checkpoint | `outputs/20260420_clean_sorted_vocab_baseline/formal/best_model.pt` |
| Clean class vocabulary | `outputs/20260420_clean_sorted_vocab_baseline/formal/class_vocabulary.json` |
| Clean training history | `outputs/20260420_clean_sorted_vocab_baseline/formal/training_history.json` |
| Clean val export | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_val_predictions.json` |
| Clean test export | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_predictions.json` |
| Readiness summaries | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/readiness_{val,test}_summary.json` |
| Clean coverage diagnostics | `reports/pre_method_clean_coverage_diagnostics/` |
| Clean minimal P3 proxy | `reports/pre_method_clean_p3_minimal/` |
| Clean real-logits P3 entry | `reports/pre_method_clean_real_logits_p3/` |

## Apr9 / Apr10 Boundary

Apr9 is retained only as prediction-level benchmark anchor:

- `outputs/20260409_learned_class_embedding/formal/eval_test_results.json`
- Test Acc@1 / Acc@5: 30.79% / 91.75%

It is not suitable as the fresh-export logits source because the original learned class-name to class-index mapping was not frozen in the checkpoint artifacts. Regenerated DistilBERT features and current sorted vocabulary do not reproduce the trusted Apr9 logits. Apr10 smoke belongs to the same learned-class family and does not repair that missing mapping. This is documented in `reports/cover3d_logits_audit/base_logits_recovery_audit.md`.

## Inventory Conclusion

All pre-method infrastructure assets now exist and are connected through the clean sorted-vocabulary baseline. The only remaining boundary is interpretive: the proxy P3 and real-logits entry are readiness checks, not formal COVER-3D method results.
