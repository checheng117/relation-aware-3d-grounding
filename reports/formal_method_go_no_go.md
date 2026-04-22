# Formal Method Go / No-Go

GO

Date: 2026-04-20

## Answers

1. Current repository satisfies the required pre-method validation: yes.
2. Clean baseline + logits/margins infrastructure exists: yes.
3. Real Base / Dense-no-cal / Dense-calibrated start conditions exist: yes, via `scripts/run_cover3d_real_logits_p3_entry.py` and the updated clean wrapper config.
4. Formal COVER-3D method validation can now start: yes.
5. Remaining blocker: none for starting formal method validation. The boundary is that current P3 numbers are readiness/proxy checks, not formal method claims.
6. Next command:

```bash
python scripts/run_cover3d_real_logits_p3_entry.py \
  --manifest data/processed/scene_disjoint/official_scene_disjoint/test_manifest.jsonl \
  --coverage reports/pre_method_clean_coverage_diagnostics/per_sample_coverage.jsonl \
  --predictions outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_predictions.json \
  --geometry-dir data/geometry \
  --output-dir reports/pre_method_clean_real_logits_p3 \
  --dense-lambda 0.5
```

This is the clean real-logits entry to branch from when replacing proxy relation evidence with the learned COVER-3D dense scorer. Do not use Apr9 fresh-export logits.

## Evidence

Clean baseline:

- Config: `repro/referit3d_baseline/configs/clean_sorted_vocab_baseline.yaml`
- Checkpoint: `outputs/20260420_clean_sorted_vocab_baseline/formal/best_model.pt`
- Class vocabulary: `outputs/20260420_clean_sorted_vocab_baseline/formal/class_vocabulary.json`
- Training history: `outputs/20260420_clean_sorted_vocab_baseline/formal/training_history.json`
- Best val Acc@1 / Acc@5: 33.20% / 92.89%
- Test Acc@1 / Acc@5: 30.83% / 91.87%

Export readiness:

- Val export: `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_val_predictions.json`
- Test export: `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_predictions.json`
- Val checker: `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/readiness_val_summary.json`, PASS
- Test checker: `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/readiness_test_summary.json`, PASS

Diagnostics:

- Coverage diagnostics: `reports/pre_method_clean_coverage_diagnostics/coverage_diagnostics_report.md`
- Dense-vs-sparse recovery: `reports/pre_method_clean_coverage_diagnostics/dense_sparse_recovery_summary.json`
- Minimal P3 proxy: `reports/pre_method_clean_p3_minimal/p3_minimal_report.md`
- Real-logits P3 entry: `reports/pre_method_clean_real_logits_p3/real_logits_p3_report.md`
- COVER-3D smoke: `reports/cover3d_phase2_smoke_diagnostics.json`, success=true

Static checks:

- `python -m py_compile ...`: PASS
- `git diff --check`: PASS

## Boundaries

- Apr9 30.79 remains the prediction-level benchmark anchor only.
- Apr9/Apr10 checkpoints are not used as fresh-export logits sources.
- Apr14 vocabulary-fix remains historical infrastructure smoke only.
- No formal COVER-3D training was run in this pass.
- No parser/LLM/VLM mainline was expanded.

## Conclusion

The formal method validation gate is open. Start from the clean sorted-vocabulary baseline and clean real-logits products, not from Apr9 hidden-mapping assets.
