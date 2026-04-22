# Pre-Method Full Validation

Date: 2026-04-20

## Inputs

- Clean checkpoint: `outputs/20260420_clean_sorted_vocab_baseline/formal/best_model.pt`
- Clean test predictions with logits: `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_predictions.json`
- Clean test results: `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_results.json`
- Test manifest: `data/processed/scene_disjoint/official_scene_disjoint/test_manifest.jsonl`

## Validation Suite

| Check | Command / Output | Status |
| --- | --- | --- |
| Coverage diagnostics | `python scripts/run_cover3d_coverage_diagnostics.py --predictions outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_predictions.json --output-dir reports/pre_method_clean_coverage_diagnostics` | PASS |
| Dense-vs-sparse recovery | `reports/pre_method_clean_coverage_diagnostics/dense_sparse_recovery_summary.json` | PASS |
| Minimal P3 proxy | `python scripts/run_cover3d_p3_minimal_verification.py --coverage reports/pre_method_clean_coverage_diagnostics/per_sample_coverage.jsonl --output-dir reports/pre_method_clean_p3_minimal` | PASS |
| Real-logits P3 entry | `python scripts/run_cover3d_real_logits_p3_entry.py --predictions outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_predictions.json --coverage reports/pre_method_clean_coverage_diagnostics/per_sample_coverage.jsonl --output-dir reports/pre_method_clean_real_logits_p3` | PASS |
| COVER-3D module smoke | `python scripts/smoke_test_cover3d_phase2.py` | PASS, 8/8 tests |
| Syntax checks | `python -m py_compile ...` | PASS |
| Whitespace check | `git diff --check` | PASS |

## Coverage Diagnostics

Output directory: `reports/pre_method_clean_coverage_diagnostics/`

Topline:

- Samples analyzed: 4,255
- Clean base Acc@1 / Acc@5: 30.83% / 91.87%
- Anchor-annotated samples: 434
- Geometry-evaluable anchor samples: 417
- Dense any/all reachability: 100.00% / 100.00%
- Any-anchor coverage@5: 67.87%
- All-anchor coverage@5: 54.20%

Dense-vs-sparse recovery at k=5:

- Baseline-wrong anchor samples: 334
- Geometry-evaluable baseline-wrong anchor samples: 324
- Sparse misses every anchor: 110 (33.95%)
- Sparse misses at least one anchor: 156 (48.15%)
- Dense recovered any/all sparse misses at candidate-set level: 100.00% / 100.00%

## Minimal P3 Proxy

Output directory: `reports/pre_method_clean_p3_minimal/`

| Variant | Acc@1 | Acc@5 |
| --- | ---: | ---: |
| Base | 30.83% | 91.87% |
| Sparse no-cal | 32.48% | 91.84% |
| Dense no-cal | 32.76% | 91.30% |
| Dense simple gate | 31.77% | 91.82% |

Dense no-cal proxy:

- recovered from base-wrong: 91
- harmed from base-correct: 9
- net correct delta: +82

Dense simple gate proxy:

- recovered from base-wrong: 40
- harmed from base-correct: 0
- net correct delta: +40

Boundary: this remains an oracle-anchor geometry proxy and is not a formal method result.

## Real-Logits Entry

Output directory: `reports/pre_method_clean_real_logits_p3/`

| Variant | Acc@1 | Acc@5 |
| --- | ---: | ---: |
| Base | 30.83% | 91.87% |
| Sparse no-cal | 33.44% | 92.20% |
| Dense no-cal | 34.29% | 92.22% |
| Dense calibrated | 34.24% | 92.17% |

This check proves the minimal real-logits comparison path can run. It still uses proxy relation evidence, so the numbers are readiness evidence only.

## Base / Dense Start Conditions

Ready:

- Base logits are exported from the clean checkpoint.
- Dense no-cal and dense calibrated entry exists and consumes the clean export.
- COVER-3D dense relation, soft anchor posterior, and calibration modules import and forward on CPU smoke.
- `configs/cover3d_referit_wrapper.yaml` now points to the clean checkpoint/logits/class vocabulary.

Not claimed:

- No formal COVER-3D training was started.
- No learned dense relation scorer result is claimed.
- No parser/LLM/VLM mainline was extended.

## Validation Conclusion

All pre-method validation checks passed. The repository is ready to start formal COVER-3D method validation from the clean sorted-vocabulary baseline.
