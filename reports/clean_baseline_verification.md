# Clean Baseline Verification

Date: 2026-04-20

## Config

- Full config: `repro/referit3d_baseline/configs/clean_sorted_vocab_baseline.yaml`
- Smoke config: `repro/referit3d_baseline/configs/clean_sorted_vocab_smoke.yaml`
- Manifest dir: `data/processed/scene_disjoint/official_scene_disjoint`
- Text feature dir: `data/text_features/full_official_nr3d`
- Class vocabulary ordering: `sorted(class_name)`
- Class vocabulary size: 516

## Checkpoints

| Purpose | Path |
| --- | --- |
| Smoke checkpoint | `outputs/20260420_clean_sorted_vocab_baseline/smoke/best_model.pt` |
| Full clean checkpoint | `outputs/20260420_clean_sorted_vocab_baseline/formal/best_model.pt` |
| Full clean final model | `outputs/20260420_clean_sorted_vocab_baseline/formal/final_model.pt` |
| Frozen class vocabulary | `outputs/20260420_clean_sorted_vocab_baseline/formal/class_vocabulary.json` |

## Training Verification

Full training completed 30 epochs.

Best epoch:

```json
{"epoch": 28, "train_loss": 1.2436291132412904, "train_acc": 0.35416358745455084, "val_acc_at_1": 0.3319684118163206, "val_acc_at_5": 0.9289265867212635}
```

Resume smoke completed from `outputs/20260420_clean_sorted_vocab_baseline/smoke/best_model.pt` and ran the remaining configured epoch.

## Evaluation Metrics

| Split | Acc@1 | Acc@5 | Samples | Export path |
| --- | ---: | ---: | ---: | --- |
| val | 33.20% | 92.89% | 3,419 | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_val_predictions.json` |
| test | 30.83% | 91.87% | 4,255 | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_predictions.json` |

The test score is an acceptable clean baseline and is aligned with the Apr9 prediction-level anchor (30.79%) while being independently reproducible from a checkpoint that stores the sorted class vocabulary.

## Export Fields

Each exported prediction row contains:

- `pred_top1`, `pred_top5`, `correct_at_1`, `correct_at_5`
- `base_logits`
- `base_margin`
- `base_entropy`
- `base_prob_margin`
- `base_top1_logit`, `base_top2_logit`
- `base_top1_probability`, `base_top2_probability`
- `base_top1_tie_count`
- `target_logit`, `target_probability`, `target_rank`

## Export Consistency Checks

| Split | Checker output | Acc@1 | Acc@5 | Issues |
| --- | --- | ---: | ---: | ---: |
| val | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/readiness_val_summary.json` | 33.20% | 92.89% | 0 |
| test | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/readiness_test_summary.json` | 30.83% | 91.87% | 0 |

Test readiness summary:

- mean base margin: 0.0999
- mean base entropy: 1.2867
- rows with top1 ties: 3,967
- max top1 tie count: 18

This confirms that tie-aware features are necessary for calibration; raw top1-top2 margin alone is weak for this baseline.

## Verification Conclusion

The clean sorted-vocabulary baseline is established and usable as the formal pre-method infrastructure base. It satisfies:

- train/resume/evaluate chain works;
- val/test metrics are in acceptable clean-baseline range;
- logits/margins/entropy/predictions export from the same run;
- export and evaluation metrics are consistent;
- class vocabulary is frozen with the checkpoint.
