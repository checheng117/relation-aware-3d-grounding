# Real-Logits P3 Entry Readiness

**Status**: pre-method infrastructure check, not a formal COVER-3D result.

## Inputs

- Manifest: `data/processed/scene_disjoint/official_scene_disjoint/test_manifest.jsonl`
- Predictions with logits: `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_predictions.json`
- Coverage rows: `reports/pre_method_clean_coverage_diagnostics/per_sample_coverage.jsonl`
- Geometry dir: `data/geometry`

## Overall

| Variant | Acc@1 | Acc@5 |
| --- | ---: | ---: |
| base | 30.83% | 91.87% |
| sparse_no_cal | 33.44% | 92.20% |
| dense_no_cal | 34.29% | 92.22% |
| dense_calibrated | 34.24% | 92.17% |

## Recovery/Harm

| Variant | Recovered | Harmed | Net Correct Delta |
| --- | ---: | ---: | ---: |
| sparse_no_cal | 121 | 10 | +111 |
| dense_no_cal | 169 | 22 | +147 |
| dense_calibrated | 167 | 22 | +145 |

## Readiness Interpretation

This entry proves that real exported base logits can drive the Base / Dense-no-cal / Dense-calibrated comparison path. The relation signal is still the existing oracle-anchor geometry proxy, so these numbers are not method evidence.

## Artifacts

- `real_logits_p3_summary.json`
- `real_logits_p3_per_sample.jsonl`
- `real_logits_p3_casebook.json` / `.csv`

## Casebook Preview

| Event | Scene | Target | Base | Dense no-cal | Dense calibrated | Gate | Utterance |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| recovered | scene0340_00 | 31 | 28 | 31 | 31 | 0.034512 | 'The lamp sitting on the floor next to the curtains.' |
| recovered | scene0340_00 | 31 | 28 | 31 | 31 | 0.035048 | This lamp is right next to a window |
| recovered | scene0340_00 | 31 | 28 | 31 | 31 | 0.034307 | It is the lamp in the middle of the curtains. |
| recovered | scene0340_00 | 31 | 28 | 31 | 31 | 0.04326 | Place the camera so the two beds are on the right side of the image. Select the lamp that is in the center, on the far wall, in front of a window. |
| recovered | scene0340_00 | 31 | 28 | 31 | 31 | 0.035673 | The correct lamp is directly in front of the window and inbetween the two curtains. |
| recovered | scene0058_00 | 28 | 10 | 28 | 28 | 0.031747 | The desk that is on the side of the room where there are not chairs against the wall. |
| recovered | scene0058_00 | 28 | 10 | 28 | 28 | 0.031555 | NOT the desk with two blue chairs, including a non-swivel chair. |
| recovered | scene0058_00 | 28 | 10 | 28 | 28 | 0.031529 | pick the smaller desk that has only one chair. |
| recovered | scene0058_00 | 28 | 10 | 28 | 28 | 0.031187 | The desk far away from the blue door |
| recovered | scene0330_00 | 15 | 10 | 15 | 15 | 0.069172 | There are two blue pillows on the bed. When facing the wall, its the blue pillow on the right. |
| recovered | scene0640_00 | 2 | 1 | 2 | 2 | 0.02993 | 'The lamp over the desk.' |
| recovered | scene0640_00 | 2 | 1 | 2 | 2 | 0.029898 | The lamp at the foot of the bed. |
