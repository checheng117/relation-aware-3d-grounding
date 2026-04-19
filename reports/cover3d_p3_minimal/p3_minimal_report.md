# COVER-3D Minimal P3 Verification

**Date**: 2026-04-19
**Training**: none.
**Status**: offline rank-proxy / oracle-anchor geometry smoke test, not a learned COVER-3D result.

## Executive Summary

This diagnostic enters the minimal P3 stage without training a new model. It compares the trusted 30.79% ReferIt3DNet predictions against a sparse relation proxy, a dense uncalibrated relation proxy, and a dense relation proxy with a simple deterministic gate.

The trusted prediction file does not contain logits, and a fresh logits export attempt did not reproduce the trusted baseline because the matching full-test BERT feature cache is not available. Therefore this report uses a rank proxy from the stored top-5 predictions rather than claiming true base-margin calibration.

These results should be read as a low-risk calibration smoke test: they can reveal whether dense relation evidence has both recoveries and harms, but they do not prove calibration necessity or final method effectiveness.

Under this proxy, dense no-cal improves Acc@1 from **30.79%** to **32.57%**, with **84** base-wrong recoveries and **8** base-correct harms. The simple gate prevents **8** dense no-cal harms, but loses **57** dense no-cal recoveries, so the current gate is useful as a risk signal but not yet a finished calibration design.

## Variants

- `Base`: frozen trusted top-1/top-5 predictions.
- `Sparse no-cal`: base rank proxy plus relation proxy using only annotated anchors that are inside sparse top-5 nearest-neighbor coverage.
- `Dense no-cal`: base rank proxy plus relation proxy using all geometry-recovered annotated anchors.
- `Dense simple gate`: dense proxy with a deterministic gate based on predicted-class ambiguity, anchor-count uncertainty, and relation-score margin.

## Overall Metrics

| Variant | Acc@1 | Acc@5 |
| --- | ---: | ---: |
| base | 30.79% | 91.75% |
| sparse_no_cal | 32.36% | 91.75% |
| dense_no_cal | 32.57% | 91.14% |
| dense_simple_gate | 31.73% | 91.63% |

## Base-Wrong Recovery vs Base-Correct Harm

| Variant | Recovered | Harmed | Net Correct Delta |
| --- | ---: | ---: | ---: |
| sparse_no_cal | 67 (2.28%) | 0 (0.00%) | +67 |
| dense_no_cal | 84 (2.85%) | 8 (0.61%) | +76 |
| dense_simple_gate | 40 (1.36%) | 0 (0.00%) | +40 |

## Gate Behavior

- Mean gate lambda: **0.14442**
- Median gate lambda: **0.136526**
- Mean relation margin: **0.188977**
- Dense no-cal harms prevented by gate: **8**
- Dense no-cal recoveries lost by gate: **57**

## Hard-Subset Metrics

| Subset | n | Base | Sparse no-cal | Dense no-cal | Dense gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| same_class_clutter | 2373 | 21.41% | 22.80% | 23.18% | 22.17% |
| same_class_high_clutter | 697 | 15.78% | 18.36% | 18.22% | 16.64% |
| multi_anchor | 93 | 19.35% | 27.96% | 34.41% | 21.51% |
| relative_position | 968 | 29.24% | 31.30% | 31.71% | 29.75% |
| relational | 2577 | 30.66% | 32.17% | 32.48% | 31.24% |
| dense_scene | 752 | 29.39% | 31.38% | 31.25% | 30.05% |
| baseline_wrong | 2945 | 0.00% | 2.28% | 2.85% | 1.36% |
| baseline_correct | 1310 | 100.00% | 100.00% | 99.39% | 100.00% |
| sparse_any_missed_at_5 | 134 | 17.91% | 17.91% | 26.12% | 29.85% |
| sparse_all_incomplete_at_5 | 191 | 17.80% | 20.94% | 25.65% | 26.70% |

## Interpretation

The useful question here is not whether this proxy beats the baseline overall. The useful question is whether dense relation evidence creates a measurable recovery pool and whether uncalibrated dense evidence also creates a harm pool on base-correct samples.

If dense no-cal recovers hard cases but also harms base-correct cases, then the calibration hypothesis has a concrete target. If the simple gate prevents some harms without destroying all recoveries, then P3 becomes worth testing with real logits and a learned dense scorer.

## Boundary

These results do not establish that a learned dense relation scorer improves final grounding accuracy, nor do they prove calibration necessity. They only test whether dense oracle-anchor geometry evidence can expose the benefit/harm pattern that calibration is supposed to control.

## Artifacts

- `p3_minimal_summary.json`: aggregate metrics and transition counts.
- `p3_minimal_per_sample.jsonl`: per-sample variant predictions and gate diagnostics.
- `p3_minimal_casebook.json` / `.csv`: recovered and harmed examples.

## Casebook Preview

| Event | Scene | Target | Base | Dense no-cal | Dense gate | Gate | Utterance |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| recovered | scene0340_00 | 31 | 28 | 31 | 28 | 0.116835 | 'The lamp sitting on the floor next to the curtains.' |
| recovered | scene0340_00 | 39 | 28 | 39 | 28 | 0.144616 | Its the lamp that's on and is located to the right of a hung up picture frame. |
| recovered | scene0340_00 | 31 | 28 | 31 | 28 | 0.116835 | This lamp is right next to a window |
| recovered | scene0058_00 | 28 | 10 | 28 | 10 | 0.124881 | The desk that is on the side of the room where there are not chairs against the wall. |
| recovered | scene0058_00 | 28 | 10 | 28 | 10 | 0.124881 | NOT the desk with two blue chairs, including a non-swivel chair. |
| recovered | scene0058_00 | 28 | 10 | 28 | 10 | 0.124881 | pick the smaller desk that has only one chair. |
| recovered | scene0058_00 | 28 | 10 | 28 | 10 | 0.124881 | The desk far away from the blue door |
| recovered | scene0330_00 | 13 | 8 | 13 | 8 | 0.100089 | pick the blue box that is on the left |
| recovered | scene0330_00 | 12 | 10 | 12 | 10 | 0.184052 | the pillow is the zebra stripe one on the left of the bed |
| recovered | scene0330_00 | 12 | 10 | 12 | 10 | 0.165751 | Facing striped pillows on couch, it's the lefthand striped pillow. |
| recovered | scene0330_00 | 12 | 10 | 12 | 10 | 0.165751 | facing the 4 pillows, the striped pillow on your left |
| recovered | scene0640_00 | 2 | 1 | 2 | 1 | 0.125485 | 'The lamp over the desk.' |
