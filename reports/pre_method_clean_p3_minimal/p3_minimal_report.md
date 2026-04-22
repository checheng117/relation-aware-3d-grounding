# COVER-3D Minimal P3 Verification

**Date**: 2026-04-19
**Training**: none.
**Status**: offline rank-proxy / oracle-anchor geometry smoke test, not a learned COVER-3D result.

## Executive Summary

This diagnostic enters the minimal P3 stage without training a new model. It compares the trusted 30.79% ReferIt3DNet predictions against a sparse relation proxy, a dense uncalibrated relation proxy, and a dense relation proxy with a simple deterministic gate.

The trusted prediction file does not contain logits, and a fresh logits export attempt did not reproduce the trusted baseline because the matching full-test BERT feature cache is not available. Therefore this report uses a rank proxy from the stored top-5 predictions rather than claiming true base-margin calibration.

These results should be read as a low-risk calibration smoke test: they can reveal whether dense relation evidence has both recoveries and harms, but they do not prove calibration necessity or final method effectiveness.

Under this proxy, dense no-cal improves Acc@1 from **30.83%** to **32.76%**, with **91** base-wrong recoveries and **9** base-correct harms. The simple gate prevents **9** dense no-cal harms, but loses **64** dense no-cal recoveries, so the current gate is useful as a risk signal but not yet a finished calibration design.

## Variants

- `Base`: frozen trusted top-1/top-5 predictions.
- `Sparse no-cal`: base rank proxy plus relation proxy using only annotated anchors that are inside sparse top-5 nearest-neighbor coverage.
- `Dense no-cal`: base rank proxy plus relation proxy using all geometry-recovered annotated anchors.
- `Dense simple gate`: dense proxy with a deterministic gate based on predicted-class ambiguity, anchor-count uncertainty, and relation-score margin.

## Overall Metrics

| Variant | Acc@1 | Acc@5 |
| --- | ---: | ---: |
| base | 30.83% | 91.87% |
| sparse_no_cal | 32.48% | 91.84% |
| dense_no_cal | 32.76% | 91.30% |
| dense_simple_gate | 31.77% | 91.82% |

## Base-Wrong Recovery vs Base-Correct Harm

| Variant | Recovered | Harmed | Net Correct Delta |
| --- | ---: | ---: | ---: |
| sparse_no_cal | 70 (2.38%) | 0 (0.00%) | +70 |
| dense_no_cal | 91 (3.09%) | 9 (0.69%) | +82 |
| dense_simple_gate | 40 (1.36%) | 0 (0.00%) | +40 |

## Gate Behavior

- Mean gate lambda: **0.145061**
- Median gate lambda: **0.140161**
- Mean relation margin: **0.188977**
- Dense no-cal harms prevented by gate: **9**
- Dense no-cal recoveries lost by gate: **64**

## Hard-Subset Metrics

| Subset | n | Base | Sparse no-cal | Dense no-cal | Dense gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| same_class_clutter | 2373 | 21.74% | 23.22% | 23.68% | 22.50% |
| same_class_high_clutter | 697 | 16.07% | 18.94% | 18.94% | 16.93% |
| multi_anchor | 93 | 19.35% | 27.96% | 37.63% | 21.51% |
| relative_position | 968 | 29.86% | 32.02% | 32.64% | 30.37% |
| relational | 2577 | 30.73% | 32.40% | 32.79% | 31.32% |
| dense_scene | 752 | 29.79% | 32.18% | 32.18% | 30.45% |
| baseline_wrong | 2943 | 0.00% | 2.38% | 3.09% | 1.36% |
| baseline_correct | 1312 | 100.00% | 100.00% | 99.31% | 100.00% |
| sparse_any_missed_at_5 | 134 | 17.91% | 17.91% | 27.61% | 29.85% |
| sparse_all_incomplete_at_5 | 191 | 18.32% | 21.47% | 27.75% | 27.23% |

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
| recovered | scene0330_00 | 12 | 10 | 12 | 10 | 0.184052 | the pillow is the zebra stripe one on the left of the bed |
| recovered | scene0330_00 | 12 | 10 | 12 | 10 | 0.165751 | Facing striped pillows on couch, it's the lefthand striped pillow. |
| recovered | scene0330_00 | 15 | 2 | 15 | 9 | 0.141517 | Facing the couch or bed, it's the largest one on the right. |
| recovered | scene0330_00 | 12 | 10 | 12 | 10 | 0.165751 | facing the 4 pillows, the striped pillow on your left |
| recovered | scene0640_00 | 2 | 1 | 2 | 1 | 0.125485 | 'The lamp over the desk.' |
