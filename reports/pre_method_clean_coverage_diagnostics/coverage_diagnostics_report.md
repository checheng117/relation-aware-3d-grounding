# COVER-3D Coverage Diagnostics Report

**Date**: 2026-04-19
**Training**: none. This report uses frozen ReferIt3DNet baseline predictions.

## Executive Summary

The first-round diagnostics provide direct support for the coverage-failure hypothesis: sparse nearest-neighbor anchor selection leaves a substantial fraction of anchor evidence uncovered, and this gap is especially severe for multi-anchor expressions, where all-anchor coverage at k=5 is only **13.98%**. This motivates dense candidate-anchor coverage, but does not yet prove that dense reranking improves final grounding accuracy.

The updated diagnostics strengthen P2 beyond aggregate coverage statistics. Among baseline-wrong, anchor-evaluable samples, **33.95%** miss all annotated anchors under sparse top-5 selection and **48.15%** miss at least one annotated anchor. Dense all-pair candidate coverage recovers these sparse-miss cases at the candidate-set level, indicating that coverage failure is not merely descriptive but directly coupled with a substantial portion of baseline errors.

These results do not yet establish that a learned dense relation scorer improves final grounding accuracy, nor do they prove calibration necessity. They show that dense coverage creates recoverable conditions; whether these conditions translate into robust gains still depends on relation scoring and calibration.

## Diagnostic Definition

This is the first direct evidence pass for the COVER-3D coverage hypothesis.
Sparse coverage is measured with a target-centric nearest-neighbor anchor proxy: candidate anchors are ranked by Euclidean distance from the target object center. Dense reachability means all annotated anchors with recovered geometry are available to an all-pair candidate-anchor scorer.

This does not prove a trained sparse relation model would rank anchors identically. It directly tests the common assumption that useful anchors are local/nearby enough for a sparse shortlist.

## Inputs

- Manifest: `data/processed/scene_disjoint/official_scene_disjoint/test_manifest.jsonl`
- Predictions: `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_predictions.json`
- Entity annotations: `data/raw/referit3d/annotations/nr3d_annotations.json`
- Geometry dir: `data/geometry`
- Annotation matches: `{'none': 3673, 'exact_utterance': 171, 'target_fallback': 411}`

## Topline

- Samples analyzed: **4255**
- Baseline Acc@1 / Acc@5: **30.83% / 91.87%**
- Anchor-annotated samples: **434** (10.20%)
- Geometry-evaluable anchor samples: **417**
- Exact-utterance anchor samples: **125**
- Target-fallback anchor samples: **309**
- Dense any-anchor reachability: **100.00%**
- Dense all-anchor reachability: **100.00%**
- Mean / median minimum anchor rank: **5.41 / 4**

## Coverage@k

| k | Any Anchor Covered | All Anchors Covered |
| ---: | ---: | ---: |
| 1 | 29.50% | 18.94% |
| 3 | 46.04% | 35.97% |
| 5 | 67.87% | 54.20% |
| 10 | 83.69% | 73.62% |

## Subset Coverage Curves

| Subset | Count | Anchor Eval | Acc@1 | Any@1 | Any@3 | Any@5 | Any@10 | Mean Min Rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| same_class_clutter | 2373 | 268 | 21.74% | 36.94% | 50.37% | 73.13% | 88.81% | 4.44 |
| same_class_high_clutter | 697 | 144 | 16.07% | 37.50% | 57.64% | 75.00% | 89.58% | 4.38 |
| multi_anchor | 93 | 93 | 19.35% | 47.31% | 53.76% | 75.27% | 100.00% | 3.34 |
| relative_position | 968 | 118 | 29.86% | 27.12% | 40.68% | 68.64% | 82.20% | 5.39 |
| dense_scene | 752 | 82 | 29.79% | 30.49% | 53.66% | 79.27% | 81.71% | 4.77 |
| baseline_wrong | 2943 | 324 | 0.00% | 28.70% | 42.90% | 66.05% | 85.49% | 5.32 |
| baseline_correct | 1312 | 93 | 100.00% | 32.26% | 56.99% | 74.19% | 77.42% | 5.73 |
| annotation_exact | 171 | 121 | 27.49% | 33.88% | 52.07% | 72.73% | 87.60% | 4.65 |
| annotation_fallback | 411 | 296 | 24.57% | 27.70% | 43.58% | 65.88% | 82.09% | 5.72 |

## Initial Interpretation

- Under the nearest-neighbor sparse proxy, top-5 covers at least one annotated anchor in **67.87%** of geometry-evaluable anchor samples.
- Requiring all annotated anchors is stricter: top-5 covers all anchors in **54.20%** of geometry-evaluable anchor samples.
- For baseline-wrong samples with anchor geometry, top-5 any-anchor coverage is **66.05%**.
- Multi-anchor samples are the stress case: top-5 any-anchor coverage is **75.27%**, while all-anchor coverage should be read separately in the JSON output.
- These numbers are evidence about sparse geometric reachability, not method gains. They should decide whether the next step deserves calibrated reranker training.

## Dense vs Sparse Recovery

Recovery is measured on baseline-wrong samples with anchor annotations, using sparse top-5 nearest-neighbor anchor selection as the sparse proxy.

| Pool | Count | Percent | Dense Candidate Recovery |
| --- | ---: | ---: | ---: |
| Baseline-wrong anchor samples | 334 | 100.00% | n/a |
| Geometry-evaluable baseline-wrong anchor samples | 324 | 97.01% | n/a |
| Sparse misses every anchor | 110 | 33.95% | 100.00% |
| Sparse misses at least one anchor | 156 | 48.15% | 100.00% |

Subset concentration for baseline-wrong samples where sparse top-k misses every anchor:

| Subset | Count |
| --- | ---: |
| same_class_clutter | 70 |
| same_class_high_clutter | 36 |
| multi_anchor | 22 |
| single_anchor | 88 |
| relative_position | 27 |
| directional | 36 |
| between | 6 |
| relational | 74 |
| dense_scene | 16 |
| annotation_exact | 26 |
| annotation_fallback | 84 |

The key distinction is candidate-set recovery, not final reranking accuracy. Dense all-pair scoring can include the missed anchors; a trained scorer still has to assign them useful evidence.

## Calibration Pre-Risk Analysis

This section is intentionally conservative. The trusted baseline predictions do not include logits or probabilities, so true base margin is unavailable. The analysis uses observable proxies only.

| Proxy Pool | Count | Percent of Anchor-Evaluable Samples |
| --- | ---: | ---: |
| Potential benefit, any-anchor criterion | 110 | 26.38% |
| Potential benefit, all-anchor criterion | 156 | 37.41% |
| Potential harm, any-anchor criterion | 24 | 5.76% |
| Potential harm, all-anchor criterion | 35 | 8.39% |

Benefit/harm ratio under the stricter all-anchor criterion: **4.46**.

Anchor-count proxy for uncertainty:

| Anchor Bin | Count | Baseline Acc@1 | Any@5 | All@5 | Mean Min Rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| single_anchor | 324 | 23.15% | 65.74% | 65.74% | 6.0 |
| multi_anchor | 93 | 19.35% | 75.27% | 13.98% | 3.34 |
| three_plus_anchors | 15 | 0.00% | 53.33% | 6.67% | 4.27 |

Interpretation for gate design: dense evidence has a real benefit pool, but there is also a non-trivial harm pool where the base prediction is already correct and additional relation evidence could perturb it. This supports calibration as a necessary next test, not as a completed claim.

## Missed-Anchor Casebook Preview

| Scene | Target | Correct | Min Rank | Anchors | Utterance |
| --- | --- | --- | ---: | --- | --- |
| scene0635_00 | 25 box | False | 7 | 14:bookshelf, 16:ladder | The small box on the top shelf of the tall shelving unit with a ladder in front of it. |
| scene0635_00 | 25 box | False | 7 | 14:bookshelf, 16:ladder | Please select the highest box on top of the book case, on the right hand side of the ladder as your facing it. |
| scene0635_00 | 25 box | False | 7 | 14:bookshelf, 16:ladder | the highest box |
| scene0635_00 | 25 box | False | 7 | 14:bookshelf, 16:ladder | The highest of the boxes. |
| scene0635_00 | 25 box | False | 7 | 14:bookshelf, 16:ladder | This box is high atop the tall bookshelf. |
| scene0635_00 | 25 box | False | 7 | 14:bookshelf, 16:ladder | Facing the door, the box is on top of the tall brown bookcase to the left, above the ladder. |
| scene0635_00 | 25 box | False | 7 | 14:bookshelf, 16:ladder | This is the highest box in the room, on top of a bookshelf next to the door. |
| scene0670_00 | 25 trash can | False | 6 | 27:cabinet, 33:picture | The trash can is small and white. It is near brown cabinets and there is a poster that says "kitchen" above it. |
| scene0670_00 | 25 trash can | False | 6 | 27:cabinet, 33:picture | Choose the garbage can  next to the  counter and under the poster that says KITCHEN |
| scene0670_00 | 25 trash can | False | 6 | 27:cabinet, 33:picture | The small trash can by itself. |

## Artifacts

- `coverage_summary.json`: aggregate and subset metrics.
- `dense_sparse_recovery_summary.json`: sparse-miss recovery statistics.
- `dense_recovery_casebook_top5.json` / `.csv`: baseline-wrong sparse-miss cases recovered by dense candidate reachability.
- `calibration_prerisk_summary.json`: pre-gate risk proxies.
- `subset_coverage_curves.csv`: coverage@k curves by subset.
- `anchor_rank_histogram.csv`: anchor distance-rank histogram.
- `per_sample_coverage.jsonl`: per-sample diagnostic records.
- `missed_anchor_casebook_top5.json` / `.csv`: cases where sparse top-5 misses all annotated anchors.

## Claim Status

This report turns the coverage claim into a measurable proposition. It does not yet validate calibration or method gains.
