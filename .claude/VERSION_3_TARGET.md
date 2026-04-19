# Version 3 Target: COVER-3D

## Working Title

**COVER-3D: Coverage-Calibrated Relational Grounding for Scene-Disjoint 3D Referring Expressions**

## Research Claim

Hard relational failures in 3D visual grounding are caused by two coupled issues:

1. **Coverage failure**: local/top-k relation neighborhoods miss useful anchors, especially in long-range, multi-anchor, and same-class clutter cases.
2. **Calibration failure**: relation branches can overpower reliable base predictions when parser or anchor evidence is noisy.

Version 3 should show that dense relation coverage plus uncertainty-calibrated fusion improves hard relational subsets while preserving overall accuracy. The contribution should be a model-agnostic reranker and diagnostic protocol, not another isolated backbone variant.

## Reviewer-Guided Positioning

Current state:

- Trusted evaluation/reproduction foundation is strong.
- Custom method evidence is not yet strong enough for AAAI main-track method claims.
- Parser v1/v2 and sparse top-k relation are negative evidence, not final methods.
- Implicit v3 is useful as the chunked dense primitive, but it is not a confirmed improvement.

Target state:

- COVER-3D wraps a strong backbone.
- Dense coverage beats sparse top-k relation.
- Calibrated dense relation beats uncalibrated dense relation.
- Gains appear in hard relational subsets and do not trade away overall accuracy.
- Final claims are supported by 3 seeds, ablations, and qualitative case studies.

## Method Modules

### 1. Backbone Adapter

Standardize backbone outputs:

```text
base_logits
object_embeddings
object_geometry
candidate_mask
utterance_features
```

The reranker should be able to wrap ReferIt3DNet first and later stronger backbones.

### 2. Relation Coverage Diagnostics

Compute:

```text
coverage@k
anchor_distance_rank
anchor_relation_rank
long_range_anchor
same_class_clutter
shared_anchor_negative
multi_anchor
```

This is the diagnostic bridge between the failed sparse v2 and final dense Version 3.

### 3. Soft Anchor Posterior

Outputs:

```text
anchor_distribution
anchor_entropy
top_anchor_id
top_anchor_margin
parser_confidence
relation_type_distribution
```

No hard parser decision should directly choose the target.

### 4. Chunked Dense Relation Evidence

Compute candidate-anchor evidence over all valid object pairs, but execute in chunks for memory control.

Required logging:

```text
chunk_size
num_candidate_anchor_pairs
peak_memory_estimate
relation_score_margin
```

### 5. Calibrated Fusion

Fusion should depend on:

```text
base_margin
anchor_entropy
parser_confidence
relation_margin
relation_evidence_strength
```

The gate should be regularized to avoid always trusting the relation branch.

### 6. Hard Relational Training

Required hard cases:

- same-class distractors
- shared-anchor negatives
- long-range anchor cases
- relation counterfactuals
- paraphrase-equivalent utterances
- ambiguous low-margin examples

## Required Ablations

- base model only
- base + uncalibrated relation
- base + calibrated relation
- sparse top-k relation
- dense chunked relation
- no hard negatives
- no paraphrase consistency
- oracle anchor
- oracle relation type
- noisy parser stress test

## Required Tables

1. Main overall table.
2. Hard relational subset table.
3. Same-class clutter table.
4. Long-range anchor table.
5. Multi-anchor table.
6. Paraphrase stability table.
7. Ablation table.
8. Runtime/memory table.

Minimum main comparison:

| Method | Required Role |
| --- | --- |
| ReferIt3DNet | Trusted primary baseline |
| SAT | Secondary reproduced baseline |
| Base + sparse top-k relation | Coverage-failure control |
| Base + dense uncalibrated relation | Calibration-failure control |
| COVER-3D | Final method |

## Required Figures

- Method overview.
- Relation coverage failure example.
- Calibration/gate behavior plot.
- Qualitative case study with target, baseline prediction, top anchors, and relation evidence.
- Failure taxonomy chart.

## Acceptance Bar

Version 3 is paper-ready only if:

- overall Acc@1 reaches at least 32.0% against the trusted 30.79% ReferIt3DNet baseline;
- Acc@5 remains near baseline and does not collapse;
- hard relational gains are meaningful and stable, preferably +3 to +5 Acc@1;
- results hold across at least 3 seeds;
- gains are not restricted to one weak baseline;
- ablations show coverage and calibration both matter;
- qualitative cases match the quantitative story.

## Paper-Framing Fallback

If the acceptance bar is not met, do not force a method-paper claim. Reframe as:

**A reproducibility and diagnostic study of scene-disjoint 3D referring-expression grounding**, with COVER-3D diagnostics explaining why parser-based and sparse relation methods fail.
