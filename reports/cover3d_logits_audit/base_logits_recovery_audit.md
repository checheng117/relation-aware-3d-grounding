# COVER-3D Base Logits Recovery Audit

**Date**: 2026-04-19  
**Purpose**: Determine whether the trusted 30.79% ReferIt3DNet baseline can be reproduced in the current workspace while exporting real base logits, margins, and target ranks for P3 calibration tests.

---

## Executive Judgment

The trusted Apr9 baseline remains valid as a **prediction-level benchmark anchor**, but its **logits cannot currently be recovered reliably** from the current repository state.

The blocking issue is not the DistilBERT feature generation path. A later `vocabulary_fix` checkpoint reproduces normally with regenerated DistilBERT features. The blocking issue is that the Apr9 learned-class checkpoint appears to depend on an unfrozen class-name-to-index mapping for the learned class embedding.

Safe conclusion:

> Do not use freshly exported logits from `outputs/20260409_learned_class_embedding/formal/best_model.pt` for P3. They do not reproduce the trusted 30.79% result. Use them only as negative audit evidence.

Practical fallback:

> Use `outputs/20260414_vocabulary_fix/verification/best_model.pt` as a reproducible logits-infrastructure smoke baseline, while keeping the Apr9 30.79 predictions as the Phase 1/P2 benchmark anchor. For method-paper P3 evidence, train a clean sorted-vocabulary baseline and export logits from the same run.

---

## Target Asset

Trusted baseline:

- Checkpoint: `outputs/20260409_learned_class_embedding/formal/best_model.pt`
- Predictions: `outputs/20260409_learned_class_embedding/formal/eval_test_predictions.json`
- Results: `outputs/20260409_learned_class_embedding/formal/eval_test_results.json`
- Stored test Acc@1 / Acc@5: **30.79% / 91.75%**
- Samples: **4,255**
- Stored BERT coverage: **4,255 / 4,255**

Checkpoint metadata:

- `epoch`: 27
- `val_acc`: 33.43%
- `use_learned_class_embedding`: true
- `num_classes`: 516
- `class_embed_dim`: 64
- `fusion.0.weight`: `(512, 576)`
- `class_embedding.weight`: `(516, 64)`

---

## Negative Reproduction Results

All runs below used the current manifest directory:

```text
data/processed/scene_disjoint/official_scene_disjoint
```

| Attempt | Split | Acc@1 | Acc@5 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Apr9 checkpoint, no BERT cache | test | 3.50 | 20.68 | Expected failure; random text fallback. |
| Apr9 checkpoint, regenerated `bert-base-uncased` test cache | test | 3.62 | 19.04 | Wrong for trusted baseline. |
| Apr9 checkpoint, regenerated `distilbert-base-uncased` test cache | test | 4.65 | 21.74 | Wrong for trusted baseline. |
| Apr9 checkpoint, regenerated DistilBERT val cache | val | 4.65 | 24.86 | Wrong for stored 33.43 val. |
| Apr10 smoke checkpoint, regenerated DistilBERT val cache | val | 5.03 | 30.30 | Wrong for stored 24.31 val. |

The Apr10 smoke checkpoint is important because it recorded `val_acc=24.31%` after epoch 0 in the same learned-class family. Its collapse to 5.03% under regenerated features shows that the Apr9/Apr10 learned-class checkpoints are not compatible with the current class-index wiring.

---

## Ruled-Out Explanations

### 1. BERT regeneration is not the primary blocker

The `vocabulary_fix` checkpoint reproduces with the regenerated DistilBERT features:

| Checkpoint | Split | Acc@1 | Acc@5 | Result |
| --- | ---: | ---: | ---: | --- |
| `outputs/20260414_vocabulary_fix/verification/best_model.pt` | val | 31.18 | 91.55 | Reproducible with regenerated DistilBERT. |
| `outputs/20260414_vocabulary_fix/verification/best_model.pt` | test | 29.02 | 90.51 | Reproducible with exported logits. |

This means the DistilBERT feature generation path can support a real logits export when the checkpoint and class vocabulary agree.

### 2. Fusion-input order is not the primary blocker

A no-edit forward reconstruction tested permutations of:

```text
object chunk / class embedding chunk / language chunk
```

and also tested encoded-object vs raw-object chunks. Best observed results were far below the trusted baseline:

| Best permutation screen | Split | Best Acc@1 |
| --- | ---: | ---: |
| Apr9 checkpoint, fusion/chunk permutations | val | 8.28 |
| Apr9 checkpoint, fusion/chunk permutations | test | 7.76 |

So the failure is not explained by simply concatenating the 256/64/256 chunks in the wrong order.

### 3. Small hash-seed search did not recover the old class vocabulary

A bounded screen over likely `PYTHONHASHSEED` values tested an unsorted Python `set` vocabulary order hypothesis. On a 768-sample val subset, the best result was:

| Screen | Best Acc@1 | Best Acc@5 |
| --- | ---: | ---: |
| Apr9 checkpoint, unsorted-set hash seed screen | 9.64 | 42.45 |

This supports the class-vocabulary-mismatch diagnosis, but does not recover the exact old mapping. If the original process used a random hash seed or an unrecorded local vocabulary file, brute-force recovery is not a reliable path.

---

## Reproducible Logits Smoke Baseline

The following output is usable for infrastructure testing, but it is **not** the trusted 30.79 baseline:

```text
outputs/cover3d_logits_audit/vocabfix_20260414_distilbert_test_logits/
```

Results:

- Checkpoint: `outputs/20260414_vocabulary_fix/verification/best_model.pt`
- Test Acc@1: **29.02%**
- Test Acc@5: **90.51%**
- Total samples: **4,255**
- BERT coverage: **4,255 / 4,255**
- Predictions SHA256: `c13df2a7ed7df23662eb6b6551df2012ca59e19d8eb730cf818437c7ee8879ef`
- Results SHA256: `3d5705b15649390be30db72ca07af0c4a44a51cef145160cd6ccd8d8965a6779`

Exported prediction rows contain:

- `base_logits`
- `base_margin`
- `base_top1_logit`
- `base_top2_logit`
- `target_logit`
- `target_rank`

Logit caveat:

- The exported margins are often exactly zero because many candidate logits are tied.
- Median `base_margin`: **0.0**
- 75th percentile `base_margin`: **0.0**
- Mean `base_margin`: **0.0922**

This makes raw top1-top2 margin a weak standalone calibration signal for this baseline. A real P3 test should log additional tie-aware features:

- number of candidates tied with top1;
- top-k target rank under tie groups;
- entropy after temperature scaling;
- class tie count for the predicted class;
- same-class clutter count.

---

## Impact on P3

The previous P3 proxy remains useful, but it should not be upgraded into a method result.

Current evidence state:

- P1: failure concentration is supported.
- P2: coverage failure is directly supported.
- P3: proxy-level recovery/harm tradeoff is supported.
- Real P3 with trusted 30.79 logits: **blocked by baseline asset/version mismatch**.

The clean next move is not to design a smarter gate. It is to restore a reproducible logits source:

1. **Preferred**: train a clean sorted-vocabulary ReferIt3DNet baseline for the full 30-epoch protocol and export logits from that exact run.
2. **Acceptable smoke**: use the 29.02% `vocabulary_fix` logits to validate P3 code paths, reporting it only as infrastructure smoke.
3. **Only if available**: recover the original Apr9 class vocabulary or full run directory, then re-evaluate the Apr9 checkpoint under the exact old mapping.

---

## Recommendation

For the paper track, keep using the Apr9 30.79 predictions as the trusted benchmark anchor for P1/P2 diagnostics, but do not claim real-logits calibration evidence from that checkpoint.

For P3 implementation, use the reproducible `vocabulary_fix` logits only to debug:

```text
Base logits -> Dense no-cal -> Dense calibrated
```

Then rerun the same P3 pipeline on a newly trained clean baseline before making any method claim.
