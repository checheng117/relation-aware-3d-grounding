# COVER-3D Phase 2: Design Freeze

**Date**: 2026-04-19
**Phase**: Verifiable-claim freeze + diagnostics implementation (NO FULL TRAINING)

---

## 0. Verifiable Proposition Freeze

The next revision must turn COVER-3D from a broad method direction into a set of claims that can be directly falsified.

### Main Claim

> Under scene-disjoint 3D grounding, failures concentrate in same-class clutter and multi-anchor hard subsets. COVER-3D is not a new relation-aware paradigm; it is a coverage-calibrated relational reranker that targets two measurable failure mechanisms: anchors are not covered by sparse relation selection, and noisy relation evidence should not be over-trusted.

This claim is now the center of the paper plan. All title, abstract, introduction, method, and experiment language should support this exact framing.

### Claim Wording Rules

| Avoid | Use Instead |
| --- | --- |
| relation-aware 3D grounding method | coverage-calibrated reranking for hard relational subsets |
| solve relational grounding | target a specific failure mode under scene-disjoint evaluation |
| multi-anchor reasoning as the main innovation | measured coverage failure plus calibrated fusion |
| new backbone / end-to-end complete system | model-agnostic reranker / wrapper |
| parser/LLM reasoning as the method highlight | weak optional signal controlled by uncertainty and calibration |

### Priority Order

1. Freeze the claim before changing the model.
2. Measure coverage failure before running new full training.
3. Keep the method minimal: base listener + dense candidate-anchor scorer + soft anchor posterior + calibrated fusion gate.
4. Design ablations as a causal chain, not a list of module toggles.
5. Promote hard-subset tables and figures to main results.
6. Maintain two paper routes: A-line method paper if the evidence is strong; B-line diagnostic/reproducibility paper if the method gate fails.

---

## 1. Final Method Story (One Paragraph)

COVER-3D is a **model-agnostic coverage-calibrated reranking wrapper** for the hard subsets where scene-disjoint 3D grounding currently fails. It does not claim to introduce the first relation-aware or anchor-aware 3D grounding model. Instead, it tests whether two mechanisms explain and reduce failures: **coverage failure**, where sparse candidate-anchor selection misses useful relational evidence, and **calibration failure**, where noisy relation evidence overrides reliable base predictions. The minimal method computes chunked dense candidate-anchor relation evidence, maintains a soft anchor posterior instead of hard parser decisions, and fuses base and relation scores through an uncertainty-aware gate.

---

## 2. Evidence Before Training: Coverage Diagnostics

Direct coverage evidence is the first implementation priority. Do not start new large model variants until these measurements exist.

| Diagnostic | Question It Answers | Required Slices |
| --- | --- | --- |
| `coverage@k` | Is the decisive or useful anchor inside sparse top-k relation candidates? | all, clutter >= 3, clutter >= 5, multi-anchor, relative-position |
| Dense vs sparse reachability | How many anchor candidates are recovered by all-pair scoring but missed by sparse selection? | same slices |
| Missed-anchor case analysis | When base or sparse relation fails, was the useful anchor absent from the shortlist? | multi-anchor and same-class clutter first |
| Anchor distance rank | Are decisive anchors often not nearest neighbors? | relative-position and dense-scene subsets |
| Subset coverage curves | Does coverage degrade exactly where accuracy drops? | clutter bins, scene size bins, relation type bins |

Minimum artifacts:

- `coverage@k` table for k in `{1, 3, 5, 10, all}`.
- Sparse-vs-dense missed-anchor table.
- Anchor distance-rank histogram.
- Per-sample casebook with baseline prediction, target, candidate anchors, sparse shortlist, dense evidence, and subset tags.

The goal is to make the coverage hypothesis measurable before relying on Acc@1 gains.

---

## 3. Minimal Method Line

Only three method components remain in the main line:

| Component | Purpose | Mechanism It Tests |
| --- | --- | --- |
| Dense relation scorer | Scores candidate-anchor evidence across all valid pairs with chunking. | Dense coverage beats sparse top-k selection. |
| Soft anchor posterior | Represents anchor uncertainty instead of committing to one parsed anchor. | Hard anchor decisions are brittle under ambiguous language. |
| Calibration gate | Controls relation influence using base margin, anchor entropy, and relation confidence. | Relation evidence helps hard cases but should not overrule reliable base predictions. |

Explicitly de-scoped from the main method:

- complex parser-centric pipelines;
- symbolic or LLM multi-stage reasoning as the core contribution;
- open-vocabulary/VLM extensions;
- a new monolithic 3D grounding backbone.

These can appear only as weak-signal variants, stress tests, or future work.

---

## 4. Module Architecture

### Module Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         COVER-3D Reranker                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌────────────────────────┐                    │
│  │ Base Model   │────▶│ Base Logits [B,N]      │                    │
│  │ (any)        │     │ Base Features [B,N,D]  │                    │
│  └──────────────┘     └────────────────────────┘                    │
│                               │                                      │
│                               ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Dense Relation Module                      │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │  Input:                                                       │   │
│  │  - object_embeddings [B,N,D]                                  │   │
│  │  - object_geometry [B,N,G] (optional, may be fallback)        │   │
│  │  - utterance_features [B,L]                                   │   │
│  │                                                               │   │
│  │  Process:                                                     │   │
│  │  - Chunked pairwise scoring (all N² pairs)                    │   │
│  │  - Per-pair relation score → aggregate → per-object score     │   │
│  │                                                               │   │
│  │  Output:                                                      │   │
│  │  - dense_relation_scores [B,N]                                │   │
│  │  - relation_evidence_tensors (for diagnostics)                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                               │                                      │
│                               ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  Soft Anchor Posterior Module                 │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │  Input:                                                       │   │
│  │  - utterance_features [B,L]                                   │   │
│  │  - object_class_features [B,N,C]                              │   │
│  │  - optional relation priors                                   │   │
│  │                                                               │   │
│  │  Process:                                                     │   │
│  │  - Soft distribution over potential anchors                   │   │
│  │  - No hard parser decision                                    │   │
│  │                                                               │   │
│  │  Output:                                                      │   │
│  │  - anchor_posterior [B,N]                                     │   │
│  │  - anchor_entropy [B]                                         │   │
│  │  - top_anchor_margin [B]                                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                               │                                      │
│                               ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  Calibrated Fusion Gate                       │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │  Input:                                                       │   │
│  │  - base_logits [B,N]                                          │   │
│  │  - dense_relation_scores [B,N]                                │   │
│  │  - anchor_entropy [B]                                         │   │
│  │  - base_margin [B]                                            │   │
│  │                                                               │   │
│  │  Process:                                                     │   │
│  │  - Compute gate value ∈ [0,1]                                 │   │
│  │  - Gate depends on calibration signals                        │   │
│  │  - Bound to prevent collapse                                  │   │
│  │                                                               │   │
│  │  Output:                                                      │   │
│  │  - fused_logits [B,N]                                         │   │
│  │  - gate_values [B]                                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                               │                                      │
│                               ▼                                      │
│                      final_logits [B,N]                              │
│                      diagnostics_dict                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Module Specifications

### 5.1 COVER-3D Wrapper (`cover3d_wrapper.py`)

| Aspect | Specification |
|--------|---------------|
| **Purpose** | Model-agnostic entry point for reranking |
| **Inputs** | base_logits, object_embeddings, object_geometry, candidate_mask, utterance_features, optional metadata |
| **Outputs** | reranked_logits, diagnostics dict |
| **Dependencies** | None (pure interface module) |
| **State** | No learnable parameters (wraps other modules) |

---

### 5.2 Dense Relation Module (`cover3d_dense_relation.py`)

| Aspect | Specification |
|--------|---------------|
| **Purpose** | Compute all-pair candidate-anchor relation evidence |
| **Inputs** | object_embeddings [B,N,D], object_geometry [B,N,G], utterance_features [B,L], candidate_mask [B,N] |
| **Outputs** | dense_relation_scores [B,N], relation_evidence dict |
| **Core Algorithm** | Chunked pairwise scoring (avoids O(N²) memory spike) |
| **Learnable** | Yes (relation scoring MLP) |
| **Chunking** | Process j dimension in chunks of chunk_size (default 16) |

**Key Constraint**: NO sparse top-k approximation. Must be dense all-pairs.

---

### 5.3 Soft Anchor Posterior Module (`cover3d_anchor_posterior.py`)

| Aspect | Specification |
|--------|---------------|
| **Purpose** | Soft distribution over potential anchors (no hard decisions) |
| **Inputs** | utterance_features [B,L], object_class_features [B,N,C], optional priors |
| **Outputs** | anchor_posterior [B,N], anchor_entropy [B], top_anchor_margin [B] |
| **Core Algorithm** | Class-utterance matching → softmax distribution |
| **Learnable** | Yes (class projection MLP) |
| **Constraint** | Structured parser outputs are weak priors only, never hard decisions |

---

### 5.4 Calibrated Fusion Gate (`cover3d_calibration.py`)

| Aspect | Specification |
|--------|---------------|
| **Purpose** | Fuse base and relation scores with calibration |
| **Inputs** | base_logits [B,N], relation_scores [B,N], anchor_entropy [B], base_margin [B] |
| **Outputs** | fused_logits [B,N], gate_values [B], calibration diagnostics |
| **Core Algorithm** | Signal-dependent gate computation |
| **Learnable** | Yes (gate MLP) |
| **Constraint** | Gate bounded [0.1, 0.9] to prevent collapse to always-base or always-relation |

---

### 5.5 Unified COVER-3D Model (`cover3d_model.py`)

| Aspect | Specification |
|--------|---------------|
| **Purpose** | Combine all modules into end-to-end reranker |
| **Inputs** | Same as wrapper |
| **Outputs** | final_logits, full diagnostics |
| **Composition** | wrapper + dense_relation + anchor_posterior + calibration |

---

## 6. In Scope (Phase 2)

| Item | Status |
|------|--------|
| Claim wording freeze | IMPLEMENT |
| Coverage diagnostics before training | IMPLEMENT |
| Coverage@k / anchor reachability / anchor rank exports | IMPLEMENT |
| Model-agnostic wrapper interface | IMPLEMENT |
| Dense relation module (chunked) | IMPLEMENT |
| Soft anchor posterior module | IMPLEMENT |
| Calibrated fusion gate | IMPLEMENT |
| Unified COVER-3D model | IMPLEMENT |
| Smoke validation (CPU) | IMPLEMENT |
| Config files for smoke tests | IMPLEMENT |
| Integration documentation | IMPLEMENT |

---

## 7. Out of Scope (Phase 2)

| Item | Status |
|------|--------|
| Full GPU training | BLOCKED (stable hardware required) |
| Hyperparameter search | OUT |
| Multi-seed experiments | OUT |
| Formal hard-subset result claims | OUT |
| Ablation studies | OUT |
| Parser dependency (hard) | FORBIDDEN |
| Sparse top-k approximation | FORBIDDEN |
| Open-vocabulary / VLM competition claims | OUT |

---

## 8. Causal Ablation Chain (Phase 3)

The main ablation table must follow the mechanism, not arbitrary module switches.

| Order | Variant | Required Interpretation |
| ---: | --- | --- |
| 1 | Base only | Trusted listener reference point. |
| 2 | Base + sparse relation | Tests whether local/top-k relation evidence is enough. |
| 3 | Base + dense relation, no calibration | Tests whether coverage alone helps and whether it hurts easy cases. |
| 4 | Base + dense relation + calibration | Tests the full coverage + calibration mechanism. |
| 5 | Dense + calibration, no soft anchor posterior | Tests whether hard or implicit anchor choice is too brittle. |
| 6 | Dense + calibration + noisy parser stress | Tests whether the gate suppresses unreliable structure. |
| 7 | Oracle anchor upper bound | Separates anchor discovery failure from relation scoring capacity. |

Expected causal story:

```text
sparse < dense on coverage-sensitive subsets
dense uncalibrated may improve hard cases but risks easy-case degradation
calibrated dense preserves easy cases while improving hard subsets
soft anchor posterior improves robustness over brittle anchor decisions
oracle anchor exposes remaining headroom
```

---

## 9. Claims to Validate Later (Phase 3)

| Claim | Validation Metric |
|-------|-------------------|
| Coverage failure is real | `coverage@k`, missed-anchor, and anchor-distance-rank evidence |
| Dense coverage improves hard subsets | +3-5 Acc@1 on clutter, +5-10 on multi-anchor |
| Calibration prevents degradation | Hard subsets improve while easy subsets and Acc@5 do not collapse |
| Soft anchors beat brittle anchors | Soft posterior variant beats no-posterior/hard-anchor variants |
| Method is model-agnostic | Works with different base backbones |

---

## 10. Target Numbers (Phase 3)

| Metric | Target | Baseline |
|--------|--------|----------|
| Overall Acc@1 | ≥ 32.0% | 30.79% |
| Clutter (≥3) Acc@1 | ≥ 25.0% | 21.96% |
| Multi-Anchor Acc@1 | ≥ 20.0% | 11.90% |
| Acc@5 | Near baseline | 91.75% |

---

## 11. Main Paper Result Layout

Hard-subset evidence must be in the main paper, not relegated to appendix.

| Result | Main-Paper Role |
| --- | --- |
| Overall Acc@1 / Acc@5 | Main table 1: confirms no average-case collapse. |
| Hard subsets | Main table 2: same-class clutter, high clutter, multi-anchor, relative-position, dense scene. |
| Failure concentration | Main figure 1: where the trusted baseline fails. |
| Coverage curves | Main figure 2: `coverage@k` and sparse-vs-dense reachability. |
| Calibration behavior | Main figure 3: gate values vs base margin, anchor entropy, relation confidence, and error type. |
| Causal ablation chain | Main table 3: validates coverage and calibration separately. |

The paper story is not "we gain a small average number." It is "we intervene where the baseline actually fails."

---

## 12. Paper Route Gate

| Route | Trigger | Paper Framing |
| --- | --- | --- |
| A-line method paper | 3 seeds, >=32.0 Acc@1, hard-subset gains, Acc@5 stable, causal ablations close. | COVER-3D as a coverage-calibrated reranking method. |
| B-line diagnostic paper | Method gains are unstable or below the acceptance bar. | Scene-disjoint reproducibility, hard-subset diagnosis, failure taxonomy, and coverage/calibration evidence as benchmark assets. |

The B-line route is not a failure state. It protects the project's strongest current asset: trustworthy evaluation and failure diagnosis.

---

## 13. Integration Target

**Primary backbone**: ReferIt3DNet (trusted baseline at 30.79%)

Integration points:
1. Base logits: from ReferIt3DNet classification head
2. Object embeddings: from ReferIt3DNet encoder (before classification)
3. Geometry: may be fallback (limitations acknowledged)
4. Utterance features: from ReferIt3DNet language encoder

**Note**: Geometry may be limited/fallback. Implementation must handle this gracefully.

---

## 14. Implementation Order

| Step | Module | Priority |
|------|--------|----------|
| 0 | Design freeze | DONE (this document) |
| 1 | Claim wording freeze | HIGH |
| 2 | Coverage diagnostics (`coverage@k`, reachability, anchor rank) | HIGH |
| 3 | Wrapper interface | HIGH |
| 4 | Dense relation module | HIGH |
| 5 | Soft anchor posterior | HIGH |
| 6 | Calibrated fusion | HIGH |
| 7 | Unified model | HIGH |
| 8 | Causal ablation config skeletons | HIGH |
| 9 | Main-result hard-subset reporting templates | HIGH |
| 10 | Smoke validation | HIGH |

---

## 15. Design Constraints

1. **Model-agnostic**: Must work with any base model that outputs logits + embeddings
2. **Memory-safe**: Chunked processing, no O(N²) tensor allocation
3. **Calibrated**: Gate bounded, not allowed to fully ignore either branch
4. **No hard parser**: Soft distributions only
5. **Dense coverage**: All N² pairs, no sparse approximation
6. **Evidence-first**: Direct coverage diagnostics must precede new method claims
7. **Hard-subset-first**: Every main result must include clutter and multi-anchor slices

---

## 16. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Geometry unavailable | Use embedding distance as proxy; document limitation |
| Relation branch collapse | Gate bounds [0.1, 0.9]; entropy regularization |
| Memory spike on large N | Chunked processing; configurable chunk_size |
| Overfitting to easy cases | Calibration based on entropy/margin |
| Reviewer calls method incremental | Lead with coverage/calibration failure mechanisms, not relation-aware novelty |
| Method result misses acceptance bar | Switch to B-line diagnostic/reproducibility framing |

---

## Final Statement

**Design frozen around verifiable propositions. Implementation may proceed only along the minimal coverage-calibrated reranking line.**

All subsequent Phase 2 code and paper drafts must follow this specification without deviation. The next priority is direct coverage diagnostics, not a larger method.
