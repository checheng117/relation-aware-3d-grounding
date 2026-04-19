# Next Task

**Active target**: Turn COVER-3D from a broad method direction into a set of verifiable propositions.

The next step is not to design a larger method. The next step is to prove or falsify the coverage/calibration story with direct evidence.

## Frozen Main Claim

Existing 3D grounding under scene-disjoint evaluation fails disproportionately on same-class clutter and multi-anchor hard subsets. COVER-3D is not a new relation-aware paradigm; it is a coverage-calibrated relational reranker designed to target two measurable failure mechanisms:

- anchors are not covered by sparse relation selection;
- noisy relation evidence should not be over-trusted.

All paper drafts and implementation tasks should support this claim.

## Immediate Task Order

1. **Claim and wording freeze**
   - Replace broad "relation-aware 3D grounding method" language with "coverage-calibrated reranking for hard relational subsets."
   - Replace "solve relational grounding" with "target a specific failure mode under scene-disjoint evaluation."
   - Replace "multi-anchor reasoning" as the main innovation with "measured coverage failure plus calibrated fusion."
   - Keep parser/LLM and open-vocabulary language out of the main contribution.

2. **Coverage diagnostics before new training**
   - Implement `coverage@k` for k in `{1, 3, 5, 10, all}`.
   - Measure sparse-vs-dense anchor reachability.
   - Export missed-anchor case analysis for base and sparse-relation failures.
   - Compute anchor distance rank to test whether decisive anchors are often not nearest neighbors.
   - Plot subset-level coverage curves for clutter >= 3, clutter >= 5, multi-anchor, relative-position, and dense-scene subsets.

3. **Minimal COVER-3D only**
   - Base listener.
   - Dense candidate-anchor scorer.
   - Soft anchor posterior.
   - Calibrated fusion gate.
   - No complex parser-centric pipeline, no symbolic/LLM multi-stage main method, no new monolithic backbone.

4. **Causal ablation chain**
   - Base only.
   - Base + sparse relation.
   - Base + dense relation, no calibration.
   - Base + dense relation + calibration.
   - Dense + calibration, no soft anchor posterior.
   - Dense + calibration + noisy parser stress.
   - Oracle anchor upper bound.

5. **Hard-subset-first reporting**
   - Treat hard-subset tables and figures as main results, not appendix material.
   - Main results must include same-class clutter, high clutter, multi-anchor, relative-position, and dense-scene slices.
   - Every overall metric must be paired with hard/easy split behavior and Acc@5 stability.

6. **A/B paper route**
   - A-line method paper only if 3 seeds, >=32.0 Acc@1, hard-subset gains, stable Acc@5, and causal ablations close.
   - B-line diagnostic paper if method gains are unstable or below bar: scene-disjoint reproducibility, hard-subset diagnosis, failure taxonomy, and trustworthy benchmark protocol.

## Required Diagnostic Outputs

Each coverage diagnostic run should produce:

- overall Acc@1 / Acc@5 for the base predictions being analyzed;
- hard/easy split metrics;
- `coverage@k`;
- sparse-vs-dense anchor reachability;
- anchor distance-rank distribution;
- same-class clutter metrics;
- multi-anchor metrics;
- relative-position metrics;
- prediction margin and entropy diagnostics;
- per-sample casebook with target, baseline prediction, sparse shortlist, dense anchors, and subset tags.

## Main Paper Tables And Figures

1. Overall comparison: ReferIt3DNet, SAT or alternate listener, sparse relation, dense uncalibrated relation, COVER-3D.
2. Hard-subset table: clutter >= 3, clutter >= 5, multi-anchor, relative-position, dense scene.
3. Failure concentration figure: where the trusted baseline fails.
4. Coverage curve figure: `coverage@k` and sparse-vs-dense reachability.
5. Calibration behavior figure: gate values vs base margin, anchor entropy, relation margin, parser confidence, and error type.
6. Causal ablation table: coverage, calibration, soft anchor posterior, noisy parser stress, oracle anchor.
7. Runtime/memory table: sparse vs dense vs chunked dense relation computation.

## Target Numbers

- Overall Acc@1: at least 32.0% against the trusted 30.79% ReferIt3DNet baseline.
- Hard relational / same-class / long-range subsets: roughly +3 to +5 Acc@1.
- Multi-anchor: ideally +5 to +10 Acc@1, with caution due to small sample size.
- Acc@5: near baseline, no collapse.
- Seeds: at least 3 with mean/std.

## Stop Conditions

Do not continue toward a method-paper claim if:

- direct coverage diagnostics do not support the coverage-failure hypothesis;
- overall accuracy drops against the selected strong backbone;
- gains only appear on a weak baseline;
- gains only appear in one seed;
- overall gain is below the +1.2 point practical threshold and hard-subset gains are weak;
- relation fusion learns to ignore the base model or saturates without calibration;
- hard relational subsets do not improve meaningfully.

If those stop conditions occur, reposition the project as a reproducibility and diagnostic evaluation project rather than a method paper.
