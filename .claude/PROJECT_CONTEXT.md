# Project Context

Project: Relation-Aware 3D Grounding

Public positioning:
- Open-source research codebase for trustworthy 3D referring-expression grounding.
- The repository combines dataset/split recovery, baseline reproduction, diagnostics, and relation-aware method exploration.
- The project should be presented honestly: current trusted contribution is the evaluation/reproduction base; Version 3 is the final method-paper target.

Current research target:
- **Version 3 / COVER-3D**: coverage-calibrated relational reranking for robust 3D visual grounding.
- Goal: improve hard relational cases without reducing overall accuracy.
- Paper angle: relation-aware 3D grounding fails when relation evidence is not covered or is poorly calibrated, not merely because a relation module is missing.

Current AAAI reviewer assessment:
- Current method-paper framing would likely be Reject / Weak Reject because custom methods do not yet have stable, confirmed gains over ReferIt3DNet.
- If Implicit v3 only reaches a small gain around 31.3-31.8% Acc@1, the likely outcome is still Weak Reject / Borderline unless supported by a stronger diagnostic contribution.
- The Accept path is COVER-3D with 3-seed overall gains, hard relational subset gains, and ablations proving coverage and calibration both matter.
- Strong Accept potential requires the same story to hold across more than one backbone, with full diagnostics, qualitative evidence, and reproducible artifacts.

Core evidence from previous phases:
- Full Nr3D recovered: 41,503 samples.
- Scene-disjoint split established and validated.
- ReferIt3DNet reproduced as primary trusted baseline: 30.79% test Acc@1, 91.75% test Acc@5.
- SAT reproduced as secondary baseline: 28.27% test Acc@1, 87.64% test Acc@5.
- Parser-assisted methods underperformed and are not the final direction.
- Sparse top-k relation approximation underperformed, supporting the relation coverage hypothesis.
- Dense pairwise relation modeling showed signal but is not yet stable enough for final claims.
- Chunked dense computation is the engineering base for Version 3.

Final development goal:
1. Build a model-agnostic backbone adapter.
2. Add relation coverage diagnostics for anchor reachability and long-range evidence.
3. Replace hard parser decisions with soft anchor posterior and uncertainty tracking.
4. Preserve all-pair candidate-anchor relation evidence with chunked dense computation.
5. Calibrate relation/base fusion using parser confidence, anchor entropy, base margin, and relation margin.
6. Train with same-class, shared-anchor, long-range, counterfactual, and paraphrase-consistency hard cases.
7. Validate with overall, hard relational, same-class clutter, long-range anchor, multi-anchor, and paraphrase stability metrics.

AAAI-level acceptance bar:
- Overall Acc@1 should improve by at least +1.2 points over the trusted ReferIt3DNet baseline (target: 32.0%+ vs 30.79%).
- Acc@5 should stay close to the trusted baseline and must not collapse.
- Hard relational subsets should improve clearly, preferably +3 to +5 Acc@1.
- Long-range anchor and multi-anchor subsets should show the strongest gains.
- At least 3 seeds with mean/std for final claims.
- Use statistical tests where appropriate.
- Include ablations for coverage, calibration, dense relation evidence, hard negatives, and paraphrase consistency.
- Do not submit as a method paper if improvements are limited to a weak/old baseline or a single run.
- If these thresholds are not met, reposition the project as a reproducibility and diagnostic evaluation paper rather than a main method paper.

Repository hygiene:
- Root directory should stay clean: metadata, environment files, README, license, Makefile only.
- Historical root-level result files were removed from the active repository during cleanup.
- Generated data, checkpoints, BERT features, outputs, and caches stay out of git.
- New reports go under `reports/`; new persistent docs go under `docs/`; new source goes under `src/rag3d/`.
