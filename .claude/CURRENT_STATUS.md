# Current Status

**Status**: AAAI upgrade planning and Version 3 / COVER-3D verifiable-claim freeze phase.

The experimentation phase that produced Parser v1/v2 and Implicit v1/v2/v3 is closed. The next active target is not another ad hoc model variant and not a larger method. It is direct evidence for the coverage/calibration hypothesis.

## Current Highest Priority

Convert the paper from "direction description" into a testable proposition:

> Existing 3D grounding under scene-disjoint evaluation fails disproportionately on same-class clutter and multi-anchor hard subsets. COVER-3D is not a new relation-aware paradigm; it is a coverage-calibrated relational reranker designed to target anchor coverage failure and relation calibration failure.

Do coverage diagnostics before new full training.

## Paper Readiness

Current strict-reviewer verdict:

| Framing | Likely Verdict | Reason |
| --- | --- | --- |
| Current method paper | Reject / Weak Reject | No stable, confirmed custom-method gain over ReferIt3DNet |
| Implicit v3 only with small gain | Weak Reject / Borderline | Engineering fix alone is not enough for a strong AAAI claim |
| COVER-3D with 3-seed overall and hard-subset gains | Accept | Coverage/calibration claims become supported |
| COVER-3D with cross-backbone validation and full diagnostics | Strong Accept potential | Method and evaluation contributions reinforce each other |

## Trusted Results

| Method | Test Acc@1 | Test Acc@5 | Status |
| --- | ---: | ---: | --- |
| ReferIt3DNet | 30.79% | 91.75% | Trusted primary baseline |
| SAT | 28.27% | 87.64% | Verified secondary baseline |

## Explored Custom Methods

| Method | Test Acc@1 | Val Acc@1 | Verdict |
| --- | ---: | ---: | --- |
| Parser v1 | 30.04% | - | Discarded |
| Parser v2 | 28.81% | - | Discarded |
| Implicit v1 dense | 31.26% | ~33.26% | Promising but crashed before completion |
| Implicit v2 sparse | 28.55% | 31.03% | Discarded |
| Implicit v3 chunked dense | 30.36% | 32.90% | Promising but unconfirmed |

## Current Interpretation

- Parser-assisted explicit reasoning is not the final route because parser noise and uncontrolled fusion hurt test performance.
- Sparse top-k relation modeling is not enough; it misses useful relation evidence.
- Dense pairwise relation coverage appears more promising, but current evidence is incomplete.
- Chunked dense computation is the correct engineering primitive for relation coverage.
- Version 3 must test coverage and calibration together.
- The final paper claim should be "coverage-calibrated reranking targets anchor coverage failure and relation over-trust under scene-disjoint hard subsets," not "parser-based reasoning improves grounding" and not "we introduce the first relation-aware 3D grounding method."

## AAAI Acceptance Gate

Do not position this as a AAAI main-track method paper unless:

- Overall Acc@1 reaches at least 32.0% against the trusted 30.79% ReferIt3DNet baseline.
- Hard relational, same-class clutter, and long-range anchor subsets gain roughly +3 to +5 Acc@1.
- Acc@5 remains near the trusted baseline level.
- Results are reported across at least 3 seeds.
- Ablations show sparse top-k < dense chunked relation and uncalibrated dense < calibrated dense.
- Qualitative cases support the quantitative story.

If this gate fails, reposition as a reproducibility and diagnostic evaluation project.

## Hardware Constraint

The current development machine showed GPU driver instability during v3 training. Do not run long formal GPU experiments on this machine.

Allowed locally:
- CPU checks
- unit tests
- smoke tests that do not stress GPU stability
- documentation and refactoring
- diagnostics over existing artifacts

Formal Version 3 training should resume only on stable hardware.

## Open-Source Project State

Root cleanup:
- Root-level experiment result artifacts were removed from the active repository.
- Local caches and `.claude` lock file removed.
- `.gitignore` updated for local text feature caches and agent lock files.

Expected root contents:
- `README.md`
- `LICENSE`
- `Makefile`
- `pyproject.toml`
- `requirements.txt`
- `environment.yml`
- `.env.example`
- project directories: `src/`, `configs/`, `scripts/`, `tests/`, `docs/`, `reports/`, `repro/`, `data/`

## Current Priority

Version 3 finalization sequence:
1. Freeze claim wording around coverage-calibrated reranking for hard relational subsets.
2. Freeze the official scene-disjoint protocol as the only paper split.
3. Implement coverage diagnostics before new full training: `coverage@k`, anchor reachability, missed-anchor cases, and anchor distance rank.
4. Keep the method minimal: base listener + dense candidate-anchor scorer + soft anchor posterior + calibrated fusion gate.
5. Implement backbone adapter interface.
6. Implement soft anchor posterior diagnostics.
7. Implement calibrated relation fusion.
8. Prepare causal ablations: base, sparse, dense-no-calibration, dense-calibrated, no-soft-anchor, noisy-parser stress, oracle anchor.
9. Promote hard-subset reports to main-result templates.
10. Run controlled smoke tests.
11. Move formal training to stable hardware.
12. Report 3-seed results, ablations, hard-subset metrics, coverage curves, calibration behavior, runtime/memory, and failure taxonomy.

## Non-Goals

- Do not revive Parser v1/v2 as the main method.
- Do not report v3 as a confirmed improvement yet.
- Do not start broad hyperparameter sweeps before coverage/calibration diagnostics exist.
- Do not expand into open-vocabulary/VLM claims as part of the current paper.
- Do not frame COVER-3D as a new backbone or a first relation-aware method.
- Do not turn this into a planning or embodied-agent project.

## Paper Route

| Route | Trigger | Framing |
| --- | --- | --- |
| A-line method paper | 3 seeds, >=32.0 Acc@1, hard-subset gains, stable Acc@5, causal ablations close. | COVER-3D as coverage-calibrated reranking. |
| B-line diagnostic paper | Method gains are unstable or below the acceptance bar. | Scene-disjoint reproducibility, hard-subset diagnosis, failure taxonomy, and trustworthy benchmark protocol. |
