# Project Context

Project: credible ReferIt3D baseline reproduction inside `rag3d`

Goal:
- Reproduce public ReferIt3D baseline credibly before pursuing stronger baselines or custom methods.
- This is not just a course project; reproducibility and engineering usefulness matter.

Current reproduction target:
- ReferIt3DNet on Nr3D
- Public target: 35.6% overall Acc@1

Rules:
- Prefer fidelity over novelty
- Keep reproduction track isolated from custom structured / parser-aware code
- Do not expand MVT or custom methods until the baseline anchor becomes trustworthy
- Avoid broad hyperparameter sweeps unless explicitly asked