# Working Rules

## Repository Hygiene

- Keep the repository root clean. Root files should be project metadata and entrypoint docs only.
- Put persistent documentation in `docs/`.
- Put experiment reports, audits, and result summaries in `reports/`.
- Put generated training/evaluation artifacts in `outputs/`; do not commit them.
- Put raw/processed data and feature caches under `data/`; do not commit generated caches.
- Historical root-level results should not be restored unless explicitly needed for a specific audit.
- Do not commit `.env`, `.claude/*.lock`, caches, checkpoints, BERT embeddings, or generated outputs.

## Development Discipline

- Read `.claude/*.md` and `README.md` before making research-direction changes.
- Prefer existing package structure under `src/rag3d/`.
- Keep changes scoped to Version 3 unless explicitly asked.
- Do not revive discarded parser-assisted methods as the main direction.
- Do not introduce unrelated embodied-agent or planning components.
- Do not start broad sweeps before diagnostics are in place.

## Version 3 Rules

- Version 3 is a coverage-calibrated relational reranker, not another standalone baseline.
- Relation evidence should be all-pair in semantics and chunked in implementation.
- Parser outputs are weak signals and must be uncertainty-calibrated.
- Fusion must include safeguards against relation branch saturation.
- Hard relational evaluation is required; overall Acc@1 alone is insufficient.
- Every final claim needs run artifacts, seeds, ablations, and failure analysis.

## Experiment Rules

- Use staged runs: unit test -> smoke -> verification -> formal.
- Do not run long formal GPU training on the unstable development machine.
- Formal Version 3 training requires stable hardware.
- Track hardware/software details for reproducibility.
- Report mean/std for multi-seed claims.
- Use statistical tests where differences are small.

## Reporting Rules

- Be explicit about confirmed results versus hypotheses.
- Do not describe Implicit v3 as a confirmed improvement.
- Do not hide failed methods; use them as evidence for the final design.
- When adding reports, include date, config, checkpoint/run path, metric definitions, and limitations.
