# Working Rules

- Do not read `outputs/`, `logs/`, `artifacts/`, `checkpoints/`, `.git/` unless explicitly required
- Do not recursively expand the whole repository tree
- Prefer reading:
  - `.claude/*.md`
  - `repro/referit3d_baseline/`
  - `configs/`
  - `scripts/`
  - `src/`
  - specific reports explicitly referenced
- Keep changes isolated to the reproduction track unless explicitly asked
- Do not introduce unrelated new methods
- Use staged runs: smoke -> verification -> formal
- Every conclusion must be backed by a run artifact or report