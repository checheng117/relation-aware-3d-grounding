# README Gain-Retention Validation Note

## Updated sections

A minimal follow-up sentence was added in `README.md` under the current-stage findings block, after the conservative second-pass statement.

No large section rewrite was made.

## Authoritative files used

- `outputs/20260331_185900_gain_retention_validation/low_lr_secondpass_extra_seed_table.csv`
- `outputs/20260331_185900_gain_retention_validation/low_lr_sweep_table.csv`
- `outputs/20260331_185900_gain_retention_validation/shortlist_rerank_combined_table_gain_validation.csv`

## Why this README follow-up is meaningful

The new narrow validation adds one important nuance:

- local LR neighborhood around the validated low-LR point is stable at `0.1154` (for seed 42)
- but one additional seed (`43`) falls to `0.1026`

So README now reflects that retained gain is reproducible under local LR perturbation in the validated seed, but not yet seed-robust enough to claim broad stability.
