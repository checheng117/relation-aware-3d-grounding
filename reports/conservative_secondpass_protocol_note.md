# Conservative Second-Pass Protocol Note

## Variants under test

Mandatory reference rows:

- `coadapted_intermediate_best`
  - fixed row from minimal co-adaptation (`0.1154`)
- `naive_secondpass_rerank`
  - prior strict second-pass row (`0.1090`)

Mandatory conservative second-pass variants:

- `low_lr_secondpass`
  - same second-pass setup, lower LR
- `short_schedule_secondpass`
  - same second-pass setup, aggressive early stopping / shorter effective schedule
- `partial_freeze_secondpass`
  - same second-pass setup, fine-module partial tuning (`fine_tune_mode=attr_rel_heads`)

Optional variant:

- `regularized_secondpass`
  - not included in this run to keep code changes minimal and runtime focused

## What makes these variants conservative

All variants keep fixed:

- architecture family
- corrected loss handling (`valid_rows`, non-finite batch skip)
- co-adapted shortlist regime
- primary model-selection metric (`val_natural_two_stage_acc@1`)

Only optimization aggressiveness is changed:

- smaller LR
- stronger early stop pressure
- fewer trainable reranker parameters

## Success criterion

The criterion is gain retention, not large movement:

- target to preserve the co-adapted intermediate natural Acc@1 (`0.1154`)
- minimum useful success is beating naive second-pass fallback (`0.1090`)

This protocol intentionally tests stability of alignment, not novelty of model design.
