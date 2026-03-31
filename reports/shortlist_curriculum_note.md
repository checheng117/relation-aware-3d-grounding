# Curriculum (entity → cluttered → full-scene)

**Status for this phase:** not implemented as an automated schedule.

**Reason:** The default `data/processed/` tree in this workspace exposes a **single** train/val manifest pair (full-scene NR3D-style). A proper curriculum requires separate manifests or batch-level regime tags with a sampler; that would expand scope beyond the bottleneck-first mandate.

**Next step if added:** reuse diagnosis manifest builders or CSV splits, then `WeightedRandomSampler` by `candidate_load` / scene size, staged over epochs—documented here for follow-up only.
