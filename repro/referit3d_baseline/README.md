# ReferIt3D Official Baseline Reproduction

This directory contains the official ReferIt3DNet baseline reproduction track.

## Target

- **Model**: ReferIt3DNet (ECCV 2020)
- **Dataset**: Nr3D
- **Target Accuracy**: 35.6% overall

## Structure

```
repro/referit3d_baseline/
├── configs/
│   └── official_baseline.yaml    # Training configuration
├── scripts/
│   ├── train.py                  # Training script
│   └── evaluate.py               # Evaluation script
├── src/
│   └── referit3d_net.py         # Model implementation
└── README.md
```

## Quick Start

### Stage A: Smoke Test

```bash
# Quick test with debug mode (10 batches)
python repro/referit3d_baseline/scripts/train.py \
    --config repro/referit3d_baseline/configs/official_baseline.yaml \
    --device cpu
```

### Stage B: Protocol Verification

```bash
# Run for a few epochs to verify pipeline
python repro/referit3d_baseline/scripts/train.py \
    --device cuda
```

### Stage C: Formal Reproduction

```bash
# Full training run
python repro/referit3d_baseline/scripts/train.py \
    --device cuda

# Evaluation
python repro/referit3d_baseline/scripts/evaluate.py \
    --checkpoint outputs/repro/referit3d_baseline/best_model.pt \
    --split test
```

## Success Criteria

| Level | Gap from Target | Accuracy Range |
|---|---|---|
| Exact | ≤ 2% | 33.6% - 37.6% |
| Acceptable | 3-5% | 30.6% - 33.6% or 37.6% - 40.6% |
| Partial | > 5% | Outside above ranges |

## Known Limitations

1. **Subset data**: Current dataset is 1,569 samples vs official ~41,503
2. **Placeholder geometry**: Centers/sizes are synthetic
3. **No PointNet++**: Using simple MLP encoder
4. **No BERT**: Using simple hash-based language encoder

## References

- Paper: "ReferIt3D: Neural Listeners for Fine-Grained 3D Object Identification" (ECCV 2020)
- Code: https://github.com/referit3d/referit3d
- Benchmark: https://referit3d.github.io/benchmarks.html