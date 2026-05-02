# Dense-calibrated Round-1 Results

**Date**: 2026-04-21
**Status**: COMPLETE

---

## Configuration

- Variant: dense-calibrated
- Epochs: 10
- Batch size: 4
- Seed: 42

---

## Test Results

| Metric | Value |
| --- | ---: |
| Acc@1 | 30.60% |
| Acc@5 | 91.89% |
| Net Recovered | -10 |
| Recovered | 434 |
| Harmed | 444 |

---

## Training History

| Epoch | Loss | Train Acc | Gate Mean |
| ---: | ---: | ---: | ---: |
| 1 | 3.5633 | 28.43% | 0.806 |
| 2 | 3.5218 | 28.26% | 0.900 |
| 3 | 3.5216 | 28.59% | 0.900 |
| 4 | 3.5216 | 28.51% | 0.900 |
| 5 | 3.5215 | 28.67% | 0.900 |
| 6 | 3.5214 | 28.50% | 0.900 |
| 7 | 3.5214 | 28.56% | 0.900 |
| 8 | 3.5213 | 28.48% | 0.900 |
| 9 | 3.5213 | 28.45% | 0.900 |
| 10 | 3.5213 | 28.39% | 0.900 |

---

## Hard Subset Performance

| Subset | Acc@1 | Correct | Total |
| --- | ---: | ---: | ---: |
| Overall | 30.60% | 1302 | 4255 |
| Easy | 30.80% | 1177 | 3821 |
| Multi-anchor | 28.80% | 125 | 434 |
| Relative position | 30.37% | 294 | 968 |

---

## Key Observations

1. **NaN Issue Fixed**: Training completed without NaN using the clamped relation_scores fix
2. **Gate Collapse**: Gate mean collapsed to ~0.9 by epoch 2 and stayed there
3. **Worse than Base**: Acc@1 (30.60%) < Base (30.83%)
4. **Negative Net Recovery**: -10 net (434 recovered, 444 harmed)

---

## Comparison with Other Variants

| Variant | Acc@1 | Acc@5 | Net |
| --- | ---: | ---: | ---: |
| Base | 30.83% | 91.87% | 0 |
| Dense-no-cal | 31.05% | 92.01% | +9 |
| Dense-calibrated | 30.60% | 91.89% | -10 |

---

## Conclusion

Dense-calibrated underperforms both Base and Dense-no-cal. The calibration gate collapses to ~0.9 (over-relying on relation scores), which harms performance. The calibration approach needs redesign before it can be effective.
