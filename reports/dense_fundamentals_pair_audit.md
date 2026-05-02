# Dense Fundamentals: Pair Ranking Usefulness Audit

**Date**: 2026-04-22
**Model**: Dense-no-cal-v1

---

## Executive Summary

**Q2-ranking Answer**: WEAK

Pair ranking shows weak effectiveness.

---

## 1. Overall Hit@K Metrics

| Metric | Value |
|--------|-------|
| Hit@1 | 7.00% |
| Hit@3 | 20.00% |
| Hit@5 | 27.20% |
| Samples | 500 |

Interpretation:
- Hit@1 > 50%: Strong pair ranking
- Hit@1 20-50%: Moderate pair ranking
- Hit@1 < 20%: Weak pair ranking

---

## 2. Multi-Anchor vs Single-Anchor Analysis

| Subset | Hit@1 | Hit@3 | Count |
|--------|-------|-------|-------|
| Multi-Anchor | 8.00% | 24.00% | 75 |
| Single-Anchor | 6.82% | N/A | 425 |

---

## 3. Pair Weight Distribution Analysis

The pair weights represent attention from candidate i to anchor j.

**Key Questions**:
1. Do correct targets attend to appropriate anchors?
2. Are high pair weights concentrated or diffuse?

---

## 4. Case Analysis

### When Pair Ranking Works
- Multi-anchor cases with clear spatial relations
- Cases with distinctive anchor objects

### When Pair Ranking Fails
- Single-anchor cases (less benefit from pair modeling)
- High-clutter scenes (many similar objects)

---

## 5. Conclusion

**Pair Ranking Assessment**:

1. **Overall Effectiveness**: Weak
2. **Multi-Anchor Benefit**: Limited
3. **Single-Anchor Limitation**: Significant

**Q2-ranking Final Answer**: WEAK

Rationale: Hit@1 of 7.0% indicates weak pair ranking ability
