# Dense Fundamentals: Final Summary

**Date**: 2026-04-22
**Model**: Dense-no-cal-v1

---

## Executive Summary

This report answers three key questions about the DenseRelationModule:

| Question | Answer |
|----------|--------|
| Q1-fundamental | **MOSTLY-NOISY** |
| Q2-ranking | **WEAK** |
| Q3-worth | **CONTINUE-WITH-CAUTION** |

**Route Decision**: **Route C-hard-stop**

---

## Q1-fundamental: Do relation scores have discriminative power?

**Answer**: MOSTLY-NOISY

### Evidence

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Score Gap (Correct - Wrong) | -0.9515 | Weak |
| Mean Margin (Top1 - Top2) | 0.3598 | Moderate |
| Base-Wrong Mean Score | -1.5342 | Dense helps where Base fails |
| Base-Correct Mean Score | -1.1135 | Dense amplifies easy cases |

### Rationale

Relation scores show little to no separation between correct and wrong targets, suggesting mostly noise.

---

## Q2-ranking: Does pair ranking work?

**Answer**: WEAK

### Evidence

| Metric | Value |
|--------|-------|
| Hit@1 | 7.0% |
| Hit@3 | 20.0% |
| Hit@5 | 27.2% |
| Multi-Anchor Hit@1 | 8.0% |
| Single-Anchor Hit@1 | 6.8% |

### Valid Subsets

- **Multi-Anchor**: Dense branch provides clear benefit, recovering more cases than it harms.

### Harmed Subsets

- **Hard Cases**: Dense branch harms hard cases, suggesting fundamental issues with pair scoring.

### Rationale

Pair ranking achieves only 7.0% Hit@1, indicating weak anchor selection.

---

## Q3-worth: Is dense scorer line worth continuing?

**Answer**: CONTINUE-WITH-CAUTION

### Evidence

| Metric | Value |
|--------|-------|
| Net Recovered | 3 |
| Recovered Count | 57 |
| Harmed Count | 54 |
| Multi-Anchor Recovery | 15 |
| Multi-Anchor Harm | 8 |

### Rationale

Dense branch shows modest net benefit (+3), but patterns are not clear enough for confident redesign.

---

## Route Decision: Route C-hard-stop

### If Route B-fundamentals

**Allowed Actions**:
- Redesign dense scorer based on findings
- Focus on multi-anchor / clutter subsets
- Protect easy cases from over-correction
- Simplify aggregation (not add complexity)

**Forbidden Actions**:
- Blind complexity (attention, geometry, focal without foundation)
- Multi-seed experiments
- Calibration without validated signal

### If Route C-hard-stop

**Next Steps**:
- Pause dense scorer line
- Focus on diagnostic paper route
- Benchmark and analyze error patterns
- Consider alternative methods (not dense-based)

---

## Appendix: Key Statistics


| Statistic | Value |
|-----------|-------|
| **Score Analysis** | |
| Correct Target Score Mean | -2.5290 |
| Wrong Target Score Mean | -1.5774 |
| Score Gap | -0.9515 |
| Margin Mean | 0.3598 |
| **Pair Ranking** | |
| Hit@1 | 7.0% |
| Hit@3 | 20.0% |
| **Contribution** | |
| Recovered | 57 |
| Harmed | 54 |
| Net | 3 |

