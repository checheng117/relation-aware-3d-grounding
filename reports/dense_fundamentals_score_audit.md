# Dense Fundamentals: Relation Score Quality Audit

**Date**: 2026-04-22
**Model**: Dense-no-cal-v1

---

## Executive Summary

**Q1-fundamental Answer**: MOSTLY-NOISY

Relation scores show mostly noisy discriminative power.

---

## 1. Score Distributions

### 1.1 Correct Target Scores

| Statistic | Value |
|-----------|-------|
| Count | 500 |
| Mean | -2.5290 |
| Std | 3.7057 |
| Median | -3.2760 |
| 25th %ile | -4.1583 |
| 75th %ile | -2.5953 |

### 1.2 Wrong Target Scores

| Statistic | Value |
|-----------|-------|
| Count | 16930 |
| Mean | -1.5774 |
| Std | 4.3200 |

### 1.3 Score Gap (Correct - Wrong Mean)

**Gap**: -0.9515

Interpretation:
- Gap > 0.5: Strong discriminative signal
- Gap 0.1-0.5: Weak but real signal
- Gap < 0.1: Mostly noisy

---

## 2. Margin Analysis

### 2.1 Top-1 vs Top-2 Margin Distribution

| Statistic | Value |
|-----------|-------|
| Count | 500 |
| Mean | 0.3598 |
| Std | 0.5189 |
| Median | 0.0361 |
| % Positive | 79.00% |

Interpretation:
- Margin > 0.5: Confident top-1 selection
- Margin 0.1-0.5: Moderate confidence
- Margin < 0.1: Uncertain selection

---

## 3. Base-Correct vs Base-Wrong Analysis

| Subset | Mean Relation Score | Count |
|--------|--------------------|-------|
| Base-Correct | -1.1135 | 148 |
| Base-Wrong | -1.5342 | 352 |

**Difference**: 0.4207

Interpretation:
- If Base-Wrong > Base-Correct: Dense helps where Base fails
- If Base-Correct > Base-Wrong: Dense amplifies already-easy cases

---

## 4. Subset-Level Analysis

| Subset | Mean Relation Score | Count |
|--------|--------------------|-------|
| Multi-Anchor | -3.1988 | 75 |
| Relative-Position | -0.2098 | 103 |
| Easy | -1.3768 | 322 |

---

## 5. Diagnostic Histograms (text-based)

### Correct Target Score Distribution
  [ -7.16- -5.08] |████████| (44)
  [ -5.08- -3.00] |██████████████████████████████████████████████████| (260)
  [ -3.00- -0.92] |██████████████████████████████| (157)
  [ -0.92-  1.16] || (1)
  [  1.16-  3.24] || (0)
  [  3.24-  5.32] || (0)
  [  5.32-  7.40] || (2)
  [  7.40-  9.48] |███| (20)
  [  9.48- 11.56] |█| (7)
  [ 11.56- 13.64] |█| (9)

### Wrong Target Score Distribution
  [ -8.59- -6.09] |█████| (715)
  [ -6.09- -3.59] |████████████████████████████████████| (4787)
  [ -3.59- -1.08] |██████████████████████████████████████████████████| (6517)
  [ -1.08-  1.42] |███████████████████████| (3045)
  [  1.42-  3.92] |██| (358)
  [  3.92-  6.42] || (105)
  [  6.42-  8.93] |█| (243)
  [  8.93- 11.43] |████| (530)
  [ 11.43- 13.93] |████| (558)
  [ 13.93- 16.44] || (72)

### Margin Distribution
  [  0.00-  0.17] |██████████████████████████████████████████████████| (276)
  [  0.17-  0.34] |████████████████| (93)
  [  0.34-  0.52] || (0)
  [  0.52-  0.69] || (0)
  [  0.69-  0.86] |█| (10)
  [  0.86-  1.03] |██████| (34)
  [  1.03-  1.20] |████████| (45)
  [  1.20-  1.38] |█| (9)
  [  1.38-  1.55] |█| (7)
  [  1.55-  1.72] |████| (26)

---

## 6. Conclusion

**Signal Quality Assessment**:

1. **Score Separation**: Weak
2. **Margin Confidence**: Moderate
3. **Subset Discrimination**: Absent

**Q1-fundamental Final Answer**: MOSTLY-NOISY

Rationale: Score gap of -0.9515 indicates limited discriminative power
