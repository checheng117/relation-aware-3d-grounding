# Formal Round-1 Base Parity Audit

**Date**: 2026-04-21
**Auditor**: Round-1 Base Parity Audit Agent

---

## Executive Summary

**Finding**: Base parity NOT restored. Round-1 Base-only shows **30.04%** Acc@1 vs clean baseline **30.83%**.

**Root Cause**: Padding artifact in `train_cover3d_round1.py` evaluate function.

**Gap**: -0.79% absolute (136 incorrect predictions out of 4255)

**Status**: Fix identified and ready to apply.

---

## 1. Clean Baseline Anchor

| Metric | Value |
| --- | --- |
| Checkpoint | `outputs/20260420_clean_sorted_vocab_baseline/formal/best_model.pt` |
| Test Acc@1 | 30.83% |
| Test Acc@5 | 91.87% |
| Export path | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_predictions.json` |
| Samples | 4255 |

---

## 2. Round-1 Base-only Measurement

| Metric | Value |
| --- | --- |
| Script | `scripts/train_cover3d_round1.py --variant base` |
| Test Acc@1 | 30.04% |
| Test Acc@5 | 89.12% |
| Output path | `outputs/cover3d_round1/base_results.json` |
| Samples | 4255 |

**Gap**: -0.79% Acc@1, -2.75% Acc@5

---

## 3. Audit Trail

### 3.1 Sample Ordering Verification

**Check**: Verify sample ordering between clean predictions and extracted embeddings.

**Result**: 0 mismatches. Ordering is correct.

### 3.2 Logits Verification

**Result**: MISMATCH! Extracted embeddings contain re-run inference logits, not clean export logits.

**Status**: `merge_embeddings_with_clean_logits` correctly replaces logits.

### 3.3 End-to-End Simulation

**Result**: Overall accuracy 30.83% (CORRECT), but individual predictions show discrepancy.

### 3.4 Deep Dive: Sample 39

**Finding**:
- Sample 39 has 19 valid objects (indices 0-18)
- Collate padding fills indices 19-40 with **0.0**
- All valid logits are negative (max = -0.94)
- **0.0 > all negative logits**, so argmax returns first padding index (19)

### 3.5 Code Analysis

**Location**: `scripts/train_cover3d_round1.py`, line 516

**Bug**:
```python
base_pred = base_logits[i].argmax().item()  # No mask applied!
```

**Model forward correctly applies mask**:
```python
fused_logits = base_logits.masked_fill(~object_mask, float("-inf"))
```

### 3.6 Scope of Impact

**Samples with all negative logits**: 1016 (23.9%)

**Actual wrong predictions**: 136 (3.2%)

**Accuracy gap**: -0.79%

---

## 4. Root Cause Summary

| Issue | Location | Impact |
| --- | --- | --- |
| Padding artifact | `train_cover3d_round1.py:516` | 136 wrong predictions |
| Missing mask | `base_pred = base_logits[i].argmax()` | -0.79% Acc@1 |

---

## 5. Fix Required

**File**: `scripts/train_cover3d_round1.py`
**Line**: 516

**Current**:
```python
base_pred = base_logits[i].argmax().item()
```

**Fix**:
```python
masked_logits = base_logits[i].masked_fill(~object_mask[i], float("-inf"))
base_pred = masked_logits.argmax().item()
```

---

## 6. Verification Plan

After applying fix:
1. Re-run `python scripts/train_cover3d_round1.py --variant base --epochs 0`
2. Verify Acc@1 = 30.83% ± 0.2%

---

## 7. Conclusion

**Base Parity Status**: NOT RESTORED (fix identified, pending application)

**Root Cause**: Padding artifact in evaluation code.

**Fix Complexity**: Minimal (single line change).
