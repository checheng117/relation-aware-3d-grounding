# Dense-calibrated NaN Audit Report

**Date**: 2026-04-21
**Status**: ROOT CAUSE IDENTIFIED

---

## Executive Summary

**Finding**: NaN originates in `CalibratedFusionGate` backward pass.

**Root Cause**: Gradient explosion through fusion formula when `relation_scores = -inf`.

**Fix Required**: Mask gradient flow for invalid positions.

---

## 1. Debug Method

### Debug Script
`scripts/debug_calibrated_nan_v2.py`

### Test Configuration
- B=4, N=20, valid positions=15, masked positions=5
- Targets always in valid range
- Three test phases:
  1. Forward only (no grad)
  2. Grad on calibration ONLY (dense frozen)
  3. Full training (both modules)

---

## 2. Results

### Test 1: Forward Only
- **Loss**: 3.3825 (finite) ✓
- **Gate mean**: 0.2646
- **Conclusion**: Forward pass is stable

### Test 2: Calibration Grad Only
- **Step 1**: Loss finite, but gate_mlp gradients are NaN
```
GRAD ISSUE: gate_mlp.0.weight: nan=True, inf=False, norm=nan
GRAD ISSUE: gate_mlp.0.bias: nan=True, inf=False, norm=nan
...
```
- **Conclusion**: NaN originates in calibration backward

### Test 3: Full Training
- **Step 1**: Both relation_mlp and gate_mlp weights become NaN
- **Conclusion**: NaN propagates to all parameters

---

## 3. Root Cause Analysis

### Fusion Formula
```python
fused_logits = (1 - gate) * base_logits + gate * relation_scores
```

### Gradient Flow
```
d(fused)/d(gate) = relation_scores - base_logits
```

### Problem
When `relation_scores[i] = -inf` (masked position):
```
d(fused[i])/d(gate) = -inf - base_logits[i] = -inf
```

This `-inf` gradient flows back through:
1. Gate MLP → NaN in gate_mlp weights
2. Then to relation module → NaN everywhere

### Why Dense-no-cal Works
Dense-no-cal uses fixed `gate = 0.5` (no learning), so no gradient flows through gate computation.

### Why Calibration Fails
Calibration learns gate values, so gradients flow through fusion formula and hit `-inf`.

---

## 4. Why Dense-no-cal Doesn't Have This Issue

| Aspect | Dense-no-cal | Dense-calibrated |
| --- | --- | --- |
| Gate | Fixed 0.5 | Learned MLP |
| Gate gradients | None | Yes (hits -inf) |
| Result | Stable | NaN |

---

## 5. Fix Options

### Option A: Mask Gradient at Fusion (Recommended)
Zero out gate contribution for masked positions before fusion:

```python
# Zero gate for masked positions to prevent gradient flow
gate_masked = gate_values * object_mask[:, 0].float()  # [B]
fused_logits = base_logits + gate_masked.unsqueeze(-1) * relation_scores
```

### Option B: Use Safe Relation Scores
Replace `-inf` with large negative value for fusion:

```python
safe_relation = relation_scores.clamp(min=-1e6)  # Instead of -inf
fused_logits = base_logits + gate_values.unsqueeze(-1) * safe_relation
```

### Option C: Masked Fusion Formula
Only apply gate to valid positions:

```python
gate_exp = gate_values.unsqueeze(-1)  # [B, 1]
fused_logits = base_logits.clone()
fused_logits[object_mask] += gate_exp[object_mask.any(dim=-1)] * relation_scores[object_mask]
```

### Option D: Gradient Clipping
Add aggressive gradient clipping:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # Very small
```

---

## 6. Recommended Fix

**Option A + Option B combined**:

1. Clamp relation_scores to safe range for fusion
2. Mask gate gradient for invalid positions

This preserves the `-inf` for argmax while preventing gradient explosion.

---

## 7. Files to Modify

- `scripts/train_cover3d_round1.py` - Fusion formula fix
- Possibly `src/rag3d/models/cover3d_calibration.py` - Add safe fusion option

---

## 8. Verification Plan

After fix:
1. Run debug script - no NaN in gradients
2. Run smoke test (2 epochs) - no NaN warnings
3. Run full 10 epoch training - complete without NaN
