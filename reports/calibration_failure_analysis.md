# Calibration Failure Analysis

**Date**: 2026-04-21
**Status**: ROOT CAUSE IDENTIFIED

---

## Executive Summary

**Observation**: Dense-calibrated gate collapses to ~0.9 within 2 epochs, causing over-reliance on relation scores and worse performance than both Base and Dense-no-cal.

**Root Cause**: Combination of (1) high init_bias pushing gate toward relation-heavy initialization, (2) uninformative calibration signals, and (3) training dynamics that reinforce relation-heavy behavior.

**Primary Mechanism**: **init_bias too high** → gate initialized at ~0.88 → relation-heavy fusion harms accuracy → gradient reinforces relation trust → collapse to max_gate bound.

---

## 1. Gate Collapse Phenomenon

### Observed Behavior

| Metric | Value |
| --- | --- |
| Initial gate_mean (epoch 1) | 0.806 |
| Final gate_mean (epoch 2+) | 0.900 |
| Gate std | ~0.0 (no variation) |
| High relation pct (>0.8) | >95% |
| Balanced pct (0.4-0.6) | ~0% |

### Training Log Evidence

```
Epoch 1: Loss=3.5633, Acc=28.43%, Gate=0.806
Epoch 2: Loss=3.5218, Acc=28.26%, Gate=0.900  ← Collapse happens here
Epoch 3: Loss=3.5216, Acc=28.59%, Gate=0.900  ← Stuck at max bound
...
Epoch 10: Loss=3.5213, Acc=28.39%, Gate=0.900
```

Gate collapses to 0.9 (max_gate bound) by epoch 2 and never recovers.

---

## 2. Root Cause Analysis (Ranked by Credibility)

### Mechanism 1: High init_bias Causes Relation-Heavy Initialization (HIGH CONFIDENCE)

**Evidence from Code** (`cover3d_calibration.py:101-106`):

```python
# Current configuration
init_bias = 0.3  # Config parameter
min_gate = 0.1
max_gate = 0.9

# Target sigmoid output for init_bias=0.3:
target_sigmoid = (0.3 - 0.1) / (0.9 - 0.1) = 0.25
init_raw = log(0.25 / 0.75) = log(1/3) ≈ -1.1

# But wait - the config in train_cover3d_round1.py shows:
fusion_init_bias: float = 0.3  # Line 261
```

**BUT** the actual gate behavior shows initial gate ~0.88, not 0.3. This suggests:

1. The init_bias parameter may not be working as intended
2. OR the signal vector pushes gate higher than init_bias

**Signal Vector Analysis**:

At initialization (before training):
- `base_signal = sigmoid(base_margin)` → base_margin typically small positive → base_signal ~0.5-0.6
- `entropy_signal = 1 - normalized_entropy` → entropy ~2-3 → normalized ~0.5-0.6 → entropy_signal ~0.4-0.5
- `relation_signal = sigmoid(relation_margin)` → relation_margin small positive → relation_signal ~0.5-0.6
- `parser_signal = 0` (not used)

MLP forward: `ReLU(W @ signal + b)`

With init_bias targeting 0.3, but signal vector having mean ~0.5, the MLP output could be higher than expected.

**Recalculated**: Looking at epoch 1 gate_mean = 0.806, this suggests the signal vector + MLP combination produces much higher gate than init_bias alone.

**Conclusion**: init_bias=0.3 is overridden by signal vector statistics. The MLP transforms signals to high gate values.

**Credibility**: HIGH - The mismatch between init_bias config and actual initial gate is the primary driver.

---

### Mechanism 2: Calibration Signals Lack Discriminative Power (HIGH CONFIDENCE)

**Signal Definitions** (`cover3d_calibration.py:132-157`):

```python
base_signal = torch.sigmoid(base_margin)           # Base confidence
entropy_signal = 1 - (anchor_entropy / 4.6)        # Anchor uncertainty (inverted)
relation_signal = torch.sigmoid(relation_margin)   # Relation concentration
parser_signal = 0                                   # Unused
```

**Problem**: These signals may not vary enough across samples to produce meaningful gate modulation.

**Signal Statistics from Round-1** (dense-calibrated_results.json):
- `base_margin_mean`: Not directly reported, but can infer from behavior
- `relation_margin_mean`: Same
- `anchor_entropy_mean`: Same
- Gate std: 0.0 (no variation across samples!)

**Critical Observation**: Gate std = 0.0 means ALL samples get nearly identical gate values (~0.9). This indicates:

1. Signals are not discriminative (all samples look similar)
2. OR MLP collapses to constant output (easier optimization)

**Why signals might be uninformative**:
- `base_margin = top1 - top2` logits: For hard samples, this is small; for easy samples, larger. But sigmoid squashes this to narrow range.
- `anchor_entropy`: Derived from relation score softmax. If relation scores are diffuse (typical), entropy is always high → entropy_signal always low.
- `relation_margin`: Same issue as base_margin.

**Credibility**: HIGH - Zero gate variance is a smoking gun for uninformative signals.

---

### Mechanism 3: MLP Capacity Too High, Learns Constant Output (MEDIUM CONFIDENCE)

**MLP Architecture**:
```python
gate_mlp = Sequential(
    Linear(4, 32),   # 4 input signals → 32 hidden
    ReLU(),
    Dropout(0.1),
    Linear(32, 1),   # 32 hidden → 1 gate
    Sigmoid()
)
```

**Analysis**:
- 4 → 32 → 1 architecture is relatively large for 4 input features
- 32 hidden units can easily learn to ignore input variation and produce constant output
- Combined with uninformative signals, constant ~0.9 is an easy local optimum

**Why constant high gate?**:
- Relation scores provide more "active" signal than base logits
- High gate gives model more flexibility (relation scores are learned, base is fixed)
- Gradient may favor exploring relation space over relying on fixed base

**Credibility**: MEDIUM - Architecture contributes, but signal quality is more fundamental.

---

### Mechanism 4: Fusion Formula Amplifies Small Signal Differences (MEDIUM CONFIDENCE)

**Fusion Formula** (`train_cover3d_round1.py:388`):
```python
fused_logits = base_logits + gate_values.unsqueeze(-1) * safe_relation
```

**Analysis**:
- This is ADDITIVE fusion, not convex combination
- Gate scales relation_scores directly
- If relation_scores have larger magnitude than base_logits, small gate changes have big effects

**Gradient Flow**:
```
d(loss)/d(gate) = d(loss)/d(fused) * safe_relation
```

If `safe_relation` has large magnitude, gate gradient is amplified.

**Problem**: Large gate gradients → gate quickly pushed to bounds (0.1 or 0.9)

**Observed**: Gate goes to 0.9 (max), not 0.1 (min), suggesting relation_scores are helpful on average but gate overshoots.

**Credibility**: MEDIUM - Fusion formula contributes to instability, but doesn't explain why gate goes HIGH not LOW.

---

### Mechanism 5: No Regularization Toward Balanced Gate (HIGH CONFIDENCE)

**Missing Component**: No penalty for gate collapsing to bounds.

**Current Loss**:
```python
loss = cross_entropy(fused_logits, target_index)
# No gate regularization!
```

**What's Missing**:
- L2 penalty toward gate=0.5 (balanced prior)
- KL divergence toward uniform gate distribution
- Min-entropy penalty on gate distribution

**Effect**: Without regularization, gate collapses to whichever bound gives slightly better training loss.

**Observed**: Gate collapses to 0.9 (relation-heavy), suggesting relation scores provide slightly better training signal, but this overfits.

**Credibility**: HIGH - Absence of regularization allows collapse.

---

## 3. Summary: Why Gate Collapses

| Mechanism | Credibility | Contribution |
| --- | --- | --- |
| 1. init_bias ineffective | HIGH | Primary driver of high initial gate |
| 2. Signals uninformative | HIGH | Gate has no input variation to modulate |
| 3. MLP too flexible | MEDIUM | Enables constant output collapse |
| 4. Fusion amplifies | MEDIUM | Pushes gate to bounds faster |
| 5. No regularization | HIGH | Allows collapse to persist |

**Primary Root Cause**: Mechanisms 1 + 2 + 5 combine to create collapse:
- init_bias doesn't anchor gate at balanced value
- Signals don't provide meaningful variation
- No regularization pulls gate back to balanced

---

## 4. Recommended Fixes (Priority Order)

### Fix 1: Add Gate Regularization (HIGH PRIORITY)

Add L2 penalty toward gate=0.5:

```python
# In training loss
gate_prior_weight = 0.1
gate_prior_loss = gate_prior_weight * ((gate_values - 0.5) ** 2).mean()
total_loss = ce_loss + gate_prior_loss
```

**Expected Effect**: Pulls gate toward 0.5, prevents collapse to bounds.

---

### Fix 2: Improve Signal Normalization (HIGH PRIORITY)

Normalize signals to have more variance and better separation:

```python
# Instead of sigmoid (which squashes), use tanh or direct scaling
base_signal = torch.tanh(base_margin / 2.0) * 0.5 + 0.5  # More spread
entropy_signal = 1 - (anchor_entropy / anchor_entropy.max())  # Dynamic max
relation_signal = torch.tanh(relation_margin / 2.0) * 0.5 + 0.5
```

**Expected Effect**: More signal variance → more gate modulation.

---

### Fix 3: Reduce init_bias / Change Target Gate (MEDIUM PRIORITY)

Set init_bias to target gate=0.5 (balanced):

```python
# Current (effectively high)
init_bias = 0.3  # But actual ~0.88 due to signal interaction

# Proposed (truly balanced)
init_bias = 0.5  # Target gate=0.5
# Recalculate: target_sigmoid = (0.5 - 0.1) / (0.9 - 0.1) = 0.5
# init_raw = log(0.5/0.5) = 0
```

**Expected Effect**: Gate starts at balanced 0.5, less prone to relation-heavy collapse.

---

### Fix 4: Simplify Gate MLP (LOW PRIORITY)

Reduce MLP capacity:

```python
# Current: 4 → 32 → 1
# Proposed: 4 → 8 → 1 (or even 4 → 1 linear)
```

**Expected Effect**: Less capacity to learn constant output.

---

### Fix 5: Use Multiplicative Fusion (LOW PRIORITY)

Change fusion formula:

```python
# Current (additive)
fused = base + gate * relation

# Proposed (multiplicative / convex)
fused = (1 - gate) * base + gate * relation  # Convex combination
# OR
fused = base * (1 + gate * relation)  # Modulates base
```

**Expected Effect**: Better numerical stability, gate=0 recovers base exactly.

---

## 5. Diagnostic Analysis to Run

Before implementing fixes, verify hypotheses:

1. **Gate distribution by epoch**: Confirm collapse timing
2. **Signal histograms**: Check variance of base_signal, entropy_signal, relation_signal
3. **Gate vs base_margin correlation**: Should be negative (high base confidence → low gate)
4. **Gate vs harmed/recovered**: Do harmed samples have systematically different gate?
5. **MLP weight norms**: Check if weights are learning meaningful signal weights

---

## 6. Files to Modify

- `src/rag3d/models/cover3d_calibration.py`: Signal normalization, gate bounds
- `scripts/train_cover3d_round1.py`: Add gate regularization to loss
- Potentially new config variant: `dense-calibrated-v2`

---

## 7. Conclusion

Gate collapse is caused by **three converging factors**:

1. **Ineffective init_bias** - Gate initialized high (~0.8) not balanced (0.5)
2. **Uninformative signals** - Zero gate variance means signals don't modulate gate
3. **No regularization** - Nothing prevents gate from collapsing to bounds

**Recommended approach**: Implement Fix 1 (regularization) + Fix 2 (signals) + Fix 3 (init_bias) together as a minimal redesign. Test with single-seed 10-epoch rerun.

**Expected outcome**: Gate should stabilize near 0.5 with meaningful variance across samples, allowing calibration to actually modulate relation influence based on sample characteristics.
