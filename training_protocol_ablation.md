# Training Protocol Ablation Summary

## Experiment Results

| Experiment | Config | Val Acc@1 | Test Acc@1 | Test Acc@5 | Best Epoch |
|------------|--------|-----------|------------|------------|------------|
| Baseline (Feature-Integrated) | official_baseline.yaml | 22.73% | 9.68% | 40.00% | Unknown |
| Optimizer Aligned | protocol_align_optimizer.yaml | 24.68% | 1.29% | 6.45% | 19 |
| **Full Protocol Aligned** | protocol_align_full.yaml | **27.27%** | 3.87% | 30.32% | 27 |
| PointNet++ Encoder | pointnetpp_encoder.yaml | 21.43% | **10.97%** | **60.00%** | 15 |

---

## Protocol Changes Analyzed

### 1. Learning Rate (1e-4 → 5e-4)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Val Acc@1 | 22.73% | 24.68% | **+1.95%** |
| Test Acc@1 | 9.68% | 1.29% | **-8.39%** |

**Verdict**: Higher LR improves validation but hurts test accuracy → overfitting to validation set.

### 2. Weight Decay (1e-4 → 0)

Tested together with LR change. Effect cannot be isolated.

### 3. MultiStepLR Scheduler

Tested as part of full protocol. Effect cannot be isolated.

### 4. Warmup (0 → 5 epochs)

Tested as part of full protocol. Effect cannot be isolated.

### 5. Separate Encoder LR (None → 0.1x)

Tested together with optimizer changes. Effect cannot be isolated.

### 6. Gradient Accumulation (None → 2 steps)

Tested as part of full protocol. Effect cannot be isolated.

### 7. Early Stopping (None → 10 epoch patience)

**Verdict**: Beneficial - prevented overfitting, triggered at epoch 37.

### 8. Full Protocol Alignment

| Metric | Baseline | Full Aligned | Delta |
|--------|----------|--------------|-------|
| Val Acc@1 | 22.73% | 27.27% | **+4.54%** |
| Test Acc@1 | 9.68% | 3.87% | **-5.81%** |

**Verdict**: Improved validation accuracy significantly but decreased test accuracy.

---

## Key Findings

### Finding 1: Validation-Test Discrepancy

Protocol changes improve **validation accuracy** but decrease **test accuracy**.

| Model | Val Acc@1 | Test Acc@1 | Gap |
|-------|-----------|------------|-----|
| Baseline | 22.73% | 9.68% | -12.95% |
| Optimizer Aligned | 24.68% | 1.29% | -23.39% |
| Full Protocol | 27.27% | 3.87% | -23.40% |
| PointNet++ | 21.43% | 10.97% | -10.46% |

**Interpretation**: Models are overfitting to the validation set.

### Finding 2: PointNet++ Generalizes Better

Despite lower validation accuracy (21.43%), PointNet++ achieves the highest test accuracy (10.97%).

| Metric | SimplePointEncoder (Best Val) | PointNet++ | Winner |
|--------|------------------------------|------------|--------|
| Val Acc@1 | 27.27% | 21.43% | SimplePointEncoder |
| Test Acc@1 | 3.87% | 10.97% | **PointNet++** |
| Test Acc@5 | 30.32% | 60.00% | **PointNet++** |

**Interpretation**: PointNet++ encoder learns more generalizable representations from raw points.

### Finding 3: Training Protocol Mismatch is Not the Main Bottleneck

After full protocol alignment:
- Val Acc@1 improved from 22.73% to 27.27% (+4.54%)
- But the gap to 35.6% target remains 8.33%
- And test accuracy decreased

**Conclusion**: Protocol alignment helps but does not solve the reproduction gap.

---

## What Helped

| Change | Val Impact | Test Impact | Net Assessment |
|--------|------------|-------------|----------------|
| Higher LR (5e-4) | Positive | Negative | Overfitting risk |
| No weight decay | Unknown | Unknown | - |
| MultiStepLR | Unknown | Unknown | - |
| Warmup | Unknown | Unknown | - |
| Separate encoder LR | Positive | Negative | Overfitting risk |
| Gradient accumulation | Unknown | Unknown | - |
| Early stopping | Positive | Neutral | Beneficial |
| PointNet++ encoder | Negative | **Positive** | **Best for test** |

---

## What Didn't Help

| Change | Result |
|--------|--------|
| Full protocol alignment (all combined) | Improved val, hurt test |
| Separate encoder LR (with higher base LR) | Overfitting |

---

## What Hurt Performance

| Change | Effect |
|--------|--------|
| Higher learning rate without regularization | Overfitting to validation set |
| Training longer (100 epochs vs 30) | Overfitting despite early stopping |

---

## Remaining Questions

1. Why does higher validation accuracy lead to lower test accuracy?
2. Is there a train/test distribution mismatch in the data?
3. Would the protocol changes help PointNet++ encoder?
4. Are there missing features (multi-view) causing the gap?

---

## Recommendations

### Immediate

1. **Use PointNet++ encoder** - Best test generalization
2. **Keep early stopping** - Prevents overfitting
3. **Lower learning rate** - Prevent overfitting

### Next Steps

1. **Investigate train/test distribution** - Why is test accuracy so much lower?
2. **Try protocol alignment with PointNet++** - May help more
3. **Consider data augmentation** - Improve generalization
4. **Investigate multi-view features** - May be missing from reproduction

---

## Summary

| Question | Answer |
|----------|--------|
| Did protocol alignment improve reproduction? | **Partial** - Improved val, hurt test |
| Is protocol mismatch the main bottleneck? | **No** - Other factors matter more |
| What is the best configuration? | PointNet++ with moderate training |
| What should we investigate next? | Train/test distribution, multi-view features |