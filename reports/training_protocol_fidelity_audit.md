# Training Protocol Fidelity Audit

## Executive Summary

This audit identifies training-protocol mismatches between the current reproduction track and the official ReferIt3D baseline.

**Key Finding**: Multiple protocol mismatches likely contribute to the remaining reproduction gap.

---

## 1. Current Training Protocol

### Configuration Summary

| Parameter | Current Value | Source |
|-----------|---------------|--------|
| Optimizer | AdamW | `train.py` |
| Learning rate | 1e-4 (0.0001) | config |
| Weight decay | 1e-4 (0.0001) | config |
| Batch size | 16 (SimplePointEncoder) / 8 (PointNet++) | config |
| Epochs | 30 | config |
| Scheduler | CosineAnnealingLR | `train.py` |
| Warmup | None | `train.py` |
| Gradient clipping | 1.0 | config |
| Seed | 42 | config |
| Loss | CrossEntropyLoss | `train.py` |

### Training Script Details

```python
# From repro/referit3d_baseline/scripts/train.py

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=training_config.get("lr", 1e-4),
    weight_decay=training_config.get("weight_decay", 1e-4),
)

# Scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# Loss
criterion = nn.CrossEntropyLoss()
```

### Data Pipeline

| Aspect | Current Implementation |
|--------|----------------------|
| Point sampling | Random sampling to 1024 points |
| Point normalization | (point - bbox_center) / bbox_size |
| Data augmentation | None |
| BERT features | Pre-computed DistilBERT embeddings |
| Shuffle | True for training, False for validation |
| Num workers | 0 |

### Checkpoint Selection

| Aspect | Current Implementation |
|--------|----------------------|
| Criterion | Best validation Acc@1 |
| Save frequency | Every epoch if best |
| Early stopping | None |

---

## 2. Official ReferIt3D Protocol

### From Original Paper and Code

Based on web research of the ReferIt3D paper (ECCV 2020) and related implementations:

| Parameter | Official Value | Source |
|-----------|---------------|--------|
| Optimizer | Adam | Paper / Code |
| Learning rate | ~5e-4 to 1e-3 | Related works |
| Batch size | 32 (single) / 64 (joint) | Paper |
| Epochs | 100 (max) with early stopping | Paper |
| Scheduler | MultiStepLR or ReduceLROnPlateau | SAT implementation |
| Warmup | 5 epochs | SAT implementation |
| Weight decay | 0 (default Adam) | SAT implementation |
| Early stopping | 10 epochs patience | Paper |
| PointNet++ points | 1024 | Paper |

### From SAT ReferIt3D Implementation

The SAT repository (`zyang-ur/SAT`) provides detailed training configuration:

| Parameter | SAT Implementation |
|-----------|-------------------|
| Optimizer | Adam |
| Learning rate | `init_lr` (backbone uses `init_lr/10`) |
| LR scheduler | `ReduceLROnPlateau` (factor=0.65, patience=5) OR `MultiStepLR` (milestones=[25,40,50,60,70,80,90], gamma=0.65) |
| Warmup | 5 epochs (`GradualWarmupScheduler`) |
| Separate LR | Backbone (encoder) uses lower LR |

### From Related Works

| Paper | Optimizer | LR | Batch | Epochs | Scheduler |
|-------|-----------|-----|-------|--------|-----------|
| 3DRefTransformer (WACV 2022) | Adam | 5e-4 | 16 | 60 | - |
| CoT3DRef (ICLR 2024) | Adam | 1e-4 | 24 | - | Decay 0.65 every 10 epochs |

---

## 3. Protocol Mismatch Analysis

### 3.1 Critical Mismatches (High Impact)

#### A. Learning Rate Schedule

| Aspect | Current | Official | Impact |
|--------|---------|----------|--------|
| Scheduler | CosineAnnealingLR | MultiStepLR / ReduceLROnPlateau | **HIGH** |
| Warmup | None | 5 epochs | **HIGH** |
| Decay schedule | Cosine (smooth) | Step-based (0.65 factor) | **MEDIUM** |

**Why this matters**: The learning rate schedule significantly affects convergence. Step-based decay with warmup is standard for 3D vision tasks.

#### B. Batch Size

| Aspect | Current | Official | Impact |
|--------|---------|----------|--------|
| Batch size | 8-16 | 32 | **HIGH** |
| Effective batch | Same | Same | - |
| Gradient accumulation | No | Unknown | **MEDIUM** |

**Why this matters**: Smaller batch sizes can lead to noisier gradients and different convergence behavior.

#### C. Separate Encoder Learning Rate

| Aspect | Current | Official | Impact |
|--------|---------|----------|--------|
| Encoder LR | Same as rest | 10x lower | **HIGH** |
| Backbone tuning | No distinction | Lower LR for backbone | **HIGH** |

**Why this matters**: Pre-trained or complex encoders often benefit from lower learning rates to prevent overfitting.

### 3.2 Moderate Mismatches (Medium Impact)

#### D. Epochs and Early Stopping

| Aspect | Current | Official | Impact |
|--------|---------|----------|--------|
| Max epochs | 30 | 100 | **MEDIUM** |
| Early stopping | None | 10 epoch patience | **MEDIUM** |

**Why this matters**: Our training may be under-training, especially with lower epochs and no early stopping.

#### E. Weight Decay

| Aspect | Current | Official | Impact |
|--------|---------|----------|--------|
| Weight decay | 1e-4 | 0 (default) | **LOW** |

**Why this matters**: Weight decay adds regularization which may affect convergence.

### 3.3 Minor Mismatches (Low Impact)

#### F. Data Augmentation

| Aspect | Current | Official | Impact |
|--------|---------|----------|--------|
| Point jitter | None | Unknown | **UNKNOWN** |
| Rotation | None | Unknown | **UNKNOWN** |
| Dropout/subsampling | None | Unknown | **UNKNOWN** |

**Why this matters**: Augmentation helps generalization but specifics for 3D grounding are unclear.

#### G. Gradient Clipping

| Aspect | Current | Official | Impact |
|--------|---------|----------|--------|
| Gradient clipping | 1.0 | Not present | **LOW** |

---

## 4. Mismatch Impact Assessment

### Ranked by Likely Impact on Reproduction Gap

| Rank | Mismatch | Expected Impact | Confidence |
|------|----------|-----------------|------------|
| 1 | No warmup | High | High |
| 2 | Wrong scheduler type | High | High |
| 3 | No separate encoder LR | High | Medium |
| 4 | Smaller batch size | Medium-High | Medium |
| 5 | Fewer epochs | Medium | High |
| 6 | No early stopping | Medium | Medium |
| 7 | Weight decay present | Low | Low |
| 8 | No augmentation | Unknown | Low |

---

## 5. Minimal Protocol-Alignment Plan

### Phase A: Optimizer/Scheduler Alignment (Highest Priority)

1. **Add warmup**: 5 epochs of gradual learning rate warmup
2. **Change scheduler**: From CosineAnnealingLR to MultiStepLR with milestones [25, 40, 50, 60, 70, 80, 90] and gamma=0.65
3. **Separate encoder LR**: Use lower learning rate (1/10) for encoder backbone

### Phase B: Batch Size Alignment (High Priority)

1. **Option 1**: Increase batch size to 32 if memory allows
2. **Option 2**: Use gradient accumulation to achieve effective batch size of 32

### Phase C: Training Duration Alignment (Medium Priority)

1. **Increase epochs**: From 30 to 100 (or use early stopping)
2. **Add early stopping**: 10 epoch patience on validation accuracy

### Phase D: Augmentation (Lower Priority)

1. **Point jitter**: Small random noise to point coordinates
2. **Random rotation**: Around vertical axis if applicable

---

## 6. Controlled Experiment Protocol

### Experiment Group A: Optimizer/Scheduler

| Config | Change | Hypothesis |
|--------|--------|------------|
| A1 | Add 5-epoch warmup | Improves initial convergence |
| A2 | MultiStepLR instead of CosineAnnealing | Better matches official training |
| A3 | Separate encoder LR (0.1x) | Prevents encoder overfitting |
| A4 | Combined (A1+A2+A3) | Full optimizer alignment |

### Experiment Group B: Batch Size

| Config | Change | Hypothesis |
|--------|--------|------------|
| B1 | Gradient accumulation (4 steps) | Matches effective batch of 32 |
| B2 | Actual batch size 32 | Direct comparison (if memory allows) |

### Experiment Group C: Training Duration

| Config | Change | Hypothesis |
|--------|--------|------------|
| C1 | 100 epochs with early stopping | Allows full convergence |
| C2 | 50 epochs (compromise) | Sufficient for convergence |

---

## 7. Implementation Requirements

### Code Changes Needed

| Change | File | Complexity |
|--------|------|------------|
| Warmup scheduler | `train.py` | Low |
| MultiStepLR scheduler | `train.py` | Low |
| Separate encoder LR | `train.py` | Medium |
| Gradient accumulation | `train.py` | Medium |
| Early stopping | `train.py` | Medium |
| Data augmentation | `collate_fn` | Medium |

### New Dependencies

None required - all changes use existing PyTorch functionality.

---

## 8. Recommendations

### Immediate Actions

1. **Implement warmup**: 5 epochs of linear warmup
2. **Change to step-based LR schedule**: MultiStepLR with official milestones
3. **Separate encoder LR**: Add parameter groups with different LRs

### Secondary Actions

1. **Increase effective batch size**: Via gradient accumulation
2. **Extend training duration**: 50-100 epochs with early stopping
3. **Consider augmentation**: If above changes don't close gap

### Do NOT Change (Yet)

- Model architecture (encoder upgrade was already tested)
- Loss function
- Data splits
- Feature pipeline

---

## 9. Expected Impact

If training protocol mismatch is the dominant bottleneck:

| Change | Expected Val Acc@1 Improvement |
|--------|-------------------------------|
| Warmup + correct scheduler | +3-5% |
| Separate encoder LR | +2-3% |
| Larger batch / accumulation | +1-2% |
| Extended training | +1-2% |
| **Combined** | **+5-10%** |

This would bring Val Acc@1 from ~22% to ~27-32%, significantly closing the gap to 35.6%.

---

## 10. Conclusion

The current training protocol differs from the official ReferIt3D baseline in several important ways:

1. **No warmup** - Critical for stable training start
2. **Wrong scheduler type** - Step-based vs cosine
3. **No separate encoder LR** - Encoder may be overfitting
4. **Smaller batch size** - Different optimization dynamics
5. **Shorter training** - May be under-training

These protocol mismatches are likely contributing to the remaining reproduction gap. A controlled series of experiments aligning these parameters should materially improve performance.

---

## Sources

- [ReferIt3D Paper (ECCV 2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460409.pdf)
- [ReferIt3D Supplementary Material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460409-supp.pdf)
- [SAT ReferIt3D Training Script](https://github.com/zyang-ur/SAT/blob/main/referit3d/scripts/train_referit3d.py)
- [3DRefTransformer (WACV 2022)](https://openaccess.thecvf.com/content/WACV2022/html/Abdelreheem_3DRefTransformer_Fine-Grained_Object_Identification_in_Real-World_Scenes_Using_Natural_Language_WACV_2022_paper.html)
- [CoT3DRef (ICLR 2024)](https://arxiv.org/html/2310.06214v4)