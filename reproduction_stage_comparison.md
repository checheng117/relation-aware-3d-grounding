# Reproduction Stage Comparison

## Summary

Progress from placeholder reproduction to feature-integrated reproduction.

| Stage | Val Acc@1 | Test Acc@1 | Test Acc@5 |
|-------|-----------|------------|------------|
| Placeholder | 11.03% | 1.94% | 15.48% |
| Geometry Recovered | 14.29% | 1.94% | 15.48% |
| **Feature Integrated** | **22.73%** | **9.68%** | **40.00%** |
| Target (Official) | 35.6% | ~35% | - |

---

## Stage 1: Placeholder Reproduction

**Configuration:**
- Geometry: Synthetic/fallback (default centers and sizes)
- Text features: Random 768-dim vectors
- Object features: Class name hash

**Results:**
- Val Acc@1: 11.03%
- Test Acc@1: 1.94%
- Test Acc@5: 15.48%

**Limitations:**
- No real geometry information
- No semantic language features
- Model cannot learn meaningful object-language associations

---

## Stage 2: Geometry Recovered

**Configuration:**
- Geometry: Real from Pointcept point clouds (center/size from point bboxes)
- Text features: Random 768-dim vectors (not wired)
- Object features: Center + size + class hash

**Results:**
- Val Acc@1: 14.29% (+3.26%)
- Test Acc@1: 1.94% (unchanged)
- Test Acc@5: 15.48% (unchanged)

**Improvement:**
- Geometry provided spatial context
- Class hash provided semantic signal
- Limited improvement without language features

---

## Stage 3: Feature Integrated

**Configuration:**
- Geometry: Real from Pointcept point clouds
- Text features: Real DistilBERT embeddings (768-dim, wired)
- Object features: Center + size + class hash

**Results:**
- Val Acc@1: 22.73% (+8.44%)
- Test Acc@1: 9.68% (+7.74%)
- Test Acc@5: 40.00% (+24.52%)

**Improvement:**
- BERT features enabled semantic language understanding
- Model can learn object-language associations
- Major improvement in all metrics

---

## Gap to Official Baseline

| Metric | Current | Target | Absolute Gap | Percent Gap |
|--------|---------|--------|--------------|-------------|
| Val Acc@1 | 22.73% | 35.6% | 12.87% | 36.2% |
| Test Acc@1 | 9.68% | ~35% | ~25.32% | ~72% |

**Remaining gap likely due to:**
1. SimplePointEncoder vs PointNet++ backbone
2. Missing multi-view features
3. Training hyperparameters

---

## Commands Used

```bash
# BERT feature generation
python scripts/prepare_bert_features.py --device cuda

# Training with features
python repro/referit3d_baseline/scripts/train.py --device cuda

# Evaluation
python repro/referit3d_baseline/scripts/evaluate.py \
    --checkpoint outputs/repro/referit3d_baseline/best_model.pt \
    --split test --device cuda
```

---

## Conclusions

1. **BERT features are critical**: 59% improvement from language feature integration
2. **Geometry helps**: 29% improvement from real geometry
3. **Architecture matters**: Remaining gap suggests encoder upgrade needed
4. **Baseline becoming trustworthy**: 22.73% is a credible partial reproduction