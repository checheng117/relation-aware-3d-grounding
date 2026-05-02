# Dense Scorer Strengthening Log

**Date**: 2026-04-22
**Phase**: Implementation and Evaluation

---

## Implementation Summary

### Modified Files

1. **src/rag3d/models/cover3d_dense_relation.py**
   - Removed Tanh activation for unbounded score output
   - Added `aggregation` parameter: 'weighted', 'max', 'hybrid', 'attention'
   - Added `use_focal` and `focal_gamma` parameters for hard-negative training
   - Added attention pooling mechanism with query/key/value projections
   - Added `attention_proj` layer to project relation_context to attention space

2. **scripts/train_cover3d_round1.py**
   - Added variant choices: dense-v2-attpool, dense-v3-geo, dense-v4-hardneg
   - Added variant configuration blocks for each strengthening variant
   - Added focal weighting support in train_epoch (focal_gamma parameter)
   - Fixed target_index type conversion bug in EmbeddingDataset

### Variant Configurations

| Variant | Aggregation | use_geometry | use_focal | focal_gamma | Notes |
|---------|-------------|--------------|-----------|-------------|-------|
| dense-no-cal | weighted | False | False | 0.0 | Anchor |
| dense-v2-attpool | attention | False | False | 0.0 | Attention pooling |
| dense-v3-geo | weighted | True | False | 0.0 | Geometry features |
| dense-v4-hardneg | hybrid | False | True | 2.0 | Focal weighting |

---

## Smoke Test Results (1 Epoch)

### Dense-v2-AttPool
```
Test Acc@1: 25.24%
Test Acc@5: 79.95%
Recovered: 514, Harmed: 752, Net: -238
```
Status: Runs successfully, but severe performance degradation

### Dense-v3-Geo
```
Test Acc@1: 24.32%
Test Acc@5: 79.06%
Recovered: 506, Harmed: 783, Net: -277
```
Status: Runs with zeros fallback for geometry (data not available)

### Dense-v4-HardNeg
```
Test Acc@1: 24.47%
Test Acc@5: 79.41%
Recovered: 506, Harmed: 777, Net: -271
```
Status: Runs successfully, but severe performance degradation

---

## Full Training Attempts

### Dense-v2-AttPool (10 epochs attempted)
- Epoch 1: Loss=2.3879, Acc=30.37%
- Epoch 2: Loss=1.5631, Acc=32.13%
- Epoch 3+: Training crashed (data loading error, fixed)
- Post-fix: Training extremely slow (~3 min/epoch due to attention)
- Decision: Stopped early due to poor 1-epoch results and computational cost

---

## Key Learnings

1. **Attention is expensive**: The attention pooling mechanism adds ~10x computational cost per epoch

2. **Geometry data not available**: The extracted embeddings don't include object centers/sizes needed for geometry features

3. **Focal weighting needs curriculum**: Starting with gamma=2.0 immediately down-weights all samples, hurting initial learning

4. **Weak signal foundation**: All variants assume the base relation scores are informative; results suggest they may not be

---

## Code Changes Summary

### cover3d_dense_relation.py

**New Parameters**:
```python
def __init__(
    self,
    object_dim: int = 320,
    language_dim: int = 256,
    geometry_dim: int = 6,
    hidden_dim: int = 256,
    aggregation: str = "weighted",  # NEW
    use_focal: bool = False,        # NEW
    focal_gamma: float = 2.0,       # NEW
    use_attention: bool = False,    # NEW
    attention_heads: int = 4,       # NEW
    attention_hidden_dim: int = 128,# NEW
    ...
)
```

**Attention Pooling** (new):
```python
elif self.aggregation == "attention":
    relation_proj = self.attention_proj(relation_context)
    query = self.attention_query.unsqueeze(0).expand(B, N, -1)
    key = self.attention_key(relation_proj)
    value = self.attention_value(relation_proj)
    attn_output = torch.bmm(pair_weights, value)
    gate = torch.sigmoid(query + attn_output)
    relation_scores = self.aggregation_mlp(gate * query).squeeze(-1)
```

**Hybrid Aggregation** (new):
```python
elif self.aggregation == "hybrid":
    relation_scores_max, _ = all_pair_scores.max(dim=-1)
    relation_scores_weighted = (pair_weights * safe_pair_scores).sum(dim=-1)
    relation_scores = relation_scores_max + 0.1 * relation_scores_weighted
```

**Focal Weighting** (in train_cover3d_round1.py):
```python
if focal_gamma > 0:
    log_probs = F.log_softmax(fused_logits, dim=-1)
    pt = torch.exp(log_probs.gather(1, target_index.unsqueeze(1)))
    focal_weight = (1 - pt) ** focal_gamma
    ce_loss = (ce_loss * focal_weight.squeeze()).mean()
```

---

## Files Generated

- `reports/dense_strengthening_diagnosis.md` - Bottleneck analysis
- `reports/dense_strengthening_results.md` - Final results and Route decision
- `reports/dense_strengthening_table.csv` - Comparison data
- `reports/dense_strengthening_log.md` - This implementation log
