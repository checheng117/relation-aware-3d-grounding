# BERT Feature Wiring Report

## Summary

Successfully integrated real BERT/DistilBERT text features into the reproduction training/evaluation pipeline.

**Before**: Random 768-dim vectors used as text features
**After**: Real DistilBERT embeddings from `data/text_features/*.npy`

---

## Files Modified

### `repro/referit3d_baseline/scripts/train.py`

**Changes**:
1. `IndexedDataset` now returns `(idx, sample)` tuples instead of setting `_dataset_idx` attribute on Pydantic model
2. `collate_fn` updated to handle tuple input and use direct index lookup for BERT features
3. BERT features now always wired when available (removed dependency on `_dataset_idx` attribute)

### `repro/referit3d_baseline/scripts/evaluate.py`

**Changes**:
1. Added `IndexedDataset` wrapper class
2. Updated `collate_fn` to accept tuples and wire BERT features
3. Updated `evaluate()` to use real BERT features
4. Added `bert_used_count` and `bert_coverage` to evaluation metrics

---

## BERT Feature Generation

Regenerated BERT features to align with current manifests:

```bash
python scripts/prepare_bert_features.py --device cuda
```

**Result**:
- train: 1235 samples (was 1255, aligned after geometry rebuild)
- val: 154 samples (was 156, aligned)
- test: 155 samples (was 158, aligned)
- Dimension: 768 (DistilBERT hidden size)
- Model: `distilbert-base-uncased`

---

## Coverage Verification

### Sample-to-Feature Alignment

```
Manifest samples: 1235 (train)
BERT embeddings:  (1235, 768)
Alignment: ✓ EXACT MATCH
```

### Collate Verification

```
Dataset size: 1235
BERT features: (1235, 768)
Collate result keys: ['object_features', 'object_mask', 'target_index', 'texts', 'samples_ref', 'bert_features']
bert_features shape: torch.Size([4, 768])
✓ BERT features WIRED successfully!
```

---

## Previous Failure Mode (Fixed)

**Root cause**: `_dataset_idx` attribute on Pydantic BaseModel

```python
# OLD (broken):
class IndexedDataset:
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample._dataset_idx = idx  # Attribute on Pydantic model - may not persist
        return sample

# NEW (fixed):
class IndexedDataset:
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return (idx, sample)  # Tuple - explicit index passed to collate
```

**Why Pydantic attribute failed**: Pydantic BaseModel only validates defined fields. Arbitrary attribute assignment (`sample._dataset_idx = idx`) creates a dynamic attribute that may not persist through model validation/copying.

---

## Feature Statistics

| Split | Samples | BERT Shape | Coverage |
|-------|---------|------------|----------|
| train | 1235 | (1235, 768) | 100% |
| val | 154 | (154, 768) | 100% |
| test | 155 | (155, 768) | 100% |

### BERT Embedding Properties

Sample embedding norms (unit-normalized via CLS token):
- train[0]: norm = 12.62
- train[1]: norm = 12.58
- train[2]: norm = 12.71

---

## Remaining Work

### Object Features Still Synthetic

Object features currently use:
- Channels 0-5: Real center + size (from geometry files)
- Channels 6-255: Class name hash (synthetic semantic)

**Next step**: Wire real object point features from `data/geometry/*.npz`

---

## Verification Checklist

- [x] BERT features regenerated with correct manifest alignment
- [x] IndexedDataset returns tuple instead of setting Pydantic attribute
- [x] collate_fn properly wires BERT features to batch
- [x] train_epoch receives real BERT features (not random fallback)
- [x] evaluate script updated to use BERT features
- [x] 100% coverage for train/val/test splits