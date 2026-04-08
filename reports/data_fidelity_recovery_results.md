# Data Fidelity Recovery Results

**Date**: 2026-04-03
**Phase**: BERT integration and fidelity rerun

---

## 1. Exact Commands Run

```bash
# BERT feature preparation
python scripts/prepare_bert_features.py --device cpu

# Training with BERT features
python repro/referit3d_baseline/scripts/train.py --device cpu
```

---

## 2. Data Assets Used

| Asset | Location | Status |
|---|---|---|
| Train manifest | `data/processed/train_manifest.jsonl` | 1,255 samples |
| Val manifest | `data/processed/val_manifest.jsonl` | 156 samples |
| Test manifest | `data/processed/test_manifest.jsonl` | 158 samples |
| Train BERT features | `data/text_features/train_bert_embeddings.npy` | (1255, 768) |
| Val BERT features | `data/text_features/val_bert_embeddings.npy` | (156, 768) |
| Test BERT features | `data/text_features/test_bert_embeddings.npy` | (158, 768) |

---

## 3. BERT Integration Summary

| Component | Details |
|---|---|
| **Model** | distilbert-base-uncased |
| **Hidden dim** | 768 |
| **Max length** | 128 tokens |
| **Aggregation** | [CLS] token |

---

## 4. Formal Rerun Result

### 4.1 Training Progress (with BERT)

| Epoch | Train Loss | Train Acc | Val Acc@1 | Val Acc@5 |
|---|---|---|---|---|
| 1 | 3.4615 | 12.91% | 14.10% | 66.67% |
| 5 | 2.3745 | 16.18% | 17.95% | 73.08% |
| 10 | 2.0221 | 24.70% | **21.79%** | 82.69% |
| 15 | 1.7867 | 23.35% | 24.36% | 86.54% |
| 20 | 1.6654 | 27.01% | 24.36% | 86.54% |
| 25 | 1.6145 | 26.14% | 24.36% | 87.82% |
| 30 | 1.5957 | 27.81% | 24.36% | 88.46% |

### 4.2 Best Results

| Metric | Value |
|---|---|
| **Best Val Acc@1** | **25.00%** |
| Best Val Acc@5 | 89.10% |
| Final Train Loss | 1.5957 |

---

## 5. Old vs New Reproduction Comparison

| Version | Acc@1 | Acc@5 | Gap from Target |
|---|---|---|---|
| **Old (placeholder)** | 16.03% | 73.08% | -19.57% |
| **New (BERT)** | **25.00%** | **89.10%** | **-10.60%** |
| **Target** | 35.6% | - | - |

### 5.1 Improvement

| Metric | Improvement |
|---|---|
| Acc@1 | **+8.97%** |
| Acc@5 | **+16.02%** |
| Gap reduction | **8.97%** closer to target |

---

## 6. Remaining Gap Analysis

### 6.1 Current Gap

**-10.60%** (25.00% vs 35.6% target)

### 6.2 Identified Bottlenecks

| Bottleneck | Estimated Impact | Status |
|---|---|---|
| **No real point clouds** | -5~8% | ❌ Missing |
| **No real geometry** | -2~4% | ❌ Missing |
| **Subset data (1,569 vs 41,503)** | -2~5% | ⚠️ Investigate |
| **Model architecture deviation** | -1~2% | ⚠️ Minor |

### 6.3 What's Working

- [x] BERT language features integrated
- [x] Training pipeline functional
- [x] Loss decreases properly
- [x] Acc@5 improved significantly (89%)

---

## 7. Final Decision

### 7.1 Decision: **B. Baseline improved but still not trustworthy**

### 7.2 Reasoning

| Factor | Status |
|---|---|
| **Accuracy** | 25.00% (below 30% threshold) |
| **Gap from target** | 10.6% (above 5% threshold) |
| **Missing components** | Real ScanNet geometry, full dataset |

**The baseline improved significantly (+8.97%) but remains below the minimum useful threshold (30%).**

### 7.3 Recommended Next Steps

**Priority 1: Obtain real ScanNet data**
- Real point clouds could add +5-8% accuracy
- This is the largest remaining bottleneck

**Priority 2: Verify dataset coverage**
- Current 1,569 samples vs official 41,503
- May need full NR3D download

**Priority 3: Consider partial progress**
- Can proceed to MVT reproduction with caveat
- Document that baseline is partially recovered

---

## 8. Summary

| Aspect | Status |
|---|---|
| **BERT integration** | ✅ Complete |
| **Geometry recovery** | ❌ Not attempted (no ScanNet access) |
| **Dataset verification** | ⚠️ Subset only |
| **Accuracy improvement** | +8.97% |
| **Trustworthiness** | Still below threshold |

**Key insight**: BERT features provided significant improvement, but real ScanNet geometry is likely required to reach credible reproduction (>30%).

---

**End of Data Fidelity Recovery Results**