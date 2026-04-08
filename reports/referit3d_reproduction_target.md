# ReferIt3D Reproduction Target

**Date**: 2026-04-03
**Objective**: Define exact reproduction target and success criteria

---

## A. Baseline Target

### Model
**ReferIt3DNet** - Official baseline from ECCV 2020 paper

### Dataset
**Nr3D** (Natural References in 3D Scenes)

### Target Metric
**Overall Accuracy**: 35.6%

### Sub-metrics
| Split | Target |
|---|---|
| Easy | 43.6% |
| Hard | 27.9% |
| View-Dependent | 32.5% |
| View-Independent | 37.1% |

---

## B. Protocol Components

### B.1 Dataset

| Component | Specification |
|---|---|
| **Name** | Nr3D (Natural References in 3D Scenes) |
| **Source** | https://referit3d.github.io/ or Hugging Face `chouss/nr3d` |
| **Total samples** | ~41,503 (paper) / 1,569 (our processed) |
| **Scenes** | ScanNet processed scenes |
| **Language** | Natural human utterances |

### B.2 Splits

| Split | Samples (Paper) | Samples (Ours) |
|---|---|---|
| Train | ~27,000 | 1,255 |
| Val | ~5,000 | 156 |
| Test | ~9,500 | 158 |

**Note**: Our processed data has significantly fewer samples. Need to verify if this is the full Nr3D or a subset.

### B.3 Candidate Object Setup

| Aspect | Official Protocol |
|---|---|
| Object source | ScanNet instance segmentation |
| Candidate set | All objects in scene OR entity-referenced objects |
| Object ID | ScanNet segment ID (objectId in aggregation.json) |

### B.4 Object Point Cloud Preprocessing

| Aspect | Official Protocol |
|---|---|
| Points per object | 1024 (typical) |
| Sampling | Random sampling from mesh/points |
| Normalization | Centered and scaled per-object |
| Color | RGB values included or not (verify) |

### B.5 Language Preprocessing

| Aspect | Official Protocol |
|---|---|
| Tokenizer | BERT/DistilBERT tokenizer |
| Max length | Typically 128 or 256 tokens |
| Embedding | BERT hidden states (768-dim or projected) |

### B.6 Evaluation Metric Definitions

| Metric | Definition |
|---|---|
| **Overall Accuracy** | Correct predictions / Total samples |
| **Acc@1** | Top-1 prediction matches target |
| **Acc@5** | Target in top-5 predictions |
| **Easy** | Single-instance objects or unambiguous references |
| **Hard** | Multi-instance objects with spatial relations |
| **View-Dependent** | References requiring viewpoint understanding |
| **View-Independent** | References not requiring viewpoint |

---

## C. Reproduction Success Levels

### C.1 Level Definitions

| Level | Gap from Target | Accuracy Range | Description |
|---|---|---|---|
| **Exact** | ≤ 2% | 33.6% - 37.6% | Near-perfect reproduction |
| **Acceptable** | 3-5% | 30.6% - 33.6% or 37.6% - 40.6% | Minor deviations |
| **Partial** | > 5% | < 30.6% or > 40.6% | Significant gap, needs analysis |

### C.2 Reproduction Checklist

For reproduction to be considered valid:

- [ ] Data split matches official
- [ ] Sample count matches official (within 5%)
- [ ] Candidate object definition matches
- [ ] Point cloud preprocessing matches
- [ ] Language encoding matches
- [ ] Model architecture matches
- [ ] Training procedure matches
- [ ] Evaluation protocol matches

### C.3 Gap Documentation Requirements

If gap > 2%, document:

1. **Data differences**: Sample counts, split definitions, object candidates
2. **Model differences**: Architecture, hyperparameters, initialization
3. **Training differences**: Epochs, batch size, learning rate, optimizer
4. **Evaluation differences**: Metric computation, candidate filtering

---

## D. Implementation Requirements

### D.1 Model Architecture

Based on official ReferIt3DNet:

```
Input:
  - Object point clouds: [B, N, 1024, 3+color]
  - Utterance tokens: [B, L]

Encoder:
  - PointNet++ backbone → per-object features [B, N, D]
  - BERT encoder → language features [B, D]

Fusion:
  - Cross-attention or concatenation
  - MLP classifier per object

Output:
  - Per-object scores: [B, N]
  - Softmax → prediction
```

### D.2 Training Configuration

| Parameter | Typical Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 to 1e-3 |
| Batch size | 16-64 |
| Epochs | 20-50 |
| Loss | Cross-entropy |

### D.3 Required Data Files

| File | Source | Status |
|---|---|---|
| Nr3D annotations | `nr3d_annotations.json` | ✅ Available |
| ScanNet scenes | ScanNet processed meshes | ❌ Need to download |
| Aggregation files | `*.aggregation.json` | ✅ Available |
| Point clouds | Extracted from meshes | ❌ Need to generate |

---

## E. Current Data Audit

### E.1 What We Have

| Data | Count | Status |
|---|---|---|
| NR3D annotations | 1,569 samples | ✅ Real utterances |
| ScanNet scenes | 270+ scenes | ⚠️ Aggregation only, no meshes |
| Train split | 1,255 samples | ✅ Processed |
| Val split | 156 samples | ✅ Processed |
| Test split | 158 samples | ✅ Processed |

### E.2 What We're Missing

| Data | Source | Action |
|---|---|---|
| ScanNet mesh files | ScanNet website | Download |
| Point cloud extraction | Custom pipeline | Implement |
| Object bounding boxes | From aggregation | Parse/derive |
| Easy/Hard labels | NR3D metadata | May need to derive |
| View-dependence labels | NR3D metadata | Need source |

---

## F. Verification Commands

### F.1 Check Data Integrity

```bash
# Count samples
wc -l data/processed/train_manifest.jsonl
wc -l data/processed/val_manifest.jsonl
wc -l data/processed/test_manifest.jsonl

# Verify scenes
ls data/raw/referit3d/scans/ | wc -l

# Check annotations
python -c "import json; data=json.load(open('data/raw/referit3d/annotations/nr3d_annotations.json')); print(len(data))"
```

### F.2 Check Model Output Shape

```python
# After implementation
model = ReferIt3DNet(...)
output = model(points, utterance)
assert output.shape == (batch_size, num_objects)
```

---

## G. References

| Resource | URL |
|---|---|
| Official Repository | https://github.com/referit3d/referit3d |
| Benchmark Page | https://referit3d.github.io/benchmarks.html |
| Paper | ECCV 2020: "ReferIt3D: Neural Listeners for Fine-Grained 3D Object Identification" |
| NR3D on HuggingFace | https://huggingface.co/datasets/chouss/nr3d |

---

**End of Reproduction Target Document**