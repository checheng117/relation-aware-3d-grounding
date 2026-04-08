# Baseline Reproduction Audit

**Date**: 2026-04-03
**Objective**: Reproduce official ReferIt3D baseline before custom method development

---

## 1. Exact Public Baseline We Are Reproducing First

### 1.1 Target Model

**ReferIt3DNet** - the official baseline from the ECCV 2020 paper "ReferIt3D: Neural Listeners for Fine-Grained 3D Object Identification"

### 1.2 Public Benchmark Target Number

| Metric | Target |
|---|---|
| **Nr3D Overall Accuracy** | 35.6% |
| Nr3D Easy | 43.6% |
| Nr3D Hard | 27.9% |
| Nr3D View-Dependent | 32.5% |
| Nr3D View-Independent | 37.1% |

**Source**: [ReferIt3D Benchmarks](https://referit3d.github.io/benchmarks.html)

### 1.3 Authoritative Code/Protocol Source

| Resource | URL |
|---|---|---|
| **Official Repository** | https://github.com/referit3d/referit3d |
| **Project Website** | https://referit3d.github.io/ |
| **Paper** | ECCV 2020: "ReferIt3D: Neural Listeners for Fine-Grained 3D Object Identification" |

---

## 2. Current Repo Analysis

### 2.1 What is Custom and Should NOT Be Used for Baseline Reproduction

| Component | Location | Why Exclude |
|---|---|---|
| `StructuredRelationModel` | `src/rag3d/relation_reasoner/structured_relation_model.py` | Custom structured reasoning with anchor selection |
| `RawTextRelationModel` | `src/rag3d/relation_reasoner/model.py` | Custom relation-aware variant |
| `FallbackController` | `src/rag3d/relation_reasoner/fallback_controller.py` | Custom fallback logic |
| `VlmParserAdapter` | `src/rag3d/parsers/vlm_parser_adapter.py` | Custom VLM parser integration |
| `HeuristicParser` | `src/rag3d/parsers/heuristic_parser.py` | Custom structured parsing |
| `TextHashEncoder` | `src/rag3d/relation_reasoner/text_encoding.py` | Hash-based text encoding (not real language model) |
| Parser-aware experiment scripts | `scripts/run_phase3_*.py` | Custom ablation infrastructure |

**Key issue**: All current models use `TextHashEncoder` which is NOT a real language encoder. This is a toy implementation.

### 2.2 What May Be Reusable

| Component | Location | Status | Reusability |
|---|---|---|---|
| `ReferIt3DManifestDataset` | `src/rag3d/datasets/referit3d.py` | ✅ Reusable | Basic dataset loader |
| `GroundingSample` schema | `src/rag3d/datasets/schemas.py` | ✅ Reusable | Sample representation |
| `collate_grounding_samples` | `src/rag3d/datasets/collate.py` | ⚠️ Check | May need alignment |
| Evaluation metrics | `src/rag3d/evaluation/metrics.py` | ⚠️ Check | Need to verify alignment |
| `PointNetToyEncoder` | `src/rag3d/encoders/point_encoder.py` | ❌ Too simple | Need real PointNet++ |
| Processed manifests | `data/processed/*.jsonl` | ⚠️ Mock | Geometry is placeholder |

### 2.3 Current Data Status

| Issue | Current State | Required State |
|---|---|---|
| Object centers | All [0.0, 0.0, 0.0] | Real ScanNet coordinates |
| Object sizes | All [0.1, 0.1, 0.1] | Real bounding boxes |
| Feature vectors | All null | Real PointNet++ features |
| Language features | Hash embeddings | BERT/DistilBERT embeddings |
| Point clouds | Not present | 1024 sampled points per object |

**Critical gap**: Current data is mock/synthetic. Real ScanNet geometry and NR3D annotations are needed.

---

## 3. Biggest Reproduction Risks

### 3.1 Data Pipeline Mismatch (HIGH RISK)

| Risk | Description | Mitigation |
|---|---|---|
| **No real ScanNet data** | Current mock geometry doesn't match real scans | Download official ScanNet processed data |
| **No real point clouds** | PointNet++ requires actual point samples | Extract point clouds from ScanNet meshes |
| **Split mismatch** | Custom split vs official split | Use official Nr3D train/val/test splits |
| **Object ID mismatch** | Custom indexing vs ScanNet object IDs | Use ScanNet segmentation IDs |

### 3.2 Model Architecture Mismatch (HIGH RISK)

| Component | Official Baseline | Current Repo |
|---|---|---|
| Point encoder | PointNet++ | Simple MLP |
| Language encoder | BERT/DistilBERT | Hash embeddings |
| Fusion | Specific architecture | Custom relation models |
| Training objective | Cross-entropy on classification | Same, but need correct features |

### 3.3 Evaluation Protocol Drift (MEDIUM RISK)

| Risk | Description |
|---|---|
| Candidate set definition | Are we evaluating on same candidate objects? |
| Metric computation | Verify Acc@1 calculation matches official |
| Easy/Hard split | Need to align with official split definitions |
| View-dependent split | Need view-dependence annotations |

### 3.4 Implementation Gaps (MEDIUM RISK)

| Gap | Description |
|---|---|
| No real PointNet++ implementation | Need to add or import PointNet++ backbone |
| No BERT integration | Need real language model for utterance encoding |
| No data preprocessing pipeline | Need to build from ScanNet raw data |

---

## 4. Implementation Plan

### Phase A: Data Alignment (Critical)

1. **Download official NR3D data**
   - Source: Hugging Face `chouss/nr3d` (already in config)
   - Contains: Real utterances, scene IDs, target object IDs

2. **Obtain ScanNet processed data**
   - Need: Scene meshes, segmentation, bounding boxes
   - Process: Extract point clouds per object segment

3. **Align splits**
   - Use official Nr3D train/val/test split
   - Verify sample counts match official benchmark

### Phase B: Baseline Model Implementation

Option 1: **Port official ReferIt3DNet** (Recommended)
- Clone official repo structure into `repro/referit3d/`
- Use their PointNet++ backbone
- Use their BERT integration
- Minimal modification to run

Option 2: **Implement from scratch**
- Add PointNet++ encoder to current repo
- Add BERT encoder
- Risk of protocol drift

**Recommendation**: Option 1 - port official implementation to ensure fidelity.

### Phase C: Evaluation Alignment

1. Verify metric computation matches official
2. Export results in official format
3. Compare to benchmark target (35.6%)

### Phase D: Gap Analysis

1. Document any remaining gap from 35.6%
2. Identify causes (data, model, training)
3. Iterate until acceptable reproduction

---

## 5. Proposed Directory Structure

```
repro/
└── referit3d_baseline/
    ├── configs/
    │   └── official_baseline.yaml
    ├── scripts/
    │   ├── train_official_baseline.py
    │   └── eval_official_baseline.py
    ├── src/
    │   └── referit3d_net.py      # Ported from official repo
    ├── data/                       # Symlinks to real data
    └── README.md                   # Reproduction log
```

This isolation ensures:
- No mixing with custom structured reasoning code
- Clear provenance from official implementation
- Easy auditing of any deviations

---

## 6. Success Criteria

| Level | Gap from Target | Description |
|---|---|---|
| **Exact reproduction** | ≤ 2% | 33.6% - 37.6% accuracy |
| **Acceptable reproduction** | 3-5% | 30.6% - 33.6% or 37.6% - 40.6% |
| **Partial reproduction** | > 5% | Pipeline runs but significant gap |

---

## 7. Next Steps

1. **Verify data availability** - Check if we have real ScanNet data or need to download
2. **Create reproduction directory** - `repro/referit3d_baseline/`
3. **Port official model** - Minimal modification to official ReferIt3DNet
4. **Run smoke test** - Verify pipeline with small subset
5. **Full reproduction run** - Train and evaluate on full Nr3D

---

## 8. Summary

| Aspect | Status |
|---|---|
| **Target baseline** | ReferIt3DNet (35.6% Nr3D) |
| **Current repo** | Custom models, mock data, placeholder encoders |
| **Biggest gap** | No real ScanNet geometry, no real language features |
| **Recommended approach** | Port official implementation to isolated `repro/` directory |
| **Risk level** | HIGH - Data and model both need significant work |

**Key insight**: The current repo is optimized for custom structured reasoning experiments, not baseline reproduction. A clean, isolated reproduction track is necessary to establish a trustworthy anchor.