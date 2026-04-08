# Data Fidelity Recovery Plan

**Date**: 2026-04-03
**Phase**: Recover data and feature fidelity for credible baseline reproduction

---

## 1. Current Achieved Result vs Target

### 1.1 Current Status

| Metric | Achieved | Target | Gap |
|---|---|---|---|
| **Nr3D Overall Acc@1** | 16.03% | 35.6% | **-19.57%** |

### 1.2 Classification

**PARTIAL REPRODUCTION** (gap > 5%)

### 1.3 Known Deficiencies

| Component | Current State | Required State |
|---|---|---|
| Object features | Hash-based synthetic | PointNet++ from real point clouds |
| Language features | Random tensors | BERT/DistilBERT embeddings |
| Geometry | Placeholder (centers [0,0,0]) | Real ScanNet coordinates |
| Dataset size | 1,569 samples | ~41,503 samples |
| Scene meshes | Not available | ScanNet .ply files |

---

## 2. High-Impact Missing Components

### 2.1 Component Impact Analysis

| Component | Estimated Impact | Effort | Priority |
|---|---|---|---|
| **Real point cloud features** | +8-12% accuracy | High | **P1** |
| **BERT language features** | +5-8% accuracy | Medium | **P2** |
| **Real geometry (centers/bboxes)** | +2-4% accuracy | Medium | **P3** |
| **Full NR3D dataset** | +3-5% accuracy | Medium | **P4** |

### 2.2 Detailed Missing Components

#### A. ScanNet Geometry

**What we have**:
- `*.aggregation.json` files (segment ID → object ID mapping)
- Segment IDs for each object

**What we need**:
- ScanNet mesh files (`.ply` or `.obj`)
- Segment-to-vertex mapping
- Ability to extract points per object

**Status**: ❌ Critical gap

#### B. Point Cloud Extraction

**Required pipeline**:
1. Load ScanNet mesh
2. Load segment assignment
3. For each object (objectId), extract vertices from segments
4. Sample 1024 points per object
5. Optionally include RGB

**Status**: ❌ Not implemented

#### C. BERT Integration

**What we need**:
- Pretrained BERT model (bert-base-uncased or distilbert)
- Tokenizer
- Forward pass to get [CLS] embedding (768-dim)
- Optional: project to 256-dim

**Status**: ❌ Using random features

#### D. Full NR3D Dataset

**Current**: 1,569 samples
**Official**: ~41,503 samples

**Possible explanations**:
1. We have a subset (test set only?)
2. Annotations were filtered
3. Missing data download

**Status**: ⚠️ Need to investigate

---

## 3. Dependency Order

### Phase A: Geometry Recovery (Highest Impact)

```
1. Obtain ScanNet meshes
   ↓
2. Implement point extraction pipeline
   ↓
3. Generate per-object point clouds
   ↓
4. Verify geometry sanity
```

### Phase B: Language Feature Recovery

```
1. Install transformers library
   ↓
2. Load pretrained BERT
   ↓
3. Pre-compute text embeddings
   ↓
4. Cache for reuse
```

### Phase C: Dataset Coverage

```
1. Verify current split sizes
   ↓
2. Compare to official
   ↓
3. Obtain missing data if needed
```

### Phase D: Model Fidelity

```
1. Update point encoder (PointNet++ or faithful equivalent)
   ↓
2. Update language encoder
   ↓
3. Verify training schedule
   ↓
4. Re-run reproduction
```

---

## 4. Implementation Risks

### 4.1 ScanNet Access

**Risk**: ScanNet requires license agreement

**Mitigation**: 
- Check if we already have access
- Use publicly available processed versions if accessible
- Document if unavailable

### 4.2 Compute Requirements

**Risk**: Point cloud extraction and BERT inference require GPU

**Mitigation**:
- Pre-compute features offline
- Cache to disk
- Use CPU fallback with patience

### 4.3 Data Version Mismatch

**Risk**: Our NR3D annotations may not match official splits

**Mitigation**:
- Document exact version used
- Cross-reference with official benchmark

### 4.4 PointNet++ Implementation

**Risk**: Full PointNet++ is complex to implement correctly

**Mitigation**:
- Start with simpler PointNet encoder
- Use existing implementations if available
- Document approximation

---

## 5. Concrete Success Thresholds

### 5.1 Threshold Definitions

| Level | Accuracy Range | Description |
|---|---|---|
| **Not yet trustworthy** | < 30% | Major data fidelity issues remain |
| **Minimum useful recovery** | 30% - 32.6% | Pipeline works, some gap remains |
| **Credible reproduction** | 32.6% - 38.6% | Within 3 points of target |
| **Exact reproduction** | 33.6% - 37.6% | Within 2 points of target |

### 5.2 Success Metrics Per Phase

| Phase | Target Accuracy | Measured By |
|---|---|---|
| After geometry recovery | 24-28% | Point cloud features working |
| After BERT integration | 28-32% | Language features working |
| After dataset coverage | 30-34% | Full data used |
| After model fidelity | 32-36% | Complete pipeline |

### 5.3 Decision Points

**After each phase, evaluate**:
- If accuracy improved by expected amount → continue
- If accuracy did not improve → investigate bottleneck
- If accuracy reaches credible range → can proceed to MVT

---

## 6. Implementation Plan

### Step 1: Geometry Recovery

**Files to create**:
- `scripts/prepare_scannet_geometry.py`
- `scripts/validate_scannet_geometry.py`
- `reports/scannet_geometry_alignment.md`

**Expected output**:
- Per-object point cloud files (`.npy` or `.pt`)
- Geometry statistics (centers, sizes, point counts)

### Step 2: NR3D Coverage Verification

**Files to create**:
- `scripts/validate_nr3d_coverage.py`
- `reports/nr3d_coverage_verification.md`

**Expected output**:
- Exact sample counts
- Scene coverage
- Missing data documentation

### Step 3: BERT Integration

**Files to create**:
- `scripts/prepare_bert_features.py`
- `scripts/validate_bert_features.py`
- `reports/bert_feature_alignment.md`

**Expected output**:
- Cached BERT embeddings
- Feature dimension validation

### Step 4: Model Fidelity Recheck

**Files to create**:
- `reports/referit3d_baseline_fidelity_recheck.md`

**Expected output**:
- Architecture comparison
- Deviation documentation

### Step 5: Formal Rerun

**Output directory**:
- `outputs/<timestamp>_referit3d_fidelity_rerun/`

**Expected output**:
- Training logs
- Evaluation results
- Comparison to placeholder run

---

## 7. Expected Outcomes

### 7.1 Pessimistic Scenario

If we cannot obtain ScanNet meshes:
- Accuracy: 20-24% (BERT + better synthetic features)
- Status: Not yet trustworthy
- Action: Document limitations, proceed with caveat

### 7.2 Moderate Scenario

If we get geometry but not full NR3D:
- Accuracy: 28-32%
- Status: Minimum useful recovery
- Action: Document dataset limitation, can proceed to MVT

### 7.3 Optimistic Scenario

If we get full ScanNet + full NR3D + BERT:
- Accuracy: 32-36%
- Status: Credible reproduction
- Action: Proceed to MVT reproduction

---

## 8. Timeline

| Step | Duration | Dependencies |
|---|---|---|
| Geometry recovery | 1-2 days | ScanNet access |
| NR3D verification | 0.5 days | None |
| BERT integration | 0.5 days | transformers |
| Model fidelity | 0.5 days | Above complete |
| Formal rerun | 0.5 days | All above |

**Total**: 3-4 days

---

## 9. Success Criteria

This phase is complete only if:

- [ ] Real geometry pipeline attempted (success or documented limitation)
- [ ] NR3D coverage audited
- [ ] BERT features integrated
- [ ] Staged reruns completed
- [ ] Old vs new comparison exported
- [ ] Final trustworthiness decision made

---

## 10. Immediate Next Steps

1. **Check ScanNet access**: Can we download meshes?
2. **Verify NR3D source**: Where did our 1,569 samples come from?
3. **Install transformers**: For BERT integration
4. **Begin geometry pipeline**: Even if approximate

---

**End of Data Fidelity Recovery Plan**