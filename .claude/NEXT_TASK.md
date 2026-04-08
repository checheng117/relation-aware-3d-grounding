# Next Task

Phase: Complete Dataset Recovery and Gap Reduction

**Status**: Trustworthy baseline established at Test Acc@1 = 26.26%. Ready for next phase.

---

## Objective

Close the remaining 9.34% gap to 35.6% target through data recovery and geometry improvement.

---

## Current Situation

- Baseline: Test Acc@1 = 26.26% (on 23,186 samples)
- Target: Test Acc@1 = 35.6%
- Gap: -9.34%
- Progress: 73.76%

---

## Priority Actions

### 1. Complete Dataset Recovery (Expected +10-15%)

- Download remaining ScanNet scenes
- Generate aggregation files for missing scenes
- Target: Full 41,503 samples

### 2. Real Geometry Generation (Expected +3-5%)

- Extract point clouds from ScanNet meshes
- Generate real geometry files (not placeholder)
- Update manifests with new geometry

### 3. PointNet++ Evaluation (Expected +1-2%)

- Test PointNet++ on recovered 23K
- Compare to SimplePointEncoder baseline
- Document encoder comparison

---

## Expected Outcomes

| Action | Current | Expected | Target |
|--------|---------|----------|--------|
| Data recovery | 26.26% | 35-40% | 35.6% |
| + Geometry | - | 38-45% | - |
| + PointNet++ | - | 40-47% | - |

---

## Files to Create

- `reports/complete_recovery_plan.md`
- `reports/geometry_generation_plan.md`
- `outputs/<timestamp>_complete_recovery_rerun/`

---

## Success Criteria

- Test Acc@1 ≥ 35% (within 0.6% of target)
- Training stable with full dataset
- Results document progress to target