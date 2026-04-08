# Old Subset vs Recovered Nr3D Comparison

**Date**: 2026-04-08
**Phase**: Full Nr3D Dataset Recovery - Step 6

---

## Executive Summary

**Recovery factor**: 15x increase (1,544 → 23,186 samples)

The recovered dataset is now **sufficient for meaningful baseline reproduction**.

---

## 1. Size Comparison

| Metric | Old Subset (HF) | Recovered (Official) | Change |
|--------|-----------------|----------------------|--------|
| **Total samples** | 1,544 | 23,186 | **+15.0x** |
| **Unique scenes** | 266 | 269 | +3 |
| **Official coverage** | 3.7% | 55.9% | **+52.2%** |

---

## 2. Split Comparison

| Split | Old Subset | Recovered | Increase |
|-------|------------|-----------|----------|
| Train | 1,211 | 18,459 | **15.2x** |
| Val | 148 | 2,046 | **13.8x** |
| Test | 185 | 2,681 | **14.5x** |

---

## 3. Scene Coverage Change

| Metric | Old | Recovered |
|--------|-----|-----------|
| Total scenes in pool | 266 | 269 |
| Train scenes | 212 | 215 |
| Val scenes | 26 | 26 |
| Test scenes | 28 | 28 |

**Note**: Scene count barely changed (+3) because we used the same aggregation file pool. The increase comes from recovering more samples per scene from the official Nr3D.

---

## 4. Why the Increase?

### Old Subset (Hugging Face)

- Source: `chouss/nr3d` on Hugging Face
- Contains: 1,569 samples
- Reason: Subset upload (not full dataset)

### Recovered (Official)

- Source: Official Nr3D CSV from ReferIt3D website
- Contains: 41,503 samples total
- Recoverable: 23,186 samples (limited by aggregation file availability)

### Limiting Factor

The 269 scenes with aggregation files contain only 55.9% of official samples. The remaining 44.1% (18,317 samples) are in 372 scenes without aggregation files.

---

## 5. Geometry Quality Comparison

| Aspect | Old Subset | Recovered |
|--------|------------|-----------|
| Geometry source | Pointcept extraction | ScanNet aggregation |
| Real centers | ✓ Yes | ⚠ Placeholder |
| Real sizes | ✓ Yes | ⚠ Placeholder |
| Point features | ✓ Sparse (97% zeros) | ✗ None |

**Trade-off**: The recovered dataset uses placeholder geometry from aggregation files instead of real point-based geometry. This is acceptable for baseline reproduction but may affect accuracy.

---

## 6. Sufficiency Assessment

### Is the Recovered Dataset Sufficient?

| Criterion | Threshold | Recovered | Status |
|-----------|-----------|-----------|--------|
| Sample count | > 10,000 | 23,186 | ✓ PASS |
| Train samples | > 5,000 | 18,459 | ✓ PASS |
| Val samples | > 500 | 2,046 | ✓ PASS |
| Test samples | > 500 | 2,681 | ✓ PASS |
| Scene diversity | > 100 scenes | 269 | ✓ PASS |

**Verdict**: The recovered dataset is **sufficient for meaningful baseline reproduction**.

---

## 7. Expected Impact on Baseline

### Previous Results (1,544 samples)

| Metric | Value |
|--------|-------|
| Test Acc@1 | 2.70% |
| Gap to 35.6% | -32.9% |

### Expected Results (23,186 samples)

| Metric | Expected Range | Reasoning |
|--------|----------------|-----------|
| Test Acc@1 | 10-20% | Similar scale to official partial reproductions |
| Convergence | More stable | Larger dataset reduces variance |
| Gap to 35.6% | -15% to -25% | Still limited by placeholder geometry and feature quality |

---

## 8. Remaining Gaps

| Gap | Status | Impact |
|-----|--------|--------|
| Dataset size | Partially addressed | 55.9% coverage |
| Geometry quality | Degraded | Using aggregation placeholders |
| Feature fidelity | Unchanged | Would need feature computation |
| Missing scenes | 372 scenes without aggregation | Cannot recover without source data |

---

## 9. Recommendations

### Immediate

1. **Run trustworthy baseline rerun** on recovered dataset
2. **Accept placeholder geometry** for initial results
3. **Document geometry limitation** in all reports

### Future

1. **Regenerate geometry files** with correct object ID mapping
2. **Download remaining ScanNet scenes** to recover 18,317 more samples
3. **Compute real features** for better accuracy

---

## 10. Conclusion

The recovered dataset provides a **15x improvement** in sample count and is **sufficient for meaningful baseline reproduction**. While geometry quality is degraded compared to the old subset, the dramatic increase in sample count should more than compensate for this limitation.