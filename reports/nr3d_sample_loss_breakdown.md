# Nr3D Sample Loss Breakdown

**Date**: 2026-04-07
**Phase**: Full Nr3D Dataset Recovery - Step 2

---

## Executive Summary

**Main bottleneck**: Missing geometry files for 372 scenes, blocking 18,317 samples.

**Recovery potential**: 23,186 samples (55.9% of official 41,503) with current geometry.

---

## 1. End-to-End Sample Flow

```
Official Nr3D CSV (41,503 samples)
         │
         ├──► Scenes with geometry (269 scenes)
         │           │
         │           └──► 23,186 samples ──► BUILDABLE
         │
         └──► Scenes without geometry (372 scenes)
                     │
                     └──► 18,317 samples ──► BLOCKED
```

---

## 2. Sample Loss by Category

| Category | Count | Percentage | Recoverable? |
|----------|-------|------------|--------------|
| **Missing geometry** | 18,317 | 44.1% | Needs ScanNet download |
| Available with geometry | 23,186 | 55.9% | **Buildable now** |
| Target not in scene objects | ~186 | ~0.4% | Annotation mismatch |

---

## 3. Scene-Level Breakdown

### Scenes WITH Geometry (269)

| Metric | Value |
|--------|-------|
| Scenes | 269 |
| Samples | 23,186 |
| Samples/scene (min) | 1 |
| Samples/scene (max) | 226 |
| Samples/scene (avg) | 86.2 |

### Scenes WITHOUT Geometry (372)

| Metric | Value |
|--------|-------|
| Scenes | 372 |
| Samples | 18,317 |
| Samples/scene (min) | 1 |
| Samples/scene (max) | 196 |
| Samples/scene (avg) | 49.2 |

---

## 4. Loss Stage Analysis

### Stage 1: Source Data

| Input | Count |
|-------|-------|
| Official Nr3D CSV rows | 41,503 |
| Unique scenes | 641 |

**Loss at this stage**: 0 (all rows valid)

### Stage 2: Geometry Filter

| Input | Output | Lost | Reason |
|-------|--------|------|--------|
| 41,503 rows | 23,186 rows | 18,317 | No geometry file for scene |

**Loss at this stage**: 44.1%

### Stage 3: Target-Object Matching

| Input | Output | Lost | Reason |
|-------|--------|------|--------|
| 23,186 rows | ~23,000 rows | ~186 | Target ID not in scene objects |

**Estimated loss at this stage**: ~0.8%

### Stage 4: Scene-Disjoint Split

| Input | Output | Lost | Reason |
|-------|--------|------|--------|
| ~23,000 rows | ~23,000 rows | 0 | All samples kept |

**Loss at this stage**: 0%

---

## 5. Projected Recovery

### With Current Geometry (269 scenes)

| Split | Estimated Samples |
|-------|-------------------|
| Train (80%) | ~18,400 |
| Val (10%) | ~2,300 |
| Test (10%) | ~2,300 |
| **Total** | **~23,000** |

### Comparison to Current State

| Metric | Current | Projected | Increase |
|--------|---------|-----------|----------|
| Total samples | 1,544 | ~23,000 | **14.9x** |
| Scenes | 266 | 269 | +3 |
| Train samples | 1,211 | ~18,400 | **15.2x** |
| Val samples | 148 | ~2,300 | **15.5x** |
| Test samples | 185 | ~2,300 | **12.4x** |

---

## 6. Unrecoverable Samples

### Reason: Missing Geometry

| Aspect | Value |
|--------|-------|
| Scenes affected | 372 |
| Samples affected | 18,317 |
| % of official | 44.1% |

### To Recover These

**Option 1**: Download ScanNet geometry for all 641 scenes
- Requires ScanNet license acceptance
- Large download (~1.5TB raw, ~50GB processed)
- Would enable full 41,503 samples

**Option 2**: Use placeholder geometry
- Would lose spatial information
- Not recommended for 3D grounding

**Option 3**: Accept current coverage
- 23,186 samples is sufficient for meaningful baseline
- Focus on other improvements (features, model)

---

## 7. Conclusion

**Immediate recovery possible**: 23,186 samples (15x current)

**Blocking issue**: Missing ScanNet geometry for 372 scenes

**Recommendation**: Build with available 23K samples now, pursue full geometry later