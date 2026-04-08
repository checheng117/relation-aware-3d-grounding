# Nr3D Source Inventory

**Date**: 2026-04-07
**Phase**: Full Nr3D Dataset Recovery - Step 1

---

## Executive Summary

**Recovery potential**: 23,186 samples (55.9% of official 41,503) with current geometry availability. This is a **15x increase** from the current 1,544 samples.

---

## 1. Source Files

### A. Annotation Sources

| File | Location | Count | Format |
|------|----------|-------|--------|
| `nr3d_official.csv` | `data/raw/referit3d/annotations/` | **41,503** | Official CSV |
| `nr3d_annotations.json` | `data/raw/referit3d/annotations/` | 1,569 | HF subset JSON |
| `train.csv` | `data/raw/referit3d/annotations/` | 1,255 | HF subset CSV |
| `val.csv` | `data/raw/referit3d/annotations/` | 156 | HF subset CSV |
| `test.csv` | `data/raw/referit3d/annotations/` | 158 | HF subset CSV |

### B. Geometry Sources

| Directory | Location | Count | Format |
|-----------|----------|-------|--------|
| Geometry files | `data/geometry/` | 269 scenes | `.npz` |
| Aggregation files | `data/raw/referit3d/scans/` | 269 scenes | `.aggregation.json` |
| Object features | `data/object_features/` | 269 scenes | `.npz` |

---

## 2. Official Nr3D Format

### Column Schema

| Column | Type | Description |
|--------|------|-------------|
| `assignmentid` | int | Unique sample ID |
| `stimulus_id` | str | Scene + object IDs combined |
| `utterance` | str | Natural language reference |
| `correct_guess` | bool | Whether listener guessed correctly |
| `speaker_id` | int | Speaker identifier |
| `listener_id` | int | Listener identifier |
| `scan_id` | str | ScanNet scene ID (e.g., `scene0525_00`) |
| `instance_type` | str | Object class name |
| `target_id` | str | Target object ID within scene |
| `tokens` | str | Tokenized utterance (JSON list) |
| `dataset` | str | Dataset marker (always `nr3d`) |
| `mentions_target_class` | bool | Language mentions class |
| `uses_object_lang` | bool | Uses object language |
| `uses_spatial_lang` | bool | Uses spatial language |
| `uses_color_lang` | bool | Uses color language |
| `uses_shape_lang` | bool | Uses shape language |

### Key Differences from HF Format

| Aspect | Official | HF Subset |
|--------|----------|-----------|
| Scene ID field | `scan_id` | `scene_id` |
| Target ID field | `target_id` | `object_id` |
| Utterance field | `utterance` | `descriptions` (list) |
| Sample ID | `assignmentid` | `unique_id` |
| Entity list | Not present | `entities` list |

---

## 3. Scene Coverage Analysis

### Official Nr3D

| Metric | Count |
|--------|-------|
| Total samples | 41,503 |
| Unique scenes | 641 |
| Avg samples/scene | 64.7 |

### Current Geometry Availability

| Metric | Count | % of Official |
|--------|-------|---------------|
| Scenes with geometry | 269 | 42.0% |
| Scenes with aggregation | 269 | 42.0% |
| **Samples with geometry** | **23,186** | **55.9%** |

### Scene Overlap Detail

| Category | Scenes | Samples |
|----------|--------|---------|
| Official scenes with geometry | 269 | 23,186 |
| Official scenes without geometry | 372 | 18,317 |
| Geometry scenes not in official | 0 | 0 |

---

## 4. Sample Loss Breakdown

### Current HF Subset Pipeline

| Stage | Input | Output | Lost | Reason |
|-------|-------|--------|------|--------|
| HF JSON rows | 1,569 | - | - | Source |
| parse_nr3d_row_meta() | 1,569 | 1,569 | 0 | All parse |
| load_scene_objects() | 1,569 | 1,544 | 25 | Target not in objects |
| Scene-disjoint split | 1,544 | 1,544 | 0 | All kept |
| **Final** | - | **1,544** | - | - |

### Official Pipeline (Projected)

| Stage | Input | Output | Lost | Reason |
|-------|-------|--------|------|--------|
| Official CSV rows | 41,503 | - | - | Source |
| Scenes with geometry | 41,503 | 23,186 | 18,317 | No geometry file |
| parse_official_row() | 23,186 | ~23,000 | ~186 | Target not in objects (est 0.8%) |
| Scene-disjoint split | ~23,000 | ~23,000 | 0 | All kept |
| **Projected Final** | - | **~23,000** | - | - |

### Loss Categories

| Loss Reason | Estimated Count | Recoverable? |
|-------------|-----------------|--------------|
| No geometry file | 18,317 | **Needs geometry download** |
| Target not in scene objects | ~186 | No (annotation-geometry mismatch) |
| Duplicate samples | 0 | N/A |

---

## 5. Geometry Gap Analysis

### Missing Scenes

- **Total missing**: 372 scenes
- **Missing samples**: 18,317

### Options to Recover Missing Geometry

| Option | Effort | Feasibility |
|--------|--------|-------------|
| Download full ScanNet geometry | High | Requires ScanNet license |
| Use placeholder geometry | Low | Already implemented |
| Focus on available 23K samples | None | Current approach |

---

## 6. Builder Script Inventory

### Current Builders

| Script | Function | Format |
|--------|----------|--------|
| `scripts/prepare_data.py` | `cmd_build_nr3d_geom_scene_disjoint()` | HF JSON |
| `src/rag3d/datasets/builder.py` | `build_records_nr3d_hf_with_scans()` | HF JSON |
| `src/rag3d/datasets/nr3d_hf.py` | `parse_nr3d_row_meta()` | HF JSON |

### Required New Builders

| Function | Purpose | Status |
|----------|---------|--------|
| `parse_nr3d_official_row()` | Parse official CSV format | **Needed** |
| `build_records_nr3d_official()` | Build from official CSV | **Needed** |
| `cmd_build_nr3d_official_scene_disjoint()` | CLI command | **Needed** |

---

## 7. File Sizes

| File | Size |
|------|------|
| `nr3d_official.csv` | 10.2 MB |
| `nr3d_annotations.json` | 913 KB |
| Geometry files (269 scenes) | ~2.5 GB total |
| Aggregation files (269 scenes) | ~150 MB total |

---

## 8. Recommended Builder Changes

### Option A: Create New Official Parser (Recommended)

```python
def parse_nr3d_official_row(row: dict) -> dict | None:
    """Parse official Nr3D CSV format."""
    return {
        'scene_id': row['scan_id'],
        'utterance': row['utterance'],
        'target_object_id': row['target_id'],
        'utterance_id': row['assignmentid'],
        'instance_type': row['instance_type'],
        # Additional metadata
        'uses_spatial_lang': row.get('uses_spatial_lang') == 'True',
        'uses_color_lang': row.get('uses_color_lang') == 'True',
    }
```

### Option B: Convert Official to HF Format

Pre-process official CSV to match HF JSON format, then use existing pipeline.

---

## 9. Next Steps

1. **Create official Nr3D parser** in `src/rag3d/datasets/nr3d_official.py`
2. **Add builder function** in `src/rag3d/datasets/builder.py`
3. **Add CLI command** in `scripts/prepare_data.py`
4. **Build new manifests** with 23K samples
5. **Validate scene-disjoint splits**