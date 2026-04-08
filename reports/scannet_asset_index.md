# ScanNet Asset Index Report

## Summary

| Metric | Count |
|--------|-------|
| Total NR3D scenes | 269 |
| Can extract geometry | 269 (100.0%) |
| Has aggregation JSON | 269 |
| Has coord.npy | 269 |
| Has instance.npy | 269 |
| Has color.npy | 269 |
| Missing all geometry | 0 |

## Geometry Sources

- **Pointcept tar**: Contains `coord.npy`, `instance.npy`, `color.npy`, `normal.npy` per scene
- **Local aggregation**: `269` scenes have `*.aggregation.json`
- **Real geometry extraction**: Possible for `269` scenes

## Missing Scenes

None

## Next Steps

1. Extract per-object geometry from Pointcept tar using `instance.npy` labels
2. Compute center/size from real point bboxes
3. Wire into `scannet_objects.py` geometry pipeline
