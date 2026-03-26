#!/usr/bin/env python3
"""Assemble ``outputs/report_bundle_blueprint/`` (non-destructive copies + README)."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/report_bundle_blueprint"
READY = ROOT / "outputs/figures/report_ready_blueprint"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    READY.mkdir(parents=True, exist_ok=True)
    copies = [
        (ROOT / "outputs/figures/diagnosis_matrix.csv", OUT / "diagnosis_matrix.csv"),
        (ROOT / "outputs/figures/diagnosis_parser_comparison.csv", OUT / "diagnosis_parser_comparison.csv"),
        (ROOT / "outputs/metrics/main_results_nr3d_geom_first.json", OUT / "main_results_nr3d_geom_first.json"),
        (ROOT / "outputs/metrics/archive_nr3d_warmup_main_results.json", OUT / "archive_nr3d_warmup_main_results.json"),
        (ROOT / "outputs/metrics/paraphrase_consistency_blueprint.json", OUT / "paraphrase_consistency_blueprint.json"),
    ]
    for src, dst in copies:
        if src.is_file():
            shutil.copy2(src, dst)
            shutil.copy2(src, READY / dst.name)
    readme = OUT / "README.md"
    readme.write_text(
        """# Report bundle (metrics snapshot)

Stable copies of selected CSV/JSON artifacts. **Regenerate** with:

```bash
python scripts/build_report_bundle_blueprint.py
python scripts/eval_paraphrase_consistency.py \\
  --manifest data/processed/diagnosis_entity_geom/val_manifest.jsonl \\
  --checkpoint outputs/checkpoints_diagnosis/diag_entity_rel_heuristic_last.pt \\
  --model relation_aware \\
  --dataset-config configs/dataset/diagnosis_entity_geom.yaml
```

| File | Meaning |
|------|---------|
| `diagnosis_matrix.csv` | Candidate-regime × model × parser diagnosis (Acc@1/5, n). |
| `diagnosis_parser_comparison.csv` | Heuristic vs structured parser (relation_aware only). |
| `main_results_nr3d_geom_first.json` | Full-scene geometry-backed val metrics (reference). |
| `archive_nr3d_warmup_main_results.json` | Warmup-stage val metrics (small candidate lists). |
| `paraphrase_consistency_blueprint.json` | Template paraphrase robustness (agreement, anchor drift). |

See also `outputs/figures/report_ready_diagnosis/` and the root README.
""",
        encoding="utf-8",
    )
    shutil.copy2(readme, READY / "README.md")
    print("Wrote", OUT, READY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
