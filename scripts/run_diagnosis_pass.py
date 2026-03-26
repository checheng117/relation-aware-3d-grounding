#!/usr/bin/env python3
"""Build diagnosis manifests, train the 8-cell matrix, eval, then summarize (non-destructive paths).

Does not overwrite ``outputs/metrics/main_results.json`` or ``outputs/checkpoints_nr3d_geom_first/``.
See ``configs/train/diagnosis/`` and ``outputs/metrics/diagnosis/``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-summarize", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    diag_m = ROOT / "outputs/metrics/diagnosis"
    diag_m.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/checkpoints_diagnosis").mkdir(parents=True, exist_ok=True)

    if not args.skip_prepare:
        _run(
            [
                py,
                "scripts/prepare_data.py",
                "--mode",
                "build-nr3d-geom",
                "--config",
                "configs/dataset/diagnosis_entity_geom.yaml",
            ]
        )
        _run(
            [
                py,
                "scripts/prepare_data.py",
                "--mode",
                "build-nr3d-geom",
                "--config",
                "configs/dataset/diagnosis_full_geom.yaml",
            ]
        )

    trains: list[tuple[str, list[str]]] = [
        ("baseline", [py, "scripts/train_baseline.py", "--config", "configs/train/diagnosis/diag_entity_baseline.yaml"]),
        ("baseline", [py, "scripts/train_baseline.py", "--config", "configs/train/diagnosis/diag_entity_raw_relation.yaml"]),
        ("main", [py, "scripts/train_main.py", "--config", "configs/train/diagnosis/diag_entity_rel_heuristic.yaml"]),
        ("main", [py, "scripts/train_main.py", "--config", "configs/train/diagnosis/diag_entity_rel_structured.yaml"]),
        ("baseline", [py, "scripts/train_baseline.py", "--config", "configs/train/diagnosis/diag_full_baseline.yaml"]),
        ("baseline", [py, "scripts/train_baseline.py", "--config", "configs/train/diagnosis/diag_full_raw_relation.yaml"]),
        ("main", [py, "scripts/train_main.py", "--config", "configs/train/diagnosis/diag_full_rel_heuristic.yaml"]),
        ("main", [py, "scripts/train_main.py", "--config", "configs/train/diagnosis/diag_full_rel_structured.yaml"]),
    ]

    if not args.skip_train:
        for _, cmd in trains:
            _run(cmd)

    evals = [
        "configs/eval/diagnosis/entity_baseline.yaml",
        "configs/eval/diagnosis/entity_raw_relation.yaml",
        "configs/eval/diagnosis/entity_rel_heuristic.yaml",
        "configs/eval/diagnosis/entity_rel_structured.yaml",
        "configs/eval/diagnosis/full_baseline.yaml",
        "configs/eval/diagnosis/full_raw_relation.yaml",
        "configs/eval/diagnosis/full_rel_heuristic.yaml",
        "configs/eval/diagnosis/full_rel_structured.yaml",
    ]
    if not args.skip_eval:
        for ec in evals:
            _run([py, "scripts/eval_all.py", "--config", ec])

    if not args.skip_summarize:
        _run([py, "scripts/summarize_diagnosis_pass.py"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
