#!/usr/bin/env python3
"""Orchestrate coarse optimization sweep: train → eval → rerank promotion → rerank eval."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COARSE_CONFIGS = [
    "configs/train/coarse/optimization/coarse_attr_baseline_fullref.yaml",
    "configs/train/coarse/optimization/coarse_geom_ce_only.yaml",
    "configs/train/coarse/optimization/coarse_geom_ce_load.yaml",
    "configs/train/coarse/optimization/coarse_geom_ce_sameclass.yaml",
    "configs/train/coarse/optimization/coarse_geom_ce_hardneg.yaml",
    "configs/train/coarse/optimization/coarse_geom_ce_spatial.yaml",
    "configs/train/coarse/optimization/coarse_geom_combined_light.yaml",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["train_coarse", "eval_coarse", "promote", "all"],
        default="all",
    )
    ap.add_argument("--skip-promote", action="store_true", help="Stop after coarse eval (no rerank train).")
    args = ap.parse_args()

    if args.phase in ("train_coarse", "all"):
        for cfg in COARSE_CONFIGS:
            p = ROOT / cfg
            if not p.is_file():
                raise FileNotFoundError(p)
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/train_coarse_stage1.py"), "--config", str(p)],
                cwd=str(ROOT),
                check=True,
            )

    if args.phase in ("eval_coarse", "all"):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/eval_coarse_optimization_sweep.py")],
            cwd=str(ROOT),
            check=True,
        )

    if args.phase == "promote" or (args.phase == "all" and not args.skip_promote):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/promote_coarse_opt_rerank.py")],
            cwd=str(ROOT),
            check=True,
        )


if __name__ == "__main__":
    main()
