#!/usr/bin/env python3
"""Orchestrate shortlist-aligned audit (K=10 coarse metrics) + promote + rerank eval."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["audit", "promote", "all"],
        default="all",
    )
    ap.add_argument("--skip-train", action="store_true", help="Promote phase: only eval (rerank ckpts exist).")
    ap.add_argument("--no-force-spatial", action="store_true")
    args = ap.parse_args()

    if args.phase in ("audit", "all"):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/eval_coarse_k10_shortlist_audit.py")],
            cwd=str(ROOT),
            check=True,
        )

    if args.phase in ("promote", "all"):
        cmd = [sys.executable, str(ROOT / "scripts/promote_shortlist_aligned_rerank.py")]
        if args.skip_train:
            cmd.append("--skip-train")
        if args.no_force_spatial:
            cmd.append("--no-force-spatial")
        subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
