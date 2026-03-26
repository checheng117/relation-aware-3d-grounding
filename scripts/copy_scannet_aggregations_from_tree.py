#!/usr/bin/env python3
"""Copy ScanNet-style ``*_vh_clean_aggregation.json`` from an official extracted tree into this repo.

Use this when you have agreed to ScanNet ToS and unpacked the official ``scans/`` directory
locally (HF mirrors remain gated for many accounts).

Example::

  python scripts/copy_scannet_aggregations_from_tree.py \\
    --source-scans /path/to/ScanNet/scans \\
    --scene-ids scene0525_00,scene0002_00

Does not read or print secrets.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.datasets.scannet_objects import resolve_aggregation_path


def _canonical_dest(scans_out: Path, scene_id: str) -> Path:
    return scans_out / scene_id / f"{scene_id}_vh_clean_aggregation.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Copy aggregation JSON from local ScanNet tree.")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--source-scans", type=Path, required=True, help="Official ScanNet scans/ root.")
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw/referit3d"))
    ap.add_argument("--scans-subdir", type=str, default="scans")
    ap.add_argument("--scene-ids", type=str, required=True, help="Comma-separated scene_id list.")
    ap.add_argument("--nr3d-json", type=Path, default=None, help="If set, union scene list with unique ids from this JSON.")
    args = ap.parse_args()

    base = args.root.resolve()
    src_root = args.source_scans.resolve()
    raw = (args.raw_root if args.raw_root.is_absolute() else base / args.raw_root).resolve()
    out = raw / args.scans_subdir

    scenes = {s.strip() for s in args.scene_ids.split(",") if s.strip()}
    if args.nr3d_json:
        jp = args.nr3d_json if args.nr3d_json.is_absolute() else base / args.nr3d_json
        data = json.loads(jp.read_text(encoding="utf-8"))
        for row in data:
            if isinstance(row, dict):
                sid = str(row.get("scene_id") or "").strip()
                if sid:
                    scenes.add(sid)

    ok = 0
    missing: list[str] = []
    for sid in sorted(scenes):
        scene_dir = src_root / sid
        found = resolve_aggregation_path(scene_dir, sid)
        if found is None:
            missing.append(sid)
            continue
        dest = _canonical_dest(out, sid)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found, dest)
        ok += 1

    print(f"Copied {ok} aggregation files into {out}")
    if missing:
        print(f"Missing source aggregation for {len(missing)} scenes (first 15): {', '.join(missing[:15])}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
