#!/usr/bin/env python3
"""Write train.csv / val.csv / test.csv from existing manifest JSONL (for ``prepare_data --mode build``).

Columns: scene_id, utterance, target_object_id, utterance_id — compatible with
``build_records_from_csv_and_scans`` once ScanNet aggregations exist under raw scans/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene_id", "utterance", "target_object_id", "utterance_id"])
        w.writeheader()
        w.writerows(rows)


def _rows_from_manifest(mpath: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in mpath.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        uid = r.get("utterance_id") or ""
        rows.append(
            {
                "scene_id": str(r["scene_id"]),
                "utterance": str(r["utterance"]),
                "target_object_id": str(r["target_object_id"]),
                "utterance_id": str(uid),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--annotations-dir", type=Path, default=Path("data/raw/referit3d/annotations"))
    args = ap.parse_args()
    base = args.root.resolve()
    proc = (args.processed_dir if args.processed_dir.is_absolute() else base / args.processed_dir).resolve()
    ann = (args.annotations_dir if args.annotations_dir.is_absolute() else base / args.annotations_dir).resolve()

    for split in ("train", "val", "test"):
        mp = proc / f"{split}_manifest.jsonl"
        if not mp.is_file():
            print(f"Missing {mp}", file=sys.stderr)
            return 1
        _write_csv(ann / f"{split}.csv", _rows_from_manifest(mp))

    print(f"Wrote {ann}/train.csv, val.csv, test.csv from manifests under {proc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
