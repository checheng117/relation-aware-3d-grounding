#!/usr/bin/env python3
"""Recompute shortlist-aligned ranks and the promoted recipe set (no training, no GPU)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.evaluation.shortlist_promote import assign_ranks, old_promote_key

SPATIAL = "coarse_geom_ce_spatial"


def _pick(rows: list[dict[str, Any]], top_n: int, force_spatial: bool) -> list[str]:
    by_score = sorted(rows, key=lambda r: float(r["promote_score_shortlist"]), reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for r in by_score[: max(1, top_n)]:
        n = str(r["name"])
        if n not in seen:
            out.append(n)
            seen.add(n)
    if force_spatial and SPATIAL not in seen:
        if any(str(r["name"]) == SPATIAL for r in rows):
            out.append(SPATIAL)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-json", type=Path, default=ROOT / "outputs/metrics/shortlist_alignment_audit_latest.json")
    ap.add_argument("--top-n", type=int, default=2)
    ap.add_argument("--no-force-spatial", action="store_true")
    ap.add_argument("--json-out", action="store_true", help="Print one JSON object to stdout.")
    args = ap.parse_args()
    doc = json.loads(args.audit_json.read_text(encoding="utf-8"))
    rows = list(doc.get("rows") or [])
    promoted = _pick(rows, args.top_n, force_spatial=not args.no_force_spatial)
    old_r = assign_ranks(rows, key_fn=old_promote_key)
    new_r = assign_ranks(rows, key_fn=lambda r: float(r["promote_score_shortlist"]))
    payload = {
        "audit_stamp": doc.get("stamp"),
        "promoted_recipes": promoted,
        "ranks_old_promote": old_r,
        "ranks_shortlist_aligned": new_r,
        "shortlist_aligned_score": {str(r["name"]): r.get("promote_score_shortlist") for r in rows},
    }
    if args.json_out:
        print(json.dumps(payload, indent=2))
    else:
        for name in promoted:
            print(name)


if __name__ == "__main__":
    main()
