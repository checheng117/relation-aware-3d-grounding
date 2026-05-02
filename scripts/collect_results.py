#!/usr/bin/env python3
"""Aggregate metrics JSON + optional train logs into SUMMARY.md and summary_metrics.csv."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.utils.logging import setup_logging

import logging

log = logging.getLogger(__name__)


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _md_table_main(main: dict[str, dict]) -> str:
    lines = ["| Model | Acc@1 | Acc@5 | n |", "|-------|-------|-------|---|"]
    for k, v in main.items():
        lines.append(f"| {k} | {v.get('acc@1', '')} | {v.get('acc@5', '')} | {v.get('n', '')} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", type=Path, default=ROOT / "outputs/metrics")
    ap.add_argument("--out-md", type=Path, default=ROOT / "outputs/metrics/RESULTS_SUMMARY.md")
    ap.add_argument("--out-csv", type=Path, default=ROOT / "outputs/metrics/summary_metrics.csv")
    args = ap.parse_args()
    mdir = args.metrics_dir
    main = _read_json(mdir / "main_results.json") or {}
    strat = _read_json(mdir / "stratified_results.json") or {}
    debug = _read_json(mdir / "debug_results.json")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "# Results summary (auto-generated)\n",
        "Regenerate: `python scripts/collect_results.py`\n\n",
        "## Main metrics\n\n",
        _md_table_main(main) if main else "_No main_results.json found._\n\n",
        "## Stratified keys present\n\n",
    ]
    if strat:
        for model, sub in strat.items():
            keys = ", ".join(sorted(sub.keys()))
            parts.append(f"- **{model}:** {keys}\n")
        parts.append("\n")
    else:
        parts.append("_No stratified_results.json found._\n\n")

    if debug:
        parts.append("## Debug bundle\n\n`debug_results.json` is present (eval with debug subdir).\n\n")

    args.out_md.write_text("".join(parts), encoding="utf-8")

    # Flat CSV: one row per model per metric key
    rows: list[dict[str, str]] = []
    for model, v in main.items():
        rows.append({"model": model, "metric": "acc@1", "value": str(v.get("acc@1", ""))})
        rows.append({"model": model, "metric": "acc@5", "value": str(v.get("acc@5", ""))})
        rows.append({"model": model, "metric": "n", "value": str(v.get("n", ""))})
    for model, sub in strat.items():
        for k, val in sub.items():
            rows.append({"model": model, "metric": k, "value": str(val)})

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "metric", "value"])
        w.writeheader()
        w.writerows(rows)

    log.info("Wrote %s and %s", args.out_md, args.out_csv)


if __name__ == "__main__":
    main()
