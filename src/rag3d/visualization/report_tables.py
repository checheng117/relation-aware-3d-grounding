"""CSV / JSON exports for report tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_main_results_csv(main: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "acc@1", "acc@5", "n"])
        for model, m in main.items():
            w.writerow([model, m.get("acc@1", ""), m.get("acc@5", ""), m.get("n", "")])


def write_stratified_csv(stratified: dict[str, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: set[str] = set()
    for m in stratified.values():
        keys.update(m.keys())
    ordered = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", *ordered])
        for model, m in stratified.items():
            w.writerow([model, *[m.get(k, "") for k in ordered]])


def write_failure_summary_json(
    failure_hist: dict[str, int],
    path: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"failure_tag_counts": failure_hist}
    if extra:
        out.update(extra)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
