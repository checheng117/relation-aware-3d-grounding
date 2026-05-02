#!/usr/bin/env python3
"""Report figures: metric bar chart + CSV tables + report_ready/ stable copies."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.utils.logging import setup_logging
from rag3d.visualization.plot_metrics import plot_bar_metrics
from rag3d.visualization.report_tables import write_main_results_csv, write_stratified_csv

import logging

log = logging.getLogger(__name__)

REPORT_READY_README = """report_ready/
  01_main_results.csv       — Acc@1 / Acc@5 / n per model
  02_stratified_results.csv — wide stratified metrics
  03_acc_at_1_bar.png       — bar chart (Acc@1)
  04_failure_summary.json   — optional; from analyze_hard_cases

Regenerate: python scripts/make_figures.py
Copy into docs/assets/figures/ if you want figures version-controlled.
"""


def _sync_report_ready(fig_dir: Path) -> None:
    rd = fig_dir / "report_ready"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "00_README.txt").write_text(REPORT_READY_README, encoding="utf-8")
    src_main = fig_dir / "main_results_table.csv"
    src_strat = fig_dir / "stratified_results_table.csv"
    src_bar = fig_dir / "metrics_bar.png"
    if src_main.is_file():
        shutil.copy2(src_main, rd / "01_main_results.csv")
    if src_strat.is_file():
        shutil.copy2(src_strat, rd / "02_stratified_results.csv")
    if src_bar.is_file():
        shutil.copy2(src_bar, rd / "03_acc_at_1_bar.png")
    fail = fig_dir / "failure_summary.json"
    if fail.is_file():
        shutil.copy2(fail, rd / "04_failure_summary.json")


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-json", type=Path, default=ROOT / "outputs/metrics/main_results.json")
    ap.add_argument("--strat-json", type=Path, default=ROOT / "outputs/metrics/stratified_results.json")
    ap.add_argument("--failure-json", type=Path, default=ROOT / "outputs/figures/failure_summary.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/figures")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    fig_dir = args.out_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    (fig_dir / "qualitative_cases").mkdir(parents=True, exist_ok=True)
    (fig_dir / "anchor_panels").mkdir(parents=True, exist_ok=True)

    if args.demo or not args.main_json.is_file():
        plot_bar_metrics(
            ["acc@1_attr", "acc@1_raw", "acc@1_rel"],
            [0.35, 0.38, 0.42],
            fig_dir / "metrics_bar.png",
            title="Demo (placeholder — run eval_all then make_figures)",
        )
        write_main_results_csv(
            {"attribute_only": {"acc@1": 0.35, "acc@5": 0.5, "n": 0}},
            fig_dir / "main_results_table.csv",
        )
        write_stratified_csv(
            {"attribute_only": {"acc@1_subset::same_class_clutter": 0.2}},
            fig_dir / "stratified_results_table.csv",
        )
        _sync_report_ready(fig_dir)
        log.info("Wrote demo artifacts under %s and %s", fig_dir, fig_dir / "report_ready")
        return

    main = json.loads(args.main_json.read_text(encoding="utf-8"))
    strat = json.loads(args.strat_json.read_text(encoding="utf-8")) if args.strat_json.is_file() else {}
    write_main_results_csv(main, fig_dir / "main_results_table.csv")
    write_stratified_csv(strat, fig_dir / "stratified_results_table.csv")

    names: list[str] = []
    vals: list[float] = []
    for model_name, m in main.items():
        names.append(f"{model_name}:acc@1")
        vals.append(float(m.get("acc@1", 0.0)))
    plot_bar_metrics(names, vals, fig_dir / "metrics_bar.png", title="Acc@1 by model")

    analyze_fail = fig_dir / "failure_summary.json"
    if analyze_fail.is_file() and analyze_fail.resolve() != args.failure_json.resolve():
        args.failure_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(analyze_fail, args.failure_json)
        log.info("Copied failure summary to %s", args.failure_json)

    _sync_report_ready(fig_dir)
    metrics_summary = ROOT / "outputs/metrics/RESULTS_SUMMARY.md"
    if metrics_summary.is_file():
        shutil.copy2(metrics_summary, fig_dir / "RESULTS_SUMMARY.md")
    log.info(
        "Wrote %s, %s, %s, report_ready/",
        fig_dir / "main_results_table.csv",
        fig_dir / "stratified_results_table.csv",
        fig_dir / "metrics_bar.png",
    )


if __name__ == "__main__":
    main()
