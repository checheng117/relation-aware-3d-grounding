#!/usr/bin/env python3
"""Train K=10 rerank heads on top-N coarse checkpoints from coarse_optimization_latest.json; eval + transfer CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.utils.logging import setup_logging

log = logging.getLogger(__name__)


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in s).strip("_")[:80]


def _sort_key(r: dict[str, Any]) -> tuple[float, float]:
    return (float(r.get("recall@20", 0.0)), float(r.get("recall@10", 0.0)))


def _rel_repo(path_str: str) -> str:
    p = Path(path_str).resolve()
    r = ROOT.resolve()
    try:
        return str(p.relative_to(r))
    except ValueError:
        return path_str


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse-json", type=Path, default=ROOT / "outputs/metrics/coarse_optimization_latest.json")
    ap.add_argument("--top-n", type=int, default=2)
    ap.add_argument("--skip-train", action="store_true", help="Only write eval yaml and run eval (checkpoints exist).")
    args = ap.parse_args()

    data = json.loads(args.coarse_json.read_text(encoding="utf-8"))
    exps: list[dict[str, Any]] = list(data.get("experiments") or [])
    exps.sort(key=_sort_key, reverse=True)
    picked = exps[: max(1, args.top_n)]

    train_dir = ROOT / "configs/train/rerank/optimization"
    train_dir.mkdir(parents=True, exist_ok=True)
    ckpt_rerank = ROOT / "outputs/checkpoints_stage1_rerank_opt"
    ckpt_rerank.mkdir(parents=True, exist_ok=True)

    fine_ckpts: list[tuple[str, str, str, str]] = []
    for i, row in enumerate(picked):
        name = str(row["name"])
        ck = row["checkpoint"]
        kind = str(row.get("coarse_model", "coarse_geom"))
        san = _sanitize(name)
        run_name = f"rerank_k10_opt_{san}"
        train_yaml = train_dir / f"generated_promote_{i}_{san}.yaml"
        tcfg = {
            "model": "relation_aware",
            "coarse_model": kind,
            "dataset_config": "configs/dataset/diagnosis_full_geom.yaml",
            "coarse_checkpoint": _rel_repo(str(ck)),
            "rerank_k": 10,
            "parser_mode": "structured",
            "parser_cache_dir": "data/parser_cache/diagnosis",
            "epochs": 8,
            "batch_size": 16,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "seed": 42,
            "num_workers": 2,
            "checkpoint_dir": "outputs/checkpoints_stage1_rerank_opt",
            "metrics_file": f"outputs/metrics/rerank_opt_promote_{i}.jsonl",
            "run_name": run_name,
            "device": "cuda",
            "debug_max_batches": None,
            "loss": {"hard_negative": {"enabled": False}},
        }
        train_yaml.write_text(yaml.safe_dump(tcfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        log.info("Wrote %s", train_yaml)
        if not args.skip_train:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/train_two_stage_rerank.py"), "--config", str(train_yaml)],
                cwd=str(ROOT),
                check=True,
            )
        fine_last = ckpt_rerank / f"{run_name}_last.pt"
        if not fine_last.is_file():
            log.error("Missing fine checkpoint %s", fine_last)
            sys.exit(1)
        rel_coarse = _rel_repo(str(ck))
        fine_ckpts.append((name, kind, rel_coarse, str(fine_last.resolve().relative_to(ROOT.resolve()))))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    eval_experiments: list[dict[str, Any]] = []
    for i, (name, kind, rel_coarse, rel_fine) in enumerate(fine_ckpts):
        san = _sanitize(name)
        eval_experiments.append(
            {
                "name": f"opt_promote_{i}_{san}_coarse_topk10",
                "type": "coarse_topk_attribute",
                "coarse_checkpoint": rel_coarse,
                "coarse_model": kind,
                "rerank_k": 10,
            }
        )
        eval_experiments.append(
            {
                "name": f"opt_promote_{i}_{san}_two_stage_k10",
                "type": "two_stage_rerank",
                "coarse_checkpoint": rel_coarse,
                "coarse_model": kind,
                "fine_checkpoint": rel_fine,
                "rerank_k": 10,
            }
        )

    eval_cfg = {
        "dataset_config": "configs/dataset/diagnosis_full_geom.yaml",
        "model_config": "configs/model/relation_aware.yaml",
        "split": "val",
        "batch_size": 16,
        "device": "cuda",
        "margin_thresh": 0.15,
        "parser_mode": "structured",
        "parser_cache_dir": "data/parser_cache/diagnosis",
        "coarse_checkpoint": "outputs/checkpoints_stage1_opt/coarse_geom_ce_only_last.pt",
        "coarse_model": "coarse_geom",
        "output_json": f"outputs/metrics/rerank_optimization_{stamp}.json",
        "reference_metrics": {},
        "experiments": eval_experiments,
    }
    eval_path = ROOT / "configs/eval/rerank_optimization_generated.yaml"
    eval_path.write_text(yaml.safe_dump(eval_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/eval_rerank_blueprint.py"), "--config", str(eval_path)],
        cwd=str(ROOT),
        check=True,
    )

    rjson = ROOT / "outputs/metrics" / f"rerank_optimization_{stamp}.json"
    latest_r = ROOT / "outputs/metrics/rerank_optimization_latest.json"
    if rjson.is_file():
        latest_r.write_text(rjson.read_text(encoding="utf-8"), encoding="utf-8")

    rdata = json.loads(rjson.read_text(encoding="utf-8"))
    rows_r = rdata.get("rows") or {}

    fig_dir = ROOT / "outputs/figures"
    report_dir = ROOT / "outputs/figures/report_ready_blueprint_coarse_opt"
    report_dir.mkdir(parents=True, exist_ok=True)

    ro = fig_dir / "rerank_optimization_table.csv"
    with ro.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "experiment",
                "acc@1",
                "acc@5",
                "coarse_target_in_topk_rate",
                "topk_recall_success_rate",
                "rerank_rescue_rate",
            ]
        )
        for ename, r in sorted(rows_r.items()):
            w.writerow(
                [
                    ename,
                    r.get("acc@1", ""),
                    r.get("acc@5", ""),
                    r.get("coarse_target_in_topk_rate", ""),
                    r.get("topk_recall_success_rate", ""),
                    r.get("rerank_rescue_rate", ""),
                ]
            )

    rt = fig_dir / "rerank_transfer_table.csv"
    with rt.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "coarse_recipe",
                "recall@20_coarse_sweep",
                "coarse_only_acc@1_topk10",
                "two_stage_acc@1",
                "delta_acc@1",
                "coarse_target_in_topk_rate",
                "rerank_rescue_rate",
            ]
        )
        for i, row in enumerate(picked):
            name = str(row["name"])
            san = _sanitize(name)
            ct = rows_r.get(f"opt_promote_{i}_{san}_coarse_topk10") or {}
            ts = rows_r.get(f"opt_promote_{i}_{san}_two_stage_k10") or {}
            r20 = float(row.get("recall@20", 0.0))
            a_coarse = float(ct.get("acc@1", 0.0) or 0.0)
            a_ts = float(ts.get("acc@1", 0.0) or 0.0)
            w.writerow(
                [
                    name,
                    r20,
                    a_coarse,
                    a_ts,
                    a_ts - a_coarse,
                    ts.get("coarse_target_in_topk_rate", ""),
                    ts.get("rerank_rescue_rate", ""),
                ]
            )

    for p in (ro, rt):
        (report_dir / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

    log.info("Wrote %s, %s, %s", rjson, ro, rt)


if __name__ == "__main__":
    main()
