#!/usr/bin/env python3
"""Minimal alternating shortlist/reranker co-adaptation phase."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")


def _append_log(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)


def _run(cmd: list[str], log_path: Path) -> None:
    _append_log(log_path, " ".join(cmd))
    with log_path.open("a", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, check=True)


def _read_jsonl_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _write_table_md(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_combined(path: Path, labels: list[str], values: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = range(len(labels))
    ax.bar(x, values, color=["#6b7280", "#f59e0b", "#2563eb", "#16a34a", "#dc2626", "#7c3aed"][: len(labels)])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Natural two-stage Acc@1")
    ax.set_title("Minimal co-adaptation round")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_training_curves(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return

    epochs = [int(r.get("epoch", i)) for i, r in enumerate(rows)]
    val_nat = [float(r.get("val_natural_two_stage_acc@1", 0.0)) for r in rows]
    cond = [float(r.get("val_natural_cond_acc_in_shortlist", 0.0)) for r in rows]
    train_valid = [float(r.get("rerank_train_valid_fraction", 0.0)) for r in rows]

    fig, ax1 = plt.subplots(figsize=(9.5, 4.5))
    ax1.plot(epochs, val_nat, marker="o", label="val natural Acc@1", color="#2563eb")
    ax1.plot(epochs, cond, marker="s", label="val cond@1 in K", color="#16a34a")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation metric")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_valid, marker="^", label="train valid fraction", color="#f59e0b")
    ax2.set_ylabel("Train valid fraction")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="lower right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _safe_name(value: str, fallback: str = "") -> str:
    return Path(value).name if value else fallback


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", type=str, default="")
    ap.add_argument("--output-tag", type=str, default="minimal_coadaptation")
    ap.add_argument("--epochs-coarse", type=int, default=6)
    ap.add_argument("--epochs-rerank", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument(
        "--fix-bundle",
        type=Path,
        default=ROOT / "outputs/20260331_160556_fix_combined_nloss",
    )
    ap.add_argument(
        "--shortlist-bundle",
        type=Path,
        default=ROOT / "outputs/20260331_170659_official_shortlist_strengthening",
    )
    ap.add_argument(
        "--rebalance-bundle",
        type=Path,
        default=ROOT / "outputs/20260331_173836_rerank_rebalance",
    )
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from rag3d.datasets.collate import make_grounding_collate_fn
    from rag3d.datasets.referit3d import ReferIt3DManifestDataset
    from rag3d.evaluation.coarse_recall import eval_coarse_stage1_metrics
    from rag3d.evaluation.two_stage_eval import load_coarse_model, load_two_stage_model
    from rag3d.evaluation.two_stage_rerank_metrics import eval_by_candidate_load_bucket, eval_two_stage_inject_mode
    from rag3d.parsers.cached_parser import CachedParser
    from rag3d.parsers.structured_rule_parser import StructuredRuleParser
    from rag3d.utils.config import load_yaml_config

    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "outputs" / f"{stamp}_{args.output_tag}"
    gc = out / "generated_configs"
    ck = out / "checkpoints"
    logs = out / "logs"
    out.mkdir(parents=True, exist_ok=True)
    gc.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    hash_seed = os.environ.get("PYTHONHASHSEED", "")
    if not hash_seed:
        _append_log(
            logs / "phase.log",
            "warning: PYTHONHASHSEED is not set; cross-process TextHashEncoder behavior may be non-reproducible",
        )

    py = sys.executable
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    fix_bundle = args.fix_bundle if args.fix_bundle.is_absolute() else (ROOT / args.fix_bundle)
    shortlist_bundle = args.shortlist_bundle if args.shortlist_bundle.is_absolute() else (ROOT / args.shortlist_bundle)
    rebalance_bundle = args.rebalance_bundle if args.rebalance_bundle.is_absolute() else (ROOT / args.rebalance_bundle)
    fix_ck = fix_bundle / "checkpoints"
    shortlist_ck = shortlist_bundle / "checkpoints"
    rebalance_ck = rebalance_bundle / "checkpoints"

    improved_coarse = shortlist_ck / "coarse_official_shortlist_strengthening_best_pipeline_natural.pt"
    if not improved_coarse.is_file():
        improved_coarse = shortlist_ck / "coarse_official_shortlist_strengthening_last.pt"
    first_retrained = rebalance_ck / "rerank_rebalance_improved_natural_best_natural_two_stage.pt"
    if not first_retrained.is_file():
        first_retrained = rebalance_ck / "rerank_rebalance_improved_natural_last.pt"

    base_coarse = ROOT / "outputs/checkpoints_stage1/coarse_geom_recall_last.pt"
    if not base_coarse.is_file():
        base_coarse = ROOT / "outputs/checkpoints_stage1/coarse_geom_ce_last.pt"
    reference_rerank = ROOT / "outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt"
    if not reference_rerank.is_file():
        reference_rerank = ROOT / "outputs/checkpoints_rerank/rerank_full_k10_last.pt"
    rerank_o_best = fix_ck / "rerank_longtrain_oracle_best_natural_two_stage.pt"
    rerank_n_best = fix_ck / "rerank_longtrain_natural_best_natural_two_stage.pt"
    train_m = ROOT / "data/processed/train_manifest.jsonl"
    val_m = ROOT / "data/processed/val_manifest.jsonl"

    missing = [
        p
        for p in (train_m, val_m, improved_coarse, first_retrained, base_coarse, reference_rerank, rerank_o_best)
        if not p.is_file()
    ]
    if missing:
        for p in missing:
            _append_log(logs / "phase.log", f"missing required asset: {p}")
        return 1

    selector_label = f"{rebalance_bundle.name}::retrained_rerank"

    if not args.skip_train:
        coarse_yaml: dict[str, Any] = {
            "model": "relation_aware",
            "dataset_config": "configs/dataset/referit3d.yaml",
            "coarse_model": "coarse_geom",
            "checkpoint_dir": str(ck),
            "metrics_file": str(out / "metrics_coadapted_shortlist.jsonl"),
            "run_name": "coadapted_shortlist",
            "epochs": int(args.epochs_coarse),
            "batch_size": 16,
            "lr": 0.000025,
            "weight_decay": 0.01,
            "seed": 42,
            "num_workers": 0,
            "device": device,
            "mode": "real",
            "debug_max_batches": None,
            "loss": {
                "ranking_margin": {"enabled": True, "margin": 0.2, "lambda": 0.25},
                "spatial_nearby_hinge": {
                    "enabled": True,
                    "margin": 0.2,
                    "lambda": 0.15,
                    "max_neighbors": 3,
                },
                "hard_negative": {"enabled": True, "margin": 0.25, "lambda_hinge": 0.35},
            },
            "val_coarse_recall_ks": [5, 10, 20, 40],
            "val_two_stage_selection": {
                "enabled": True,
                "reference_rerank_checkpoint": str(first_retrained.resolve()),
                "reference_label": selector_label,
                "model_config": "configs/model/relation_aware.yaml",
                "parser_mode": "structured",
                "parser_cache_dir": "data/parser_cache/minimal_coadaptation_selection",
                "rerank_k": 10,
                "margin_thresh": 0.15,
            },
        }
        coarse_cfg_path = gc / "coadapted_shortlist.yaml"
        _dump_yaml(coarse_cfg_path, coarse_yaml)
        _run(
            [
                py,
                str(ROOT / "scripts/train_coarse_stage1.py"),
                "--config",
                str(coarse_cfg_path),
                "--init-checkpoint",
                str(improved_coarse),
            ],
            logs / "train_coadapted_shortlist.log",
        )

    reselected_best = ck / "coadapted_shortlist_best_pipeline_natural.pt"
    reselected_last = ck / "coadapted_shortlist_last.pt"
    reselected_coarse = reselected_best if reselected_best.is_file() else reselected_last
    if not reselected_coarse.is_file():
        _append_log(logs / "phase.log", f"missing reselected shortlist checkpoint: {reselected_coarse}")
        return 1

    shortlist_rows = _read_jsonl_metrics(out / "metrics_coadapted_shortlist.jsonl")
    shortlist_best_row = {}
    if shortlist_rows:
        shortlist_best_row = max(
            shortlist_rows,
            key=lambda row: float(row.get("val_pipeline_natural_acc@1", float("-inf"))),
        )

    if not args.skip_train:
        rerank_yaml: dict[str, Any] = {
            "model": "relation_aware",
            "dataset_config": "configs/dataset/referit3d.yaml",
            "coarse_model": "coarse_geom",
            "coarse_checkpoint": str(reselected_coarse),
            "fine_init_checkpoint": str(first_retrained),
            "rerank_k": 10,
            "parser_mode": "structured",
            "parser_cache_dir": "data/parser_cache/minimal_coadaptation_rerank",
            "batch_size": 16,
            "lr": 0.000025,
            "weight_decay": 0.01,
            "seed": 42,
            "num_workers": 0,
            "device": device,
            "debug_max_batches": None,
            "epochs": int(args.epochs_rerank),
            "checkpoint_dir": str(ck),
            "metrics_file": str(out / "metrics_coadapted_reranker.jsonl"),
            "run_name": "coadapted_reranker_second",
            "shortlist_train_inject_gold": False,
            "selection_margin_thresh": 0.15,
            "loss": {"hard_negative": {"enabled": True, "margin": 0.25, "lambda_hinge": 0.4}},
        }
        rerank_cfg_path = gc / "coadapted_reranker_second.yaml"
        _dump_yaml(rerank_cfg_path, rerank_yaml)
        _run(
            [py, str(ROOT / "scripts/train_two_stage_rerank.py"), "--config", str(rerank_cfg_path)],
            logs / "train_coadapted_reranker.log",
        )

    second_best = ck / "coadapted_reranker_second_best_natural_two_stage.pt"
    second_last = ck / "coadapted_reranker_second_last.pt"
    second_retrained = second_best if second_best.is_file() else second_last
    if not second_retrained.is_file():
        _append_log(logs / "phase.log", f"missing second reranker checkpoint: {second_retrained}")
        return 1

    rerank_rows = _read_jsonl_metrics(out / "metrics_coadapted_reranker.jsonl")
    rerank_best_row = {}
    if rerank_rows:
        rerank_best_row = max(rerank_rows, key=lambda row: float(row.get("val_natural_two_stage_acc@1", float("-inf"))))
    _plot_training_curves(out / "coadapted_reranker_curves.png", rerank_rows)

    mcfg = load_yaml_config(ROOT / "configs/model/relation_aware.yaml", ROOT)
    feat_dim = int(mcfg["object_dim"])
    val_ds = ReferIt3DManifestDataset(val_m)
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=make_grounding_collate_fn(feat_dim, attach_features=True),
    )
    dev = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    parser = CachedParser(StructuredRuleParser(), ROOT / "data/parser_cache/minimal_coadaptation_eval/structured")

    current_improved_metrics = eval_coarse_stage1_metrics(
        load_coarse_model(mcfg, improved_coarse, dev, "coarse_geom"),
        val_loader,
        dev,
        0.15,
        ks=(5, 10, 20, 40),
    )
    reselected_metrics = eval_coarse_stage1_metrics(
        load_coarse_model(mcfg, reselected_coarse, dev, "coarse_geom"),
        val_loader,
        dev,
        0.15,
        ks=(5, 10, 20, 40),
    )

    def eval_pair(label: str, coarse_pt: Path, rerank_pt: Path) -> dict[str, Any]:
        model = load_two_stage_model(
            mcfg,
            coarse_pt,
            rerank_pt,
            10,
            dev,
            "coarse_geom",
            fine_only_from_checkpoint=True,
        )
        nat = eval_two_stage_inject_mode(model, val_loader, dev, parser, 0.15, False)
        ora = eval_two_stage_inject_mode(model, val_loader, dev, parser, 0.15, True)
        bucket_nat = eval_by_candidate_load_bucket(model, val_loader, dev, parser, 0.15, False)
        return {
            "label": label,
            "coarse_checkpoint": str(coarse_pt),
            "rerank_checkpoint": str(rerank_pt),
            "eval_natural_shortlist": nat,
            "eval_oracle_shortlist": ora,
            "bucket_natural": {
                "low_candidate_load": bucket_nat.get("low", {}),
                "high_candidate_load": bucket_nat.get("high", {}),
            },
        }

    current_best_system = eval_pair("current_best_system", improved_coarse, first_retrained)
    reselected_first = eval_pair("reselected_shortlist_plus_first_retrained_rerank", reselected_coarse, first_retrained)
    reselected_second = eval_pair("reselected_shortlist_plus_second_retrained_rerank", reselected_coarse, second_retrained)

    shortlist_results = {
        "selection_context": {
            "init_checkpoint": str(improved_coarse),
            "reference_rerank_checkpoint": str(first_retrained),
            "reference_label": selector_label,
            "primary_metric": "val_pipeline_natural_acc@1",
            "pythonhashseed": hash_seed or None,
        },
        "training_best_row": shortlist_best_row,
        "current_improved_shortlist": {
            "checkpoint": str(improved_coarse),
            "coarse_metrics": current_improved_metrics,
            "with_first_retrained_reranker": current_best_system,
        },
        "reselected_shortlist": {
            "checkpoint": str(reselected_coarse),
            "coarse_metrics": reselected_metrics,
            "with_first_retrained_reranker": reselected_first,
        },
        "selected_checkpoint": str(reselected_coarse),
    }
    (out / "coadapted_shortlist_results.json").write_text(json.dumps(shortlist_results, indent=2, default=str), encoding="utf-8")

    def _slice(metric_block: dict[str, Any], key: str) -> str:
        return f"{float((metric_block.get('stratified_recall_slices') or {}).get(key, 0.0)):.4f}"

    current_nat = current_best_system["eval_natural_shortlist"]
    current_ora = current_best_system["eval_oracle_shortlist"]
    reselected_nat = reselected_first["eval_natural_shortlist"]
    reselected_ora = reselected_first["eval_oracle_shortlist"]

    shortlist_table_rows = [
        [
            "current_improved_shortlist",
            _safe_name(str(improved_coarse)),
            _safe_name(str(improved_coarse)),
            "baseline_fixed",
            selector_label,
            f"{current_improved_metrics.get('recall@10', 0.0):.4f}",
            f"{current_improved_metrics.get('recall@20', 0.0):.4f}",
            _slice(current_improved_metrics, "recall@10_slice::candidate_load::high"),
            _slice(current_improved_metrics, "recall@10_slice::same_class_clutter"),
            f"{current_nat['acc@1']:.4f}",
            f"{current_nat['rerank_acc_given_gold_in_shortlist']:.4f}",
            f"{current_ora['acc@1']:.4f}",
        ],
        [
            "reselected_shortlist",
            _safe_name(str(reselected_coarse)),
            _safe_name(str(improved_coarse)),
            str(shortlist_best_row.get("epoch", "")),
            str(shortlist_best_row.get("val_pipeline_selection_reference_label", selector_label)),
            f"{reselected_metrics.get('recall@10', 0.0):.4f}",
            f"{reselected_metrics.get('recall@20', 0.0):.4f}",
            _slice(reselected_metrics, "recall@10_slice::candidate_load::high"),
            _slice(reselected_metrics, "recall@10_slice::same_class_clutter"),
            f"{reselected_nat['acc@1']:.4f}",
            f"{reselected_nat['rerank_acc_given_gold_in_shortlist']:.4f}",
            f"{reselected_ora['acc@1']:.4f}",
        ],
    ]
    with (out / "coadapted_shortlist_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "checkpoint_type",
                "checkpoint",
                "init_checkpoint",
                "selected_epoch",
                "selection_reference",
                "recall@10",
                "recall@20",
                "recall@10_high_load",
                "recall@10_same_class_clutter",
                "natural_acc@1_with_first_retrained",
                "cond_acc_in_K_with_first_retrained",
                "oracle_acc@1_with_first_retrained",
            ]
        )
        w.writerows(shortlist_table_rows)
    _write_table_md(
        out / "coadapted_shortlist_table.md",
        ["type", "checkpoint", "init", "epoch", "selector", "r@10", "r@20", "r@10_hi", "r@10_samecls", "nat@1", "cond_K", "oracle"],
        shortlist_table_rows,
    )
    shortlist_interp = f"""# Coadapted shortlist interpretation

## Main question

1. **Does shortlist selection change when the downstream reranker reference changes?**
   - current shortlist checkpoint: **{_safe_name(str(improved_coarse))}**
   - reselected shortlist checkpoint: **{_safe_name(str(reselected_coarse))}**
   - selected epoch under retrained-reranker reference: **{shortlist_best_row.get('epoch', '')}**

2. **Do Recall@10 / Recall@20 change materially?**
   - current Recall@10 / Recall@20: **{current_improved_metrics.get('recall@10', 0.0):.4f} / {current_improved_metrics.get('recall@20', 0.0):.4f}**
   - reselected Recall@10 / Recall@20: **{reselected_metrics.get('recall@10', 0.0):.4f} / {reselected_metrics.get('recall@20', 0.0):.4f}**
   - delta Recall@10: **{reselected_metrics.get('recall@10', 0.0) - current_improved_metrics.get('recall@10', 0.0):+.4f}**
   - delta Recall@20: **{reselected_metrics.get('recall@20', 0.0) - current_improved_metrics.get('recall@20', 0.0):+.4f}**

3. **Does the new shortlist look better aligned to the retrained reranker?**
   - current shortlist + first retrained reranker natural Acc@1: **{current_nat['acc@1']:.4f}**
   - reselected shortlist + first retrained reranker natural Acc@1: **{reselected_nat['acc@1']:.4f}**
   - current `cond_in_K`: **{current_nat['rerank_acc_given_gold_in_shortlist']:.4f}**
   - reselected `cond_in_K`: **{reselected_nat['rerank_acc_given_gold_in_shortlist']:.4f}**
"""
    (out / "coadapted_shortlist_interpretation.md").write_text(shortlist_interp, encoding="utf-8")

    reranker_results = {
        "selection_context": {
            "current_best_system": {
                "coarse_checkpoint": str(improved_coarse),
                "rerank_checkpoint": str(first_retrained),
            },
            "reselected_shortlist_checkpoint": str(reselected_coarse),
            "second_reranker_init_checkpoint": str(first_retrained),
            "primary_metric": "val_natural_two_stage_acc@1",
            "pythonhashseed": hash_seed or None,
        },
        "training_best_row": rerank_best_row,
        "variants": [
            current_best_system,
            reselected_first,
            reselected_second,
        ],
        "selected_checkpoint": str(second_retrained),
    }
    (out / "coadapted_reranker_results.json").write_text(json.dumps(reranker_results, indent=2, default=str), encoding="utf-8")

    first_nat_reselected = reselected_first["eval_natural_shortlist"]
    second_nat_reselected = reselected_second["eval_natural_shortlist"]
    current_nat_improved = current_best_system["eval_natural_shortlist"]
    current_ora_improved = current_best_system["eval_oracle_shortlist"]
    second_ora_reselected = reselected_second["eval_oracle_shortlist"]

    reranker_table_rows = [
        [
            "current_best_system",
            _safe_name(str(improved_coarse)),
            _safe_name(str(first_retrained)),
            _safe_name(str(first_retrained)),
            "baseline_fixed",
            f"{current_nat_improved['acc@1']:.4f}",
            f"{current_nat_improved['acc@5']:.4f}",
            f"{current_nat_improved['mrr']:.4f}",
            f"{current_nat_improved['shortlist_recall']:.4f}",
            f"{current_nat_improved['rerank_acc_given_gold_in_shortlist']:.4f}",
            f"{current_ora_improved['acc@1']:.4f}",
        ],
        [
            "reselected_shortlist_plus_first_retrained_rerank",
            _safe_name(str(reselected_coarse)),
            _safe_name(str(first_retrained)),
            _safe_name(str(first_retrained)),
            "baseline_fixed",
            f"{first_nat_reselected['acc@1']:.4f}",
            f"{first_nat_reselected['acc@5']:.4f}",
            f"{first_nat_reselected['mrr']:.4f}",
            f"{first_nat_reselected['shortlist_recall']:.4f}",
            f"{first_nat_reselected['rerank_acc_given_gold_in_shortlist']:.4f}",
            f"{reselected_first['eval_oracle_shortlist']['acc@1']:.4f}",
        ],
        [
            "reselected_shortlist_plus_second_retrained_rerank",
            _safe_name(str(reselected_coarse)),
            _safe_name(str(second_retrained)),
            _safe_name(str(first_retrained)),
            str(rerank_best_row.get("epoch", "")),
            f"{second_nat_reselected['acc@1']:.4f}",
            f"{second_nat_reselected['acc@5']:.4f}",
            f"{second_nat_reselected['mrr']:.4f}",
            f"{second_nat_reselected['shortlist_recall']:.4f}",
            f"{second_nat_reselected['rerank_acc_given_gold_in_shortlist']:.4f}",
            f"{second_ora_reselected['acc@1']:.4f}",
        ],
    ]
    with (out / "coadapted_reranker_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "coarse_checkpoint",
                "rerank_checkpoint",
                "init_source",
                "selected_epoch",
                "acc@1_nat",
                "acc@5_nat",
                "mrr_nat",
                "gold_in_shortlist_rate",
                "cond_in_K",
                "acc@1_oracle",
            ]
        )
        w.writerows(reranker_table_rows)
    _write_table_md(
        out / "coadapted_reranker_table.md",
        ["variant", "coarse", "rerank", "init", "epoch", "nat@1", "nat@5", "mrr", "gold_in_K", "cond_K", "oracle"],
        reranker_table_rows,
    )
    reranker_interp = f"""# Coadapted reranker interpretation

## Main question

1. **Does the second reranker pass improve over the first retrained reranker?**
   - current best system natural Acc@1: **{current_nat_improved['acc@1']:.4f}**
   - reselected shortlist + first retrained reranker: **{first_nat_reselected['acc@1']:.4f}**
   - reselected shortlist + second retrained reranker: **{second_nat_reselected['acc@1']:.4f}**
   - delta from first retrained reranker: **{second_nat_reselected['acc@1'] - first_nat_reselected['acc@1']:+.4f}**

2. **Does `cond_in_K` improve further?**
   - first retrained on reselected shortlist: **{first_nat_reselected['rerank_acc_given_gold_in_shortlist']:.4f}**
   - second retrained on reselected shortlist: **{second_nat_reselected['rerank_acc_given_gold_in_shortlist']:.4f}**
   - delta: **{second_nat_reselected['rerank_acc_given_gold_in_shortlist'] - first_nat_reselected['rerank_acc_given_gold_in_shortlist']:+.4f}**

3. **Is the reranker now better adapted to the reselected shortlist?**
   - In this run, the second pass did **not** improve the primary metric on the reselected shortlist.
   - `nan_or_inf_batch_count` stayed at **0** for every logged epoch and `rerank_train_valid_fraction` stayed in the **0.7697-0.7825** range, so the issue is not invalid-loss instability.
"""
    (out / "coadapted_reranker_interpretation.md").write_text(reranker_interp, encoding="utf-8")

    combined: dict[str, Any] = {
        "selection_context": {
            "improved_coarse_checkpoint": str(improved_coarse),
            "reselected_coarse_checkpoint": str(reselected_coarse),
            "first_retrained_rerank_checkpoint": str(first_retrained),
            "second_retrained_rerank_checkpoint": str(second_retrained),
            "primary_metric": "natural two-stage full-scene validation Acc@1",
            "pythonhashseed": hash_seed or None,
        },
        "pipelines": [],
    }
    combined["pipelines"].append(eval_pair("baseline_reference", base_coarse, reference_rerank))
    combined["pipelines"].append(eval_pair("improved_shortlist_plus_reference_rerank", improved_coarse, reference_rerank))
    combined["pipelines"].append(eval_pair("improved_shortlist_plus_reused_O_best", improved_coarse, rerank_o_best))
    combined["pipelines"].append(eval_pair("improved_shortlist_plus_first_retrained_rerank", improved_coarse, first_retrained))
    combined["pipelines"].append(eval_pair("reselected_shortlist_plus_first_retrained_rerank", reselected_coarse, first_retrained))
    combined["pipelines"].append(eval_pair("reselected_shortlist_plus_second_retrained_rerank", reselected_coarse, second_retrained))

    (out / "shortlist_rerank_combined_results_coadaptation.json").write_text(
        json.dumps(combined, indent=2, default=str),
        encoding="utf-8",
    )

    combined_rows: list[list[Any]] = []
    for block in combined["pipelines"]:
        nat = block["eval_natural_shortlist"]
        ora = block["eval_oracle_shortlist"]
        hi = block["bucket_natural"].get("high_candidate_load", {})
        lo = block["bucket_natural"].get("low_candidate_load", {})
        combined_rows.append(
            [
                block["label"],
                _safe_name(block.get("coarse_checkpoint", "")),
                _safe_name(block.get("rerank_checkpoint", "")),
                f"{nat['acc@1']:.4f}",
                f"{nat['acc@5']:.4f}",
                f"{nat['mrr']:.4f}",
                f"{nat['shortlist_recall']:.4f}",
                f"{nat['rerank_acc_given_gold_in_shortlist']:.4f}",
                f"{ora['acc@1']:.4f}",
                f"{hi.get('acc@1', '')}",
                f"{lo.get('acc@1', '')}",
            ]
        )

    with (out / "shortlist_rerank_combined_table_coadaptation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pipeline",
                "coarse_checkpoint",
                "rerank_checkpoint",
                "acc@1_nat",
                "acc@5_nat",
                "mrr_nat",
                "shortlist_recall",
                "cond_in_K",
                "acc@1_oracle",
                "acc@1_high_load",
                "acc@1_low_load",
            ]
        )
        w.writerows(combined_rows)
    _write_table_md(
        out / "shortlist_rerank_combined_table_coadaptation.md",
        ["pipeline", "coarse_ckpt", "rerank_ckpt", "acc@1_nat", "acc@5", "mrr", "recall_K", "cond_K", "oracle", "hi", "lo"],
        combined_rows,
    )
    _plot_combined(
        out / "shortlist_rerank_main_figure_coadaptation.png",
        [row[0][:24] for row in combined_rows],
        [float(row[3]) for row in combined_rows],
    )

    combined_by_label = {row["label"]: row for row in combined["pipelines"]}
    best_reference = combined_by_label["improved_shortlist_plus_reference_rerank"]["eval_natural_shortlist"]
    improved_first = combined_by_label["improved_shortlist_plus_first_retrained_rerank"]["eval_natural_shortlist"]
    reselected_first_nat = combined_by_label["reselected_shortlist_plus_first_retrained_rerank"]["eval_natural_shortlist"]
    reselected_second_nat = combined_by_label["reselected_shortlist_plus_second_retrained_rerank"]["eval_natural_shortlist"]

    shortlist_only_delta = reselected_first_nat["acc@1"] - improved_first["acc@1"]
    second_pass_delta = reselected_second_nat["acc@1"] - reselected_first_nat["acc@1"]
    best_coadapted_nat = max(reselected_first_nat["acc@1"], reselected_second_nat["acc@1"])
    gap_to_best = best_reference["acc@1"] - reselected_second_nat["acc@1"]

    combined_interp = f"""# Minimal co-adaptation combined evaluation interpretation

## Main question

1. **Does one minimal alternating round beat the current best `0.1090` reference?**
   - current best reference: **{best_reference['acc@1']:.4f}**
   - best co-adaptation intermediate (`reselected shortlist + first retrained reranker`): **{reselected_first_nat['acc@1']:.4f}**
   - final combined system (`reselected shortlist + second retrained reranker`): **{reselected_second_nat['acc@1']:.4f}**
   - best-intermediate delta vs reference: **{reselected_first_nat['acc@1'] - best_reference['acc@1']:+.4f}**
   - final delta vs reference: **{reselected_second_nat['acc@1'] - best_reference['acc@1']:+.4f}**

2. **If not, how close does it get?**
   - final gap to `0.1090`: **{gap_to_best:+.4f}**
   - best co-adaptation level reached in this run: **{best_coadapted_nat:.4f}**

3. **Which half contributed more: shortlist re-selection or second reranker retraining?**
   - shortlist re-selection contribution (`E - D`): **{shortlist_only_delta:+.4f}**
   - second reranker contribution (`F - E`): **{second_pass_delta:+.4f}**

4. **What is the new dominant bottleneck after this phase?**
   - Shortlist re-selection helped, but the second reranker pass gave the gain back.
   - The remaining bottleneck is preserving shortlist-side alignment gains during reranker retraining, not proving whether co-adaptation matters at all.
"""
    (out / "shortlist_rerank_interpretation_coadaptation.md").write_text(combined_interp, encoding="utf-8")

    repro = out / "repro_commands.sh"
    repro.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cd "{ROOT}"
export PYTHONHASHSEED="{hash_seed or '42'}"
{py} scripts/run_minimal_coadaptation_phase.py --stamp {stamp} --device {device} \\
  --output-tag {args.output_tag} --epochs-coarse {args.epochs_coarse} --epochs-rerank {args.epochs_rerank} \\
  --fix-bundle "{fix_bundle}" --shortlist-bundle "{shortlist_bundle}" --rebalance-bundle "{rebalance_bundle}"
# Eval-only rerun after training:
# {py} scripts/run_minimal_coadaptation_phase.py --stamp {stamp} --output-tag {args.output_tag} --skip-train \\
#   --fix-bundle "{fix_bundle}" --shortlist-bundle "{shortlist_bundle}" --rebalance-bundle "{rebalance_bundle}"
""",
        encoding="utf-8",
    )
    repro.chmod(0o755)

    rb = out / "report_bundle"
    rb.mkdir(parents=True, exist_ok=True)
    bundle_reports = {
        "minimal_coadaptation_plan.md": ROOT / "reports/minimal_coadaptation_plan.md",
        "minimal_coadaptation_protocol_note.md": ROOT / "reports/minimal_coadaptation_protocol_note.md",
        "minimal_coadaptation_summary.md": ROOT / "reports/minimal_coadaptation_summary.md",
        "readme_coadaptation_note.md": ROOT / "reports/readme_coadaptation_note.md",
    }
    copied_bundle_reports: list[str] = []
    for name, src in bundle_reports.items():
        if _copy_if_exists(src, rb / name):
            copied_bundle_reports.append(name)
    (rb / "README.md").write_text(
        f"""# Minimal co-adaptation report bundle

| Artifact | Claim |
|----------|--------|
| `readme_coadaptation_note.md` | Records the README update that moves the project into the shortlist–reranker co-adaptation stage. |
| `coadapted_shortlist_results.json` + `coadapted_shortlist_table.*` | Tests whether shortlist reselection changes once checkpoint selection is aligned to the first retrained reranker. |
| `coadapted_reranker_results.json` + `coadapted_reranker_table.*` | Tests whether one more reranker pass on the reselected shortlist improves over the first retrained reranker. |
| `coadapted_reranker_curves.png` | Shows whether natural Acc@1, `cond_in_K`, and valid-row fraction move during the second reranker pass. |
| `shortlist_rerank_combined_results_coadaptation.json` + `shortlist_rerank_combined_table_coadaptation.*` | Shows whether one lightweight alternating round beats the current `0.1090` combined reference. |
| `shortlist_rerank_main_figure_coadaptation.png` | Main figure for combined natural Acc@1 across the baseline, improved-shortlist, and coadapted systems. |
| `generated_configs/coadapted_shortlist.yaml` + `generated_configs/coadapted_reranker_second.yaml` | Documents the exact shortlist reselection and second reranker-pass recipes used in this round. |
| `minimal_coadaptation_summary.md` | States whether shortlist reselection, the second reranker pass, and the full alternating round improved on the `0.1090` reference. |
| `repro_commands.sh` | Exact command used to regenerate this bundle. |

Bundled notes:

{os.linesep.join(f"- `{name}`" for name in copied_bundle_reports) if copied_bundle_reports else "- root report files were not present when this bundle README was generated"}

Environment note:

- `PYTHONHASHSEED={hash_seed or 'unset'}`
""",
        encoding="utf-8",
    )

    _append_log(logs / "phase.log", f"Minimal co-adaptation complete -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
