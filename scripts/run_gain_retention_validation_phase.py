#!/usr/bin/env python3
"""Narrow gain-retention validation around low_lr_secondpass."""

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


def _safe_name(value: str, fallback: str = "") -> str:
    return Path(value).name if value else fallback


def _write_table_md(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_lr_sweep(path: Path, lrs: list[float], vals: list[float], center_lr: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.plot(lrs, vals, marker="o", color="#2563eb")
    ax.axvline(center_lr, color="#dc2626", linestyle="--", linewidth=1.2, label="validated low_lr")
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel("Natural two-stage Acc@1")
    ax.set_title("Local low_lr sweep")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_combined(path: Path, labels: list[str], vals: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    x = range(len(labels))
    ax.bar(x, vals, color=["#f59e0b", "#2563eb", "#16a34a", "#0891b2", "#7c3aed", "#dc2626"][: len(labels)])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("Natural two-stage Acc@1")
    ax.set_title("Gain-retention validation combined comparison")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", type=str, default="")
    ap.add_argument("--output-tag", type=str, default="gain_retention_validation")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--extra-seed", type=int, default=43)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--base-lr", type=float, default=5.0e-06)
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
    ap.add_argument(
        "--coadapt-bundle",
        type=Path,
        default=ROOT / "outputs/20260331_180150_minimal_coadaptation",
    )
    ap.add_argument(
        "--conservative-bundle",
        type=Path,
        default=ROOT / "outputs/20260331_183805_conservative_secondpass",
    )
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from rag3d.datasets.collate import make_grounding_collate_fn
    from rag3d.datasets.referit3d import ReferIt3DManifestDataset
    from rag3d.evaluation.two_stage_eval import load_two_stage_model
    from rag3d.evaluation.two_stage_rerank_metrics import eval_two_stage_inject_mode
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

    py = sys.executable
    hash_seed = os.environ.get("PYTHONHASHSEED", "")
    dev_name = args.device
    if dev_name == "cuda" and not torch.cuda.is_available():
        dev_name = "cpu"

    fix_bundle = args.fix_bundle if args.fix_bundle.is_absolute() else (ROOT / args.fix_bundle)
    shortlist_bundle = args.shortlist_bundle if args.shortlist_bundle.is_absolute() else (ROOT / args.shortlist_bundle)
    rebalance_bundle = args.rebalance_bundle if args.rebalance_bundle.is_absolute() else (ROOT / args.rebalance_bundle)
    coadapt_bundle = args.coadapt_bundle if args.coadapt_bundle.is_absolute() else (ROOT / args.coadapt_bundle)
    conservative_bundle = (
        args.conservative_bundle if args.conservative_bundle.is_absolute() else (ROOT / args.conservative_bundle)
    )
    fix_ck = fix_bundle / "checkpoints"
    shortlist_ck = shortlist_bundle / "checkpoints"
    rebalance_ck = rebalance_bundle / "checkpoints"
    coadapt_ck = coadapt_bundle / "checkpoints"
    conservative_ck = conservative_bundle / "checkpoints"

    base_coarse = ROOT / "outputs/checkpoints_stage1/coarse_geom_recall_last.pt"
    if not base_coarse.is_file():
        base_coarse = ROOT / "outputs/checkpoints_stage1/coarse_geom_ce_last.pt"
    reference_rerank = ROOT / "outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt"
    if not reference_rerank.is_file():
        reference_rerank = ROOT / "outputs/checkpoints_rerank/rerank_full_k10_last.pt"
    improved_coarse = shortlist_ck / "coarse_official_shortlist_strengthening_best_pipeline_natural.pt"
    if not improved_coarse.is_file():
        improved_coarse = shortlist_ck / "coarse_official_shortlist_strengthening_last.pt"
    reselected_coarse = coadapt_ck / "coadapted_shortlist_best_pipeline_natural.pt"
    if not reselected_coarse.is_file():
        reselected_coarse = coadapt_ck / "coadapted_shortlist_last.pt"
    first_retrained = rebalance_ck / "rerank_rebalance_improved_natural_best_natural_two_stage.pt"
    if not first_retrained.is_file():
        first_retrained = rebalance_ck / "rerank_rebalance_improved_natural_last.pt"
    conservative_low_lr = conservative_ck / "low_lr_secondpass_best_natural_two_stage.pt"
    if not conservative_low_lr.is_file():
        conservative_low_lr = conservative_ck / "low_lr_secondpass_last.pt"
    train_m = ROOT / "data/processed/train_manifest.jsonl"
    val_m = ROOT / "data/processed/val_manifest.jsonl"

    missing = [
        p
        for p in (
            base_coarse,
            reference_rerank,
            improved_coarse,
            reselected_coarse,
            first_retrained,
            conservative_low_lr,
            train_m,
            val_m,
        )
        if not p.is_file()
    ]
    if missing:
        for p in missing:
            _append_log(logs / "phase.log", f"missing required asset: {p}")
        return 1

    common = {
        "model": "relation_aware",
        "dataset_config": "configs/dataset/referit3d.yaml",
        "coarse_model": "coarse_geom",
        "coarse_checkpoint": str(reselected_coarse),
        "fine_init_checkpoint": str(first_retrained),
        "rerank_k": 10,
        "parser_mode": "structured",
        "batch_size": 16,
        "weight_decay": 0.01,
        "num_workers": 0,
        "device": dev_name,
        "debug_max_batches": None,
        "shortlist_train_inject_gold": False,
        "selection_margin_thresh": 0.15,
        "min_delta": 0.0,
        "epochs": 8,
        "early_stop_patience": 0,
        "fine_tune_mode": "full",
        "loss": {"hard_negative": {"enabled": True, "margin": 0.25, "lambda_hinge": 0.4}},
    }

    extra_seed_variant = {
        "name": f"low_lr_secondpass_seed{args.extra_seed}",
        "seed": int(args.extra_seed),
        "lr": float(args.base_lr),
    }
    sweep_lrs = [float(args.base_lr * 0.5), float(args.base_lr), float(args.base_lr * 2.0)]
    sweep_variants = [
        {
            "name": f"low_lr_sweep_lr{idx}",
            "seed": int(args.base_seed),
            "lr": lr,
            "lr_factor": factor,
        }
        for idx, (lr, factor) in enumerate(zip(sweep_lrs, [0.5, 1.0, 2.0]))
    ]

    trained: dict[str, dict[str, Any]] = {}

    def _train_variant(spec: dict[str, Any], cache_prefix: str) -> None:
        run_name = str(spec["name"])
        cfg = dict(common)
        cfg.update(
            {
                "seed": int(spec["seed"]),
                "lr": float(spec["lr"]),
                "run_name": run_name,
                "metrics_file": str(out / f"metrics_{run_name}.jsonl"),
                "checkpoint_dir": str(ck),
                "parser_cache_dir": f"data/parser_cache/gain_retention_validation/{cache_prefix}/{run_name}",
            }
        )
        cfg_path = gc / f"{run_name}.yaml"
        _dump_yaml(cfg_path, cfg)
        _run([py, str(ROOT / "scripts/train_two_stage_rerank.py"), "--config", str(cfg_path)], logs / f"train_{run_name}.log")
        best = ck / f"{run_name}_best_natural_two_stage.pt"
        last = ck / f"{run_name}_last.pt"
        chosen = best if best.is_file() else last
        rows = _read_jsonl_metrics(out / f"metrics_{run_name}.jsonl")
        best_row = {}
        if rows:
            best_row = max(rows, key=lambda r: float(r.get("val_natural_two_stage_acc@1", float("-inf"))))
        trained[run_name] = {
            "name": run_name,
            "spec": spec,
            "config": cfg,
            "checkpoint": str(chosen),
            "rows": rows,
            "best_row": best_row,
        }

    if not args.skip_train:
        _train_variant(extra_seed_variant, "extra_seed")
        for spec in sweep_variants:
            _train_variant(spec, "sweep")
    else:
        for spec in [extra_seed_variant] + sweep_variants:
            run_name = str(spec["name"])
            best = ck / f"{run_name}_best_natural_two_stage.pt"
            last = ck / f"{run_name}_last.pt"
            chosen = best if best.is_file() else last
            rows = _read_jsonl_metrics(out / f"metrics_{run_name}.jsonl")
            best_row = {}
            if rows:
                best_row = max(rows, key=lambda r: float(r.get("val_natural_two_stage_acc@1", float("-inf"))))
            trained[run_name] = {
                "name": run_name,
                "spec": spec,
                "checkpoint": str(chosen),
                "rows": rows,
                "best_row": best_row,
            }

    mcfg = load_yaml_config(ROOT / "configs/model/relation_aware.yaml", ROOT)
    feat_dim = int(mcfg["object_dim"])
    val_ds = ReferIt3DManifestDataset(val_m)
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=make_grounding_collate_fn(feat_dim, attach_features=True),
    )
    device = torch.device(dev_name if dev_name == "cuda" and torch.cuda.is_available() else "cpu")
    parser = CachedParser(StructuredRuleParser(), ROOT / "data/parser_cache/gain_retention_validation_eval/structured")

    def eval_pair(label: str, coarse_pt: Path, rerank_pt: Path) -> dict[str, Any]:
        model = load_two_stage_model(
            mcfg,
            coarse_pt,
            rerank_pt,
            10,
            device,
            "coarse_geom",
            fine_only_from_checkpoint=True,
        )
        nat = eval_two_stage_inject_mode(model, val_loader, device, parser, 0.15, False)
        ora = eval_two_stage_inject_mode(model, val_loader, device, parser, 0.15, True)
        return {
            "label": label,
            "coarse_checkpoint": str(coarse_pt),
            "rerank_checkpoint": str(rerank_pt),
            "eval_natural_shortlist": nat,
            "eval_oracle_shortlist": ora,
        }

    inter = eval_pair("coadapted_intermediate_best", reselected_coarse, first_retrained)
    validated_low = eval_pair("validated_low_lr_secondpass", reselected_coarse, conservative_low_lr)

    extra_label = "low_lr_secondpass_extra_seed"
    extra_block = eval_pair(extra_label, reselected_coarse, Path(trained[extra_seed_variant["name"]]["checkpoint"]))

    extra_results = {
        "selection_context": {
            "base_lr": float(args.base_lr),
            "validated_seed": int(args.base_seed),
            "extra_seed": int(args.extra_seed),
            "coadapted_shortlist_checkpoint": str(reselected_coarse),
            "fine_init_checkpoint": str(first_retrained),
            "pythonhashseed": hash_seed or None,
        },
        "reference_rows": [inter, validated_low],
        "extra_seed_variant": extra_block,
        "extra_seed_training_best_row": trained[extra_seed_variant["name"]]["best_row"],
    }
    (out / "low_lr_secondpass_extra_seed_results.json").write_text(json.dumps(extra_results, indent=2, default=str), encoding="utf-8")

    extra_nat = extra_block["eval_natural_shortlist"]
    inter_nat = inter["eval_natural_shortlist"]
    val_nat = validated_low["eval_natural_shortlist"]
    extra_best = trained[extra_seed_variant["name"]]["best_row"] or {}
    extra_table_rows = [
        [
            "coadapted_intermediate_best",
            _safe_name(inter["rerank_checkpoint"]),
            "baseline_fixed",
            "42",
            f"{inter_nat['acc@1']:.4f}",
            f"{inter_nat['rerank_acc_given_gold_in_shortlist']:.4f}",
            f"{inter_nat['acc@5']:.4f}",
            f"{inter_nat['mrr']:.4f}",
            f"{inter_nat['shortlist_recall']:.4f}",
        ],
        [
            "validated_low_lr_secondpass",
            _safe_name(validated_low["rerank_checkpoint"]),
            "baseline_fixed",
            "42",
            f"{val_nat['acc@1']:.4f}",
            f"{val_nat['rerank_acc_given_gold_in_shortlist']:.4f}",
            f"{val_nat['acc@5']:.4f}",
            f"{val_nat['mrr']:.4f}",
            f"{val_nat['shortlist_recall']:.4f}",
        ],
        [
            extra_label,
            _safe_name(extra_block["rerank_checkpoint"]),
            str(extra_best.get("epoch", "")),
            str(args.extra_seed),
            f"{extra_nat['acc@1']:.4f}",
            f"{extra_nat['rerank_acc_given_gold_in_shortlist']:.4f}",
            f"{extra_nat['acc@5']:.4f}",
            f"{extra_nat['mrr']:.4f}",
            f"{extra_nat['shortlist_recall']:.4f}",
        ],
    ]
    with (out / "low_lr_secondpass_extra_seed_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "rerank_checkpoint",
                "best_epoch",
                "seed",
                "acc@1_nat",
                "cond_in_K",
                "acc@5_nat",
                "mrr_nat",
                "gold_in_shortlist_rate",
            ]
        )
        w.writerows(extra_table_rows)

    extra_interp = f"""# low_lr_secondpass extra-seed interpretation

1. Does the extra seed stay closer to `0.1154` or collapse toward `0.1090`?
   - coadapted intermediate: **{inter_nat['acc@1']:.4f}**
   - validated low_lr (seed 42): **{val_nat['acc@1']:.4f}**
   - extra seed (seed {args.extra_seed}): **{extra_nat['acc@1']:.4f}**
   - delta vs 0.1154: **{extra_nat['acc@1'] - inter_nat['acc@1']:+.4f}**
   - delta vs 0.1090: **{extra_nat['acc@1'] - 0.1090:+.4f}**

2. Does `cond_in_K` remain closer to preserved-gain regime?
   - preserved-gain reference cond_in_K: **{inter_nat['rerank_acc_given_gold_in_shortlist']:.4f}**
   - extra-seed cond_in_K: **{extra_nat['rerank_acc_given_gold_in_shortlist']:.4f}**
   - delta: **{extra_nat['rerank_acc_given_gold_in_shortlist'] - inter_nat['rerank_acc_given_gold_in_shortlist']:+.4f}**

3. Confidence impact on low_lr_secondpass as preferred strategy
   - This extra-seed result is interpreted conservatively against the `0.1154` retained-gain target and `0.1090` fallback reference.
"""
    (out / "low_lr_secondpass_extra_seed_interpretation.md").write_text(extra_interp, encoding="utf-8")

    sweep_blocks: list[dict[str, Any]] = []
    sweep_rows: list[list[Any]] = []
    for spec in sweep_variants:
        name = str(spec["name"])
        block = eval_pair(name, reselected_coarse, Path(trained[name]["checkpoint"]))
        sweep_blocks.append(block)
        nat = block["eval_natural_shortlist"]
        best_row = trained[name]["best_row"] or {}
        sweep_rows.append(
            [
                name,
                f"{float(spec['lr']):.8g}",
                str(spec["lr_factor"]),
                str(best_row.get("epoch", "")),
                f"{nat['acc@1']:.4f}",
                f"{nat['rerank_acc_given_gold_in_shortlist']:.4f}",
                f"{nat['acc@5']:.4f}",
                f"{nat['mrr']:.4f}",
                f"{nat['shortlist_recall']:.4f}",
            ]
        )
    sweep_results = {
        "selection_context": {
            "center_lr": float(args.base_lr),
            "seed": int(args.base_seed),
            "coadapted_shortlist_checkpoint": str(reselected_coarse),
            "fine_init_checkpoint": str(first_retrained),
            "sweep_lrs": sweep_lrs,
            "pythonhashseed": hash_seed or None,
        },
        "reference_rows": [inter, validated_low],
        "sweep_variants": sweep_blocks,
        "sweep_training_best_rows": {name: trained[name]["best_row"] for name in [s["name"] for s in sweep_variants]},
    }
    (out / "low_lr_sweep_results.json").write_text(json.dumps(sweep_results, indent=2, default=str), encoding="utf-8")

    with (out / "low_lr_sweep_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "lr",
                "lr_factor",
                "best_epoch",
                "acc@1_nat",
                "cond_in_K",
                "acc@5_nat",
                "mrr_nat",
                "gold_in_shortlist_rate",
            ]
        )
        w.writerows(sweep_rows)
    _write_table_md(
        out / "low_lr_sweep_table.md",
        ["variant", "lr", "factor", "best_ep", "nat@1", "cond_K", "nat@5", "mrr", "gold_in_K"],
        sweep_rows,
    )

    lr_vals = [float(r[1]) for r in sweep_rows]
    acc_vals = [float(r[4]) for r in sweep_rows]
    _plot_lr_sweep(out / "low_lr_sweep_plot.png", lr_vals, acc_vals, float(args.base_lr))

    best_sweep_row = max(sweep_rows, key=lambda r: float(r[4]))
    best_sweep_name = str(best_sweep_row[0])
    best_sweep_block = next(b for b in sweep_blocks if b["label"] == best_sweep_name)

    sweep_interp = f"""# low_lr sweep interpretation

1. Is current low_lr locally stable?
   - center LR ({args.base_lr:.8g}) natural Acc@1: **{next(float(r[4]) for r in sweep_rows if float(r[1]) == float(args.base_lr)):.4f}**
   - half LR ({sweep_lrs[0]:.8g}) natural Acc@1: **{next(float(r[4]) for r in sweep_rows if float(r[1]) == sweep_lrs[0]):.4f}**
   - double LR ({sweep_lrs[2]:.8g}) natural Acc@1: **{next(float(r[4]) for r in sweep_rows if float(r[1]) == sweep_lrs[2]):.4f}**

2. Is there nearby LR preserving or slightly improving `0.1154`?
   - best local LR variant: **{best_sweep_name}** at LR **{best_sweep_row[1]}**
   - best local natural Acc@1: **{best_sweep_row[4]}**

3. Does performance collapse quickly when LR increases?
   - compare center vs 2x LR delta: **{next(float(r[4]) for r in sweep_rows if float(r[1]) == sweep_lrs[2]) - next(float(r[4]) for r in sweep_rows if float(r[1]) == float(args.base_lr)):+.4f}**

4. Is gain retention narrow LR-sensitive or somewhat stable local regime?
   - Decision is based on this 3-point local sweep only; no broad search is introduced.
"""
    (out / "low_lr_sweep_interpretation.md").write_text(sweep_interp, encoding="utf-8")

    combined_blocks = [
        eval_pair("improved_shortlist_plus_reference_rerank", improved_coarse, reference_rerank),
        inter,
        validated_low,
        extra_block,
    ]
    for b in sweep_blocks:
        combined_blocks.append(
            {
                "label": f"sweep_{b['label']}",
                "coarse_checkpoint": b["coarse_checkpoint"],
                "rerank_checkpoint": b["rerank_checkpoint"],
                "eval_natural_shortlist": b["eval_natural_shortlist"],
                "eval_oracle_shortlist": b["eval_oracle_shortlist"],
            }
        )
    combined_blocks.append(
        {
            "label": "best_LR_from_micro_sweep",
            "coarse_checkpoint": best_sweep_block["coarse_checkpoint"],
            "rerank_checkpoint": best_sweep_block["rerank_checkpoint"],
            "eval_natural_shortlist": best_sweep_block["eval_natural_shortlist"],
            "eval_oracle_shortlist": best_sweep_block["eval_oracle_shortlist"],
        }
    )

    combined = {
        "selection_context": {
            "primary_metric": "natural two-stage full-scene validation Acc@1",
            "center_lr": float(args.base_lr),
            "extra_seed": int(args.extra_seed),
            "best_micro_lr_variant": best_sweep_name,
            "pythonhashseed": hash_seed or None,
        },
        "pipelines": combined_blocks,
    }
    (out / "shortlist_rerank_combined_results_gain_validation.json").write_text(
        json.dumps(combined, indent=2, default=str),
        encoding="utf-8",
    )

    combined_rows = []
    for b in combined_blocks:
        nat = b["eval_natural_shortlist"]
        ora = b["eval_oracle_shortlist"]
        combined_rows.append(
            [
                b["label"],
                _safe_name(b["coarse_checkpoint"]),
                _safe_name(b["rerank_checkpoint"]),
                f"{nat['acc@1']:.4f}",
                f"{nat['rerank_acc_given_gold_in_shortlist']:.4f}",
                f"{nat['acc@5']:.4f}",
                f"{nat['mrr']:.4f}",
                f"{nat['shortlist_recall']:.4f}",
                f"{ora['acc@1']:.4f}",
            ]
        )
    with (out / "shortlist_rerank_combined_table_gain_validation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pipeline",
                "coarse_checkpoint",
                "rerank_checkpoint",
                "acc@1_nat",
                "cond_in_K",
                "acc@5_nat",
                "mrr_nat",
                "gold_in_shortlist_rate",
                "acc@1_oracle",
            ]
        )
        w.writerows(combined_rows)
    _write_table_md(
        out / "shortlist_rerank_combined_table_gain_validation.md",
        ["pipeline", "coarse", "rerank", "nat@1", "cond_K", "nat@5", "mrr", "gold_in_K", "oracle@1"],
        combined_rows,
    )

    focus_labels = [
        "improved_shortlist_plus_reference_rerank",
        "coadapted_intermediate_best",
        "validated_low_lr_secondpass",
        "low_lr_secondpass_extra_seed",
        "best_LR_from_micro_sweep",
    ]
    fig_rows = [r for r in combined_rows if r[0] in focus_labels]
    _plot_combined(out / "shortlist_rerank_main_figure_gain_validation.png", [r[0] for r in fig_rows], [float(r[3]) for r in fig_rows])

    best_micro_nat = best_sweep_block["eval_natural_shortlist"]["acc@1"]
    gain_interp = f"""# gain-retention combined interpretation

1. Does low_lr_secondpass remain the best narrow strategy under extra seed and local sweep?
   - validated low_lr (seed 42): **{val_nat['acc@1']:.4f}**
   - extra seed (seed {args.extra_seed}): **{extra_nat['acc@1']:.4f}**
   - best micro-LR variant (`{best_sweep_name}`): **{best_micro_nat:.4f}**

2. Is `0.1154` robust enough to present as retained-gain result?
   - coadapted intermediate: **{inter_nat['acc@1']:.4f}**
   - improved_shortlist_plus_reference_rerank: **{eval_pair('tmp', improved_coarse, reference_rerank)['eval_natural_shortlist']['acc@1']:.4f}**
   - evidence is judged from this extra-seed + micro-sweep only.

3. Does any nearby LR slightly improve on `0.1154`?
   - best micro-sweep Acc@1: **{best_micro_nat:.4f}**
   - delta vs 0.1154: **{best_micro_nat - inter_nat['acc@1']:+.4f}**

4. Confidence level for low_lr_secondpass as current best
   - confidence is updated conservatively using only this narrow validation phase.
"""
    (out / "shortlist_rerank_interpretation_gain_validation.md").write_text(gain_interp, encoding="utf-8")

    repro = out / "repro_commands.sh"
    repro.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cd "{ROOT}"
export PYTHONHASHSEED="{hash_seed or '42'}"
{py} scripts/run_gain_retention_validation_phase.py --stamp {stamp} --output-tag {args.output_tag} --device {dev_name} \\
  --extra-seed {args.extra_seed} --base-seed {args.base_seed} --base-lr {args.base_lr} \\
  --fix-bundle "{fix_bundle}" --shortlist-bundle "{shortlist_bundle}" --rebalance-bundle "{rebalance_bundle}" \\
  --coadapt-bundle "{coadapt_bundle}" --conservative-bundle "{conservative_bundle}"
""",
        encoding="utf-8",
    )
    repro.chmod(0o755)

    rb = out / "report_bundle"
    rb.mkdir(parents=True, exist_ok=True)
    bundle_reports = {
        "gain_retention_validation_plan.md": ROOT / "reports/gain_retention_validation_plan.md",
        "gain_retention_validation_summary.md": ROOT / "reports/gain_retention_validation_summary.md",
        "readme_gain_retention_validation_note.md": ROOT / "reports/readme_gain_retention_validation_note.md",
    }
    copied_bundle_reports: list[str] = []
    for name, src in bundle_reports.items():
        if _copy_if_exists(src, rb / name):
            copied_bundle_reports.append(name)
    (rb / "README.md").write_text(
        f"""# Gain retention validation report bundle

| Artifact | Claim |
|----------|--------|
| `low_lr_secondpass_extra_seed_*` | Extra-seed validation checks whether low_lr_secondpass remains stable beyond the validated seed-42 run. |
| `low_lr_sweep_*` | Tiny 3-point local LR sweep checks whether `0.1154` retention is locally robust around low LR. |
| `shortlist_rerank_combined_table_gain_validation.*` | Corrected combined table summarizes confidence in retained-gain behavior across references and validation variants. |
| `shortlist_rerank_main_figure_gain_validation.png` | Main figure for narrow gain-retention validation comparison. |
| `generated_configs/*.yaml` | Exact extra-seed and LR-sweep configs used in this validation run. |
| `repro_commands.sh` | Reproduction command for this validation directory. |

Bundled notes:

{os.linesep.join(f"- `{name}`" for name in copied_bundle_reports) if copied_bundle_reports else "- root report files were not present when this bundle README was generated"}

Environment note:

- `PYTHONHASHSEED={hash_seed or 'unset'}`
""",
        encoding="utf-8",
    )

    _append_log(logs / "phase.log", f"Gain-retention validation phase complete -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
