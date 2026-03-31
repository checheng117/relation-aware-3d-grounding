#!/usr/bin/env python3
"""Official shortlist-strengthening phase on top of the corrected March 31 pipeline."""

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

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = range(len(labels))
    ax.bar(x, values, color=["#6b7280", "#2563eb", "#f59e0b", "#16a34a", "#7c3aed"][: len(labels)])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Natural two-stage Acc@1")
    ax.set_title("Official shortlist strengthening")
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
    ap.add_argument("--output-tag", type=str, default="official_shortlist_strengthening")
    ap.add_argument("--epochs-coarse", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--fix-bundle",
        type=Path,
        default=ROOT / "outputs/20260331_160556_fix_combined_nloss",
    )
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from rag3d.datasets.collate import make_grounding_collate_fn
    from rag3d.datasets.referit3d import ReferIt3DManifestDataset
    from rag3d.evaluation.coarse_recall import eval_coarse_stage1_metrics
    from rag3d.evaluation.two_stage_eval import load_coarse_model, load_two_stage_model
    from rag3d.evaluation.two_stage_rerank_metrics import (
        eval_by_candidate_load_bucket,
        eval_two_stage_inject_mode,
    )
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
    fix_ck = fix_bundle / "checkpoints"
    base_coarse = ROOT / "outputs/checkpoints_stage1/coarse_geom_recall_last.pt"
    if not base_coarse.is_file():
        base_coarse = ROOT / "outputs/checkpoints_stage1/coarse_geom_ce_last.pt"
    baseline_rerank = ROOT / "outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt"
    if not baseline_rerank.is_file():
        baseline_rerank = ROOT / "outputs/checkpoints_rerank/rerank_full_k10_last.pt"
    rerank_o_best = fix_ck / "rerank_longtrain_oracle_best_natural_two_stage.pt"
    rerank_n_best = fix_ck / "rerank_longtrain_natural_best_natural_two_stage.pt"
    train_m = ROOT / "data/processed/train_manifest.jsonl"
    val_m = ROOT / "data/processed/val_manifest.jsonl"

    missing = [
        p
        for p in (train_m, val_m, base_coarse, baseline_rerank, rerank_o_best)
        if not p.is_file()
    ]
    if missing:
        for p in missing:
            _append_log(logs / "phase.log", f"missing required asset: {p}")
        return 1

    selector_label = f"{fix_bundle.name}::rerank_O_best"

    if not args.skip_train:
        coarse_yaml: dict[str, Any] = {
            "model": "relation_aware",
            "dataset_config": "configs/dataset/referit3d.yaml",
            "coarse_model": "coarse_geom",
            "checkpoint_dir": str(ck),
            "metrics_file": str(out / "metrics_coarse_official.jsonl"),
            "run_name": "coarse_official_shortlist_strengthening",
            "epochs": int(args.epochs_coarse),
            "batch_size": 16,
            "lr": 0.00005,
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
                "reference_rerank_checkpoint": str(rerank_o_best.resolve()),
                "reference_label": selector_label,
                "model_config": "configs/model/relation_aware.yaml",
                "parser_mode": "structured",
                "parser_cache_dir": "data/parser_cache/official_shortlist_selection",
                "rerank_k": 10,
                "margin_thresh": 0.15,
            },
        }
        coarse_cfg_path = gc / "coarse_official_shortlist_strengthening.yaml"
        _dump_yaml(coarse_cfg_path, coarse_yaml)
        _run(
            [
                py,
                str(ROOT / "scripts/train_coarse_stage1.py"),
                "--config",
                str(coarse_cfg_path),
                "--init-checkpoint",
                str(base_coarse),
            ],
            logs / "train_coarse_official.log",
        )

    coarse_best = ck / "coarse_official_shortlist_strengthening_best_pipeline_natural.pt"
    coarse_last = ck / "coarse_official_shortlist_strengthening_last.pt"
    improved_coarse = coarse_best if coarse_best.is_file() else coarse_last
    if not improved_coarse.is_file():
        _append_log(logs / "phase.log", f"missing trained coarse checkpoint: {improved_coarse}")
        return 1

    metrics_rows = _read_jsonl_metrics(out / "metrics_coarse_official.jsonl")
    best_row = {}
    if metrics_rows:
        best_row = max(
            metrics_rows,
            key=lambda row: float(row.get("val_pipeline_natural_acc@1", float("-inf"))),
        )

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
    parser = CachedParser(StructuredRuleParser(), ROOT / "data/parser_cache/official_shortlist_combined/structured")

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

    combined: dict[str, Any] = {
        "selection_reference": {
            "checkpoint": str(rerank_o_best),
            "label": selector_label,
            "primary_metric": "val_pipeline_natural_acc@1",
            "pythonhashseed": hash_seed or None,
        },
        "pipelines": [],
    }
    combined["pipelines"].append(eval_pair("baseline_reference", base_coarse, baseline_rerank))
    combined["pipelines"].append(eval_pair("rerank_O_best", base_coarse, rerank_o_best))
    combined["pipelines"].append(
        eval_pair("improved_shortlist_plus_reference_rerank", improved_coarse, baseline_rerank)
    )
    combined["pipelines"].append(
        eval_pair("improved_shortlist_plus_rerank_O_best", improved_coarse, rerank_o_best)
    )
    if rerank_n_best.is_file():
        combined["pipelines"].append(
            eval_pair("improved_shortlist_plus_rerank_N_best", improved_coarse, rerank_n_best)
        )

    combined_json = out / "shortlist_rerank_combined_results_official.json"
    combined_json.write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")

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

    combined_csv = out / "shortlist_rerank_combined_table_official.csv"
    with combined_csv.open("w", newline="", encoding="utf-8") as f:
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
        out / "shortlist_rerank_combined_table_official.md",
        ["pipeline", "coarse_ckpt", "rerank_ckpt", "acc@1_nat", "acc@5", "mrr", "recall_K", "cond_K", "oracle", "hi", "lo"],
        combined_rows,
    )
    _plot_combined(
        out / "shortlist_rerank_main_figure_official.png",
        [row[0][:22] for row in combined_rows],
        [float(row[3]) for row in combined_rows],
    )

    by_label = {row["label"]: row for row in combined["pipelines"]}
    baseline_o = by_label["rerank_O_best"]["eval_natural_shortlist"]
    improved_o = by_label["improved_shortlist_plus_rerank_O_best"]["eval_natural_shortlist"]
    improved_o_oracle = by_label["improved_shortlist_plus_rerank_O_best"]["eval_oracle_shortlist"]
    baseline_o_oracle = by_label["rerank_O_best"]["eval_oracle_shortlist"]
    baseline_ref = by_label["baseline_reference"]["eval_natural_shortlist"]
    improved_ref = by_label["improved_shortlist_plus_reference_rerank"]["eval_natural_shortlist"]
    n_best_block = by_label.get("improved_shortlist_plus_rerank_N_best")

    combined_interp = f"""# Official combined evaluation interpretation

## Main question

1. **Does shortlist strengthening increase end-to-end gain beyond corrected `rerank_O_best` alone?**
   - `rerank_O_best` natural Acc@1: **{baseline_o['acc@1']:.4f}**
   - `improved_shortlist_plus_rerank_O_best` natural Acc@1: **{improved_o['acc@1']:.4f}**
   - delta: **{improved_o['acc@1'] - baseline_o['acc@1']:+.4f}**

2. **Is the new shortlist stage helping materially, or only improving coarse-only metrics?**
   - `baseline_reference` natural Acc@1: **{baseline_ref['acc@1']:.4f}**
   - `improved_shortlist_plus_reference_rerank` natural Acc@1: **{improved_ref['acc@1']:.4f}**
   - `rerank_O_best` shortlist recall: **{baseline_o['shortlist_recall']:.4f}**
   - `improved_shortlist_plus_rerank_O_best` shortlist recall: **{improved_o['shortlist_recall']:.4f}**

3. **Oracle upper bound**
   - `rerank_O_best` oracle Acc@1: **{baseline_o_oracle['acc@1']:.4f}**
   - `improved_shortlist_plus_rerank_O_best` oracle Acc@1: **{improved_o_oracle['acc@1']:.4f}**
   - delta: **{improved_o_oracle['acc@1'] - baseline_o_oracle['acc@1']:+.4f}**

4. **Candidate-load high slice**
   - `rerank_O_best` high-load Acc@1: **{by_label['rerank_O_best']['bucket_natural'].get('high_candidate_load', {}).get('acc@1', float('nan'))}**
   - `improved_shortlist_plus_rerank_O_best` high-load Acc@1: **{by_label['improved_shortlist_plus_rerank_O_best']['bucket_natural'].get('high_candidate_load', {}).get('acc@1', float('nan'))}**

## Bottleneck readout

- If natural Acc@1 and shortlist recall both rise for the `improved_shortlist_plus_rerank_O_best` row, shortlist strengthening is helping the stronger reranker translate.
- If shortlist recall rises but natural Acc@1 stays flat, shortlist quality improved less than needed or reranking remains the tighter limiter.
- If oracle stays much higher than natural after the shortlist update, shortlist is still a major bottleneck.
"""
    if n_best_block is not None:
        n_nat = n_best_block["eval_natural_shortlist"]
        combined_interp += f"""

## Auxiliary `N_best`

- `improved_shortlist_plus_rerank_N_best` natural Acc@1: **{n_nat['acc@1']:.4f}**
- This is auxiliary only; the main selector and main combined row remain tied to `O_best`.
"""
    (out / "shortlist_rerank_interpretation_official.md").write_text(combined_interp, encoding="utf-8")

    baseline_coarse_metrics = eval_coarse_stage1_metrics(
        load_coarse_model(mcfg, base_coarse, dev, "coarse_geom"),
        val_loader,
        dev,
        0.15,
        ks=(5, 10, 20, 40),
    )
    improved_coarse_metrics = eval_coarse_stage1_metrics(
        load_coarse_model(mcfg, improved_coarse, dev, "coarse_geom"),
        val_loader,
        dev,
        0.15,
        ks=(5, 10, 20, 40),
    )

    retrieval = {
        "selection_reference": {
            "checkpoint": str(rerank_o_best),
            "label": selector_label,
            "primary_metric": "val_pipeline_natural_acc@1",
            "pythonhashseed": hash_seed or None,
        },
        "training_best_row": best_row,
        "baseline_coarse_val": baseline_coarse_metrics,
        "improved_coarse_val": improved_coarse_metrics,
        "baseline_coarse_plus_rerank_O_best": by_label["rerank_O_best"],
        "improved_coarse_plus_rerank_O_best": by_label["improved_shortlist_plus_rerank_O_best"],
        "selected_checkpoint": str(improved_coarse),
    }
    (out / "shortlist_retrieval_results_official.json").write_text(
        json.dumps(retrieval, indent=2, default=str),
        encoding="utf-8",
    )

    def _slice(metric_block: dict[str, Any], key: str) -> str:
        return f"{float((metric_block.get('stratified_recall_slices') or {}).get(key, 0.0)):.4f}"

    retrieval_rows = [
        [
            "baseline",
            _safe_name(str(base_coarse)),
            "baseline_fixed",
            selector_label,
            f"{baseline_coarse_metrics.get('recall@5', 0.0):.4f}",
            f"{baseline_coarse_metrics.get('recall@10', 0.0):.4f}",
            f"{baseline_coarse_metrics.get('recall@20', 0.0):.4f}",
            f"{baseline_coarse_metrics.get('recall@40', 0.0):.4f}",
            _slice(baseline_coarse_metrics, "recall@10_slice::candidate_load::high"),
            _slice(baseline_coarse_metrics, "recall@10_slice::same_class_clutter"),
            f"{baseline_o['acc@1']:.4f}",
            f"{baseline_o_oracle['acc@1']:.4f}",
            f"{baseline_o['rerank_acc_given_gold_in_shortlist']:.4f}",
        ],
        [
            "improved",
            _safe_name(str(improved_coarse)),
            str(best_row.get("epoch", "")),
            str(best_row.get("val_pipeline_selection_reference_label", selector_label)),
            f"{improved_coarse_metrics.get('recall@5', 0.0):.4f}",
            f"{improved_coarse_metrics.get('recall@10', 0.0):.4f}",
            f"{improved_coarse_metrics.get('recall@20', 0.0):.4f}",
            f"{improved_coarse_metrics.get('recall@40', 0.0):.4f}",
            _slice(improved_coarse_metrics, "recall@10_slice::candidate_load::high"),
            _slice(improved_coarse_metrics, "recall@10_slice::same_class_clutter"),
            f"{improved_o['acc@1']:.4f}",
            f"{improved_o_oracle['acc@1']:.4f}",
            f"{improved_o['rerank_acc_given_gold_in_shortlist']:.4f}",
        ],
    ]
    retrieval_csv = out / "shortlist_retrieval_table_official.csv"
    with retrieval_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "checkpoint_type",
                "checkpoint",
                "selected_epoch",
                "selection_reference",
                "recall@5",
                "recall@10",
                "recall@20",
                "recall@40",
                "recall@10_high_load",
                "recall@10_same_class_clutter",
                "natural_acc@1_with_O_best",
                "oracle_acc@1_with_O_best",
                "cond_acc_in_K_with_O_best",
            ]
        )
        w.writerows(retrieval_rows)
    _write_table_md(
        out / "shortlist_retrieval_table_official.md",
        [
            "type",
            "checkpoint",
            "epoch",
            "selector",
            "r@5",
            "r@10",
            "r@20",
            "r@40",
            "r@10_hi",
            "r@10_samecls",
            "nat@1_O",
            "oracle_O",
            "cond_K_O",
        ],
        retrieval_rows,
    )

    retrieval_interp = f"""# Official shortlist retrieval interpretation

## Did shortlist Recall@5/10/20 improve?

- baseline Recall@5/10/20/40: **{baseline_coarse_metrics.get('recall@5', 0.0):.4f} / {baseline_coarse_metrics.get('recall@10', 0.0):.4f} / {baseline_coarse_metrics.get('recall@20', 0.0):.4f} / {baseline_coarse_metrics.get('recall@40', 0.0):.4f}**
- improved Recall@5/10/20/40: **{improved_coarse_metrics.get('recall@5', 0.0):.4f} / {improved_coarse_metrics.get('recall@10', 0.0):.4f} / {improved_coarse_metrics.get('recall@20', 0.0):.4f} / {improved_coarse_metrics.get('recall@40', 0.0):.4f}**

## Did corrected end-to-end natural Acc@1 improve?

- with fixed `O_best`, baseline natural Acc@1: **{baseline_o['acc@1']:.4f}**
- with fixed `O_best`, improved-shortlist natural Acc@1: **{improved_o['acc@1']:.4f}**
- delta: **{improved_o['acc@1'] - baseline_o['acc@1']:+.4f}**

## Did oracle upper bounds improve?

- with fixed `O_best`, baseline oracle Acc@1: **{baseline_o_oracle['acc@1']:.4f}**
- with fixed `O_best`, improved-shortlist oracle Acc@1: **{improved_o_oracle['acc@1']:.4f}**
- delta: **{improved_o_oracle['acc@1'] - baseline_o_oracle['acc@1']:+.4f}**

## Did shortlist strengthening help the stronger reranker translate into more final gain?

- `baseline_reference` natural Acc@1: **{baseline_ref['acc@1']:.4f}**
- `rerank_O_best` natural Acc@1: **{baseline_o['acc@1']:.4f}**
- `improved_shortlist_plus_rerank_O_best` natural Acc@1: **{improved_o['acc@1']:.4f}**

Interpretation rule:

- If `improved_shortlist_plus_rerank_O_best` exceeds `rerank_O_best`, shortlist strengthening is helping the already-validated reranker translate into more final gain.
- If Recall@K improves but `improved_shortlist_plus_rerank_O_best` does not, shortlist-only gains are not yet translating strongly enough under the corrected objective.
"""
    (out / "shortlist_retrieval_interpretation_official.md").write_text(retrieval_interp, encoding="utf-8")

    repro = out / "repro_commands.sh"
    repro.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cd "{ROOT}"
export PYTHONHASHSEED="{hash_seed or '42'}"
{py} scripts/run_official_shortlist_strengthening_phase.py --stamp {stamp} --device {device} \\
  --output-tag {args.output_tag} --epochs-coarse {args.epochs_coarse} --fix-bundle "{fix_bundle}"
# Eval-only rerun after training:
# {py} scripts/run_official_shortlist_strengthening_phase.py --stamp {stamp} --output-tag {args.output_tag} --skip-train --fix-bundle "{fix_bundle}"
""",
        encoding="utf-8",
    )
    repro.chmod(0o755)

    rb = out / "report_bundle"
    rb.mkdir(parents=True, exist_ok=True)
    bundle_reports = {
        "official_shortlist_strengthening_plan.md": ROOT / "reports/official_shortlist_strengthening_plan.md",
        "shortlist_selection_alignment_note.md": ROOT / "reports/shortlist_selection_alignment_note.md",
        "official_shortlist_hard_negative_note.md": ROOT / "reports/official_shortlist_hard_negative_note.md",
        "official_shortlist_strengthening_summary.md": ROOT / "reports/official_shortlist_strengthening_summary.md",
        "readme_official_shortlist_strengthening_note.md": ROOT / "reports/readme_official_shortlist_strengthening_note.md",
    }
    copied_bundle_reports: list[str] = []
    for name, src in bundle_reports.items():
        if _copy_if_exists(src, rb / name):
            copied_bundle_reports.append(name)
    (rb / "README.md").write_text(
        f"""# Official shortlist strengthening report bundle

| Artifact | Claim |
|----------|--------|
| `shortlist_retrieval_results_official.json` + `shortlist_retrieval_table_official.*` | Retrieval quality is compared under the March 31 corrected `O_best` selector, not Recall@K alone. |
| `shortlist_rerank_combined_results_official.json` + `shortlist_rerank_combined_table_official.*` | Corrected combined evaluation shows whether the stronger shortlist actually increases end-to-end natural Acc@1 with `O_best`. |
| `shortlist_rerank_main_figure_official.png` | Main figure for natural two-stage Acc@1 across the baseline, rerank-only, and improved-shortlist pipelines. |
| `generated_configs/coarse_official_shortlist_strengthening.yaml` | Documents the targeted shortlist hard-negative recipe used in the official run. |
| `logs/train_coarse_official.log` | Full training log for the official shortlist-strengthening coarse run. |
| `repro_commands.sh` | Exact command used to regenerate this bundle. |

Bundled notes:

{os.linesep.join(f"- `{name}`" for name in copied_bundle_reports) if copied_bundle_reports else "- root report files were not present when this bundle README was generated"}

Environment note:

- `PYTHONHASHSEED={hash_seed or 'unset'}`
""",
        encoding="utf-8",
    )

    _append_log(logs / "phase.log", f"Official shortlist strengthening complete -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
