#!/usr/bin/env python3
"""Reranker rebalance phase on top of the official improved shortlist."""

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
    ax.bar(x, values, color=["#6b7280", "#f59e0b", "#2563eb", "#16a34a", "#7c3aed"][: len(labels)])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Natural two-stage Acc@1")
    ax.set_title("Rerank rebalance on improved shortlist")
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
    ap.add_argument("--output-tag", type=str, default="rerank_rebalance")
    ap.add_argument("--epochs-rerank", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda")
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
    ap.add_argument("--skip-train", action="store_true")
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
    fix_ck = fix_bundle / "checkpoints"
    shortlist_ck = shortlist_bundle / "checkpoints"

    improved_coarse = shortlist_ck / "coarse_official_shortlist_strengthening_best_pipeline_natural.pt"
    if not improved_coarse.is_file():
        improved_coarse = shortlist_ck / "coarse_official_shortlist_strengthening_last.pt"

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
        for p in (train_m, val_m, improved_coarse, base_coarse, reference_rerank, rerank_o_best)
        if not p.is_file()
    ]
    if missing:
        for p in missing:
            _append_log(logs / "phase.log", f"missing required asset: {p}")
        return 1

    if not args.skip_train:
        rerank_yaml: dict[str, Any] = {
            "model": "relation_aware",
            "dataset_config": "configs/dataset/referit3d.yaml",
            "coarse_model": "coarse_geom",
            "coarse_checkpoint": str(improved_coarse),
            "fine_init_checkpoint": str(rerank_o_best),
            "rerank_k": 10,
            "parser_mode": "structured",
            "parser_cache_dir": "data/parser_cache/rerank_rebalance",
            "batch_size": 16,
            "lr": 0.00005,
            "weight_decay": 0.01,
            "seed": 42,
            "num_workers": 0,
            "device": device,
            "debug_max_batches": None,
            "epochs": int(args.epochs_rerank),
            "checkpoint_dir": str(ck),
            "metrics_file": str(out / "metrics_rerank_rebalance.jsonl"),
            "run_name": "rerank_rebalance_improved_natural",
            "shortlist_train_inject_gold": False,
            "selection_margin_thresh": 0.15,
            "loss": {"hard_negative": {"enabled": True, "margin": 0.25, "lambda_hinge": 0.4}},
        }
        rerank_cfg_path = gc / "rerank_rebalance_improved_natural.yaml"
        _dump_yaml(rerank_cfg_path, rerank_yaml)
        _run(
            [py, str(ROOT / "scripts/train_two_stage_rerank.py"), "--config", str(rerank_cfg_path)],
            logs / "train_rerank_rebalance.log",
        )

    retrained_best = ck / "rerank_rebalance_improved_natural_best_natural_two_stage.pt"
    retrained_last = ck / "rerank_rebalance_improved_natural_last.pt"
    retrained_rerank = retrained_best if retrained_best.is_file() else retrained_last
    if not retrained_rerank.is_file():
        _append_log(logs / "phase.log", f"missing retrained rerank checkpoint: {retrained_rerank}")
        return 1

    metrics_rows = _read_jsonl_metrics(out / "metrics_rerank_rebalance.jsonl")
    best_row = {}
    if metrics_rows:
        best_row = max(metrics_rows, key=lambda row: float(row.get("val_natural_two_stage_acc@1", float("-inf"))))
    _plot_training_curves(out / "rerank_rebalance_curves.png", metrics_rows)

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
    parser = CachedParser(StructuredRuleParser(), ROOT / "data/parser_cache/rerank_rebalance_eval/structured")

    improved_coarse_metrics = eval_coarse_stage1_metrics(
        load_coarse_model(mcfg, improved_coarse, dev, "coarse_geom"),
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

    rerank_variants: list[dict[str, Any]] = [
        eval_pair("reference_rerank_on_improved_shortlist", improved_coarse, reference_rerank),
        eval_pair("O_best_on_improved_shortlist", improved_coarse, rerank_o_best),
        eval_pair("retrained_rerank_on_improved_shortlist", improved_coarse, retrained_rerank),
    ]
    if rerank_n_best.is_file():
        rerank_variants.append(eval_pair("fixed_N_best_on_improved_shortlist", improved_coarse, rerank_n_best))

    rerank_results = {
        "selection_context": {
            "improved_coarse_checkpoint": str(improved_coarse),
            "primary_metric": "val_natural_two_stage_acc@1",
            "training_mode": "natural_improved_shortlist",
            "fine_init_checkpoint": str(rerank_o_best),
            "pythonhashseed": hash_seed or None,
        },
        "improved_shortlist_context": improved_coarse_metrics,
        "training_best_row": best_row,
        "variants": rerank_variants,
        "selected_checkpoint": str(retrained_rerank),
    }
    (out / "rerank_rebalance_results.json").write_text(json.dumps(rerank_results, indent=2, default=str), encoding="utf-8")

    variant_by_label = {row["label"]: row for row in rerank_variants}
    ref_nat = variant_by_label["reference_rerank_on_improved_shortlist"]["eval_natural_shortlist"]
    o_nat = variant_by_label["O_best_on_improved_shortlist"]["eval_natural_shortlist"]
    new_nat = variant_by_label["retrained_rerank_on_improved_shortlist"]["eval_natural_shortlist"]
    ref_ora = variant_by_label["reference_rerank_on_improved_shortlist"]["eval_oracle_shortlist"]
    o_ora = variant_by_label["O_best_on_improved_shortlist"]["eval_oracle_shortlist"]
    new_ora = variant_by_label["retrained_rerank_on_improved_shortlist"]["eval_oracle_shortlist"]

    rebalance_rows: list[list[Any]] = []
    for block in rerank_variants:
        nat = block["eval_natural_shortlist"]
        ora = block["eval_oracle_shortlist"]
        hi = block["bucket_natural"].get("high_candidate_load", {})
        if block["label"] == "retrained_rerank_on_improved_shortlist":
            selected_epoch = str(best_row.get("epoch", ""))
            init_source = _safe_name(str(rerank_o_best))
        else:
            selected_epoch = "baseline_fixed"
            init_source = "n/a"
        rebalance_rows.append(
            [
                block["label"],
                _safe_name(block.get("rerank_checkpoint", "")),
                init_source,
                selected_epoch,
                f"{improved_coarse_metrics.get('recall@10', 0.0):.4f}",
                f"{improved_coarse_metrics.get('recall@20', 0.0):.4f}",
                f"{nat['acc@1']:.4f}",
                f"{nat['acc@5']:.4f}",
                f"{nat['mrr']:.4f}",
                f"{nat['shortlist_recall']:.4f}",
                f"{nat['rerank_acc_given_gold_in_shortlist']:.4f}",
                f"{ora['acc@1']:.4f}",
                f"{hi.get('acc@1', '')}",
            ]
        )

    rebalance_csv = out / "rerank_rebalance_table.csv"
    with rebalance_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "rerank_checkpoint",
                "init_source",
                "selected_epoch",
                "recall@10_context",
                "recall@20_context",
                "acc@1_nat",
                "acc@5_nat",
                "mrr_nat",
                "gold_in_shortlist_rate",
                "cond_in_K",
                "acc@1_oracle",
                "acc@1_high_load",
            ]
        )
        w.writerows(rebalance_rows)
    _write_table_md(
        out / "rerank_rebalance_table.md",
        ["variant", "checkpoint", "init", "epoch", "r@10", "r@20", "nat@1", "nat@5", "mrr", "gold_in_K", "cond_K", "oracle", "hi"],
        rebalance_rows,
    )

    rebalance_interp = f"""# Rerank rebalance interpretation

## Main question

1. **Does retraining on the improved shortlist beat `reference_rerank_on_improved_shortlist`?**
   - reference natural Acc@1: **{ref_nat['acc@1']:.4f}**
   - retrained natural Acc@1: **{new_nat['acc@1']:.4f}**
   - delta: **{new_nat['acc@1'] - ref_nat['acc@1']:+.4f}**

2. **Does retraining beat `O_best_on_improved_shortlist`?**
   - `O_best` natural Acc@1: **{o_nat['acc@1']:.4f}**
   - retrained natural Acc@1: **{new_nat['acc@1']:.4f}**
   - delta: **{new_nat['acc@1'] - o_nat['acc@1']:+.4f}**

3. **Does `cond_in_K` recover under the healthier shortlist?**
   - reference `cond_in_K`: **{ref_nat['rerank_acc_given_gold_in_shortlist']:.4f}**
   - `O_best` `cond_in_K`: **{o_nat['rerank_acc_given_gold_in_shortlist']:.4f}**
   - retrained `cond_in_K`: **{new_nat['rerank_acc_given_gold_in_shortlist']:.4f}**

4. **Is the reranker now better matched to the improved shortlist distribution?**
   - improved shortlist Recall@10 / Recall@20 context: **{improved_coarse_metrics.get('recall@10', 0.0):.4f} / {improved_coarse_metrics.get('recall@20', 0.0):.4f}**
   - If the retrained row improves natural Acc@1 and `cond_in_K` on this fixed shortlist, the reranker is better matched.
"""
    (out / "rerank_rebalance_interpretation.md").write_text(rebalance_interp, encoding="utf-8")

    combined: dict[str, Any] = {
        "selection_context": {
            "improved_coarse_checkpoint": str(improved_coarse),
            "training_mode": "natural_improved_shortlist",
            "fine_init_checkpoint": str(rerank_o_best),
            "primary_metric": "val_natural_two_stage_acc@1",
            "pythonhashseed": hash_seed or None,
        },
        "pipelines": [],
    }
    combined["pipelines"].append(eval_pair("baseline_reference", base_coarse, reference_rerank))
    combined["pipelines"].append(eval_pair("improved_shortlist_plus_reference_rerank", improved_coarse, reference_rerank))
    combined["pipelines"].append(eval_pair("improved_shortlist_plus_rerank_O_best", improved_coarse, rerank_o_best))
    combined["pipelines"].append(eval_pair("improved_shortlist_plus_retrained_rerank", improved_coarse, retrained_rerank))
    if rerank_n_best.is_file():
        combined["pipelines"].append(eval_pair("improved_shortlist_plus_fixed_N_best", improved_coarse, rerank_n_best))

    (out / "shortlist_rerank_combined_results_rebalance.json").write_text(
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

    combined_csv = out / "shortlist_rerank_combined_table_rebalance.csv"
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
        out / "shortlist_rerank_combined_table_rebalance.md",
        ["pipeline", "coarse_ckpt", "rerank_ckpt", "acc@1_nat", "acc@5", "mrr", "recall_K", "cond_K", "oracle", "hi", "lo"],
        combined_rows,
    )
    _plot_combined(
        out / "shortlist_rerank_main_figure_rebalance.png",
        [row[0][:24] for row in combined_rows],
        [float(row[3]) for row in combined_rows],
    )

    combined_by_label = {row["label"]: row for row in combined["pipelines"]}
    best_current = combined_by_label["improved_shortlist_plus_reference_rerank"]["eval_natural_shortlist"]
    new_combined_nat = combined_by_label["improved_shortlist_plus_retrained_rerank"]["eval_natural_shortlist"]
    old_o_nat = combined_by_label["improved_shortlist_plus_rerank_O_best"]["eval_natural_shortlist"]

    combined_interp = f"""# Rebalance combined evaluation interpretation

## Main question

1. **Does the retrained reranker beat the current best result `0.1090`?**
   - `improved_shortlist_plus_reference_rerank`: **{best_current['acc@1']:.4f}**
   - `improved_shortlist_plus_retrained_rerank`: **{new_combined_nat['acc@1']:.4f}**
   - delta: **{new_combined_nat['acc@1'] - best_current['acc@1']:+.4f}**

2. **If not, how close does it get?**
   - gap to current best: **{best_current['acc@1'] - new_combined_nat['acc@1']:+.4f}**

3. **Does reranker re-balance convert more shortlist gain into end-to-end gain?**
   - `improved_shortlist_plus_rerank_O_best`: **{old_o_nat['acc@1']:.4f}**
   - `improved_shortlist_plus_retrained_rerank`: **{new_combined_nat['acc@1']:.4f}**
   - delta vs reused `O_best`: **{new_combined_nat['acc@1'] - old_o_nat['acc@1']:+.4f}**

4. **What is the new dominant bottleneck after this phase?**
   - If retrained rerank materially improves `acc@1_nat` and `cond_in_K`, reranker mismatch was a real bottleneck and partial fix.
   - If it stays below the `0.1090` reference row, shortlist quality is healthier but reranker tuning alone is not yet enough.
"""
    (out / "shortlist_rerank_interpretation_rebalance.md").write_text(combined_interp, encoding="utf-8")

    repro = out / "repro_commands.sh"
    repro.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cd "{ROOT}"
export PYTHONHASHSEED="{hash_seed or '42'}"
{py} scripts/run_rerank_rebalance_phase.py --stamp {stamp} --device {device} \\
  --output-tag {args.output_tag} --epochs-rerank {args.epochs_rerank} \\
  --fix-bundle "{fix_bundle}" --shortlist-bundle "{shortlist_bundle}"
# Eval-only rerun after training:
# {py} scripts/run_rerank_rebalance_phase.py --stamp {stamp} --output-tag {args.output_tag} \\
#   --skip-train --fix-bundle "{fix_bundle}" --shortlist-bundle "{shortlist_bundle}"
""",
        encoding="utf-8",
    )
    repro.chmod(0o755)

    rb = out / "report_bundle"
    rb.mkdir(parents=True, exist_ok=True)
    bundle_reports = {
        "rerank_rebalance_plan.md": ROOT / "reports/rerank_rebalance_plan.md",
        "rerank_rebalance_protocol_note.md": ROOT / "reports/rerank_rebalance_protocol_note.md",
        "rerank_rebalance_summary.md": ROOT / "reports/rerank_rebalance_summary.md",
        "readme_rerank_rebalance_note.md": ROOT / "reports/readme_rerank_rebalance_note.md",
    }
    copied_bundle_reports: list[str] = []
    for name, src in bundle_reports.items():
        if _copy_if_exists(src, rb / name):
            copied_bundle_reports.append(name)
    (rb / "README.md").write_text(
        f"""# Rerank rebalance report bundle

| Artifact | Claim |
|----------|--------|
| `rerank_rebalance_results.json` + `rerank_rebalance_table.*` | Tests whether reranker retraining can better exploit the healthier shortlist than the current reference or reused `O_best`. |
| `rerank_rebalance_curves.png` | Shows whether natural improved-shortlist Acc@1, `cond_in_K`, and valid-row fraction improved during retraining. |
| `shortlist_rerank_combined_results_rebalance.json` + `shortlist_rerank_combined_table_rebalance.*` | Corrected combined evaluation tests whether the retrained reranker beats the current `0.1090` reference on the improved shortlist. |
| `shortlist_rerank_main_figure_rebalance.png` | Main figure for combined natural Acc@1 across baseline and improved-shortlist reranker variants. |
| `generated_configs/rerank_rebalance_improved_natural.yaml` | Documents the improved-shortlist reranker training recipe used in this phase. |
| `logs/train_rerank_rebalance.log` | Full training log for the rerank rebalance run. |
| `repro_commands.sh` | Exact command used to regenerate this bundle. |

Bundled notes:

{os.linesep.join(f"- `{name}`" for name in copied_bundle_reports) if copied_bundle_reports else "- root report files were not present when this bundle README was generated"}

Environment note:

- `PYTHONHASHSEED={hash_seed or 'unset'}`
""",
        encoding="utf-8",
    )

    _append_log(logs / "phase.log", f"Rerank rebalance complete -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
