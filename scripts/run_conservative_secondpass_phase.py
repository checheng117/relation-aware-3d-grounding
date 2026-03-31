#!/usr/bin/env python3
"""Conservative second-pass reranker tuning on fixed co-adapted shortlist."""

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


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _plot_variant_curves(path: Path, curve_blocks: dict[str, list[dict[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 8.2), sharex=False)
    colors = {
        "low_lr_secondpass": "#2563eb",
        "short_schedule_secondpass": "#dc2626",
        "partial_freeze_secondpass": "#16a34a",
    }
    for name, rows in curve_blocks.items():
        if not rows:
            continue
        xs = [int(r.get("epoch", i)) for i, r in enumerate(rows)]
        nat = [float(r.get("val_natural_two_stage_acc@1", 0.0)) for r in rows]
        cond = [float(r.get("val_natural_cond_acc_in_shortlist", 0.0)) for r in rows]
        ax1.plot(xs, nat, marker="o", label=f"{name} nat@1", color=colors.get(name))
        ax2.plot(xs, cond, marker="s", label=f"{name} cond_in_K", color=colors.get(name))
    ax1.set_title("Conservative second-pass curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Natural Acc@1")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8, loc="lower right")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("cond_in_K")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_combined(path: Path, labels: list[str], values: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.8))
    xs = range(len(labels))
    ax.bar(xs, values, color=["#6b7280", "#f59e0b", "#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#0891b2"][: len(labels)])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("Natural two-stage Acc@1")
    ax.set_title("Corrected combined evaluation (conservative second-pass)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", type=str, default="")
    ap.add_argument("--output-tag", type=str, default="conservative_secondpass")
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
    ap.add_argument(
        "--coadapt-bundle",
        type=Path,
        default=ROOT / "outputs/20260331_180150_minimal_coadaptation",
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
    fix_ck = fix_bundle / "checkpoints"
    shortlist_ck = shortlist_bundle / "checkpoints"
    rebalance_ck = rebalance_bundle / "checkpoints"
    coadapt_ck = coadapt_bundle / "checkpoints"

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
    naive_second = coadapt_ck / "coadapted_reranker_second_best_natural_two_stage.pt"
    if not naive_second.is_file():
        naive_second = coadapt_ck / "coadapted_reranker_second_last.pt"
    train_m = ROOT / "data/processed/train_manifest.jsonl"
    val_m = ROOT / "data/processed/val_manifest.jsonl"

    missing = [
        p
        for p in (base_coarse, reference_rerank, improved_coarse, reselected_coarse, first_retrained, naive_second, train_m, val_m)
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
        "seed": 42,
        "num_workers": 0,
        "device": dev_name,
        "debug_max_batches": None,
        "shortlist_train_inject_gold": False,
        "selection_margin_thresh": 0.15,
        "min_delta": 0.0,
        "loss": {"hard_negative": {"enabled": True, "margin": 0.25, "lambda_hinge": 0.4}},
    }
    variants: list[dict[str, Any]] = [
        {
            "name": "low_lr_secondpass",
            "epochs": 8,
            "lr": 0.000005,
            "early_stop_patience": 0,
            "fine_tune_mode": "full",
        },
        {
            "name": "short_schedule_secondpass",
            "epochs": 8,
            "lr": 0.000025,
            "early_stop_patience": 1,
            "fine_tune_mode": "full",
            "min_delta": 0.0005,
        },
        {
            "name": "partial_freeze_secondpass",
            "epochs": 8,
            "lr": 0.000025,
            "early_stop_patience": 2,
            "fine_tune_mode": "attr_rel_heads",
            "min_delta": 0.0005,
        },
    ]

    trained: dict[str, dict[str, Any]] = {}
    if not args.skip_train:
        for v in variants:
            cfg = dict(common)
            cfg.update(v)
            run_name = str(v["name"])
            cfg["epochs"] = int(v["epochs"])
            cfg["run_name"] = run_name
            cfg["metrics_file"] = str(out / f"metrics_{run_name}.jsonl")
            cfg["checkpoint_dir"] = str(ck)
            cfg["parser_cache_dir"] = f"data/parser_cache/conservative_secondpass/{run_name}"
            cfg_path = gc / f"{run_name}.yaml"
            _dump_yaml(cfg_path, cfg)
            _run(
                [py, str(ROOT / "scripts/train_two_stage_rerank.py"), "--config", str(cfg_path)],
                logs / f"train_{run_name}.log",
            )

    for v in variants:
        name = str(v["name"])
        best = ck / f"{name}_best_natural_two_stage.pt"
        last = ck / f"{name}_last.pt"
        chosen = best if best.is_file() else last
        if not chosen.is_file():
            _append_log(logs / "phase.log", f"missing variant checkpoint: {name} -> {chosen}")
            return 1
        rows = _read_jsonl_metrics(out / f"metrics_{name}.jsonl")
        best_row: dict[str, Any] = {}
        if rows:
            best_row = max(rows, key=lambda r: float(r.get("val_natural_two_stage_acc@1", float("-inf"))))
        trained[name] = {
            "name": name,
            "config": v,
            "checkpoint": str(chosen),
            "metrics_path": str(out / f"metrics_{name}.jsonl"),
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
    parser = CachedParser(StructuredRuleParser(), ROOT / "data/parser_cache/conservative_secondpass_eval/structured")

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

    mandatory_rows: list[dict[str, Any]] = [
        eval_pair("coadapted_intermediate_best", reselected_coarse, first_retrained),
        eval_pair("naive_secondpass_rerank", reselected_coarse, naive_second),
    ]
    conservative_rows = [eval_pair(name, reselected_coarse, Path(row["checkpoint"])) for name, row in trained.items()]

    by_label = {r["label"]: r for r in (mandatory_rows + conservative_rows)}
    best_variant = max(
        conservative_rows,
        key=lambda r: float(r["eval_natural_shortlist"]["acc@1"]),
    )
    best_variant_label = str(best_variant["label"])

    results = {
        "selection_context": {
            "coadapted_shortlist_checkpoint": str(reselected_coarse),
            "first_retrained_checkpoint": str(first_retrained),
            "naive_secondpass_checkpoint": str(naive_second),
            "primary_metric": "val_natural_two_stage_acc@1",
            "pythonhashseed": hash_seed or None,
        },
        "mandatory_reference_rows": mandatory_rows,
        "conservative_variants": conservative_rows,
        "variant_training": {
            name: {
                "config": row["config"],
                "checkpoint": row["checkpoint"],
                "best_row": row["best_row"],
            }
            for name, row in trained.items()
        },
        "best_conservative_variant": best_variant_label,
    }
    (out / "conservative_secondpass_results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    table_rows: list[list[Any]] = []
    order = ["coadapted_intermediate_best", "naive_secondpass_rerank"] + [v["name"] for v in variants]
    for label in order:
        block = by_label[label]
        nat = block["eval_natural_shortlist"]
        ora = block["eval_oracle_shortlist"]
        if label in trained:
            best_row = trained[label]["best_row"] or {}
            best_epoch = str(best_row.get("epoch", ""))
            valid_frac = best_row.get("rerank_train_valid_fraction", "")
            nan_count = best_row.get("nan_or_inf_batch_count", "")
            mode = str(best_row.get("fine_tune_mode", trained[label]["config"].get("fine_tune_mode", "")))
        else:
            best_epoch = "baseline_fixed"
            valid_frac = ""
            nan_count = ""
            mode = "n/a"
        table_rows.append(
            [
                label,
                _safe_name(block["rerank_checkpoint"]),
                mode,
                best_epoch,
                f"{nat['acc@1']:.4f}",
                f"{nat['rerank_acc_given_gold_in_shortlist']:.4f}",
                f"{nat['acc@5']:.4f}",
                f"{nat['mrr']:.4f}",
                f"{nat['shortlist_recall']:.4f}",
                valid_frac,
                nan_count,
                f"{ora['acc@1']:.4f}",
            ]
        )
    with (out / "conservative_secondpass_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "rerank_checkpoint",
                "fine_tune_mode",
                "best_epoch",
                "acc@1_nat",
                "cond_in_K",
                "acc@5_nat",
                "mrr_nat",
                "gold_in_shortlist_rate",
                "rerank_train_valid_fraction",
                "nan_or_inf_batch_count",
                "acc@1_oracle",
            ]
        )
        w.writerows(table_rows)
    _write_table_md(
        out / "conservative_secondpass_table.md",
        ["variant", "ckpt", "mode", "best_ep", "nat@1", "cond_K", "nat@5", "mrr", "gold_in_K", "train_valid", "nan_inf", "oracle@1"],
        table_rows,
    )

    _plot_variant_curves(
        out / "conservative_secondpass_curves.png",
        {name: row["rows"] for name, row in trained.items()},
    )

    inter = by_label["coadapted_intermediate_best"]["eval_natural_shortlist"]
    naive = by_label["naive_secondpass_rerank"]["eval_natural_shortlist"]
    best_nat = best_variant["eval_natural_shortlist"]
    interp = f"""# Conservative second-pass interpretation

1. Which conservative variant best preserves `0.1154`?
   - best conservative variant: **{best_variant_label}**
   - coadapted intermediate (`0.1154` reference): **{inter['acc@1']:.4f}**
   - best conservative variant: **{best_nat['acc@1']:.4f}**
   - gap to 0.1154: **{best_nat['acc@1'] - inter['acc@1']:+.4f}**

2. Does any variant beat naive second-pass (`0.1090`)?
   - naive second-pass: **{naive['acc@1']:.4f}**
   - best conservative: **{best_nat['acc@1']:.4f}**
   - delta vs naive: **{best_nat['acc@1'] - naive['acc@1']:+.4f}**

3. Does any variant match or exceed `0.1154`?
   - max conservative natural Acc@1: **{best_nat['acc@1']:.4f}**
   - threshold: **{inter['acc@1']:.4f}**
   - matched_or_exceeded: **{str(best_nat['acc@1'] >= inter['acc@1'])}**

4. How do `cond_in_K`, `Acc@5`, and `MRR` behave?
   - intermediate (`coadapted_intermediate_best`): `cond_in_K={inter['rerank_acc_given_gold_in_shortlist']:.4f}`, `Acc@5={inter['acc@5']:.4f}`, `MRR={inter['mrr']:.4f}`
   - naive second-pass: `cond_in_K={naive['rerank_acc_given_gold_in_shortlist']:.4f}`, `Acc@5={naive['acc@5']:.4f}`, `MRR={naive['mrr']:.4f}`
   - best conservative (`{best_variant_label}`): `cond_in_K={best_nat['rerank_acc_given_gold_in_shortlist']:.4f}`, `Acc@5={best_nat['acc@5']:.4f}`, `MRR={best_nat['mrr']:.4f}`
"""
    (out / "conservative_secondpass_interpretation.md").write_text(interp, encoding="utf-8")

    combined_rows_blocks: list[dict[str, Any]] = [
        eval_pair("corrected_baseline_reference", base_coarse, reference_rerank),
        eval_pair("improved_shortlist_plus_reference_rerank", improved_coarse, reference_rerank),
        eval_pair("coadapted_intermediate_best", reselected_coarse, first_retrained),
        eval_pair("naive_secondpass_rerank", reselected_coarse, naive_second),
    ]
    for row in conservative_rows:
        combined_rows_blocks.append(row)
    combined_rows_blocks.append(eval_pair("best_conservative_secondpass_variant", reselected_coarse, Path(by_label[best_variant_label]["rerank_checkpoint"])))
    # Remove accidental duplicate label rows while preserving order.
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for r in combined_rows_blocks:
        if r["label"] in seen:
            continue
        seen.add(r["label"])
        merged.append(r)

    combined = {
        "selection_context": {
            "primary_metric": "natural two-stage full-scene validation Acc@1",
            "best_conservative_variant": best_variant_label,
            "pythonhashseed": hash_seed or None,
        },
        "pipelines": merged,
    }
    (out / "shortlist_rerank_combined_results_secondpass.json").write_text(
        json.dumps(combined, indent=2, default=str),
        encoding="utf-8",
    )

    combined_table_rows: list[list[Any]] = []
    for block in merged:
        nat = block["eval_natural_shortlist"]
        ora = block["eval_oracle_shortlist"]
        combined_table_rows.append(
            [
                block["label"],
                _safe_name(block["coarse_checkpoint"]),
                _safe_name(block["rerank_checkpoint"]),
                f"{nat['acc@1']:.4f}",
                f"{nat['rerank_acc_given_gold_in_shortlist']:.4f}",
                f"{nat['acc@5']:.4f}",
                f"{nat['mrr']:.4f}",
                f"{nat['shortlist_recall']:.4f}",
                f"{ora['acc@1']:.4f}",
            ]
        )
    with (out / "shortlist_rerank_combined_table_secondpass.csv").open("w", newline="", encoding="utf-8") as f:
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
        w.writerows(combined_table_rows)
    _write_table_md(
        out / "shortlist_rerank_combined_table_secondpass.md",
        ["pipeline", "coarse", "rerank", "nat@1", "cond_K", "nat@5", "mrr", "gold_in_K", "oracle@1"],
        combined_table_rows,
    )
    _plot_combined(
        out / "shortlist_rerank_main_figure_secondpass.png",
        [row[0][:30] for row in combined_table_rows],
        [float(row[3]) for row in combined_table_rows],
    )

    best_combined_nat = by_label[best_variant_label]["eval_natural_shortlist"]["acc@1"]
    combined_interp = f"""# Combined interpretation (conservative second-pass)

1. Does any conservative second-pass variant preserve the `0.1154` intermediate gain?
   - intermediate (`coadapted_intermediate_best`): **{inter['acc@1']:.4f}**
   - best conservative variant (`{best_variant_label}`): **{best_combined_nat:.4f}**
   - preserved: **{str(best_combined_nat >= inter['acc@1'])}**

2. If not, how close does it get?
   - gap to `0.1154`: **{best_combined_nat - inter['acc@1']:+.4f}**

3. Which conservative strategy works best?
   - best strategy by natural Acc@1: **{best_variant_label}**
   - value: **{best_combined_nat:.4f}**

4. New dominant bottleneck after this phase
   - the shortlist-side co-adaptation gain exists, but second-pass reranker adaptation still determines whether that gain survives into the strict final system.
"""
    (out / "shortlist_rerank_interpretation_secondpass.md").write_text(combined_interp, encoding="utf-8")

    repro = out / "repro_commands.sh"
    repro.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cd "{ROOT}"
export PYTHONHASHSEED="{hash_seed or '42'}"
{py} scripts/run_conservative_secondpass_phase.py --stamp {stamp} --output-tag {args.output_tag} --device {dev_name} \\
  --fix-bundle "{fix_bundle}" --shortlist-bundle "{shortlist_bundle}" --rebalance-bundle "{rebalance_bundle}" \\
  --coadapt-bundle "{coadapt_bundle}"
# Eval-only rerun:
# {py} scripts/run_conservative_secondpass_phase.py --stamp {stamp} --output-tag {args.output_tag} --skip-train \\
#   --fix-bundle "{fix_bundle}" --shortlist-bundle "{shortlist_bundle}" --rebalance-bundle "{rebalance_bundle}" \\
#   --coadapt-bundle "{coadapt_bundle}"
""",
        encoding="utf-8",
    )
    repro.chmod(0o755)

    rb = out / "report_bundle"
    rb.mkdir(parents=True, exist_ok=True)
    bundle_reports = {
        "conservative_secondpass_plan.md": ROOT / "reports/conservative_secondpass_plan.md",
        "conservative_secondpass_protocol_note.md": ROOT / "reports/conservative_secondpass_protocol_note.md",
        "conservative_secondpass_summary.md": ROOT / "reports/conservative_secondpass_summary.md",
        "readme_conservative_secondpass_note.md": ROOT / "reports/readme_conservative_secondpass_note.md",
    }
    copied_bundle_reports: list[str] = []
    for name, src in bundle_reports.items():
        if _copy_if_exists(src, rb / name):
            copied_bundle_reports.append(name)
    (rb / "README.md").write_text(
        f"""# Conservative second-pass report bundle

| Artifact | Claim |
|----------|--------|
| `readme_conservative_secondpass_note.md` | README now records that the project has entered a gain-retention phase after co-adaptation validation. |
| `conservative_secondpass_table.*` + `conservative_secondpass_results.json` | Tests whether careful second-pass reranker tuning can preserve the `0.1154` co-adapted intermediate gain. |
| `conservative_secondpass_curves.png` | Shows natural Acc@1 and `cond_in_K` trajectories for conservative second-pass variants. |
| `shortlist_rerank_combined_table_secondpass.*` + `shortlist_rerank_combined_results_secondpass.json` | Corrected combined table compares baseline, co-adapted intermediate, naive second-pass, and conservative second-pass variants. |
| `shortlist_rerank_main_figure_secondpass.png` | Main figure for corrected combined natural Acc@1 after conservative second-pass tuning. |
| `conservative_secondpass_summary.md` | States whether conservative tuning preserves `0.1154` and what the next bottleneck is. |
| `generated_configs/*.yaml` | Exact low-LR / short-schedule / partial-freeze variant configs used. |
| `repro_commands.sh` | Exact command used to regenerate this bundle. |

Bundled notes:

{os.linesep.join(f"- `{name}`" for name in copied_bundle_reports) if copied_bundle_reports else "- root report files were not present when this bundle README was generated"}

Environment note:

- `PYTHONHASHSEED={hash_seed or 'unset'}`
""",
        encoding="utf-8",
    )

    _append_log(logs / "phase.log", f"Conservative second-pass phase complete -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
