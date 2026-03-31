#!/usr/bin/env python3
"""Long-train shortlist + rerank strengthening: protocols N/O rerank, focused coarse, combined eval.

Writes under outputs/<timestamp>_longtrain_shortlist_rerank/. See reports/long_train_shortlist_rerank_plan.md.
Does not run geometry phase D or hybrid router training.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_rerank_curves(path: Path, natural_rows: list[dict], oracle_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    def _epochs(rs: list[dict]) -> list[int]:
        return [int(r["epoch"]) for r in rs]

    if natural_rows:
        ax = axes[0, 0]
        ax.plot(_epochs(natural_rows), [r.get("train_loss_mean", 0) for r in natural_rows], label="N train loss")
        ax.set_title("Train loss (protocol N)")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    if oracle_rows:
        ax = axes[0, 1]
        ax.plot(_epochs(oracle_rows), [r.get("train_loss_mean", 0) for r in oracle_rows], label="O train loss", color="orange")
        ax.set_title("Train loss (protocol O)")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)

    ax = axes[1, 0]
    if natural_rows:
        ax.plot(
            _epochs(natural_rows),
            [r.get("val_natural_two_stage_acc@1", r.get("val_acc@1", 0)) for r in natural_rows],
            label="N val natural Acc@1",
        )
    if oracle_rows:
        ax.plot(
            _epochs(oracle_rows),
            [r.get("val_natural_two_stage_acc@1", r.get("val_acc@1", 0)) for r in oracle_rows],
            label="O val natural Acc@1",
            color="orange",
        )
    ax.set_title("Val natural two-stage Acc@1")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    if natural_rows:
        ax.plot(
            _epochs(natural_rows),
            [r.get("val_oracle_shortlist_acc@1", 0) for r in natural_rows],
            label="N val oracle Acc@1",
        )
    if oracle_rows:
        ax.plot(
            _epochs(oracle_rows),
            [r.get("val_oracle_shortlist_acc@1", 0) for r in oracle_rows],
            label="O val oracle Acc@1",
            color="orange",
        )
    ax.set_title("Val oracle-shortlist Acc@1")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_combined(path: Path, labels: list[str], values: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(labels))
    ax.bar(x, values, color="steelblue")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Acc@1 (natural shortlist, val)")
    ax.set_title("Combined longtrain — natural pipeline")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", type=str, default="")
    ap.add_argument("--output-tag", type=str, default="longtrain_shortlist_rerank")
    ap.add_argument("--epochs-rerank", type=int, default=12)
    ap.add_argument("--epochs-coarse", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from rag3d.datasets.collate import make_grounding_collate_fn
    from rag3d.datasets.referit3d import ReferIt3DManifestDataset
    from rag3d.evaluation.coarse_recall import eval_coarse_stage1_metrics
    from rag3d.evaluation.two_stage_rerank_metrics import (
        eval_by_candidate_load_bucket,
        eval_two_stage_inject_mode,
    )
    from rag3d.evaluation.two_stage_eval import load_coarse_model, load_two_stage_model
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
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    train_m = ROOT / "data/processed/train_manifest.jsonl"
    val_m = ROOT / "data/processed/val_manifest.jsonl"
    mcfg = load_yaml_config(ROOT / "configs/model/relation_aware.yaml", ROOT)
    feat_dim = int(mcfg["object_dim"])

    base_coarse = ROOT / "outputs/checkpoints_stage1/coarse_geom_recall_last.pt"
    if not base_coarse.is_file():
        base_coarse = ROOT / "outputs/checkpoints_stage1/coarse_geom_ce_last.pt"
    base_rerank = ROOT / "outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt"
    if not base_rerank.is_file():
        base_rerank = ROOT / "outputs/checkpoints_rerank/rerank_full_k10_last.pt"

    if not train_m.is_file() or not val_m.is_file():
        print("Missing manifests", train_m, val_m)
        return 1
    if not base_coarse.is_file() or not base_rerank.is_file():
        print("Missing baseline checkpoints. Need coarse + rerank stage-1 last.pt, e.g.:")
        print(" ", ROOT / "outputs/checkpoints_stage1/coarse_geom_recall_last.pt")
        print(" ", ROOT / "outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt")
        return 1

    common_rerank: dict[str, Any] = {
        "model": "relation_aware",
        "dataset_config": "configs/dataset/referit3d.yaml",
        "coarse_model": "coarse_geom",
        "coarse_checkpoint": str(base_coarse),
        "fine_init_checkpoint": str(base_rerank),
        "rerank_k": 10,
        "parser_mode": "structured",
        "parser_cache_dir": "data/parser_cache/longtrain_rerank",
        "batch_size": 16,
        "lr": 0.0001,
        "weight_decay": 0.01,
        "seed": 42,
        "num_workers": 0,
        "device": device,
        "debug_max_batches": None,
        "epochs": int(args.epochs_rerank),
        "checkpoint_dir": str(ck),
        "loss": {"hard_negative": {"enabled": True, "margin": 0.25, "lambda_hinge": 0.4}},
        "selection_margin_thresh": 0.15,
    }

    if not args.skip_train:
        _dump_yaml(
            gc / "rerank_longtrain_oracle.yaml",
            {
                **common_rerank,
                "metrics_file": str(out / "metrics_rerank_oracle.jsonl"),
                "run_name": "rerank_longtrain_oracle",
                "shortlist_train_inject_gold": True,
            },
        )
        _dump_yaml(
            gc / "rerank_longtrain_natural.yaml",
            {
                **common_rerank,
                "metrics_file": str(out / "metrics_rerank_natural.jsonl"),
                "run_name": "rerank_longtrain_natural",
                "shortlist_train_inject_gold": False,
            },
        )
        _run(
            [py, str(ROOT / "scripts/train_two_stage_rerank.py"), "--config", str(gc / "rerank_longtrain_oracle.yaml")],
            logs / "train_rerank_oracle.log",
        )
        _run(
            [py, str(ROOT / "scripts/train_two_stage_rerank.py"), "--config", str(gc / "rerank_longtrain_natural.yaml")],
            logs / "train_rerank_natural.log",
        )

        coarse_yaml: dict[str, Any] = {
            "model": "relation_aware",
            "dataset_config": "configs/dataset/referit3d.yaml",
            "coarse_model": "coarse_geom",
            "checkpoint_dir": str(ck),
            "metrics_file": str(out / "metrics_coarse_focused.jsonl"),
            "run_name": "coarse_focused_hardneg_longtrain",
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
                "ranking_margin": {"enabled": True, "margin": 0.2, "lambda": 0.2},
                "spatial_nearby_hinge": {
                    "enabled": True,
                    "margin": 0.2,
                    "lambda": 0.15,
                    "max_neighbors": 4,
                },
                "hard_negative": {"enabled": True, "margin": 0.3, "lambda_hinge": 0.75},
            },
            "val_coarse_recall_ks": [5, 10, 20, 40],
            "val_two_stage_selection": {
                "enabled": True,
                "reference_rerank_checkpoint": str(base_rerank.resolve()),
                "model_config": "configs/model/relation_aware.yaml",
                "parser_mode": "structured",
                "parser_cache_dir": "data/parser_cache/longtrain_selection",
                "rerank_k": 10,
                "margin_thresh": 0.15,
            },
        }
        _dump_yaml(gc / "coarse_focused_longtrain.yaml", coarse_yaml)
        _run(
            [
                py,
                str(ROOT / "scripts/train_coarse_stage1.py"),
                "--config",
                str(gc / "coarse_focused_longtrain.yaml"),
                "--init-checkpoint",
                str(base_coarse),
            ],
            logs / "train_coarse_focused.log",
        )

    rerank_nat_best = ck / "rerank_longtrain_natural_best_natural_two_stage.pt"
    rerank_nat_last = ck / "rerank_longtrain_natural_last.pt"
    rerank_oracle_best = ck / "rerank_longtrain_oracle_best_natural_two_stage.pt"
    rerank_oracle_last = ck / "rerank_longtrain_oracle_last.pt"
    coarse_best = ck / "coarse_focused_hardneg_longtrain_best_pipeline_natural.pt"
    coarse_last = ck / "coarse_focused_hardneg_longtrain_last.pt"
    new_coarse = coarse_best if coarse_best.is_file() else coarse_last

    nat_rows = _read_jsonl_metrics(out / "metrics_rerank_natural.jsonl")
    ora_rows = _read_jsonl_metrics(out / "metrics_rerank_oracle.jsonl")
    oracle_bundle = {
        "protocol_N_natural_train": {
            "metrics_path": str(out / "metrics_rerank_natural.jsonl"),
            "epochs": nat_rows,
            "best_checkpoint": str(rerank_nat_best) if rerank_nat_best.is_file() else str(rerank_nat_last),
        },
        "protocol_O_oracle_train": {
            "metrics_path": str(out / "metrics_rerank_oracle.jsonl"),
            "epochs": ora_rows,
            "best_checkpoint": str(rerank_oracle_best) if rerank_oracle_best.is_file() else str(rerank_oracle_last),
        },
        "protocol_M_mixed": "skipped (would need per-batch oracle mix probability in the model forward; not added in this phase)",
    }
    (out / "oracle_reranker_results_longtrain.json").write_text(json.dumps(oracle_bundle, indent=2, default=str), encoding="utf-8")

    n_last = nat_rows[-1] if nat_rows else {}
    o_last = ora_rows[-1] if ora_rows else {}
    n0 = nat_rows[0] if nat_rows else {}
    o0 = ora_rows[0] if ora_rows else {}
    rrows = [
        [
            "N_natural_train",
            f"{n0.get('val_natural_two_stage_acc@1', n0.get('val_acc@1', ''))}",
            f"{n_last.get('val_natural_two_stage_acc@1', n_last.get('val_acc@1', ''))}",
            f"{n_last.get('val_oracle_shortlist_acc@1', '')}",
            f"{n_last.get('val_natural_cond_acc_in_shortlist', '')}",
            f"{n_last.get('val_natural_two_stage_mrr', '')}",
        ],
        [
            "O_oracle_train",
            f"{o0.get('val_natural_two_stage_acc@1', o0.get('val_acc@1', ''))}",
            f"{o_last.get('val_natural_two_stage_acc@1', o_last.get('val_acc@1', ''))}",
            f"{o_last.get('val_oracle_shortlist_acc@1', '')}",
            f"{o_last.get('val_natural_cond_acc_in_shortlist', '')}",
            f"{o_last.get('val_natural_two_stage_mrr', '')}",
        ],
    ]
    with (out / "oracle_reranker_table_longtrain.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "acc@1_nat_epoch0", "acc@1_nat_last", "acc@1_oracle_last", "cond_in_K_last", "mrr_nat_last"])
        w.writerows(rrows)
    _write_table_md(
        out / "oracle_reranker_table_longtrain.md",
        ["protocol", "nat_e0", "nat_last", "oracle_last", "cond_K", "mrr"],
        rrows,
    )
    _plot_rerank_curves(out / "oracle_reranker_curves.png", nat_rows, ora_rows)

    best_nat_n = max((float(r.get("val_natural_two_stage_acc@1", r.get("val_acc@1", 0))) for r in nat_rows), default=0.0)
    best_nat_o = max((float(r.get("val_natural_two_stage_acc@1", r.get("val_acc@1", 0))) for r in ora_rows), default=0.0)
    (out / "oracle_reranker_interpretation_longtrain.md").write_text(
        f"""# Long-train reranker interpretation

## 1. Does rerank improve materially with longer training?

- Compare **first vs last epoch** `val_natural_two_stage_acc@1` in `oracle_reranker_table_longtrain.csv` for protocols **N** and **O**.
- Best natural Acc@1 seen across epochs (N): **{best_nat_n:.4f}**, (O): **{best_nat_o:.4f}**.

## 2. Oracle-trained vs natural-trained reranker

- Under **natural** val, protocol **O** vs **N** last-epoch natural Acc@1 shows whether oracle-conditioned training helps **deployed** performance.
- **Oracle val Acc@1** columns isolate rerank headroom when gold ∈ K.

## 3. Which protocol best supports the real natural pipeline?

- Primary row: **higher `acc@1_nat_last` under protocol N** ⇒ natural training better aligned with deployment; if **O** wins on natural val, oracle training still transferred.

## 4. Is rerank still a bottleneck after longer training?

- Compare `val_oracle_shortlist_acc@1` vs `val_natural_two_stage_acc@1`. A large gap ⇒ **shortlist/retrieval** still dominates; low conditional `cond_in_K` ⇒ **rerank** still weak even when gold is in K.
""",
        encoding="utf-8",
    )

    dev = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    val_ds = ReferIt3DManifestDataset(val_m)
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=make_grounding_collate_fn(feat_dim, attach_features=True),
    )
    parser = CachedParser(StructuredRuleParser(), ROOT / "data/parser_cache/longtrain_combined/structured")

    def eval_pair(label: str, coarse_pt: Path, twostage_pt: Path, rerank_k: int = 10) -> dict[str, Any]:
        model = load_two_stage_model(
            mcfg,
            coarse_pt,
            twostage_pt,
            rerank_k,
            dev,
            "coarse_geom",
            fine_only_from_checkpoint=True,
        )
        nat = eval_two_stage_inject_mode(model, val_loader, dev, parser, 0.15, False)
        ora = eval_two_stage_inject_mode(model, val_loader, dev, parser, 0.15, True)
        buck_nat = eval_by_candidate_load_bucket(model, val_loader, dev, parser, 0.15, False)
        buck_hi = buck_nat.get("high", {})
        buck_lo = buck_nat.get("low", {})
        return {
            "label": label,
            "coarse_checkpoint": str(coarse_pt),
            "rerank_checkpoint": str(twostage_pt),
            "eval_natural_shortlist": nat,
            "eval_oracle_shortlist": ora,
            "bucket_natural": {"low_candidate_load": buck_lo, "high_candidate_load": buck_hi},
        }

    results: dict[str, Any] = {"runs": []}

    def _try(label: str, c_pt: Path, t_pt: Path) -> None:
        if not c_pt.is_file():
            _append_log(logs / "phase.log", f"skip {label}: missing coarse {c_pt}")
            return
        if not t_pt.is_file():
            _append_log(logs / "phase.log", f"skip {label}: missing two-stage {t_pt}")
            return
        results["runs"].append(eval_pair(label, c_pt, t_pt))

    def _resolve_ckpt(best_pt: Path, last_pt: Path) -> tuple[str | None, Path | None]:
        if best_pt.is_file():
            return "best", best_pt
        if last_pt.is_file():
            return "last", last_pt
        return None, None

    n_kind, n_ckpt = _resolve_ckpt(rerank_nat_best, rerank_nat_last)
    o_kind, o_ckpt = _resolve_ckpt(rerank_oracle_best, rerank_oracle_last)

    _try("baseline_reference", base_coarse, base_rerank)
    if n_ckpt is not None:
        _try(f"rerank_N_{n_kind}", base_coarse, n_ckpt)
    if o_ckpt is not None:
        _try(f"rerank_O_{o_kind}", base_coarse, o_ckpt)
    if rerank_nat_last.is_file() and n_kind != "last":
        _try("rerank_N_last", base_coarse, rerank_nat_last)
    if rerank_oracle_last.is_file() and o_kind != "last":
        _try("rerank_O_last", base_coarse, rerank_oracle_last)
    _try("focused_coarse_plus_old_rerank", Path(new_coarse), base_rerank)
    if o_ckpt is not None:
        _try(f"focused_coarse_plus_rerank_O_{o_kind}", Path(new_coarse), o_ckpt)

    (out / "shortlist_rerank_combined_results_longtrain.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    crows: list[list[Any]] = []
    for block in results["runs"]:
        lab = block["label"]
        en = block["eval_natural_shortlist"]
        eo = block["eval_oracle_shortlist"]
        hi = block["bucket_natural"].get("high_candidate_load", {})
        lo = block["bucket_natural"].get("low_candidate_load", {})
        crows.append(
            [
                lab,
                Path(block.get("coarse_checkpoint", "")).name,
                Path(block.get("rerank_checkpoint", "")).name,
                f"{en['acc@1']:.4f}",
                f"{en['acc@5']:.4f}",
                f"{en['mrr']:.4f}",
                f"{en['shortlist_recall']:.4f}",
                f"{en['rerank_acc_given_gold_in_shortlist']:.4f}",
                f"{eo['acc@1']:.4f}",
                f"{hi.get('acc@1', '')}",
                f"{lo.get('acc@1', '')}",
            ]
        )
    with (out / "shortlist_rerank_combined_table_longtrain.csv").open("w", newline="", encoding="utf-8") as f:
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
        w.writerows(crows)
    if crows:
        _write_table_md(
            out / "shortlist_rerank_combined_table_longtrain.md",
            ["pipeline", "coarse_ckpt", "rerank_ckpt", "acc@1_nat", "acc@5", "mrr", "recall_K", "cond_K", "oracle", "hi", "lo"],
            crows,
        )
        _plot_combined(
            out / "shortlist_rerank_main_figure_longtrain.png",
            [r[0][:20] for r in crows],
            [float(r[3]) for r in crows],
        )

    base_recall = eval_coarse_stage1_metrics(
        load_coarse_model(mcfg, base_coarse, dev, "coarse_geom"),
        val_loader,
        dev,
        0.15,
        ks=(5, 10, 20, 40),
    )
    new_recall: dict[str, Any] = {}
    if new_coarse.is_file():
        new_recall = eval_coarse_stage1_metrics(
            load_coarse_model(mcfg, new_coarse, dev, "coarse_geom"),
            val_loader,
            dev,
            0.15,
            ks=(5, 10, 20, 40),
        )
    retr = {
        "baseline_coarse_val": {k: base_recall[k] for k in base_recall if str(k).startswith("recall@") or k == "n"},
        "focused_coarse_val": new_recall,
        "coarse_checkpoint_used": str(new_coarse),
    }
    (out / "shortlist_retrieval_results_longtrain.json").write_text(json.dumps(retr, indent=2, default=str), encoding="utf-8")
    rrows = [["checkpoint", "recall@5", "recall@10", "recall@20", "recall@40"]]
    rrows.append(
        [
            "baseline",
            f"{base_recall.get('recall@5', 0):.4f}",
            f"{base_recall.get('recall@10', 0):.4f}",
            f"{base_recall.get('recall@20', 0):.4f}",
            f"{base_recall.get('recall@40', 0):.4f}",
        ]
    )
    if new_recall:
        rrows.append(
            [
                "focused_longtrain",
                f"{new_recall.get('recall@5', 0):.4f}",
                f"{new_recall.get('recall@10', 0):.4f}",
                f"{new_recall.get('recall@20', 0):.4f}",
                f"{new_recall.get('recall@40', 0):.4f}",
            ]
        )
    with (out / "shortlist_retrieval_table_longtrain.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rrows)
    _write_table_md(out / "shortlist_retrieval_table_longtrain.md", rrows[0], rrows[1:])

    (out / "shortlist_retrieval_interpretation_longtrain.md").write_text(
        """# Focused coarse retrieval (longtrain) — interpretation

1. **Recall@10 / @20:** compare `baseline` vs `focused_longtrain` in `shortlist_retrieval_table_longtrain.csv`.
2. **Downstream natural two-stage:** compare `focused_coarse_plus_old_rerank` vs `baseline_reference` in `shortlist_rerank_combined_results_longtrain.json`.
3. **Oracle upper bound:** compare `acc@1_oracle` for `focused_coarse_plus_old_rerank` vs `baseline_reference`.
4. **Stability:** if Recall moves but natural Acc@1 does not, keep retrieval change only if the end-to-end goal is long-term curriculum — otherwise prefer the checkpoint with higher **val_pipeline_natural_acc@1** during coarse training.
""",
        encoding="utf-8",
    )

    base_nat = 0.0
    best_nat = 0.0
    best_lab = ""
    if results["runs"]:
        base_nat = float(results["runs"][0]["eval_natural_shortlist"]["acc@1"])
        best_row = max(results["runs"], key=lambda r: float(r["eval_natural_shortlist"]["acc@1"]))
        best_nat = float(best_row["eval_natural_shortlist"]["acc@1"])
        best_lab = str(best_row.get("label", ""))
    (out / "shortlist_rerank_interpretation_longtrain.md").write_text(
        f"""# Combined longtrain interpretation

- `baseline_reference` is the historical baseline checkpoint pair.
- Explicit `rerank_N_*` / `rerank_O_*` rows isolate the reranker choice under the same baseline coarse checkpoint.
- `focused_coarse_plus_old_rerank` isolates the focused coarse checkpoint with the old reranker.
- `focused_coarse_plus_rerank_O_*` tests the strongest available oracle-trained reranker with the focused coarse checkpoint.
- Baseline natural val Acc@1 ≈ **{base_nat:.4f}**; best pipeline in this bundle ≈ **{best_nat:.4f}** (`{best_lab}`).

## Bottleneck readout

1. Compare `rerank_O_best` vs `baseline_reference` to test whether the actually strongest reranker improves the deployed natural pipeline.
2. **Natural Acc@1** (main) still far below **oracle** columns ⇒ mixed bottleneck (retrieval + rerank).
3. Next limiting factor: if conditional in-K stays low ⇒ rerank; if recall@K is low ⇒ shortlist.
""",
        encoding="utf-8",
    )

    repro = out / "repro_commands.sh"
    repro.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cd "{ROOT}"
{py} scripts/run_longtrain_shortlist_rerank_phase.py --stamp {stamp} --device {device} \\
  --output-tag {args.output_tag} --epochs-rerank {args.epochs_rerank} --epochs-coarse {args.epochs_coarse}
# Eval-only after training:
# {py} scripts/run_longtrain_shortlist_rerank_phase.py --stamp {stamp} --output-tag {args.output_tag} --skip-train
""",
        encoding="utf-8",
    )
    repro.chmod(0o755)

    rb = out / "report_bundle"
    rb.mkdir(parents=True, exist_ok=True)
    (rb / "README.md").write_text(
        """# Longtrain shortlist + rerank report bundle

| Artifact | Claim |
|----------|--------|
| `oracle_reranker_results_longtrain.json` + curves | Longer N vs O rerank training; **natural two-stage val** drives `*_best_natural_two_stage.pt`. |
| `shortlist_retrieval_*_longtrain*` | Focused three-term coarse objective; Recall@K vs baseline. |
| `shortlist_rerank_combined_*_longtrain*` | Explicit downstream comparison of `baseline_reference`, `rerank_N_best`, `rerank_O_best`, and focused-coarse combinations. |

Re-run via `repro_commands.sh` in the parent directory.
""",
        encoding="utf-8",
    )

    _append_log(logs / "phase.log", f"Longtrain phase complete -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
