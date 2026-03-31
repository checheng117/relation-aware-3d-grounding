#!/usr/bin/env python3
"""Shortlist-aware retrieval + oracle-shortlist reranker upgrade phase.

Writes under outputs/<timestamp>_shortlist_rerank_upgrade/. See reports/shortlist_rerank_upgrade_plan.md.
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


def _read_jsonl_last_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.is_file():
        return {}
    last = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            continue
    return last


def _write_table_md(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_combined(path: Path, labels: list[str], values: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(labels))
    ax.bar(x, values, color="teal")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Acc@1 (natural shortlist)")
    ax.set_title("Combined pipeline — val (natural eval)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", type=str, default="")
    ap.add_argument("--epochs-rerank", type=int, default=4)
    ap.add_argument("--epochs-coarse", type=int, default=3)
    ap.add_argument("--epochs-rerank2", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--skip-train", action="store_true", help="Only eval using existing checkpoints in output dir")
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
    out = ROOT / "outputs" / f"{stamp}_shortlist_rerank_upgrade"
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

    base_coarse = ROOT / "outputs/checkpoints_stage1/coarse_geom_recall_last.pt"
    base_rerank = ROOT / "outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt"
    train_m = ROOT / "data/processed/train_manifest.jsonl"
    val_m = ROOT / "data/processed/val_manifest.jsonl"
    mcfg = load_yaml_config(ROOT / "configs/model/relation_aware.yaml", ROOT)
    feat_dim = int(mcfg["object_dim"])

    if not train_m.is_file() or not val_m.is_file():
        print("Missing manifests", train_m, val_m)
        return 1
    if not base_coarse.is_file() or not base_rerank.is_file():
        print("Missing baseline checkpoints", base_coarse, base_rerank)
        return 1

    common_rerank = {
        "model": "relation_aware",
        "dataset_config": "configs/dataset/referit3d.yaml",
        "coarse_model": "coarse_geom",
        "coarse_checkpoint": str(base_coarse),
        "fine_init_checkpoint": str(base_rerank),
        "rerank_k": 10,
        "parser_mode": "structured",
        "parser_cache_dir": "data/parser_cache/shortlist_upgrade",
        "batch_size": 16,
        "lr": 0.0001,
        "weight_decay": 0.01,
        "seed": 42,
        "num_workers": 0,
        "device": device,
        "debug_max_batches": None,
        "loss": {"hard_negative": {"enabled": True, "margin": 0.25, "lambda_hinge": 0.4}},
    }

    if not args.skip_train:
        _dump_yaml(
            gc / "rerank_oracle_train.yaml",
            {
                **common_rerank,
                "checkpoint_dir": str(ck),
                "metrics_file": str(out / "metrics_rerank_oracle.jsonl"),
                "run_name": "rerank_train_oracle",
                "epochs": int(args.epochs_rerank),
                "shortlist_train_inject_gold": True,
            },
        )
        _dump_yaml(
            gc / "rerank_natural_train.yaml",
            {
                **common_rerank,
                "checkpoint_dir": str(ck),
                "metrics_file": str(out / "metrics_rerank_natural.jsonl"),
                "run_name": "rerank_train_natural",
                "epochs": int(args.epochs_rerank),
                "shortlist_train_inject_gold": False,
            },
        )
        _run(
            [py, str(ROOT / "scripts/train_two_stage_rerank.py"), "--config", str(gc / "rerank_oracle_train.yaml")],
            logs / "train_rerank_oracle.log",
        )
        _run(
            [py, str(ROOT / "scripts/train_two_stage_rerank.py"), "--config", str(gc / "rerank_natural_train.yaml")],
            logs / "train_rerank_natural.log",
        )

        coarse_yaml = {
            "model": "relation_aware",
            "dataset_config": "configs/dataset/referit3d.yaml",
            "coarse_model": "coarse_geom",
            "checkpoint_dir": str(ck),
            "metrics_file": str(out / "metrics_coarse_upgrade.jsonl"),
            "run_name": "coarse_shortlist_aware",
            "epochs": int(args.epochs_coarse),
            "batch_size": 16,
            "lr": 0.00005,
            "weight_decay": 0.01,
            "seed": 42,
            "num_workers": 0,
            "device": device,
            "mode": "real",
            "loss": {
                "candidate_load_weight": {"enabled": True},
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
        }
        _dump_yaml(gc / "coarse_shortlist_upgrade.yaml", coarse_yaml)
        _run(
            [
                py,
                str(ROOT / "scripts/train_coarse_stage1.py"),
                "--config",
                str(gc / "coarse_shortlist_upgrade.yaml"),
                "--init-checkpoint",
                str(base_coarse),
            ],
            logs / "train_coarse_upgrade.log",
        )

        new_coarse = ck / "coarse_shortlist_aware_last.pt"
        _dump_yaml(
            gc / "rerank_newcoarse_oracle.yaml",
            {
                **common_rerank,
                "coarse_checkpoint": str(new_coarse),
                "fine_init_checkpoint": str(ck / "rerank_train_oracle_last.pt"),
                "checkpoint_dir": str(ck),
                "metrics_file": str(out / "metrics_rerank_newc_oracle.jsonl"),
                "run_name": "rerank_newcoarse_oracle",
                "epochs": int(args.epochs_rerank2),
                "shortlist_train_inject_gold": True,
            },
        )
        _dump_yaml(
            gc / "rerank_newcoarse_natural.yaml",
            {
                **common_rerank,
                "coarse_checkpoint": str(new_coarse),
                "fine_init_checkpoint": str(ck / "rerank_train_natural_last.pt"),
                "checkpoint_dir": str(ck),
                "metrics_file": str(out / "metrics_rerank_newc_natural.jsonl"),
                "run_name": "rerank_newcoarse_natural",
                "epochs": int(args.epochs_rerank2),
                "shortlist_train_inject_gold": False,
            },
        )
        _run(
            [
                py,
                str(ROOT / "scripts/train_two_stage_rerank.py"),
                "--config",
                str(gc / "rerank_newcoarse_oracle.yaml"),
            ],
            logs / "train_rerank_newcoarse_oracle.log",
        )
        _run(
            [
                py,
                str(ROOT / "scripts/train_two_stage_rerank.py"),
                "--config",
                str(gc / "rerank_newcoarse_natural.yaml"),
            ],
            logs / "train_rerank_newcoarse_natural.log",
        )

    dev = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    val_ds = ReferIt3DManifestDataset(val_m)
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=make_grounding_collate_fn(feat_dim, attach_features=True),
    )
    parser = CachedParser(StructuredRuleParser(), ROOT / "data/parser_cache/shortlist_upgrade/structured")

    def eval_pair(
        label: str,
        coarse_pt: Path,
        twostage_pt: Path,
        rerank_k: int = 10,
    ) -> dict[str, Any]:
        model = load_two_stage_model(mcfg, coarse_pt, twostage_pt, rerank_k, dev, "coarse_geom")
        nat = eval_two_stage_inject_mode(model, val_loader, dev, parser, 0.15, False)
        ora = eval_two_stage_inject_mode(model, val_loader, dev, parser, 0.15, True)
        buck_nat = eval_by_candidate_load_bucket(model, val_loader, dev, parser, 0.15, False)
        buck_hi = buck_nat.get("high", {})
        buck_lo = buck_nat.get("low", {})
        return {
            "label": label,
            "eval_natural_shortlist": nat,
            "eval_oracle_shortlist": ora,
            "bucket_natural": {"low_candidate_load": buck_lo, "high_candidate_load": buck_hi},
        }

    results_oracle: dict[str, Any] = {"runs": []}
    new_coarse = ck / "coarse_shortlist_aware_last.pt"

    def _try_eval(label: str, c_pt: Path, t_pt: Path) -> None:
        if not t_pt.is_file():
            _append_log(logs / "phase.log", f"skip eval {label}: missing {t_pt}")
            return
        results_oracle["runs"].append(eval_pair(label, c_pt, t_pt))

    _try_eval("A_baseline_pipeline", base_coarse, base_rerank)
    _try_eval("B_rerank_trained_oracle_protocol", base_coarse, ck / "rerank_train_oracle_last.pt")
    _try_eval("B_rerank_trained_natural_protocol", base_coarse, ck / "rerank_train_natural_last.pt")
    if new_coarse.is_file():
        _try_eval(
            "C_upgraded_coarse_rerank_natural_finetuned",
            new_coarse,
            ck / "rerank_newcoarse_natural_last.pt",
        )
        _try_eval(
            "D_upgraded_coarse_rerank_oracle_finetuned",
            new_coarse,
            ck / "rerank_newcoarse_oracle_last.pt",
        )

    (out / "oracle_reranker_results.json").write_text(json.dumps(results_oracle, indent=2), encoding="utf-8")

    rows = []
    for block in results_oracle["runs"]:
        lab = block["label"]
        en = block["eval_natural_shortlist"]
        eo = block["eval_oracle_shortlist"]
        rows.append(
            [
                lab,
                f"{en['acc@1']:.4f}",
                f"{en['acc@5']:.4f}",
                f"{en['mrr']:.4f}",
                f"{en['shortlist_recall']:.4f}",
                f"{en['rerank_acc_given_gold_in_shortlist']:.4f}",
                f"{eo['acc@1']:.4f}",
                f"{eo['mrr']:.4f}",
            ]
        )
    with (out / "oracle_reranker_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "acc@1_nat",
                "acc@5_nat",
                "mrr_nat",
                "shortlist_recall_nat",
                "rerank_acc|in_K_nat",
                "acc@1_oracle_eval",
                "mrr_oracle_eval",
            ]
        )
        w.writerows(rows)
    _write_table_md(
        out / "oracle_reranker_table.md",
        [
            "model",
            "acc@1_nat",
            "acc@5_nat",
            "mrr_nat",
            "recall_nat",
            "rerank|in_K_nat",
            "acc@1_oracle",
            "mrr_oracle",
        ],
        rows,
    )

    bl = results_oracle["runs"][0]["eval_natural_shortlist"]
    bo = results_oracle["runs"][0]["eval_oracle_shortlist"]
    ro = next((r for r in results_oracle["runs"] if r["label"] == "B_rerank_trained_oracle_protocol"), None)
    interp_o = f"""# Oracle reranker interpretation

## How weak is the baseline reranker under natural shortlist?

- **Natural** Acc@1 ≈ **{bl['acc@1']:.4f}**, shortlist recall ≈ **{bl['shortlist_recall']:.4f}**.
- **Conditional** rerank Acc@1 given gold ∈ K ≈ **{bl['rerank_acc_given_gold_in_shortlist']:.4f}** (still low ⇒ rerank head is weak even when retrieval succeeds).

## Headroom when shortlist is not the limiter

- **Oracle shortlist eval** on the **same** checkpoint: Acc@1 ≈ **{bo['acc@1']:.4f}**, MRR ≈ **{bo['mrr']:.4f}**.
- Gap (oracle − natural) Acc@1 ≈ **{bo['acc@1'] - bl['acc@1']:.4f}** ⇒ large **retrieval** contribution; remaining error under oracle is **rerank** headroom.

## Does oracle-shortlist training help?

"""
    if ro:
        rn = ro["eval_natural_shortlist"]
        roe = ro["eval_oracle_shortlist"]
        interp_o += f"- **Trained with gold in shortlist** — natural eval Acc@1 **{rn['acc@1']:.4f}**, oracle eval Acc@1 **{roe['acc@1']:.4f}**.\n"
    interp_o += """
## Investment takeaway

- If oracle-eval Acc@1 stays well above natural while conditional in-K accuracy stays moderate, **rerank** still deserves capacity; if oracle-eval is also low, prioritize **representation / fine architecture** or **K and features** before deeper routing.
"""
    (out / "oracle_reranker_interpretation.md").write_text(interp_o, encoding="utf-8")

    # Retrieval metrics
    coarse_metrics_path = out / "metrics_coarse_upgrade.jsonl"
    last_coarse = _read_jsonl_last_metrics(coarse_metrics_path)
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
        "baseline_coarse_val": {k: base_recall[k] for k in base_recall if k.startswith("recall@") or k == "n"},
        "upgraded_coarse_val_file_metrics": last_coarse,
        "upgraded_coarse_re_eval": new_recall,
    }
    (out / "shortlist_retrieval_results.json").write_text(json.dumps(retr, indent=2, default=str), encoding="utf-8")
    rrows = [["checkpoint", "recall@5", "recall@10", "recall@20", "recall@40"]]
    rrows.append(
        [
            "baseline_coarse",
            f"{base_recall.get('recall@5', 0):.4f}",
            f"{base_recall.get('recall@10', 0):.4f}",
            f"{base_recall.get('recall@20', 0):.4f}",
            f"{base_recall.get('recall@40', 0):.4f}",
        ]
    )
    if new_recall:
        rrows.append(
            [
                "upgraded_coarse",
                f"{new_recall.get('recall@5', 0):.4f}",
                f"{new_recall.get('recall@10', 0):.4f}",
                f"{new_recall.get('recall@20', 0):.4f}",
                f"{new_recall.get('recall@40', 0):.4f}",
            ]
        )
    with (out / "shortlist_retrieval_table.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rrows)
    _write_table_md(out / "shortlist_retrieval_table.md", rrows[0], rrows[1:])

    slice_lines = []
    for key in sorted(base_recall.get("stratified_recall_slices", {})):
        if "same_class_clutter" in key or "candidate_load::high" in key:
            slice_lines.append(
                f"- {key}: baseline={base_recall['stratified_recall_slices'][key]:.4f}"
                + (
                    f", upgraded={new_recall.get('stratified_recall_slices', {}).get(key, 'n/a')}"
                    if new_recall
                    else ""
                )
            )
    (out / "shortlist_retrieval_interpretation.md").write_text(
        f"""# Shortlist retrieval interpretation

## Recall@K (val, coarse only)

See `shortlist_retrieval_table.csv`. Upgraded coarse is finetuned from the baseline with same-class + spatial + ranking hinges and **load weighting**.

## Slices (from `eval_coarse_stage1_metrics`)

{chr(10).join(slice_lines) if slice_lines else "- (no clutter/high-load keys in this val run)"}

## Questions

1. **Does Recall@K improve?** Compare upgraded vs baseline columns.
2. **Oracle upper bound for two-stage** rises if recall@K rises **without** changing rerank; check `oracle_reranker_results` natural vs oracle columns for baseline vs upgraded coarse pipelines.
3. **Downstream:** If recall improves but pipeline Acc@1 (natural) barely moves, rerank is still limiting; if pipeline moves with recall, retrieval was binding.
""",
        encoding="utf-8",
    )

    # Combined
    combined = {"pipelines": []}
    for block in results_oracle["runs"]:
        combined["pipelines"].append(
            {
                "name": block["label"],
                "acc@1_natural": block["eval_natural_shortlist"]["acc@1"],
                "acc@1_oracle_eval": block["eval_oracle_shortlist"]["acc@1"],
                "acc@1_high_load": block["bucket_natural"].get("high_candidate_load", {}).get("acc@1"),
                "acc@1_low_load": block["bucket_natural"].get("low_candidate_load", {}).get("acc@1"),
            }
        )
    (out / "shortlist_rerank_combined_results.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    def _fmt(x: Any) -> str:
        return f"{float(x):.4f}" if isinstance(x, (int, float)) and x == x else ""

    crows = [
        [p["name"], f"{p['acc@1_natural']:.4f}", _fmt(p.get("acc@1_low_load")), _fmt(p.get("acc@1_high_load"))]
        for p in combined["pipelines"]
    ]
    with (out / "shortlist_rerank_combined_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pipeline", "acc@1_val_natural", "acc@1_low_load", "acc@1_high_load"])
        w.writerows(crows)
    _write_table_md(out / "shortlist_rerank_combined_table.md", ["pipeline", "acc@1_nat", "low_load", "high_load"], crows)

    labels = [p["name"][:18] for p in combined["pipelines"]]
    vals = [float(p["acc@1_natural"]) for p in combined["pipelines"]]
    _plot_combined(out / "shortlist_rerank_main_figure.png", labels, vals)

    base_nat = combined["pipelines"][0]["acc@1_natural"]
    best_nat = max(float(p["acc@1_natural"]) for p in combined["pipelines"])
    (out / "shortlist_rerank_interpretation.md").write_text(
        f"""# Combined pipeline interpretation

- **Baseline (A)** natural Acc@1 ≈ **{base_nat:.4f}**; best row in this run ≈ **{best_nat:.4f}**.
- **Component attribution:** compare **B_*** (rerank-only retrain, same coarse) vs **C/D_*** (new coarse + refit fine). Larger jump on B rows ⇒ rerank training mattered; larger jump only after C/D ⇒ retrieval mattered.
- **Regimes:** `low_load` / `high_load` proxy controlled vs crowded scenes (candidate_load tag).

## Bottleneck readout

1. If oracle-eval Acc@1 ≫ natural for all models, **retrieval** remains a major gap.
2. If conditional rerank given in-K is low in `oracle_reranker_results`, **rerank** remains weak.
3. **Both** can bind; this table shows which intervention moved **natural** Acc@1 more.
""",
        encoding="utf-8",
    )

    # Dataset note
    (out / "oracle_shortlist_rerank_dataset_note.md").write_text(
        """# Oracle-shortlist rerank dataset protocol

**Manifests:** Same `train_manifest.jsonl` / `val_manifest.jsonl` as standard NR3D processed data.

**Oracle shortlist at train:** `TwoStageCoarseRerankModel` gathers coarse top-K, then **replaces** the lowest coarse-scoring slot with the gold index when `shortlist_train_inject_gold: true`. Distractors remain **real** coarse top-(K-1) neighbors plus gold.

**Natural shortlist at train:** Pure coarse top-K (gold may be absent); CE + hinge on **full-scene** scattered logits unchanged.

**Eval modes:** `inject_gold_in_shortlist=False` (deploy-like) vs `True` (oracle ceiling for rerank isolation).

**K:** Primary experiments use **K=10** aligned with existing `rerank_k10_stage1` artifacts.
""",
        encoding="utf-8",
    )

    # repro + report bundle
    repro = out / "repro_commands.sh"
    repro.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cd "{ROOT}"
# Full phase (train + eval)
{py} scripts/run_shortlist_rerank_upgrade_phase.py --stamp {stamp} --device {device} \\
  --epochs-rerank {args.epochs_rerank} --epochs-coarse {args.epochs_coarse} --epochs-rerank2 {args.epochs_rerank2}
# Eval only (after training)
# {py} scripts/run_shortlist_rerank_upgrade_phase.py --stamp {stamp} --skip-train
""",
        encoding="utf-8",
    )
    repro.chmod(0o755)

    rb = out / "report_bundle"
    rb.mkdir(parents=True, exist_ok=True)
    (rb / "README.md").write_text(
        """# Shortlist / rerank upgrade bundle

| Artifact | Claim |
|----------|--------|
| `oracle_reranker_*` | Reranker strength vs **oracle shortlist** eval; conditional accuracy; protocol comparison (natural vs oracle **training**). |
| `shortlist_retrieval_*` | Coarse **Recall@K** and slices after shortlist-aware finetune. |
| `shortlist_rerank_combined_*` + `shortlist_rerank_main_figure.png` | Whether bottleneck work improved **natural** full pipeline Acc@1 and load-stratified accuracy. |
| `oracle_shortlist_rerank_dataset_note.md` | Definition of oracle-shortlist training/eval (no synthetic cheating). |

Re-run training+eval via `repro_commands.sh` in the parent directory.
""",
        encoding="utf-8",
    )

    _append_log(logs / "phase.log", f"Phase complete -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
