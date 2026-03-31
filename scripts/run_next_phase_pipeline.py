#!/usr/bin/env python3
"""Next-phase experiment bundle: train 8-cell matrix, eval, diagnostics, plots, report (timestamped dir)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.datasets.collate import make_grounding_collate_fn
from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.evaluation.result_bundle import (
    build_main_comparison_table,
    build_official_main_lines,
    extract_hard_case_slice,
    hard_case_results_to_csv,
    merge_main_stratified,
    summary_csv_rows,
    write_json,
    write_main_table_csv,
    write_main_table_md,
    write_official_main_table_csv,
    write_official_main_table_md,
    write_summary_csv,
)
from rag3d.evaluation.shortlist_bottleneck import coarse_recall_curve, eval_two_stage_bottleneck
from rag3d.evaluation.two_stage_eval import load_coarse_model, load_two_stage_model
from rag3d.parsers.cached_parser import CachedParser
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.parsers.structured_rule_parser import StructuredRuleParser
from rag3d.utils.config import deep_merge, load_yaml_config
from rag3d.utils.logging import setup_logging

import logging
import torch
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

TRAIN_JOBS: list[tuple[str, str, str, str, dict[str, Any]]] = [
    ("entity", "A_baseline", "diag_entity_baseline.yaml", "baseline", {}),
    ("entity", "B_raw_relation", "diag_entity_raw_relation.yaml", "baseline", {}),
    ("entity", "C_structured", "diag_entity_rel_structured.yaml", "main", {}),
    (
        "entity",
        "D_hardneg",
        "diag_entity_rel_structured.yaml",
        "main",
        {"loss": {"hard_negative": {"enabled": True, "margin": 0.25, "lambda_hinge": 0.5}}},
    ),
    ("full", "A_baseline", "diag_full_baseline.yaml", "baseline", {}),
    ("full", "B_raw_relation", "diag_full_raw_relation.yaml", "baseline", {}),
    ("full", "C_structured", "diag_full_rel_structured.yaml", "main", {}),
    (
        "full",
        "D_hardneg",
        "diag_full_rel_structured.yaml",
        "main",
        {"loss": {"hard_negative": {"enabled": True, "margin": 0.25, "lambda_hinge": 0.5}}},
    ),
]

ROW_LABELS: list[tuple[str, str]] = [
    ("A_baseline", "A attribute-only"),
    ("B_raw_relation", "B raw-text relation"),
    ("C_structured", "C structured relation-aware"),
    ("D_hardneg", "D structured + hard-negative"),
]


def _jobs_for_scope(scope: str) -> list[tuple[str, str, str, str, dict[str, Any]]]:
    """Filter TRAIN_JOBS; for `bc` append full-scene A for shortlist coarse (np_f_attr)."""
    want_map = {
        "bc": {"B_raw_relation", "C_structured"},
        "bca": {"A_baseline", "B_raw_relation", "C_structured"},
        "all": {"A_baseline", "B_raw_relation", "C_structured", "D_hardneg"},
    }
    want = want_map[scope]
    jobs = [j for j in TRAIN_JOBS if j[1] in want]
    if scope == "bc":
        extra = ("full", "A_baseline", "diag_full_baseline.yaml", "baseline", {})
        if not any(j[0] == "full" and j[1] == "A_baseline" for j in jobs):
            jobs = list(jobs) + [extra]
    return jobs


def _row_specs_for_scope(scope: str) -> list[tuple[str, str]]:
    if scope == "bc":
        return [ROW_LABELS[1], ROW_LABELS[2]]
    if scope == "bca":
        return [ROW_LABELS[0], ROW_LABELS[1], ROW_LABELS[2]]
    return list(ROW_LABELS)


def _run_name(regime: str, row: str) -> str:
    pr = "e" if regime == "entity" else "f"
    if row == "A_baseline":
        return f"np_{pr}_attr"
    if row == "B_raw_relation":
        return f"np_{pr}_raw"
    if row == "C_structured":
        return f"np_{pr}_struct"
    if row == "D_hardneg":
        return f"np_{pr}_hn"
    raise ValueError(row)


def _model_and_parser(row: str) -> tuple[str, str]:
    if row == "A_baseline":
        return "attribute_only", "heuristic"
    if row == "B_raw_relation":
        return "raw_text_relation", "heuristic"
    return "relation_aware", "structured"


def _append_repro(sh: Path, cmd: list[str]) -> None:
    line = " ".join(shlex_quote(c) for c in cmd)
    with sh.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def shlex_quote(s: str) -> str:
    if not s or any(c in s for c in " \"'\\"):
        return repr(s)
    return s


def _write_train_cfg(
    exp: Path,
    base_yaml: str,
    run_name: str,
    extra: dict[str, Any],
    smoke: bool,
    device: str,
    seed_override: int | None = None,
) -> Path:
    base = load_yaml_config(ROOT / "configs/train/diagnosis" / base_yaml, ROOT)
    ov: dict[str, Any] = {
        "run_name": run_name,
        "checkpoint_dir": str((exp / "checkpoints").relative_to(ROOT)),
        "metrics_file": str((exp / "train_logs" / f"{run_name}.jsonl").relative_to(ROOT)),
    }
    if seed_override is not None:
        ov["seed"] = int(seed_override)
    if smoke:
        ov.update(
            {
                "epochs": 1,
                "batch_size": 8,
                "num_workers": 0,
                "debug_max_batches": 64,
                "device": device,
            }
        )
    else:
        ov["device"] = device
    merged = deep_merge(base, deep_merge(ov, extra))
    gen = exp / "generated_configs"
    gen.mkdir(parents=True, exist_ok=True)
    out = gen / f"train_{run_name}.yaml"
    out.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def _write_eval_cfg(
    exp: Path,
    regime: str,
    row: str,
    run_name: str,
    device: str,
    metrics_suffix: str = "",
) -> Path:
    dset = (
        "configs/dataset/diagnosis_entity_geom.yaml"
        if regime == "entity"
        else "configs/dataset/diagnosis_full_geom.yaml"
    )
    model, parser_mode = _model_and_parser(row)
    main_p = exp / "metrics" / f"main_{regime}_{row}{metrics_suffix}.json"
    strat_p = exp / "metrics" / f"strat_{regime}_{row}{metrics_suffix}.json"
    ck = (exp / "checkpoints").relative_to(ROOT)
    cfg = {
        "dataset_config": dset,
        "split": "val",
        "use_debug_subdir": False,
        "device": device,
        "batch_size": 16,
        "margin_thresh": 0.15,
        "checkpoint_dir": str(ck),
        "main_results_path": str(main_p.relative_to(ROOT)),
        "stratified_results_path": str(strat_p.relative_to(ROOT)),
        "parser_mode": parser_mode,
        "parser_cache_dir": "data/parser_cache/diagnosis",
        "models": [model],
        "checkpoint_run_names": {model: run_name},
    }
    gen = exp / "generated_configs"
    p = gen / f"eval_{regime}_{row}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def _run_cmd(py: str, args: list[str], cwd: Path, repro: Path) -> None:
    cmd = [py, *args]
    _append_repro(repro, cmd)
    log.info("Running %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd))


def _shortlist_and_optional_two_stage(
    exp: Path,
    device: torch.device,
    repro: Path,
    py: str,
    attr_ckpt_name: str = "np_f_attr_last.pt",
) -> dict[str, Any]:
    out: dict[str, Any] = {"note": "", "coarse_recall_at_k": {}, "two_stage": None}
    ck_attr = exp / "checkpoints" / attr_ckpt_name
    mcfg = load_yaml_config(ROOT / "configs/model/relation_aware.yaml", ROOT)
    dcfg = load_yaml_config(ROOT / "configs/dataset/diagnosis_full_geom.yaml", ROOT)
    proc = Path(dcfg["processed_dir"])
    if not proc.is_absolute():
        proc = ROOT / proc
    manifest = proc / "val_manifest.jsonl"
    if not ck_attr.is_file() or not manifest.is_file():
        out["note"] = "Missing full baseline checkpoint or manifest; skip shortlist curve."
        return out
    feat_dim = int(mcfg["object_dim"])
    ds = ReferIt3DManifestDataset(manifest)
    loader = DataLoader(
        ds,
        batch_size=16,
        shuffle=False,
        collate_fn=make_grounding_collate_fn(feat_dim, attach_features=True),
    )
    coarse = load_coarse_model(mcfg, ck_attr, device, "attribute_only")
    bundle = coarse_recall_curve(coarse, loader, device, 0.15, ks=(1, 5, 10, 20, 40))
    out["coarse_recall_at_k"] = {k: bundle[k] for k in bundle if k.startswith("recall@")}
    out["coarse_acc@1"] = bundle.get("acc@1")
    out["note"] = f"Coarse model = {attr_ckpt_name} (attribute-only) on full-scene val."

    fine_globs = [
        ROOT / "outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt",
        ROOT / "outputs/checkpoints_stage1_rerank/rerank_k20_stage1_last.pt",
    ]
    fine_ckpt = next((p for p in fine_globs if p.is_file()), None)
    coarse_geom = ROOT / "outputs/checkpoints_stage1/coarse_geom_recall_last.pt"
    if fine_ckpt and coarse_geom.is_file():
        parser = CachedParser(StructuredRuleParser(), ROOT / "data/parser_cache/diagnosis/structured")
        ts = load_two_stage_model(mcfg, coarse_geom, fine_ckpt, 10, device, "coarse_geom")
        aux = eval_two_stage_bottleneck(ts, loader, device, parser, 0.15)
        out["two_stage"] = aux
        out["final_acc@1"] = aux.get("acc@1")
        out["oracle_upper_bound_perfect_rerank"] = aux.get("oracle_upper_bound_perfect_rerank")
        out["rerank_acc_given_target_in_shortlist"] = aux.get("rerank_acc_given_target_in_shortlist")
    else:
        out["two_stage_note"] = "No bundled two-stage checkpoints; see configs/eval/stage1_recall_pass.yaml to train."

    csv_path = exp / "shortlist_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in sorted(out.get("coarse_recall_at_k", {}).items()):
            w.writerow([k, v])
        if out.get("two_stage"):
            for k in ("acc@1", "shortlist_recall", "rerank_acc_given_target_in_shortlist"):
                w.writerow([k, out["two_stage"].get(k, "")])

    md = exp / "shortlist_interpretation.md"
    md.write_text(
        "## Shortlist bottleneck (auto)\n\n"
        "- **Stage-1 recall@K**: fraction of val samples where the coarse model places the target in top-K.\n"
        "- **Oracle upper bound** (two-stage block, when present): equals **shortlist recall** if reranking were perfect on the shortlist.\n"
        "- **rerank_acc_given_target_in_shortlist**: top-1 accuracy restricted to samples where the target appears in the shortlist — rerank quality conditional on retrieval.\n\n"
        "If recall@K is low but conditional rerank is high, the bottleneck is **retrieval**; if conditional rerank is low, **reranking / scoring** is limiting.\n",
        encoding="utf-8",
    )
    _append_repro(repro, [py, "-c", f"# shortlist diagnostics written to {csv_path}"])
    return out


def _relation_table_csv(strat: dict[str, Any], run_key: str, model: str, path: Path) -> None:
    block = strat.get(run_key, {}).get(model, {})
    if not block:
        return
    rels = [(k.replace("acc@1_rel::", ""), v) for k, v in block.items() if k.startswith("acc@1_rel::")]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relation_type", "acc@1"])
        for a, b in sorted(rels, key=lambda x: x[0]):
            w.writerow([a, b])


def _geometry_table_json(strat: dict[str, Any], run_key: str, model: str) -> dict[str, Any]:
    block = strat.get(run_key, {}).get(model, {})
    keys = [k for k in block if "geometry" in k or "fallback" in k or "weak_feature" in k or "real_box" in k]
    return {k: block[k] for k in sorted(keys)}


def _write_shortlist_interpretation(
    exp: Path,
    diag: dict[str, Any],
    smoke_shortlist_path: Path | None,
    prior_official_shortlist: Path | None = None,
) -> None:
    lines: list[str] = [
        "# Shortlist bottleneck — interpretation\n",
        "\n## This official run\n\n",
    ]
    cr = diag.get("coarse_recall_at_k") or {}
    if cr:
        lines.append("**Coarse attribute model** (`np_f_attr`) on full-scene val — recall@K (target in top-K):\n\n")
        for k in sorted(cr, key=lambda x: int(x.replace("recall@", ""))):
            lines.append(f"- `{k}`: **{float(cr[k]):.4f}**\n")
        lines.append("\n")
    if diag.get("coarse_acc@1") is not None:
        lines.append(f"- Coarse full-scene Acc@1: **{float(diag['coarse_acc@1']):.4f}**\n\n")
    ts = diag.get("two_stage")
    if isinstance(ts, dict):
        lines.append(
            f"**Two-stage** (K={ts.get('rerank_k')}): final Acc@1 **{float(ts.get('acc@1', 0)):.4f}**; "
            f"oracle upper bound (perfect rerank on shortlist) **{float(ts.get('oracle_upper_bound_perfect_rerank', 0)):.4f}**; "
            f"conditional rerank Acc@1 given target in shortlist **{ts.get('rerank_acc_given_target_in_shortlist')}**.\n\n"
        )
    else:
        lines.append("_No two-stage checkpoints loaded; see `two_stage_note` in `diagnostics_results.json`._\n\n")

    lines.append("### Bottleneck (retrieval vs rerank vs both)\n\n")
    r10 = float(cr.get("recall@10", 0) or 0)
    r40 = float(cr.get("recall@40", 0) or 0)
    cond = ts.get("rerank_acc_given_target_in_shortlist") if isinstance(ts, dict) else None
    lines.append(
        "- **Retrieval**: At small K, recall stays relatively low while recall@40 is much higher → many targets need a **larger shortlist** to enter the rerank set.\n"
        "- **Reranking**: If conditional rerank Acc@1 is **well below 1**, reranking is **also** weak even when the target is in the shortlist.\n"
        "- **Verdict for this run**: "
    )
    if isinstance(cond, float) and not (cond != cond):  # not NaN
        if r10 < 0.5 and cond < 0.5:
            lines.append("**mixed** — both retrieval (at K=10) and reranking appear limiting.\n\n")
        elif r10 < 0.4:
            lines.append("**retrieval-heavy** at K≈10; check larger K and training alignment.\n\n")
        elif cond < 0.35:
            lines.append("**rerank-heavy** conditional on being in the shortlist.\n\n")
        else:
            lines.append("see numeric table above; combine with main Acc@1.\n\n")
    else:
        lines.append("see coarse recall curve; add two-stage checkpoints for conditional rerank.\n\n")

    if prior_official_shortlist and prior_official_shortlist.is_file():
        po = json.loads(prior_official_shortlist.read_text(encoding="utf-8"))
        pcr = po.get("coarse_recall_at_k") or {}
        lines.append("## Comparison to prior single-seed official\n\n")
        lines.append("| Metric | This run | Prior official (1 seed) |\n|:---|---:|---:|\n")
        for k in sorted(set(cr) | set(pcr), key=lambda x: int(x.replace("recall@", ""))):
            o = cr.get(k, "")
            p = pcr.get(k, "")
            ov = f"{float(o):.4f}" if o != "" else ""
            pv = f"{float(p):.4f}" if p != "" else ""
            lines.append(f"| {k} | {ov} | {pv} |\n")
        lines.append(f"\n_Prior: `{prior_official_shortlist}`._\n\n")

    lines.append("## Comparison to smoke run\n\n")
    if smoke_shortlist_path and smoke_shortlist_path.is_file():
        sm = json.loads(smoke_shortlist_path.read_text(encoding="utf-8"))
        scr = sm.get("coarse_recall_at_k") or {}
        lines.append("| Metric | Official | Smoke |\n|:---|---:|---:|\n")
        keys = sorted(set(cr) | set(scr), key=lambda x: int(x.replace("recall@", "")))
        for k in keys:
            o = cr.get(k, "")
            s = scr.get(k, "")
            ov = f"{float(o):.4f}" if o != "" else ""
            sv = f"{float(s):.4f}" if s != "" else ""
            lines.append(f"| {k} | {ov} | {sv} |\n")
        lines.append("\n_Smoke reference: `outputs/20260326_093621_next_phase/shortlist_diagnostics.json` (1 epoch, capped batches)._ \n")
    else:
        lines.append("_Smoke shortlist JSON not found; skip table._\n")

    (exp / "shortlist_interpretation.md").write_text("".join(lines), encoding="utf-8")


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Short training (1 epoch, batch cap)")
    ap.add_argument("--full-train", action="store_true", help="Use diagnosis YAML epochs (no smoke caps)")
    ap.add_argument(
        "--official-full-train",
        action="store_true",
        help="Official package: no smoke caps, default stamp *_full_train_official, default scope bc",
    )
    ap.add_argument("--scope", default=None, choices=("bc", "bca", "all"), help="Job filter (with --official-full-train)")
    ap.add_argument("--seeds", default="42", help="Comma-separated RNG seeds (B/C aggregation for mean±std)")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--stamp", default="", help="UTC subfolder name; default auto")
    args = ap.parse_args()

    official = bool(args.official_full_train)
    scope = args.scope if args.scope is not None else ("bc" if official else "all")
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if not seeds:
        seeds = [42]
    multi = len(seeds) > 1

    if official:
        smoke = False
        stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_full_train_official"
    else:
        smoke = bool(args.smoke or not args.full_train)
        stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_next_phase"

    exp = ROOT / "outputs" / stamp
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "checkpoints").mkdir(exist_ok=True)
    (exp / "train_logs").mkdir(exist_ok=True)
    (exp / "metrics").mkdir(exist_ok=True)
    (exp / "report_bundle").mkdir(exist_ok=True)

    repro = exp / "repro_commands.sh"
    repro.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")/../..\"\n", encoding="utf-8")
    os.chmod(repro, repro.stat().st_mode | 0o111)
    _append_repro(repro, ["#", f"official={official}", f"scope={scope}", f"seeds={seeds}", f"smoke={smoke}"])

    dev_s = args.device
    if dev_s == "auto":
        dev_s = "cuda" if torch.cuda.is_available() else "cpu"
    dev_eval = "cpu" if not torch.cuda.is_available() else dev_s

    py = sys.executable
    jobs = _jobs_for_scope(scope) if official else list(TRAIN_JOBS)

    seed_metrics: dict[int, list[dict[str, Any]]] = {}
    seeds_ok: list[int] = []

    for seed in seeds:
        suf = f"_s{seed}" if multi else ""
        err_log = exp / f"ERROR_seed_{seed}.txt"
        try:
            if not args.skip_train:
                for regime, row, base_yaml, kind, extra in jobs:
                    base_rn = _run_name(regime, row)
                    run_name = f"{base_rn}{suf}" if multi else base_rn
                    cfg_path = _write_train_cfg(exp, base_yaml, run_name, extra, smoke, dev_s, seed_override=seed)
                    script = "train_baseline.py" if kind == "baseline" else "train_main.py"
                    _run_cmd(py, [f"scripts/{script}", "--config", str(cfg_path.relative_to(ROOT))], ROOT, repro)

            runs_seed: list[dict[str, Any]] = []
            if not args.skip_eval:
                for regime, row, _base, _k, _e in jobs:
                    base_rn = _run_name(regime, row)
                    run_name = f"{base_rn}{suf}" if multi else base_rn
                    ecfg = _write_eval_cfg(exp, regime, row, run_name, dev_eval, metrics_suffix=suf)
                    _run_cmd(py, ["scripts/eval_all.py", "--config", str(ecfg.relative_to(ROOT))], ROOT, repro)
                    main_p = exp / "metrics" / f"main_{regime}_{row}{suf}.json"
                    strat_p = exp / "metrics" / f"strat_{regime}_{row}{suf}.json"
                    main = json.loads(main_p.read_text(encoding="utf-8")) if main_p.is_file() else {}
                    strat = json.loads(strat_p.read_text(encoding="utf-8")) if strat_p.is_file() else {}
                    runs_seed.append(
                        {"id": row, "regime": regime, "main": main, "stratified": strat, "seed": seed}
                    )
            seed_metrics[seed] = runs_seed
            seeds_ok.append(seed)
            if err_log.is_file():
                err_log.unlink()
        except subprocess.CalledProcessError as e:
            log.exception("Seed %s failed", seed)
            err_log.write_text(f"exit={e.returncode}\ncmd={e.cmd}\n", encoding="utf-8")
            seed_metrics[seed] = []

    if not seeds_ok:
        log.error("No seeds completed successfully.")
        return 1

    write_json(
        exp / "multiseed_run_status.json",
        {"requested_seeds": seeds, "completed_seeds": seeds_ok, "failed": [s for s in seeds if s not in seeds_ok]},
    )

    runs = seed_metrics[seeds_ok[-1]]
    main_merged, strat_merged = merge_main_stratified(runs)

    write_json(exp / "main_results.json", main_merged)
    write_json(exp / "stratified_results.json", strat_merged)
    hc_slice = extract_hard_case_slice(strat_merged)
    write_json(exp / "hard_case_results.json", hc_slice)
    hard_case_results_to_csv(hc_slice, exp / "hard_case_table.csv")
    Path(exp / "hard_case_table.md").write_text(
        "## Hard-case stratified metrics\n\n"
        "Flat table: `hard_case_table.csv`. Nested: `hard_case_results.json`. "
        "Prioritize **same_class_clutter**, **anchor_confusion**, **low_model_margin**, "
        "**parser_failure**, and **geometry / fallback** slices for the report.\n",
        encoding="utf-8",
    )
    write_summary_csv(exp / "summary.csv", summary_csv_rows(runs))

    lines = build_main_comparison_table(runs)
    write_main_table_csv(exp / "main_table.csv", lines)
    write_main_table_md(exp / "main_table.md", lines)

    off_lines: list[dict[str, Any]] = []
    if official:
        row_specs = _row_specs_for_scope(scope)
        off_lines = build_official_main_lines(seed_metrics, row_specs, seeds_ok)
        write_official_main_table_csv(exp / "main_table_official.csv", off_lines)
        write_official_main_table_md(exp / "main_table_official.md", off_lines)

    mk = "relation_aware"
    rk_e = "entity::C_structured"
    rk_f = "full::C_structured"
    mflat_e = strat_merged.get(rk_e, {}).get(mk)
    mflat_f = strat_merged.get(rk_f, {}).get(mk)
    raw_text_key_e = "entity::B_raw_relation"
    raw_text_key_f = "full::B_raw_relation"
    rt_e = strat_merged.get(raw_text_key_e, {}).get("raw_text_relation")
    rt_f = strat_merged.get(raw_text_key_f, {}).get("raw_text_relation")

    if mflat_e:
        write_json(exp / "_plot_relation_metrics.json", mflat_e)
        write_json(exp / "_plot_hard_case_metrics.json", mflat_e)
        _relation_table_csv(strat_merged, rk_e, mk, exp / "relation_stratified_table.csv")
        rel_md = exp / "relation_stratified_table.md"
        rel_md.write_text(
            "| relation_type | acc@1 |\n|---:|---:|\n"
            + "\n".join(
                f"| {a} | {b} |"
                for a, b in sorted(
                    ((k.replace("acc@1_rel::", ""), v) for k, v in mflat_e.items() if k.startswith("acc@1_rel::")),
                    key=lambda x: x[0],
                )
            )
            + "\n",
            encoding="utf-8",
        )
    elif mflat_f:
        write_json(exp / "_plot_relation_metrics.json", mflat_f)
        write_json(exp / "_plot_hard_case_metrics.json", mflat_f)
        _relation_table_csv(strat_merged, rk_f, mk, exp / "relation_stratified_table.csv")
        exp.joinpath("relation_stratified_table.md").write_text(
            "| relation_type | acc@1 (full-scene C) |\n|---:|---:|\n"
            + "\n".join(
                f"| {a} | {b} |"
                for a, b in sorted(
                    ((k.replace("acc@1_rel::", ""), v) for k, v in mflat_f.items() if k.startswith("acc@1_rel::")),
                    key=lambda x: x[0],
                )
            )
            + "\n",
            encoding="utf-8",
        )

    gq_pack: dict[str, Any] = {}
    if mflat_e:
        gq_pack["controlled_C_structured"] = _geometry_table_json(strat_merged, rk_e, mk)
    if mflat_f:
        gq_pack["full_scene_C_structured"] = _geometry_table_json(strat_merged, rk_f, mk)
    if rt_e:
        gq_pack["controlled_B_raw_text"] = _geometry_table_json(strat_merged, raw_text_key_e, "raw_text_relation")
    if rt_f:
        gq_pack["full_scene_B_raw_text"] = _geometry_table_json(strat_merged, raw_text_key_f, "raw_text_relation")
    if gq_pack:
        write_json(exp / "geometry_quality_results.json", gq_pack)
        with (exp / "geometry_quality_table.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["slice", "metric", "value"])
            for slice_name, block in gq_pack.items():
                for k, v in block.items():
                    w.writerow([slice_name, k, v])
        Path(exp / "geometry_interpretation.md").write_text(
            "## Geometry-quality slices\n\n"
            "Blocks **controlled_*** come from entity-only val; **full_scene_*** from full-scene val. "
            "Compare `geometry_fallback_gt_half` vs `geometry_fallback_le_half` and `geometry_high_fallback` "
            "to argue for **input-quality** limits on full-scene Acc@1.\n",
            encoding="utf-8",
        )

    last_suf = f"_s{seeds_ok[-1]}" if multi else ""
    attr_ckpt = exp / "checkpoints" / f"{_run_name('full', 'A_baseline')}{last_suf}_last.pt"

    device_t = torch.device(dev_eval if dev_eval == "cuda" else "cpu")
    diag = _shortlist_and_optional_two_stage(exp, device_t, repro, py, attr_ckpt_name=attr_ckpt.name)
    shortlist_only = {
        k: diag[k]
        for k in (
            "note",
            "coarse_recall_at_k",
            "coarse_acc@1",
            "two_stage",
            "two_stage_note",
            "final_acc@1",
            "oracle_upper_bound_perfect_rerank",
            "rerank_acc_given_target_in_shortlist",
        )
        if k in diag
    }
    write_json(exp / "shortlist_diagnostics.json", shortlist_only)
    smoke_ref = ROOT / "outputs/20260326_093621_next_phase/shortlist_diagnostics.json"
    prior_off = ROOT / "outputs/20260326_095106_full_train_official/shortlist_diagnostics.json"
    _write_shortlist_interpretation(
        exp,
        diag,
        smoke_ref if smoke_ref.is_file() else None,
        prior_official_shortlist=prior_off if prior_off.is_file() else None,
    )
    diag["paraphrase"] = None
    struct_ckpt = exp / "checkpoints" / f"{_run_name('entity', 'C_structured')}{last_suf}_last.pt"
    ent_manifest = ROOT / "data/processed/diagnosis_entity_geom/val_manifest.jsonl"
    if struct_ckpt.is_file() and ent_manifest.is_file():
        pj = exp / "paraphrase_results.json"
        try:
            _run_cmd(
                py,
                [
                    "scripts/eval_paraphrase_consistency.py",
                    "--manifest",
                    str(ent_manifest.relative_to(ROOT)),
                    "--checkpoint",
                    str(struct_ckpt.relative_to(ROOT)),
                    "--model",
                    "relation_aware",
                    "--dataset-config",
                    "configs/dataset/diagnosis_entity_geom.yaml",
                    "--max-batches",
                    "12" if smoke else "40",
                    "--out-json",
                    str(pj.relative_to(ROOT)),
                ],
                ROOT,
                repro,
            )
            diag["paraphrase"] = json.loads(pj.read_text(encoding="utf-8")) if pj.is_file() else None
            if pj.is_file():
                pd = json.loads(pj.read_text(encoding="utf-8"))
                with (exp / "paraphrase_table.csv").open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["key", "value"])
                    for k, v in pd.items():
                        if k != "note":
                            w.writerow([k, v])
        except subprocess.CalledProcessError as e:
            log.warning("Paraphrase eval failed: %s", e)
            diag["paraphrase_error"] = str(e)
    write_json(exp / "diagnostics_results.json", diag)

    import importlib.util

    import matplotlib

    matplotlib.use("Agg")

    def _load_plots():
        spec = importlib.util.spec_from_file_location("next_phase_plots", ROOT / "scripts/next_phase_plots.py")
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    try:
        plots = _load_plots()
        pr = exp / "_plot_relation_metrics.json"
        if pr.is_file():
            plots.plot_relation_stratified_flat(
                json.loads(pr.read_text(encoding="utf-8")),
                exp / "relation_stratified_plot.png",
                "Relation-stratified (C, controlled)",
            )
        ph = exp / "_plot_hard_case_metrics.json"
        if ph.is_file():
            plots.plot_hard_case_bars_flat(json.loads(ph.read_text(encoding="utf-8")), exp / "hard_case_plot.png")
        if pr.is_file():
            plots.plot_geometry_bars_flat(json.loads(pr.read_text(encoding="utf-8")), exp / "geometry_quality_plot.png")
        plots.plot_shortlist_curve(diag, exp / "shortlist_curve.png")
        pp = exp / "paraphrase_results.json"
        if pp.is_file():
            plots.plot_paraphrase_bars(pp, exp / "paraphrase_plot.png")
        fail_json = exp / "failure_summary.json"
        if struct_ckpt.is_file() and ent_manifest.is_file():
            try:
                subprocess.run(
                    [
                        py,
                        "scripts/analyze_hard_cases.py",
                        "--manifest",
                        str(ent_manifest.relative_to(ROOT)),
                        "--checkpoint",
                        str(struct_ckpt.relative_to(ROOT)),
                        "--fig-dir",
                        str((exp / "case_figures").relative_to(ROOT)),
                        "--out-dir",
                        str((exp / "case_studies").relative_to(ROOT)),
                        "--max-batches",
                        "20",
                    ],
                    cwd=str(ROOT),
                    check=False,
                )
                src_fail = exp / "case_figures" / "failure_summary.json"
                if src_fail.is_file():
                    fail_json.write_text(src_fail.read_text(encoding="utf-8"), encoding="utf-8")
            except OSError as e:
                log.warning("Hard case viz: %s", e)
        if fail_json.is_file():
            plots.plot_failure_taxonomy(fail_json, exp / "failure_taxonomy_plot.png")
    except (ImportError, FileNotFoundError, AttributeError) as e:
        log.warning("Plotting skipped: %s", e)

    rb = exp / "report_bundle"
    for name in (
        "main_table.md",
        "main_table.csv",
        "main_table_official.md",
        "main_table_official.csv",
        "relation_stratified_table.md",
        "relation_stratified_table.csv",
        "hard_case_results.json",
        "hard_case_table.csv",
        "hard_case_table.md",
        "shortlist_metrics.csv",
        "shortlist_diagnostics.json",
        "shortlist_interpretation.md",
        "geometry_quality_results.json",
        "paraphrase_results.json",
        "paraphrase_table.csv",
        "diagnostics_results.json",
        "summary.csv",
    ):
        src = exp / name
        if src.is_file():
            dst = rb / name
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    for png in (
        "relation_stratified_plot.png",
        "hard_case_plot.png",
        "shortlist_curve.png",
        "geometry_quality_plot.png",
        "paraphrase_plot.png",
        "failure_taxonomy_plot.png",
    ):
        p = exp / png
        if p.is_file():
            rb.joinpath(png).write_bytes(p.read_bytes())

    src_hc_md = exp / "hard_case_table.md"
    if src_hc_md.is_file():
        (rb / "hard_case_table.md").write_text(src_hc_md.read_text(encoding="utf-8"), encoding="utf-8")

    rb_readme = rb / "README.md"
    claim_main = (
        "`main_table_official.md`"
        if official and (rb / "main_table_official.md").is_file()
        else "`main_table.md`"
    )
    rb_readme.write_text(
        f"# Report bundle (`{stamp}`)\n\n"
        "| Artifact | Generator | Report claim |\n"
        "|----------|-----------|---------------|\n"
        f"| {claim_main} | `run_next_phase_pipeline.py` | **Overall B vs C** (controlled vs full), seeds noted |\n"
        "| `hard_case_table.csv` / `hard_case_results.json` | stratified eval | **Where C helps** under clutter / anchor / low-margin / parser |\n"
        "| `geometry_quality_*` | stratified eval | **Input-quality** bottleneck (fallback vs clean geometry) |\n"
        "| `shortlist_curve.png`, `shortlist_interpretation.md` | `shortlist_bottleneck` | **Retrieval vs rerank** (vs smoke table inside MD) |\n"
        "| `relation_stratified_table.*` | optional | Relation-type slices if `acc@1_rel::*` present |\n"
        "| `failure_taxonomy_plot.png` | `analyze_hard_cases.py` | **Failure taxonomy** (sampled batches) |\n"
        "| `paraphrase_results.json` | `eval_paraphrase_consistency.py` | Template paraphrase stability |\n"
        "| `summary.csv` | `result_bundle` | Flat join |\n\n"
        "Regenerate official: `python scripts/run_next_phase_pipeline.py --official-full-train --scope bc --device auto`.\n",
        encoding="utf-8",
    )

    summ_path = ROOT / "reports/next_phase_experiment_summary.md"
    summ_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_note = "**Smoke training** (1 epoch, capped batches)." if smoke else "**Full** diagnosis-style epochs."
    if not official:
        summ_path.write_text(
            f"# Next-phase experiment summary\n\n"
            f"- **Output directory**: `outputs/{stamp}/`\n"
            f"- **Mode**: {smoke_note}\n\n"
            "## What ran\n\n"
            "- Default: 8 training jobs (entity/full × A–D) unless `--skip-train` / `--scope` with `--official-full-train`.\n"
            "- Official full-train: see `reports/full_train_execution_plan.md`.\n"
            "- Shortlist diagnostics on full val with attribute coarse; optional two-stage if checkpoints exist.\n\n"
            "## Findings\n\n"
            "- Read `main_table.md` under the stamp directory.\n",
            encoding="utf-8",
        )

    if official and not multi:
        off_md = (exp / "main_table_official.md").read_text(encoding="utf-8") if (exp / "main_table_official.md").is_file() else ""
        cr = diag.get("coarse_recall_at_k") or {}
        ts = diag.get("two_stage")
        ts_note = ""
        if isinstance(ts, dict):
            ts_note = (
                f"Two-stage K={ts.get('rerank_k')}: oracle **{ts.get('oracle_upper_bound_perfect_rerank')}**, "
                f"conditional rerank Acc@1 **{ts.get('rerank_acc_given_target_in_shortlist')}**.\n"
            )
        geo_note = ""
        if gq_pack:
            geo_note = "See `geometry_quality_results.json` for controlled vs full-scene geometry slices (B/C).\n"
        (ROOT / "reports/full_train_official_summary.md").write_text(
            f"# Full-train official summary\n\n"
            f"- **Output**: `outputs/{stamp}/`\n"
            f"- **Scope**: `{scope}` | **Seeds**: {seeds_ok} | **Official full-train** (no smoke caps)\n\n"
            "## Main table (embedded)\n\n"
            f"{off_md}\n"
            "## Shortlist (coarse recall excerpt)\n\n"
            f"```json\n{json.dumps(cr, indent=2)}\n```\n\n"
            f"{ts_note}\n"
            f"{geo_note}\n"
            "## Interpretation pointers\n\n"
            "- **B vs C**: compare controlled Acc@1 in `main_table_official`.\n"
            "- **Full-scene drop**: main full columns + recall@K vs Acc@1 + geometry slices.\n"
            "- **Bottleneck**: `shortlist_interpretation.md` states retrieval vs rerank vs mixed.\n\n"
            "## Skipped / optional\n\n"
            "- **D (hard-negative)**: not run unless `--scope all`.\n"
            "- **Multi-seed**: use `--seeds 42,43,44`; see `reports/bc_multiseed_claim_check.md` after postprocess.\n\n"
            "## Next actions\n\n"
            "1. If B≈C, consider longer training or ablation on parser; if C>B, emphasize structured slices in paper.\n"
            "2. Align stage-1 K with rerank training; refresh two-stage numbers.\n"
            "3. Add curated paraphrase set if claiming robustness.\n",
            encoding="utf-8",
        )

    if official and multi:
        post_cmd = [
            py,
            str(ROOT / "scripts/postprocess_bc_multiseed_report.py"),
            "--exp-dir",
            str(exp.relative_to(ROOT)),
            "--prior-official",
            "outputs/20260326_095106_full_train_official",
        ]
        _append_repro(repro, post_cmd)
        log.info("Running postprocess for multi-seed B/C tables and figures")
        r = subprocess.run(post_cmd, cwd=str(ROOT))
        if r.returncode != 0:
            log.warning("postprocess_bc_multiseed_report exited %s", r.returncode)

    log.info("Done. Artifacts under %s", exp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
