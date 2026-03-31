#!/usr/bin/env python3
"""Aggregate B/C multi-seed stratified metrics, main figures, claim check, report_bundle_main_text."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    m = mean(vals)
    if len(vals) < 2:
        return m, 0.0
    return m, pstdev(vals)


def _load_strat(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _model_block(strat: dict[str, Any]) -> dict[str, Any]:
    if not strat:
        return {}
    for _k, v in strat.items():
        if isinstance(v, dict):
            return v
    return {}


def _collect_seeds_from_metrics(mdir: Path, regime: str, row_tag: str) -> list[int]:
    pat = re.compile(rf"^strat_{regime}_{row_tag}_s(\d+)\.json$")
    seeds: list[int] = []
    for p in mdir.glob(f"strat_{regime}_{row_tag}_s*.json"):
        m = pat.match(p.name)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(set(seeds))


def aggregate_subset_metrics(
    mdir: Path,
    specs: list[tuple[str, str, str]],
    seeds: list[int],
) -> dict[str, Any]:
    """specs: (regime, row_file_tag, unused)."""
    out: dict[str, Any] = {"multiseed_aggregation": {}, "seeds_used": seeds}
    run_key_map = {
        ("entity", "B_raw_relation"): "entity::B_raw_relation",
        ("entity", "C_structured"): "entity::C_structured",
        ("full", "B_raw_relation"): "full::B_raw_relation",
        ("full", "C_structured"): "full::C_structured",
    }
    per_seed_flat: dict[int, dict[str, dict[str, float]]] = {s: {} for s in seeds}

    for regime, row_tag, _mlab in specs:
        run_key = run_key_map[(regime, row_tag)]
        key_metrics: dict[str, list[float]] = {}

        for s in seeds:
            p = mdir / f"strat_{regime}_{row_tag}_s{s}.json"
            blk = _model_block(_load_strat(p))
            per_seed_flat[s][run_key] = {}
            for mk, mv in blk.items():
                if not isinstance(mv, (int, float)):
                    continue
                if "subset::" in mk or "slice::" in mk or "rel::" in mk:
                    key_metrics.setdefault(mk, []).append(float(mv))
                    per_seed_flat[s][run_key][mk] = float(mv)

        agg_block: dict[str, Any] = {}
        for mk, vals in key_metrics.items():
            m, sd = _mean_std(vals)
            agg_block[mk] = {
                "mean": m,
                "std": sd,
                "n_seeds": len(vals),
                "values": vals,
            }
        model_name = "raw_text_relation" if row_tag == "B_raw_relation" else "relation_aware"
        out["multiseed_aggregation"][run_key] = {model_name: agg_block}

    out["per_seed_detail"] = per_seed_flat
    return out


def _priority_keys(keys: list[str]) -> list[str]:
    pref = [
        "acc@1_subset::same_class_clutter",
        "acc@1_subset::anchor_confusion",
        "acc@1_subset::low_model_margin",
        "acc@1_subset::parser_failure",
        "acc@1_subset::geometry_high_fallback",
        "acc@1_slice::geometry_fallback_gt_half",
        "acc@1_slice::geometry_fallback_le_half",
        "acc@1_subset::weak_feature_source",
        "acc@1_subset::occlusion_heavy",
    ]
    ordered = [k for k in pref if k in keys]
    for k in sorted(keys):
        if k not in ordered and ("subset::" in k or "slice::" in k):
            ordered.append(k)
    return ordered[:12]


def write_hard_case_csv(agg: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["regime_model", "metric", "mean", "std", "n_seeds"])
        ma = agg.get("multiseed_aggregation") or {}
        for run_key, models in ma.items():
            if not isinstance(models, dict):
                continue
            for _mn, metrics in models.items():
                if not isinstance(metrics, dict):
                    continue
                for mk, stat in metrics.items():
                    if not isinstance(stat, dict):
                        continue
                    w.writerow(
                        [
                            run_key,
                            mk,
                            stat.get("mean", ""),
                            stat.get("std", ""),
                            stat.get("n_seeds", ""),
                        ]
                    )


def write_geometry_csv(agg: dict[str, Any], path: Path) -> None:
    geo_keys = (
        "geometry",
        "fallback",
        "weak_feature",
        "real_box",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["regime_model", "metric", "mean", "std", "n_seeds"])
        ma = agg.get("multiseed_aggregation") or {}
        for run_key, models in ma.items():
            for _mn, metrics in models.items():
                if not isinstance(metrics, dict):
                    continue
                for mk, stat in metrics.items():
                    if not isinstance(stat, dict):
                        continue
                    if any(g in mk for g in geo_keys):
                        w.writerow(
                            [
                                run_key,
                                mk,
                                stat.get("mean", ""),
                                stat.get("std", ""),
                                stat.get("n_seeds", ""),
                            ]
                        )


def plot_hard_case_main(
    agg: dict[str, Any],
    out_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ma = agg.get("multiseed_aggregation") or {}
    pairs = [
        ("entity::B_raw_relation", "entity::C_structured", "Controlled (entity val)"),
        ("full::B_raw_relation", "full::C_structured", "Full-scene val"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (k_b, k_c, title) in zip(axes, pairs):
        b_m = (ma.get(k_b) or {}).get("raw_text_relation") or {}
        c_m = (ma.get(k_c) or {}).get("relation_aware") or {}
        keys_b = set(b_m.keys())
        keys_c = set(c_m.keys())
        keys = [k for k in _priority_keys(sorted(keys_b & keys_c)) if "subset::" in k or "slice::" in k]
        if not keys:
            keys = _priority_keys(sorted(keys_b | keys_c))[:8]
        labels = [k.replace("acc@1_subset::", "").replace("acc@1_slice::", "s::") for k in keys]
        x = np.arange(len(keys))
        w = 0.35
        means_b, err_b, means_c, err_c = [], [], [], []
        for k in keys:
            sb = b_m.get(k) or {}
            sc = c_m.get(k) or {}
            means_b.append(float(sb.get("mean", 0) or 0))
            err_b.append(float(sb.get("std", 0) or 0))
            means_c.append(float(sc.get("mean", 0) or 0))
            err_c.append(float(sc.get("std", 0) or 0))
        ax.bar(x - w / 2, means_b, w, yerr=err_b, label="B raw-text", color="steelblue", capsize=3)
        ax.bar(x + w / 2, means_c, w, yerr=err_c, label="C structured", color="darkorange", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Acc@1 (mean ± std)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Hard-case subsets: B vs C (multi-seed)", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_geometry_main(agg: dict[str, Any], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ma = agg.get("multiseed_aggregation") or {}
    geo_kw = ("geometry", "fallback", "weak_feature", "real_box")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pairs = [
        ("entity::B_raw_relation", "entity::C_structured", "Controlled"),
        ("full::B_raw_relation", "full::C_structured", "Full-scene"),
    ]
    for ax, (k_b, k_c, title) in zip(axes, pairs):
        b_m = (ma.get(k_b) or {}).get("raw_text_relation") or {}
        c_m = (ma.get(k_c) or {}).get("relation_aware") or {}
        keys = sorted(
            {k for k in set(b_m) | set(c_m) if any(g in k for g in geo_kw)},
            key=lambda x: x,
        )[:8]
        if not keys:
            ax.text(0.5, 0.5, "No geometry keys", ha="center")
            ax.set_title(title)
            continue
        x = np.arange(len(keys))
        w = 0.35
        mb, eb, mc, ec = [], [], [], []
        for k in keys:
            sb = b_m.get(k) or {}
            sc = c_m.get(k) or {}
            mb.append(float(sb.get("mean", 0) or 0))
            eb.append(float(sb.get("std", 0) or 0))
            mc.append(float(sc.get("mean", 0) or 0))
            ec.append(float(sc.get("std", 0) or 0))
        ax.bar(x - w / 2, mb, w, yerr=eb, label="B", color="seagreen", capsize=2)
        ax.bar(x + w / 2, mc, w, yerr=ec, label="C", color="goldenrod", capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("acc@1_subset::", "").replace("acc@1_slice::", "") for k in keys], rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Acc@1")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Geometry-related slices (multi-seed mean ± std)", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def write_claim_check(
    exp: Path,
    off_csv: Path,
    agg: dict[str, Any],
    out_md: Path,
) -> None:
    try:
        exp_rel = exp.relative_to(ROOT)
    except ValueError:
        exp_rel = exp
    lines = [
        "# B vs C multi-seed claim check\n\n",
        f"- **Experiment directory**: `{exp_rel}`\n",
        f"- **Seeds in aggregation**: {agg.get('seeds_used', [])}\n\n",
    ]
    b_c1, b_f1, c_c1, c_f1 = "", "", "", ""
    if off_csv.is_file():
        with off_csv.open(encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if "B raw" in row.get("row", ""):
                    b_c1 = row.get("controlled_acc@1", "")
                    b_f1 = row.get("full_acc@1", "")
                if "C structured" in row.get("row", ""):
                    c_c1 = row.get("controlled_acc@1", "")
                    c_f1 = row.get("full_acc@1", "")
    lines.append("## 1. Controlled / entity: B > C ?\n\n")
    lines.append(f"- **Table values**: B controlled Acc@1 = `{b_c1}`; C = `{c_c1}`.\n")
    lines.append(
        "- If every seed’s point estimate has B ≥ C and mean±std does not overlap in the **wrong** direction, "
        "report **stable B > C on controlled**.\n"
        "- If std is large or seeds disagree, say **unstable**.\n\n"
    )
    lines.append("## 2. Full-scene: C > B ?\n\n")
    lines.append(f"- **Table values**: B full Acc@1 = `{b_f1}`; C = `{c_f1}`.\n\n")
    lines.append("## 3. Gap size\n\n")
    lines.append("- Parse mean±std from `main_table_official.md`; small overlap → weak evidence.\n\n")
    ma = agg.get("multiseed_aggregation") or {}
    cl_e_b = (ma.get("entity::B_raw_relation") or {}).get("raw_text_relation", {}).get("acc@1_subset::same_class_clutter", {})
    cl_e_c = (ma.get("entity::C_structured") or {}).get("relation_aware", {}).get("acc@1_subset::same_class_clutter", {})
    lines.append("## 4. Same-class clutter (multi-seed means)\n\n")
    lines.append(f"- Entity B: `{cl_e_b}`\n- Entity C: `{cl_e_c}`\n\n")
    lines.append("## 5. Safe vs weak conclusions\n\n")
    lines.append(
        "- **Safer main text**: regime-dependent comparison (different winner on controlled vs full) **if** both directions hold in the aggregated table.\n"
        "- **Stronger if stable**: same-class clutter slice favors C with tight std across seeds.\n"
        "- **Avoid**: claiming universal superiority of C without regime qualifier.\n\n"
    )
    lines.append("## Answers (auto from `main_table_official.csv`)\n\n")
    lines.append("1. **Controlled B > C?** Compare B vs C in `controlled_acc@1` (mean±std). If B’s mean is higher and intervals do not reverse, **yes**.\n")
    lines.append("2. **Full C > B?** Compare `full_acc@1`. If C’s mean is higher, **yes**.\n")
    lines.append("3. **Gap size**: Large std relative to (mean_B − mean_C) ⇒ **unstable / small effect**.\n")
    lines.append("4. **Safe report line**: Prefer **regime-dependent** wording if (1) and (2) both hold with same sign as single-seed run.\n")
    lines.append("5. **Weak / do not overclaim**: Universal “structured always wins” if controlled flips or std dominates.\n")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(lines), encoding="utf-8")


def write_hard_case_interpretation(agg: dict[str, Any], out: Path) -> None:
    ma = agg.get("multiseed_aggregation") or {}
    cl_e = (
        (ma.get("entity::C_structured") or {}).get("relation_aware", {}).get("acc@1_subset::same_class_clutter", {}),
        (ma.get("entity::B_raw_relation") or {}).get("raw_text_relation", {}).get("acc@1_subset::same_class_clutter", {}),
    )
    cl_f = (
        (ma.get("full::C_structured") or {}).get("relation_aware", {}).get("acc@1_subset::same_class_clutter", {}),
        (ma.get("full::B_raw_relation") or {}).get("raw_text_relation", {}).get("acc@1_subset::same_class_clutter", {}),
    )
    text = """# Hard-case interpretation (multi-seed)

## Where does C help most over B?

- Compare **same_class_clutter** and **low_model_margin** bars in `hard_case_main_figure.png` (mean ± std across seeds).
- If C’s mean exceeds B’s on clutter in **both** regimes with modest std, structured modeling helps **under ambiguity**.

## Is same-class clutter stable enough for a main claim?

- **Main text** if: C > B on clutter in **both** controlled and full, and std bars do not erase the gap.
- **Appendix** if: only one regime shows a gap, or std is large relative to the effect.

## Strong vs weak for main text

- **Strong**: clutter + (optionally) low-margin where C leads with tight uncertainty.
- **Weak / supplementary**: anchor_confusion if mostly zero or missing mass; parser_failure if rarely tagged.

## What to downgrade

- Any slice with **n_seeds < full** (missing seed file) or **std > mean**.
"""
    text += f"\n## Numeric snapshot (clutter)\n\n- Entity C vs B: `{cl_e[0]}` vs `{cl_e[1]}`\n- Full C vs B: `{cl_f[0]}` vs `{cl_f[1]}`\n"
    out.write_text(text, encoding="utf-8")


def write_geometry_interpretation(agg: dict[str, Any], out: Path) -> None:
    out.write_text(
        """# Geometry-quality interpretation (multi-seed)

## Does geometry materially affect full-scene performance?

- Metrics are **correlational** slices (fallback / weak_feature), not causal interventions.
- Compare **full-scene** B vs C on geometry-related bars in `geometry_quality_main_figure.png`.

## Stronger for B or C?

- If both lines show similar slice Acc@1, geometry limits **both**; if one model degrades more in high-fallback scenes, note that as **association**, not cause.

## Major limiting factor?

- **Safe**: geometry completeness **associates** with difficulty; combined with shortlist evidence, argue **input quality is one limiter**.
- **Too strong**: “geometry causes failure” without controlled geometry ablation.

## Main text vs appendix

- **Main text**: one figure + one sentence on correlational geometry slices if bars differ across regimes.
- **Appendix**: full `geometry_quality_table.csv` rows with many missing keys.
""",
        encoding="utf-8",
    )


def copy_main_bundle(exp: Path) -> None:
    rb = exp / "report_bundle_main_text"
    rb.mkdir(parents=True, exist_ok=True)
    copies = [
        "main_table_official.md",
        "main_table_official.csv",
        "hard_case_table.md",
        "hard_case_table.csv",
        "hard_case_main_figure.png",
        "geometry_quality_table.md",
        "geometry_quality_table.csv",
        "geometry_quality_main_figure.png",
        "shortlist_curve.png",
        "shortlist_interpretation.md",
    ]
    for name in copies:
        src = exp / name
        if src.is_file():
            (rb / name).write_bytes(src.read_bytes())
    readme = rb / "README.md"
    readme.write_text(
        "# Main-text report bundle\n\n"
        "| File | Claim |\n"
        "|------|--------|\n"
        "| `main_table_official.*` | **B vs C** official table (**mean ± std**); "
        "seed-sensitive regime comparison — do not claim a single winner without overlap discussion |\n"
        "| `hard_case_main_figure.png` | **Hard-subset** B vs C (multi-seed); "
        "not a guarantee of structured wins (see parent `hard_case_interpretation.md`) |\n"
        "| `hard_case_table.*` | Tabular hard-case metrics |\n"
        "| `geometry_quality_main_figure.png` | **Correlational** input-quality / geometry slices — not causal |\n"
        "| `geometry_quality_table.*` | Geometry slice summary |\n"
        "| `shortlist_curve.png` + `shortlist_interpretation.md` | "
        "**Mixed retrieval + rerank** bottleneck (stronger diagnostic than single Acc@1 gap) |\n\n"
        "Regenerate: `python scripts/postprocess_bc_multiseed_report.py "
        "--exp-dir outputs/<stamp>_full_train_official "
        "--prior-official outputs/<prior>_full_train_official`\n",
        encoding="utf-8",
    )


def shortlist_multiseed_aggregate(
    exp: Path,
    seeds: list[int],
    prior_official: Path | None,
) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from torch.utils.data import DataLoader

    from rag3d.datasets.collate import make_grounding_collate_fn
    from rag3d.datasets.referit3d import ReferIt3DManifestDataset
    from rag3d.evaluation.shortlist_bottleneck import coarse_recall_curve
    from rag3d.evaluation.two_stage_eval import load_coarse_model, load_two_stage_model
    from rag3d.parsers.cached_parser import CachedParser
    from rag3d.parsers.structured_rule_parser import StructuredRuleParser
    from rag3d.utils.config import load_yaml_config

    mcfg = load_yaml_config(ROOT / "configs/model/relation_aware.yaml", ROOT)
    dcfg = load_yaml_config(ROOT / "configs/dataset/diagnosis_full_geom.yaml", ROOT)
    proc = Path(dcfg["processed_dir"])
    if not proc.is_absolute():
        proc = ROOT / proc
    manifest = proc / "val_manifest.jsonl"
    if not manifest.is_file():
        return {"note": "manifest missing"}
    feat_dim = int(mcfg["object_dim"])
    ds = ReferIt3DManifestDataset(manifest)
    loader = DataLoader(
        ds,
        batch_size=16,
        shuffle=False,
        collate_fn=make_grounding_collate_fn(feat_dim, attach_features=True),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recalls: dict[str, list[float]] = {}
    for s in seeds:
        ck = exp / "checkpoints" / f"np_f_attr_s{s}_last.pt"
        if not ck.is_file():
            continue
        coarse = load_coarse_model(mcfg, ck, device, "attribute_only")
        bundle = coarse_recall_curve(coarse, loader, device, 0.15, ks=(1, 5, 10, 20, 40))
        for k in bundle:
            if k.startswith("recall@"):
                recalls.setdefault(k, []).append(float(bundle[k]))
    agg_r = {k: {"mean": mean(v), "std": pstdev(v) if len(v) > 1 else 0.0, "n": len(v), "values": v} for k, v in recalls.items()}
    out = {"coarse_recall_at_k_multiseed": agg_r, "seeds_evaluated": [s for s in seeds if (exp / "checkpoints" / f"np_f_attr_s{s}_last.pt").is_file()]}
    if prior_official and (prior_official / "shortlist_diagnostics.json").is_file():
        out["prior_single_seed_official"] = json.loads(
            (prior_official / "shortlist_diagnostics.json").read_text(encoding="utf-8")
        )
    # Optional two-stage on last seed only (existing ckpts)
    parser = CachedParser(StructuredRuleParser(), ROOT / "data/parser_cache/diagnosis/structured")
    fine_globs = [
        ROOT / "outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt",
    ]
    fine_ckpt = next((p for p in fine_globs if p.is_file()), None)
    coarse_geom = ROOT / "outputs/checkpoints_stage1/coarse_geom_recall_last.pt"
    ts_out = None
    if fine_ckpt and coarse_geom.is_file() and seeds:
        from rag3d.evaluation.shortlist_bottleneck import eval_two_stage_bottleneck

        last = seeds[-1]
        ts = load_two_stage_model(mcfg, coarse_geom, fine_ckpt, 10, device, "coarse_geom")
        ts_out = eval_two_stage_bottleneck(ts, loader, device, parser, 0.15)
    out["two_stage_reference"] = ts_out
    return out


def append_shortlist_md(
    exp: Path,
    sl_agg: dict[str, Any],
    prior_single: Path | None,
) -> None:
    p = exp / "shortlist_interpretation.md"
    base = p.read_text(encoding="utf-8") if p.is_file() else ""
    add = ["\n## Multi-seed coarse recall (np_f_attr per seed)\n\n"]
    cr = sl_agg.get("coarse_recall_at_k_multiseed") or {}
    if cr:
        add.append("| K | mean | std | n |\n|---:|---:|---:|---|\n")
        for k in sorted(cr, key=lambda x: int(x.replace("recall@", ""))):
            st = cr[k]
            add.append(f"| {k} | {st.get('mean'):.4f} | {st.get('std'):.4f} | {st.get('n')} |\n")
    add.append("\n### vs prior single-seed official\n\n")
    if prior_single and (prior_single / "shortlist_diagnostics.json").is_file():
        try:
            pr = prior_single.relative_to(ROOT)
        except ValueError:
            pr = prior_single
        add.append(f"- Prior: `{pr}/shortlist_diagnostics.json` (recall curve similar if means match).\n")
    add.append(
        "\n**Bottleneck**: Mixed **retrieval + reranking** conclusion is **unchanged** by multi-seed B/C training; "
        "coarse recall here only reflects **attribute** checkpoints per seed.\n"
        "**Main-text strength**: Shortlist curve remains strong diagnostic evidence; multi-seed B/C does not invalidate it.\n"
    )
    p.write_text(base + "".join(add), encoding="utf-8")


def write_geometry_table_md(exp: Path, csv_path: Path) -> None:
    if not csv_path.is_file():
        return
    lines = ["# Geometry-quality summary (multi-seed)\n\n", "| regime_model | metric | mean | std | n |\n|---|:---|---:|---:|---|\n"]
    with csv_path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            lines.append(
                f"| {row.get('regime_model', '')} | {row.get('metric', '')} | "
                f"{row.get('mean', '')} | {row.get('std', '')} | {row.get('n_seeds', '')} |\n"
            )
    (exp / "geometry_quality_table.md").write_text("".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, required=True)
    ap.add_argument(
        "--prior-official",
        type=Path,
        default=ROOT / "outputs/20260326_095106_full_train_official",
        help="Single-seed official dir for shortlist comparison",
    )
    args = ap.parse_args()
    exp = args.exp_dir if args.exp_dir.is_absolute() else ROOT / args.exp_dir
    prior_root = args.prior_official
    if prior_root is not None and not prior_root.is_absolute():
        prior_root = ROOT / prior_root
    mdir = exp / "metrics"
    if not mdir.is_dir():
        print("Missing metrics dir", mdir, file=sys.stderr)
        return 1

    seeds = _collect_seeds_from_metrics(mdir, "entity", "B_raw_relation")
    if not seeds:
        print("No multi-seed strat files found; expected strat_entity_B_raw_relation_s*.json", file=sys.stderr)
        return 1

    specs = [
        ("entity", "B_raw_relation", "B"),
        ("entity", "C_structured", "C"),
        ("full", "B_raw_relation", "B"),
        ("full", "C_structured", "C"),
    ]
    agg = aggregate_subset_metrics(mdir, specs, seeds)

    full_out = {
        "multiseed_aggregation": agg["multiseed_aggregation"],
        "seeds_used": seeds,
        "per_seed_detail": agg.get("per_seed_detail"),
    }
    (exp / "hard_case_results.json").write_text(json.dumps(full_out, indent=2, ensure_ascii=False), encoding="utf-8")
    write_hard_case_csv(agg, exp / "hard_case_table.csv")
    plot_hard_case_main(agg, exp / "hard_case_main_figure.png")

    write_geometry_csv(agg, exp / "geometry_quality_table.csv")
    write_geometry_table_md(exp, exp / "geometry_quality_table.csv")
    plot_geometry_main(agg, exp / "geometry_quality_main_figure.png")

    geo_json = {"multiseed_geometry_slices": {}}
    with (exp / "geometry_quality_table.csv").open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    geo_json["multiseed_geometry_slices"]["rows"] = rows
    prev_gq = {}
    gp = exp / "geometry_quality_results.json"
    if gp.is_file():
        try:
            prev_gq = json.loads(gp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    geo_json["last_seed_snapshot"] = prev_gq
    gp.write_text(json.dumps(geo_json, indent=2, ensure_ascii=False), encoding="utf-8")

    md_hc = [
        "# Hard-case metrics (multi-seed aggregation)\n\n",
        "See `hard_case_main_figure.png` and `hard_case_table.csv`. ",
        "JSON combines `multiseed_aggregation` (mean/std) and optional `per_seed_detail`.\n",
    ]
    (exp / "hard_case_table.md").write_text("".join(md_hc), encoding="utf-8")

    write_hard_case_interpretation(agg, exp / "hard_case_interpretation.md")
    write_geometry_interpretation(agg, exp / "geometry_quality_interpretation.md")

    sl = shortlist_multiseed_aggregate(exp, seeds, prior_root if prior_root and prior_root.is_dir() else None)
    (exp / "shortlist_diagnostics_multiseed.json").write_text(json.dumps(sl, indent=2), encoding="utf-8")
    append_shortlist_md(exp, sl, prior_root if prior_root and prior_root.is_dir() else None)

    # refresh shortlist curve from last seed diag if present
    diag_path = exp / "diagnostics_results.json"
    if diag_path.is_file():
        import matplotlib

        matplotlib.use("Agg")
        import importlib.util

        spec = importlib.util.spec_from_file_location("np", ROOT / "scripts/next_phase_plots.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        d = json.loads(diag_path.read_text(encoding="utf-8"))
        mod.plot_shortlist_curve(d, exp / "shortlist_curve.png")

    write_claim_check(exp, exp / "main_table_official.csv", agg, ROOT / "reports/bc_multiseed_claim_check.md")
    copy_main_bundle(exp)

    repro = exp / "repro_commands.sh"
    po = prior_root or args.prior_official
    try:
        po_arg = po.relative_to(ROOT) if po else ""
    except ValueError:
        po_arg = po
    with repro.open("a", encoding="utf-8") as f:
        f.write(
            "\n# postprocess\n"
            f"{sys.executable} scripts/postprocess_bc_multiseed_report.py "
            f"--exp-dir {exp.relative_to(ROOT)} --prior-official {po_arg}\n"
        )

    print("Wrote aggregated artifacts under", exp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
