#!/usr/bin/env python3
"""Oracle ceiling analyses (shortlist, geometry, branch) + lightweight hybrid B/C router.

Writes under outputs/<timestamp>_hybrid_router_phase/. See README and reports/hybrid_router_phase_summary.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from rag3d.datasets.collate import make_grounding_collate_fn
from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.datasets.transforms import compute_stratification_tags
from rag3d.datasets.schemas import ParsedUtterance
from rag3d.diagnostics.confidence import anchor_entropy, logits_to_confidence_masked
from rag3d.evaluation.metrics import logit_top12_margin, per_sample_correct_at1, per_sample_correct_at5
from rag3d.evaluation.shortlist_bottleneck import eval_two_stage_bottleneck
from rag3d.evaluation.stratified_eval import (
    augment_meta_geometry_fallback_tags,
    augment_meta_with_model_margins,
)
from rag3d.evaluation.two_stage_eval import coarse_forward, load_two_stage_model, to_dev_batch
from rag3d.parsers.cached_parser import CachedParser
from rag3d.parsers.structured_rule_parser import StructuredRuleParser
from rag3d.relation_reasoner.geom_context import batch_geom_context_tensor8
from rag3d.relation_reasoner.model import RawTextRelationModel, RelationAwareModel
from rag3d.relation_reasoner.two_stage_rerank import (
    GEOM_DIM,
    TwoStageCoarseRerankModel,
    _effective_topk,
    _topk_union_target,
)
from rag3d.utils.config import load_yaml_config
from rag3d.utils.logging import setup_logging


def _resolve(p: Path, base: Path) -> Path:
    return p if p.is_absolute() else (base / p).resolve()


def _load_bc(
    kind: str,
    mcfg: dict,
    ckpt: Path | None,
    device: torch.device,
) -> nn.Module:
    if kind == "raw_text_relation":
        m = RawTextRelationModel(
            int(mcfg["object_dim"]),
            int(mcfg["language_dim"]),
            int(mcfg["hidden_dim"]),
            int(mcfg["relation_dim"]),
            dropout=float(mcfg.get("dropout", 0.1)),
        )
    else:
        m = RelationAwareModel(
            int(mcfg["object_dim"]),
            int(mcfg["language_dim"]),
            int(mcfg["hidden_dim"]),
            int(mcfg["relation_dim"]),
            anchor_temperature=float(mcfg.get("anchor_temperature", 1.0)),
            dropout=float(mcfg.get("dropout", 0.1)),
        )
    m = m.to(device)
    if ckpt is not None and ckpt.is_file():
        try:
            data = torch.load(ckpt, map_location=device, weights_only=False)
        except TypeError:
            data = torch.load(ckpt, map_location=device)
        sd = data["model"] if isinstance(data, dict) and "model" in data else data
        m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def _forward_b(model: RawTextRelationModel, batch: dict, device: torch.device) -> torch.Tensor:
    b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    sub = {k: b[k] for k in ("object_features", "object_mask", "raw_texts")}
    return model(sub)


def _forward_c(
    model: RelationAwareModel,
    batch: dict,
    parser: CachedParser,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    sub = {k: b[k] for k in ("object_features", "object_mask", "raw_texts")}
    samples = batch["samples_ref"]
    parsed_list = [parser.parse(s.utterance) for s in samples]
    logits, p_anchor = model(sub, parsed_list=parsed_list)
    return logits, p_anchor


@torch.no_grad()
def _two_stage_logits_oracle_shortlist(
    model: TwoStageCoarseRerankModel,
    batch: dict[str, Any],
    parsed_list: list[ParsedUtterance],
    device: torch.device,
) -> torch.Tensor:
    """Same as TwoStageCoarseRerankModel.forward but forces gold into top-K via _topk_union_target(training=True)."""
    model.eval()
    obj = batch["object_features"]
    mask = batch["object_mask"]
    samples = batch["samples_ref"]
    b, n, d = obj.shape
    dtype = obj.dtype
    geom = batch_geom_context_tensor8(samples, n, device, dtype)
    sub: dict[str, Any] = {k: batch[k] for k in ("object_features", "object_mask", "raw_texts")}
    if getattr(model.coarse, "uses_geometry_context", False):
        sub["samples_ref"] = batch["samples_ref"]
    coarse_logits = model.coarse(sub)
    target_index = batch["target_index"]
    k_eff = _effective_topk(mask, model.rerank_k)
    idx = _topk_union_target(coarse_logits, mask, target_index, k_eff, training=True)
    exp = idx.unsqueeze(-1).expand(-1, -1, d)
    gexp = idx.unsqueeze(-1).expand(-1, -1, GEOM_DIM)
    sub_obj = torch.gather(obj, 1, exp)
    sub_geom = torch.gather(geom, 1, gexp)
    sub_mask = torch.ones(b, k_eff, dtype=torch.bool, device=device)
    fine_logits, _ = model.fine(sub_obj, sub_geom, sub_mask, parsed_list)
    full_logits = torch.full((b, n), float("-inf"), device=device, dtype=dtype)
    full_logits.scatter_(1, idx, fine_logits)
    full_logits = full_logits.masked_fill(~mask, float("-inf"))
    return full_logits


@dataclass
class RowAgg:
    """Per-sample scalars for router + oracles (built from batches)."""

    feature_rows: list[list[float]] = field(default_factory=list)
    logits_b: list[torch.Tensor] = field(default_factory=list)
    logits_c: list[torch.Tensor] = field(default_factory=list)
    mask_rows: list[torch.Tensor] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)
    correct_b: list[bool] = field(default_factory=list)
    correct_c: list[bool] = field(default_factory=list)
    meta_snap: list[dict[str, Any]] = field(default_factory=list)


def _relation_density(parsed: ParsedUtterance) -> float:
    reln = len(parsed.relation_types) if parsed.relation_types else 0
    ah = (parsed.anchor_head or "").strip().lower()
    extra = 1.0 if ah and ah != "object" else 0.0
    return float(reln) + extra


def collect_bc_rows(
    loader: DataLoader,
    model_b: RawTextRelationModel,
    model_c: RelationAwareModel,
    coarse_geom: torch.nn.Module | None,
    parser: CachedParser,
    device: torch.device,
    margin_thresh: float,
    rerank_k: int,
) -> RowAgg:
    agg = RowAgg()
    for batch in loader:
        b = to_dev_batch(batch, device)
        mask = b["object_mask"]
        target = b["target_index"]
        samples = b["samples_ref"]
        meta = json.loads(json.dumps(b["meta"]))  # deep copy without copy.deepcopy dependency issues
        lb = _forward_b(model_b, b, device)
        lc, p_anchor = _forward_c(model_c, b, parser, device)
        augment_meta_geometry_fallback_tags(meta, samples)
        augment_meta_with_model_margins(lb.detach().cpu(), mask.cpu(), meta, margin_thresh=margin_thresh)
        # second margin tag would overwrite — store B margin in features from raw logits below

        parsed_list = [parser.parse(s.utterance) for s in samples]
        gold_in_topk: list[bool] = []
        if coarse_geom is not None:
            with torch.no_grad():
                cl = coarse_forward(coarse_geom, b)
            k_eff = _effective_topk(mask, rerank_k)
            _, idx = torch.topk(cl.masked_fill(~mask, float("-inf")), k=k_eff, dim=1)
            for bi in range(target.size(0)):
                t = int(target[bi].item())
                gold_in_topk.append(bool((idx[bi] == t).any().item()))
        else:
            gold_in_topk = [False] * target.size(0)

        pred_b = lb.argmax(dim=-1)
        pred_c = lc.argmax(dim=-1)
        bs = target.size(0)
        for i in range(bs):
            mrow = mask[i]
            ti = int(target[i].item())
            margin_b = logit_top12_margin(lb[i].detach(), mrow)
            margin_c = logit_top12_margin(lc[i].detach(), mrow)
            p_i = parsed_list[i]
            pc = float(p_i.parser_confidence)
            ent = anchor_entropy(p_anchor[i], mrow)
            stags = compute_stratification_tags(samples[i], parser_confidence=pc)
            nobj = int(mrow.sum().item())
            gf = float(stags.get("geometry_fallback_fraction") or 0.0)
            clutter = 1.0 if stags.get("same_class_clutter") else 0.0
            cand = str(stags.get("candidate_load") or "")
            high_load = 1.0 if cand == "high" else 0.0
            geo_high_fb = 1.0 if gf > 0.5 else 0.0
            rd = _relation_density(p_i)
            pb = int(pred_b[i].item())
            pc_idx = int(pred_c[i].item())
            conf_b = logits_to_confidence_masked(lb[i], mrow, pb)
            conf_c = logits_to_confidence_masked(lc[i], mrow, pc_idx)
            ghit = 1.0 if gold_in_topk[i] else 0.0

            feat = [
                pc,
                ent,
                math.log(nobj + 1.0),
                gf,
                clutter,
                high_load,
                geo_high_fb,
                rd,
                margin_b,
                margin_c,
                conf_b,
                conf_c,
                ghit,
            ]
            agg.feature_rows.append(feat)
            agg.logits_b.append(lb[i].detach().cpu())
            agg.logits_c.append(lc[i].detach().cpu())
            agg.mask_rows.append(mrow.detach().cpu())
            agg.targets.append(ti)
            agg.correct_b.append(pb == ti)
            agg.correct_c.append(pc_idx == ti)
            agg.meta_snap.append(
                {
                    "candidate_load": cand,
                    "geometry_high_fallback": bool(geo_high_fb),
                    "geometry_fallback_fraction": gf,
                    "same_class_clutter": bool(stags.get("same_class_clutter")),
                    "n_objects": nobj,
                }
            )
    return agg


def _pad_logits(rows: list[torch.Tensor], masks: list[torch.Tensor], n_max: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(rows)
    lb = torch.full((n, n_max), float("-inf"), device=device)
    m = torch.zeros((n, n_max), dtype=torch.bool, device=device)
    for i, (row, mk) in enumerate(zip(rows, masks)):
        le = int(mk.sum().item())
        lb[i, :le] = row[:le].to(device)
        m[i, :le] = True
    return lb, m


def train_alpha_router(
    feat: torch.Tensor,
    logits_b: torch.Tensor,
    logits_c: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    steps: int = 400,
    lr: float = 0.02,
    device: torch.device | None = None,
) -> tuple[nn.Module, tuple[torch.Tensor, torch.Tensor]]:
    """Linear sigmoid alpha(x); minimize NLL under mixture p = a*softmax(B)+(1-a)*softmax(C) (stable for miscalibrated B/C scales)."""
    dev = device or feat.device
    n, fdim = feat.shape
    mean = feat.mean(dim=0)
    std = feat.std(dim=0).clamp_min(1e-3)
    feat_n = (feat - mean) / std
    torch.manual_seed(42)
    router = nn.Sequential(nn.Linear(fdim, 1), nn.Sigmoid()).to(dev)
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    feat_n = feat_n.to(dev)
    lb = logits_b.to(dev).detach()
    lc = logits_c.to(dev).detach()
    mask = mask.to(dev)
    target = target.to(dev)
    neg = torch.finfo(lb.dtype).min / 4
    log_pb = F.log_softmax(lb.masked_fill(~mask, neg), dim=-1)
    log_pc = F.log_softmax(lc.masked_fill(~mask, neg), dim=-1)
    router.train()
    for _ in range(steps):
        opt.zero_grad()
        a = router(feat_n).squeeze(-1).clamp(0.05, 0.95)
        # log p_mix = log( a exp(log_pb) + (1-a) exp(log_pc) )
        log_mix = torch.logsumexp(
            torch.stack(
                [
                    torch.log(a).unsqueeze(-1) + log_pb,
                    torch.log(1.0 - a).unsqueeze(-1) + log_pc,
                ],
                dim=0,
            ),
            dim=0,
        )
        nll = -log_mix[torch.arange(n, device=dev), target]
        loss = nll.mean()
        if torch.isnan(loss):
            break
        loss.backward()
        opt.step()
    router.eval()
    return router, (mean, std)


@torch.no_grad()
def eval_fusion(
    router: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    feat: torch.Tensor,
    logits_b: torch.Tensor,
    logits_c: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> tuple[float, list[float]]:
    router.eval()
    feat_n = ((feat - mean) / std).to(device)
    a = router(feat_n).squeeze(-1).clamp(0.05, 0.95)
    neg = torch.finfo(logits_b.dtype).min / 4
    lb = logits_b.to(device)
    lc = logits_c.to(device)
    m = mask.to(device)
    pb = F.softmax(lb.masked_fill(~m, neg), dim=-1)
    pc = F.softmax(lc.masked_fill(~m, neg), dim=-1)
    pm = a.unsqueeze(-1) * pb + (1.0 - a.unsqueeze(-1)) * pc
    pred = pm.argmax(dim=-1)
    correct = (pred == target.to(device)).float()
    return float(correct.mean().item()), a.cpu().tolist()


def _acc(correct: list[bool]) -> float:
    return sum(correct) / max(len(correct), 1)


def _subset_indices(meta_snap: list[dict], pred: callable) -> list[int]:
    return [i for i, m in enumerate(meta_snap) if pred(m)]


def regime_oracle_branch(correct_b: list[bool], correct_c: list[bool], meta: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}

    def stat(idxs: list[int]) -> dict[str, float]:
        if not idxs:
            return {"acc": float("nan"), "n": 0.0}
        acc = sum(correct_b[i] or correct_c[i] for i in idxs) / len(idxs)
        return {"acc": float(acc), "n": float(len(idxs))}

    n = len(correct_b)
    out["all"] = stat(list(range(n)))
    low = _subset_indices(meta, lambda m: m.get("candidate_load") == "low")
    high = _subset_indices(meta, lambda m: m.get("candidate_load") == "high")
    clutter = _subset_indices(meta, lambda m: m.get("same_class_clutter"))
    geo_hi = _subset_indices(meta, lambda m: m.get("geometry_high_fallback"))
    geo_lo = _subset_indices(meta, lambda m: not m.get("geometry_high_fallback"))
    out["candidate_load_low"] = stat(low)
    out["candidate_load_high"] = stat(high)
    out["same_class_clutter"] = stat(clutter)
    out["geometry_high_fallback"] = stat(geo_hi)
    out["geometry_not_high_fallback"] = stat(geo_lo)
    return out


def _write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)


def _plot_main(fig_path: Path, labels: list[str], values: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(labels))
    ax.bar(x, values, color="steelblue")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Acc@1 (val)")
    ax.set_ylim(0, max(0.05, max(values) * 1.15) if values else 1.0)
    ax.set_title("Hybrid router phase (validation)")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", type=str, default="", help="Output folder name suffix; default now")
    ap.add_argument("--manifest-train", type=Path, default=ROOT / "data/processed/train_manifest.jsonl")
    ap.add_argument("--manifest-val", type=Path, default=ROOT / "data/processed/val_manifest.jsonl")
    ap.add_argument("--ckpt-b", type=Path, default=ROOT / "outputs/checkpoints_nr3d_geom_first/raw_relation_last.pt")
    ap.add_argument("--ckpt-c", type=Path, default=ROOT / "outputs/checkpoints_nr3d_geom_first/relation_aware_last.pt")
    ap.add_argument("--two-stage-coarse", type=Path, default=ROOT / "outputs/checkpoints_stage1/coarse_geom_recall_last.pt")
    ap.add_argument("--two-stage-full", type=Path, default=ROOT / "outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt")
    ap.add_argument("--rerank-k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--margin-thresh", type=float, default=0.15)
    ap.add_argument("--router-steps", type=int, default=400)
    args = ap.parse_args()

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)
    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = stamp if str(stamp).endswith("_hybrid_router_phase") else f"{stamp}_hybrid_router_phase"
    out_root = ROOT / "outputs" / out_name
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "hybrid_router_log.txt"

    def log(msg: str) -> None:
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    mcfg = load_yaml_config(ROOT / "configs/model/relation_aware.yaml", ROOT)
    feat_dim = int(mcfg["object_dim"])
    parser = CachedParser(StructuredRuleParser(), ROOT / "data/parser_cache/hybrid_phase/structured")

    if not args.manifest_train.is_file() or not args.manifest_val.is_file():
        log("Missing train or val manifest — run prepare_data.py first.")
        return 1

    train_ds = ReferIt3DManifestDataset(_resolve(args.manifest_train, ROOT))
    val_ds = ReferIt3DManifestDataset(_resolve(args.manifest_val, ROOT))
    collate = make_grounding_collate_fn(feat_dim, attach_features=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    ckpt_b = _resolve(args.ckpt_b, ROOT)
    ckpt_c = _resolve(args.ckpt_c, ROOT)
    if not ckpt_b.is_file() or not ckpt_c.is_file():
        log(f"Missing B/C checkpoint: B={ckpt_b} C={ckpt_c}")
        return 1

    model_b = _load_bc("raw_text_relation", mcfg, ckpt_b, device)
    model_c = _load_bc("relation_aware", mcfg, ckpt_c, device)

    coarse_geom = None
    ts_coarse = _resolve(args.two_stage_coarse, ROOT)
    ts_full = _resolve(args.two_stage_full, ROOT)
    if ts_coarse.is_file() and ts_full.is_file():
        try:
            ts_model = load_two_stage_model(
                mcfg, ts_coarse, ts_full, int(args.rerank_k), device, "coarse_geom"
            )
        except Exception as e:
            log(f"Two-stage load failed: {e}")
            ts_model = None
    else:
        ts_model = None
        log("Two-stage checkpoints missing — skipping shortlist oracles.")

    from rag3d.evaluation.two_stage_eval import load_coarse_model

    if ts_coarse.is_file():
        try:
            coarse_geom = load_coarse_model(mcfg, ts_coarse, device, "coarse_geom")
        except Exception as e:
            log(f"Coarse geom load failed (shortlist proxy): {e}")
            coarse_geom = None

    log("Collecting train...")
    train_agg = collect_bc_rows(
        train_loader, model_b, model_c, coarse_geom, parser, device, args.margin_thresh, int(args.rerank_k)
    )
    log("Collecting val...")
    val_agg = collect_bc_rows(
        val_loader, model_b, model_c, coarse_geom, parser, device, args.margin_thresh, int(args.rerank_k)
    )

    n_max = max(
        max((int(m.sum().item()) for m in train_agg.mask_rows), default=0),
        max((int(m.sum().item()) for m in val_agg.mask_rows), default=0),
    )

    feat_t = torch.tensor(train_agg.feature_rows, dtype=torch.float32)
    lb_t, m_t = _pad_logits(train_agg.logits_b, train_agg.mask_rows, n_max, torch.device("cpu"))
    lc_t, _ = _pad_logits(train_agg.logits_c, train_agg.mask_rows, n_max, torch.device("cpu"))
    tgt_t = torch.tensor(train_agg.targets, dtype=torch.long)

    feat_v = torch.tensor(val_agg.feature_rows, dtype=torch.float32)
    lb_v, m_v = _pad_logits(val_agg.logits_b, val_agg.mask_rows, n_max, torch.device("cpu"))
    lc_v, _ = _pad_logits(val_agg.logits_c, val_agg.mask_rows, n_max, torch.device("cpu"))
    tgt_v = torch.tensor(val_agg.targets, dtype=torch.long)

    acc_b_val = _acc(val_agg.correct_b)
    acc_c_val = _acc(val_agg.correct_c)
    oracle_branch = [val_agg.correct_b[i] or val_agg.correct_c[i] for i in range(len(val_agg.correct_b))]
    acc_oracle_branch = _acc(oracle_branch)

    regime = regime_oracle_branch(val_agg.correct_b, val_agg.correct_c, val_agg.meta_snap)

    # Oracle geometry table: B, C, oracle on geo hi / lo
    def _geo_block(idxs: list[int]) -> dict[str, float]:
        if not idxs:
            return {"B": float("nan"), "C": float("nan"), "oracle_branch": float("nan"), "n": 0}
        cb = [val_agg.correct_b[i] for i in idxs]
        cc = [val_agg.correct_c[i] for i in idxs]
        ob = [oracle_branch[i] for i in idxs]
        return {"B": _acc(cb), "C": _acc(cc), "oracle_branch": _acc(ob), "n": len(idxs)}

    geo_hi_i = _subset_indices(val_agg.meta_snap, lambda m: m.get("geometry_high_fallback"))
    geo_lo_i = _subset_indices(val_agg.meta_snap, lambda m: not m.get("geometry_high_fallback"))
    gfs_vals = [float(m.get("geometry_fallback_fraction") or 0.0) for m in val_agg.meta_snap]
    uniq_gf = sorted(set(gfs_vals))
    if len(uniq_gf) >= 2:
        med_gf = float(uniq_gf[len(uniq_gf) // 2])
        geo_hi_median_i = _subset_indices(
            val_agg.meta_snap, lambda m: float(m.get("geometry_fallback_fraction") or 0.0) > med_gf
        )
        geo_lo_median_i = _subset_indices(
            val_agg.meta_snap, lambda m: float(m.get("geometry_fallback_fraction") or 0.0) <= med_gf
        )
    else:
        med_gf = float(uniq_gf[0]) if uniq_gf else 0.0
        geo_hi_median_i = []
        geo_lo_median_i = list(range(len(val_agg.meta_snap)))
    geo_results = {
        "subset_geometry_high_fallback": _geo_block(geo_hi_i),
        "subset_geometry_lower_fallback": _geo_block(geo_lo_i),
        "subset_geometry_fallback_gt_median": _geo_block(geo_hi_median_i),
        "subset_geometry_fallback_le_median": _geo_block(geo_lo_median_i),
        "geometry_fallback_median": med_gf,
        "definition": "Primary: geometry_high_fallback (>50% objects fallback_centroid). Fallback slices: median split on geometry_fallback_fraction when primary high-fallback slice is empty.",
    }

    # Train fusion router
    router, (mean, std) = train_alpha_router(
        feat_t, lb_t, lc_t, m_t, tgt_t, steps=int(args.router_steps), device=device
    )
    acc_fusion_val, alphas_val = eval_fusion(router, mean, std, feat_v, lb_v, lc_v, m_v, tgt_v, device)

    # Oracle shortlist (two-stage)
    shortlist_pack: dict[str, Any] = {"note": "two_stage not loaded"}
    if ts_model is not None:
        bottleneck = eval_two_stage_bottleneck(ts_model, val_loader, device, parser, args.margin_thresh)
        c1_oracle: list[bool] = []
        c5_oracle: list[bool] = []
        for batch in val_loader:
            b = to_dev_batch(batch, device)
            samples = b["samples_ref"]
            parsed_list = [parser.parse(s.utterance) for s in samples]
            logits_o = _two_stage_logits_oracle_shortlist(
                ts_model,
                {k: b[k] for k in ("object_features", "object_mask", "raw_texts", "samples_ref", "target_index")},
                parsed_list,
                device,
            )
            c1_oracle.extend(per_sample_correct_at1(logits_o, b["target_index"], b["object_mask"]))
            c5_oracle.extend(per_sample_correct_at5(logits_o, b["target_index"], b["object_mask"]))
        shortlist_pack = {
            "definition_natural": "eval_two_stage_bottleneck: coarse top-K as at eval time, no gold injection",
            "definition_oracle_shortlist": "Same rerank head but coarse shortlist always includes gold (_topk_union_target training=True, fine eval)",
            "natural": bottleneck,
            "oracle_shortlist_acc@1": sum(c1_oracle) / max(len(c1_oracle), 1),
            "oracle_shortlist_acc@5": sum(c5_oracle) / max(len(c5_oracle), 1),
            "n": len(c1_oracle),
        }

    # First linear weight magnitude as coarse feature importance proxy
    imp = router[0].weight.detach().abs().mean(dim=0).cpu().tolist()
    feat_names = [
        "parser_conf",
        "anchor_entropy",
        "log_n_objects",
        "geometry_fallback_fraction",
        "same_class_clutter",
        "candidate_load_high",
        "geometry_high_fallback",
        "relation_density",
        "margin_b",
        "margin_c",
        "conf_b_pred",
        "conf_c_pred",
        "coarse_gold_in_topk",
    ]
    importance = dict(zip(feat_names, [float(x) if math.isfinite(float(x)) else None for x in imp]))

    hybrid_results = {
        "val_n": len(val_agg.targets),
        "train_n": len(train_agg.targets),
        "acc_b": acc_b_val,
        "acc_c": acc_c_val,
        "acc_fusion_router": acc_fusion_val,
        "acc_oracle_branch_discrete": acc_oracle_branch,
        "oracle_branch_by_regime": regime,
        "feature_importance_mean_abs_weight": importance,
        "checkpoints": {"B": str(ckpt_b), "C": str(ckpt_c)},
    }

    (out_root / "oracle_branch_selector_results.json").write_text(
        json.dumps({"oracle_branch_acc": acc_oracle_branch, "by_regime": regime, "n": len(oracle_branch)}, indent=2),
        encoding="utf-8",
    )
    br_rows: list[list[Any]] = [["all", regime["all"]["acc"], int(regime["all"]["n"])]]
    for k in sorted(regime.keys()):
        if k == "all":
            continue
        br_rows.append([k, regime[k]["acc"], int(regime[k]["n"])])
    _write_csv(out_root / "oracle_branch_selector_table.csv", ["slice", "acc_oracle_branch", "n"], br_rows)

    (out_root / "oracle_geometry_results.json").write_text(json.dumps(geo_results, indent=2), encoding="utf-8")
    _write_csv(
        out_root / "oracle_geometry_table.csv",
        ["subset", "acc_B", "acc_C", "acc_oracle_branch", "n"],
        [
            ["geometry_high_fallback", geo_results["subset_geometry_high_fallback"]["B"], geo_results["subset_geometry_high_fallback"]["C"], geo_results["subset_geometry_high_fallback"]["oracle_branch"], geo_results["subset_geometry_high_fallback"]["n"]],
            ["lower_fallback", geo_results["subset_geometry_lower_fallback"]["B"], geo_results["subset_geometry_lower_fallback"]["C"], geo_results["subset_geometry_lower_fallback"]["oracle_branch"], geo_results["subset_geometry_lower_fallback"]["n"]],
            ["fallback_gt_median", geo_results["subset_geometry_fallback_gt_median"]["B"], geo_results["subset_geometry_fallback_gt_median"]["C"], geo_results["subset_geometry_fallback_gt_median"]["oracle_branch"], geo_results["subset_geometry_fallback_gt_median"]["n"]],
            ["fallback_le_median", geo_results["subset_geometry_fallback_le_median"]["B"], geo_results["subset_geometry_fallback_le_median"]["C"], geo_results["subset_geometry_fallback_le_median"]["oracle_branch"], geo_results["subset_geometry_fallback_le_median"]["n"]],
        ],
    )

    (out_root / "oracle_shortlist_results.json").write_text(json.dumps(shortlist_pack, indent=2), encoding="utf-8")
    nat = shortlist_pack.get("natural") or {}
    _write_csv(
        out_root / "oracle_shortlist_table.csv",
        ["metric", "value"],
        [
            ["shortlist_recall_natural", nat.get("shortlist_recall", "")],
            ["rerank_acc_given_target_in_shortlist_natural", nat.get("rerank_acc_given_target_in_shortlist", "")],
            ["acc@1_pipeline_natural", nat.get("acc@1", "")],
            ["oracle_shortlist_acc@1", shortlist_pack.get("oracle_shortlist_acc@1", "")],
            ["oracle_shortlist_acc@5", shortlist_pack.get("oracle_shortlist_acc@5", "")],
        ],
    )

    (out_root / "hybrid_router_results.json").write_text(json.dumps(hybrid_results, indent=2), encoding="utf-8")
    _write_csv(
        out_root / "hybrid_router_table.csv",
        ["method", "val_acc@1"],
        [
            ["B_raw_text", acc_b_val],
            ["C_structured", acc_c_val],
            ["fusion_alpha_router", acc_fusion_val],
            ["oracle_branch", acc_oracle_branch],
        ],
    )

    # Interpretation markdowns
    (out_root / "oracle_shortlist_interpretation.md").write_text(
        f"""# Oracle shortlist interpretation

## Definitions

- **Natural shortlist**: coarse top-K as deployed at evaluation time (`eval_two_stage_bottleneck`).
- **Oracle shortlist**: gold is **forced** into the K-slot shortlist; rerank head stays in eval mode (no dropout), then Acc@1/5 on full scene logits.

## Numbers (validation)

| Metric | Value |
|--------|------:|
| Shortlist recall (natural) | {nat.get("shortlist_recall", "n/a")} |
| Rerank Acc@1 given target in shortlist (natural) | {nat.get("rerank_acc_given_target_in_shortlist", "n/a")} |
| Pipeline Acc@1 (natural) | {nat.get("acc@1", "n/a")} |
| Acc@1 (oracle shortlist) | {shortlist_pack.get("oracle_shortlist_acc@1", "n/a")} |
| Acc@5 (oracle shortlist) | {shortlist_pack.get("oracle_shortlist_acc@5", "n/a")} |

## Questions

1. **Is retrieval the main bottleneck?** Compare **shortlist recall (natural)** to the gap between **oracle shortlist Acc@1** and **pipeline Acc@1**. Low recall with large oracle gap ⇒ retrieval limits the stack.
2. **If the target is in the shortlist, is rerank strong?** Use **rerank_acc_given_target_in_shortlist** from the natural run. Values well below oracle shortlist Acc@1 suggest rerank headroom even when gold is present; values near oracle shortlist (conditional) suggest rerank is doing most of what it can once gold is included.
""",
        encoding="utf-8",
    )

    b_hi, c_hi = geo_results["subset_geometry_high_fallback"]["B"], geo_results["subset_geometry_high_fallback"]["C"]
    b_lo, c_lo = geo_results["subset_geometry_lower_fallback"]["B"], geo_results["subset_geometry_lower_fallback"]["C"]
    b_gtm, c_gtm = geo_results["subset_geometry_fallback_gt_median"]["B"], geo_results["subset_geometry_fallback_gt_median"]["C"]
    b_lem, c_lem = geo_results["subset_geometry_fallback_le_median"]["B"], geo_results["subset_geometry_fallback_le_median"]["C"]
    (out_root / "oracle_geometry_interpretation.md").write_text(
        f"""# Oracle geometry interpretation

**Primary subsets:** **geometry_high_fallback** (>50% objects `fallback_centroid`) vs complement.

| Subset | Acc B | Acc C |
|--------|------:|------:|
| High fallback | {b_hi} | {c_hi} |
| Lower fallback | {b_lo} | {c_lo} |

**Median split** on `geometry_fallback_fraction` (used when the primary high-fallback slice has n=0): median = {med_gf:.4f}.

| Subset | Acc B | Acc C |
|--------|------:|------:|
| Above median fallback | {b_gtm} | {c_gtm} |
| At/below median | {b_lem} | {c_lem} |

## Questions

1. **Does C benefit more from good geometry than B?** Compare (Acc C − Acc B) on **lower fallback** / **below-median** vs **high fallback** / **above-median**.
2. **Is geometry limiting C?** If C gains vs B shrink on worse-geometry slices, that is **consistent with** geometry as a limiter (correlational).
""",
        encoding="utf-8",
    )

    gap_to_oracle = acc_oracle_branch - acc_fusion_val
    regime_lines = "\n".join(
        f"| `{k}` | {v.get('acc', float('nan')):.4f} | {int(v.get('n', 0))} |"
        for k, v in sorted(regime.items())
    )
    (out_root / "oracle_branch_selector_interpretation.md").write_text(
        f"""# Oracle branch selector interpretation

- **Oracle branch Acc@1** (val): **{acc_oracle_branch:.4f}** — per sample correct if **either** B or C is correct.
- **Standalone B**: {acc_b_val:.4f}; **standalone C**: {acc_c_val:.4f}.

| Regime | Oracle Acc@1 | n |
|--------|-------------:|--:|
{regime_lines}

## Questions

1. **Is complementarity enough to justify a router?** Oracle gain over max(B,C) = **{acc_oracle_branch - max(acc_b_val, acc_c_val):.4f}**. Non-trivial gain implies disagreement where exactly one line is right.
2. **Where is oracle gain largest?** Compare regimes above (candidate load, clutter, geometry).
""",
        encoding="utf-8",
    )

    (out_root / "hybrid_router_interpretation.md").write_text(
        f"""# Hybrid router interpretation

- **Val Acc@1 B**: {acc_b_val:.4f}
- **Val Acc@1 C**: {acc_c_val:.4f}
- **Val Acc@1 fusion** (learned $\\alpha(x)$): {acc_fusion_val:.4f}
- **Oracle branch upper bound**: {acc_oracle_branch:.4f}
- **Gap (oracle − fusion)**: {gap_to_oracle:.4f}

## Questions

1. **Does fusion beat both lines?** Check whether fusion exceeds **both** B and C; beating only one still can be useful.
2. **Gain size**: Compare fusion − max(B,C) to the oracle − max(B,C) **ceiling**; small gap suggests routing is already capturing most achievable complementarity with this feature set.
3. **Features**: see `hybrid_router_results.json` → `feature_importance_mean_abs_weight` (mean absolute weight per normalized input; coarse proxy only).
""",
        encoding="utf-8",
    )

    # Per-sample dump (val)
    per_lines = []
    for i in range(len(val_agg.targets)):
        per_lines.append(
            {
                "i": i,
                "target": val_agg.targets[i],
                "correct_b": val_agg.correct_b[i],
                "correct_c": val_agg.correct_c[i],
                "oracle_branch": oracle_branch[i],
                "alpha_fusion": alphas_val[i],
            }
        )
    (out_root / "hybrid_router_per_sample_val.jsonl").write_text(
        "\n".join(json.dumps(x) for x in per_lines), encoding="utf-8"
    )

    labels = ["B", "C", "fusion", "oracle_branch"]
    vals = [acc_b_val, acc_c_val, acc_fusion_val, acc_oracle_branch]
    _plot_main(out_root / "hybrid_router_main_figure.png", labels, vals)

    # repro_commands.sh
    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    repro = out_root / "repro_commands.sh"
    repro.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cd "{ROOT}"
PYTHONPATH=src python scripts/run_hybrid_router_phase.py \\
  --stamp {stamp} \\
  --manifest-train {_rel(args.manifest_train)} \\
  --manifest-val {_rel(args.manifest_val)} \\
  --ckpt-b {_rel(ckpt_b)} \\
  --ckpt-c {_rel(ckpt_c)} \\
  --two-stage-coarse {_rel(ts_coarse)} \\
  --two-stage-full {_rel(ts_full)} \\
  --rerank-k {args.rerank_k} --device {args.device} --router-steps {args.router_steps}
""",
        encoding="utf-8",
    )
    repro.chmod(0o755)

    bundle = out_root / "report_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "README.md").write_text(
        """# Report bundle — hybrid router phase

| Artifact | Claim supported |
|----------|-----------------|
| `oracle_branch_selector_*` | **Complementarity**: upper bound if an oracle picked the better B/C branch per sample. |
| `oracle_geometry_*` | **Geometry and C**: whether structured line tracks geometry quality differently from B (correlational). |
| `oracle_shortlist_*` | **Retrieval vs rerank**: recall under natural shortlist vs rerank strength when gold is guaranteed in K. |
| `hybrid_router_*` | **Actionable hybrid**: whether a lightweight $\\alpha(x)$ fusion on frozen B/C recovers part of the oracle gain. |
| `hybrid_router_main_figure.png` | One-slide comparison of B, C, fusion, oracle branch on val. |

Regenerate the parent directory via `repro_commands.sh` in this folder’s parent.
""",
        encoding="utf-8",
    )

    log(f"Done. Outputs -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
