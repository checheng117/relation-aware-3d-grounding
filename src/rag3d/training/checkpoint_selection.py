"""End-to-end two-stage validation metrics for checkpoint selection (coarse + fixed rerank)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rag3d.evaluation.two_stage_rerank_metrics import eval_two_stage_inject_mode
from rag3d.parsers.cached_parser import CachedParser
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.parsers.structured_rule_parser import StructuredRuleParser
from rag3d.relation_reasoner.two_stage_rerank import RelationAwareGeomModel, TwoStageCoarseRerankModel
from rag3d.utils.config import load_yaml_config


@dataclass
class CoarsePipelineSelectionConfig:
    """Select coarse checkpoints by natural two-stage val Acc@1 with a frozen reference reranker."""

    enabled: bool = False
    reference_rerank_checkpoint: Path | None = None
    reference_label: str | None = None
    model_config: Path | None = None  # e.g. configs/model/relation_aware.yaml
    parser_mode: str = "structured"
    parser_cache_dir: Path | None = None
    rerank_k: int = 10
    margin_thresh: float = 0.15
    coarse_kind: str = "coarse_geom"


def _load_fine_weights_from_two_stage_ckpt(fine: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location=device)
    sd = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    fine_sd = {k[len("fine.") :]: v for k, v in sd.items() if k.startswith("fine.")}
    if not fine_sd:
        raise ValueError(f"No fine.* keys in checkpoint {ckpt_path}")
    fine.load_state_dict(fine_sd, strict=True)


def build_parser_for_selection(cfg: CoarsePipelineSelectionConfig, base_dir: Path) -> CachedParser:
    pdir = cfg.parser_cache_dir or (base_dir / "data/parser_cache/selection")
    if not pdir.is_absolute():
        pdir = base_dir / pdir
    mode = str(cfg.parser_mode).lower()
    if mode == "structured":
        inner = StructuredRuleParser()
        pdir = pdir / "structured"
    elif mode == "heuristic":
        inner = HeuristicParser()
        pdir = pdir / "heuristic"
    else:
        raise ValueError(f"Unknown parser_mode {cfg.parser_mode!r}")
    return CachedParser(inner, pdir)


def evaluate_coarse_with_fixed_rerank(
    coarse_train: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    base_dir: Path,
    sel: CoarsePipelineSelectionConfig,
) -> dict[str, Any]:
    """Natural + oracle shortlist metrics; primary key: val_pipeline_natural_acc@1."""
    if not sel.reference_rerank_checkpoint or not sel.reference_rerank_checkpoint.is_file():
        return {}
    mcfg_path = sel.model_config
    if mcfg_path is None:
        mcfg_path = base_dir / "configs/model/relation_aware.yaml"
    elif not mcfg_path.is_absolute():
        mcfg_path = base_dir / mcfg_path
    mcfg = load_yaml_config(mcfg_path, base_dir=base_dir)

    fine = RelationAwareGeomModel(
        int(mcfg["object_dim"]),
        int(mcfg["language_dim"]),
        int(mcfg["hidden_dim"]),
        int(mcfg["relation_dim"]),
        anchor_temperature=float(mcfg.get("anchor_temperature", 1.0)),
        dropout=float(mcfg.get("dropout", 0.1)),
    ).to(device)
    _load_fine_weights_from_two_stage_ckpt(fine, sel.reference_rerank_checkpoint, device)

    wrapped = TwoStageCoarseRerankModel(
        coarse_train,
        fine,
        rerank_k=int(sel.rerank_k),
        freeze_coarse=False,
    ).to(device)
    wrapped.eval()
    parser = build_parser_for_selection(sel, base_dir)

    nat = eval_two_stage_inject_mode(
        wrapped,
        val_loader,
        device,
        parser,
        float(sel.margin_thresh),
        inject_gold_in_shortlist=False,
    )
    ora = eval_two_stage_inject_mode(
        wrapped,
        val_loader,
        device,
        parser,
        float(sel.margin_thresh),
        inject_gold_in_shortlist=True,
    )
    out: dict[str, Any] = {
        "val_pipeline_natural_acc@1": float(nat["acc@1"]),
        "val_pipeline_natural_acc@5": float(nat["acc@5"]),
        "val_pipeline_natural_mrr": float(nat["mrr"]),
        "val_pipeline_natural_shortlist_recall": float(nat["shortlist_recall"]),
        "val_pipeline_natural_cond_acc_in_k": float(nat["rerank_acc_given_gold_in_shortlist"]),
        "val_pipeline_oracle_acc@1": float(ora["acc@1"]),
        "val_pipeline_oracle_mrr": float(ora["mrr"]),
        "val_pipeline_selection_reference_checkpoint": str(sel.reference_rerank_checkpoint),
    }
    if sel.reference_label:
        out["val_pipeline_selection_reference_label"] = str(sel.reference_label)
    return out


def coarse_selection_config_from_yaml(
    raw: dict[str, Any] | None,
    base_dir: Path,
    coarse_kind: str,
) -> CoarsePipelineSelectionConfig | None:
    if not raw or not raw.get("enabled"):
        return None
    ref = raw.get("reference_rerank_checkpoint")
    if not ref:
        return None
    rp = Path(ref)
    if not rp.is_absolute():
        rp = base_dir / rp
    mc = raw.get("model_config")
    mp = Path(mc) if mc else None
    if mp and not mp.is_absolute():
        mp = base_dir / mp
    pcd = raw.get("parser_cache_dir")
    pp = Path(pcd) if pcd else None
    if pp and not pp.is_absolute():
        pp = base_dir / pp
    return CoarsePipelineSelectionConfig(
        enabled=True,
        reference_rerank_checkpoint=rp,
        reference_label=str(raw.get("reference_label", "")).strip() or None,
        model_config=mp,
        parser_mode=str(raw.get("parser_mode", "structured")),
        parser_cache_dir=pp,
        rerank_k=int(raw.get("rerank_k", 10)),
        margin_thresh=float(raw.get("margin_thresh", 0.15)),
        coarse_kind=str(coarse_kind),
    )
