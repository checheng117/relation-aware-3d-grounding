#!/usr/bin/env python3
"""Train fine relation-aware reranker on top-K coarse candidates (coarse checkpoint frozen)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from rag3d.datasets.collate import make_grounding_collate_fn
from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.parsers.cached_parser import CachedParser
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.parsers.structured_rule_parser import StructuredRuleParser
from rag3d.relation_reasoner.losses import compute_batch_training_loss
from rag3d.relation_reasoner.model import AttributeOnlyModel, CoarseGeomAttributeModel
from rag3d.relation_reasoner.two_stage_rerank import (
    RelationAwareGeomModel,
    TwoStageCoarseRerankModel,
    forward_two_stage_rerank,
)
from rag3d.evaluation.two_stage_rerank_metrics import eval_two_stage_inject_mode
from rag3d.training.runner import TrainingConfig, _to_device, build_loaders
from rag3d.utils.config import load_yaml_config
from rag3d.utils.logging import setup_logging
from rag3d.utils.seed import set_seed
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def _resolve(p: Path | None, base: Path) -> Path | None:
    if p is None:
        return None
    return p if p.is_absolute() else (base / p).resolve()


def _manifest_paths(tcfg: dict, dcfg: dict, base: Path) -> tuple[Path, Path | None]:
    proc = Path(dcfg.get("processed_dir", "data/processed"))
    if not proc.is_absolute():
        proc = base / proc
    tr = tcfg.get("train_manifest")
    va = tcfg.get("val_manifest")
    train_p = _resolve(Path(tr), base) if tr else proc / "train_manifest.jsonl"
    val_p = _resolve(Path(va), base) if va else proc / "val_manifest.jsonl"
    return train_p, val_p if val_p.is_file() else None


def _gold_in_shortlist_mask(aux: dict[str, object], target_index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    idx = aux.get("rerank_idx")
    if not isinstance(idx, torch.Tensor):
        raise KeyError("Two-stage rerank aux is missing tensor rerank_idx")
    valid_target = (target_index >= 0) & (target_index < mask.size(1))
    safe_target = target_index.clamp(min=0, max=max(mask.size(1) - 1, 0))
    target_mask = mask.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    return valid_target & target_mask & (idx == target_index.unsqueeze(1)).any(dim=1)


def _snapshot_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Stable checkpoint snapshot detached from live GPU parameter storage."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _apply_fine_tune_mode(model: TwoStageCoarseRerankModel, mode: str) -> str:
    """Configure trainable fine-submodule parameters for conservative second-pass variants."""
    mode_norm = (mode or "full").strip().lower()
    for p in model.fine.parameters():
        p.requires_grad = False

    if mode_norm == "full":
        for p in model.fine.parameters():
            p.requires_grad = True
    elif mode_norm == "attr_rel_heads":
        for p in model.fine.attr.parameters():
            p.requires_grad = True
        for p in model.fine.rel.parameters():
            p.requires_grad = True
    elif mode_norm == "relation_head_only":
        for p in model.fine.rel.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown fine_tune_mode={mode!r}")

    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    total = sum(int(p.numel()) for p in model.parameters())
    log.info("fine_tune_mode=%s trainable_params=%d total_params=%d", mode_norm, trainable, total)
    return mode_norm


def run_two_stage_training(
    model: TwoStageCoarseRerankModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: TrainingConfig,
    parser: CachedParser,
    model_name: str,
    *,
    selection_margin_thresh: float = 0.15,
    early_stop_patience: int = 0,
    min_delta: float = 0.0,
    fine_tune_mode: str = "full",
) -> None:
    set_seed(cfg.seed)
    dev = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    used_fine_tune_mode = _apply_fine_tune_mode(model, fine_tune_mode)
    model = model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_natural = -1.0
    best_natural_epoch = -1
    no_improve_epochs = 0
    margin_sel = float(selection_margin_thresh)
    stop_reason = "max_epochs_reached"
    for epoch in range(cfg.epochs):
        model.train()
        losses: list[float] = []
        n_batches = 0
        train_rows = 0
        train_valid_rows = 0
        nan_or_inf_batch_count = 0
        for batch in train_loader:
            if cfg.debug_max_batches is not None and n_batches >= cfg.debug_max_batches:
                break
            n_batches += 1
            batch_d = _to_device(batch, dev)
            opt.zero_grad()
            logits, aux = forward_two_stage_rerank(model, batch_d, parser, return_aux=True)
            valid_rows = _gold_in_shortlist_mask(aux, batch_d["target_index"], batch_d["object_mask"])
            train_rows += int(valid_rows.numel())
            train_valid_rows += int(valid_rows.sum().item())
            samples = batch_d.get("samples_ref")
            loss = compute_batch_training_loss(
                logits,
                batch_d["target_index"],
                batch_d["object_mask"],
                cfg.loss,
                samples if isinstance(samples, list) else None,
                meta=batch_d.get("meta"),
                valid_rows=valid_rows,
            )
            if not torch.isfinite(loss):
                nan_or_inf_batch_count += 1
                log.warning(
                    "Skipping non-finite rerank batch loss at epoch=%s batch=%s valid_rows=%s/%s",
                    epoch,
                    n_batches - 1,
                    int(valid_rows.sum().item()),
                    int(valid_rows.numel()),
                )
                continue
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        train_valid_fraction = train_valid_rows / max(train_rows, 1)
        row: dict = {
            "epoch": epoch,
            "train_loss_mean": sum(losses) / max(len(losses), 1),
            "fine_tune_mode": used_fine_tune_mode,
            "rerank_train_valid_fraction": train_valid_fraction,
            "gold_in_shortlist_rate_train": train_valid_fraction,
            "train_rows": train_rows,
            "train_valid_rows": train_valid_rows,
            "nan_or_inf_batch_count": nan_or_inf_batch_count,
        }
        if val_loader is not None:
            model.eval()
            nat = eval_two_stage_inject_mode(model, val_loader, dev, parser, margin_sel, False)
            row["val_acc@1"] = float(nat["acc@1"])
            ora = eval_two_stage_inject_mode(model, val_loader, dev, parser, margin_sel, True)
            row["val_natural_two_stage_acc@1"] = float(nat["acc@1"])
            row["val_natural_two_stage_acc@5"] = float(nat["acc@5"])
            row["val_natural_two_stage_mrr"] = float(nat["mrr"])
            row["val_natural_shortlist_recall"] = float(nat["shortlist_recall"])
            row["val_natural_cond_acc_in_shortlist"] = float(nat["rerank_acc_given_gold_in_shortlist"])
            row["gold_in_shortlist_rate_val"] = float(nat["shortlist_recall"])
            row["val_oracle_shortlist_acc@1"] = float(ora["acc@1"])
            row["val_oracle_shortlist_mrr"] = float(ora["mrr"])
            primary = float(nat["acc@1"])
            if primary > (best_natural + float(min_delta)):
                best_natural = primary
                best_natural_epoch = epoch
                no_improve_epochs = 0
                bp = cfg.checkpoint_dir / f"{model_name}_best_natural_two_stage.pt"
                torch.save({"model": _snapshot_state_dict(model), "epoch": epoch, "metrics": row}, bp)
                log.info("New best natural two-stage val acc@1=%.6f -> %s", primary, bp)
            else:
                no_improve_epochs += 1
            row["best_natural_two_stage_acc@1_so_far"] = float(best_natural)
            row["best_natural_two_stage_epoch_so_far"] = int(best_natural_epoch)
            row["no_improve_epochs"] = int(no_improve_epochs)
            row["early_stop_patience"] = int(early_stop_patience)
        log.info("epoch %s metrics %s", epoch, row)
        history.append(row)
        with cfg.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        ckpt_path = cfg.checkpoint_dir / f"{model_name}_epoch{epoch}.pt"
        torch.save({"model": _snapshot_state_dict(model), "epoch": epoch, "metrics": row}, ckpt_path)
        if val_loader is not None and early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
            stop_reason = f"early_stop_no_improve_{early_stop_patience}"
            log.info("Early stop at epoch=%s (no_improve_epochs=%s)", epoch, no_improve_epochs)
            break

    torch.save(
        {
            "model": _snapshot_state_dict(model),
            "history": history,
            "best_natural_two_stage_acc@1": float(best_natural),
            "best_natural_two_stage_epoch": int(best_natural_epoch),
            "stop_reason": stop_reason,
            "fine_tune_mode": used_fine_tune_mode,
        },
        cfg.checkpoint_dir / f"{model_name}_last.pt",
    )
    log.info("Saved %s", cfg.checkpoint_dir / f"{model_name}_last.pt")


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/train/rerank/rerank_full_k10.yaml")
    args = ap.parse_args()
    tcfg = load_yaml_config(args.config, base_dir=ROOT)
    dcfg = load_yaml_config(ROOT / tcfg["dataset_config"], base_dir=ROOT)
    mcfg = load_yaml_config(ROOT / "configs/model" / f"{tcfg['model']}.yaml", base_dir=ROOT)
    set_seed(int(tcfg.get("seed", 42)))
    device_s = str(tcfg.get("device", "cpu"))
    device = torch.device(device_s if torch.cuda.is_available() and device_s == "cuda" else "cpu")
    feat_dim = int(mcfg["object_dim"])
    run_name = str(tcfg.get("run_name", "rerank_full_k10"))

    cache_dir = Path(tcfg.get("parser_cache_dir", "data/parser_cache"))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    parser_mode = str(tcfg.get("parser_mode", "structured")).lower()
    if parser_mode == "structured":
        inner = StructuredRuleParser()
        cache_dir = cache_dir / "structured"
    elif parser_mode == "heuristic":
        inner = HeuristicParser()
        cache_dir = cache_dir / "heuristic"
    else:
        log.error("Unknown parser_mode %r", parser_mode)
        sys.exit(1)
    parser = CachedParser(inner, cache_dir)

    coarse_ckpt = _resolve(Path(tcfg["coarse_checkpoint"]), ROOT)
    if coarse_ckpt is None or not coarse_ckpt.is_file():
        log.error("coarse_checkpoint missing or not a file: %s", tcfg.get("coarse_checkpoint"))
        sys.exit(1)

    coarse_kind = str(tcfg.get("coarse_model", "attribute_only")).lower()
    if coarse_kind == "coarse_geom":
        coarse = CoarseGeomAttributeModel(
            int(mcfg["object_dim"]),
            int(mcfg["language_dim"]),
            int(mcfg["hidden_dim"]),
            dropout=float(mcfg.get("dropout", 0.1)),
        )
    else:
        coarse = AttributeOnlyModel(
            int(mcfg["object_dim"]),
            int(mcfg["language_dim"]),
            int(mcfg["hidden_dim"]),
            dropout=float(mcfg.get("dropout", 0.1)),
        )
    try:
        coarse_sd = torch.load(coarse_ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        coarse_sd = torch.load(coarse_ckpt, map_location="cpu")
    if isinstance(coarse_sd, dict) and "model" in coarse_sd:
        coarse_sd = coarse_sd["model"]
    coarse.load_state_dict(coarse_sd, strict=True)
    log.info("Loaded coarse weights from %s", coarse_ckpt)

    fine = RelationAwareGeomModel(
        int(mcfg["object_dim"]),
        int(mcfg["language_dim"]),
        int(mcfg["hidden_dim"]),
        int(mcfg["relation_dim"]),
        anchor_temperature=float(mcfg.get("anchor_temperature", 1.0)),
        dropout=float(mcfg.get("dropout", 0.1)),
    )
    rerank_k = int(tcfg.get("rerank_k", 10))
    model = TwoStageCoarseRerankModel(coarse, fine, rerank_k=rerank_k)
    model.shortlist_train_inject_gold = bool(tcfg.get("shortlist_train_inject_gold", True))

    fi = tcfg.get("fine_init_checkpoint")
    if fi:
        fp = _resolve(Path(fi), ROOT)
        if fp is not None and fp.is_file():
            try:
                payload = torch.load(fp, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(fp, map_location="cpu")
            sd = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            fine_sd = {k[len("fine.") :]: v for k, v in sd.items() if k.startswith("fine.")}
            if fine_sd:
                model.fine.load_state_dict(fine_sd, strict=True)
                log.info("Loaded fine submodule init from %s (%d tensors)", fp, len(fine_sd))

    train_p, val_p = _manifest_paths(tcfg, dcfg, ROOT)
    if not train_p.is_file():
        log.error("Train manifest missing: %s", train_p)
        sys.exit(1)

    tconf = TrainingConfig(
        epochs=int(tcfg.get("epochs", 8)),
        batch_size=int(tcfg.get("batch_size", 16)),
        lr=float(tcfg.get("lr", 1e-4)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        seed=int(tcfg.get("seed", 42)),
        feat_dim=feat_dim,
        checkpoint_dir=_resolve(Path(tcfg["checkpoint_dir"]), ROOT) or ROOT / "outputs/checkpoints_rerank",
        metrics_path=_resolve(Path(tcfg.get("metrics_file", "outputs/metrics/rerank_default.jsonl")), ROOT)
        or ROOT / "outputs/metrics/rerank_default.jsonl",
        device=device_s if device.type == "cuda" else "cpu",
        debug_max_batches=tcfg.get("debug_max_batches"),
        loss=dict(tcfg.get("loss") or {}),
    )
    tconf.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if tconf.metrics_path.is_file():
        tconf.metrics_path.unlink()

    train_loader, val_loader = build_loaders(
        train_p,
        val_p,
        tconf,
        num_workers=int(tcfg.get("num_workers", 0)),
    )
    run_two_stage_training(
        model,
        train_loader,
        val_loader,
        tconf,
        parser,
        run_name,
        selection_margin_thresh=float(tcfg.get("selection_margin_thresh", 0.15)),
        early_stop_patience=int(tcfg.get("early_stop_patience", 0)),
        min_delta=float(tcfg.get("min_delta", 0.0)),
        fine_tune_mode=str(tcfg.get("fine_tune_mode", "full")),
    )


if __name__ == "__main__":
    main()
