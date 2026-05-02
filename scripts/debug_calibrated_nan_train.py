#!/usr/bin/env python3
"""Debug script to locate NaN during training step."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.models.cover3d_dense_relation import DenseRelationModule
from rag3d.models.cover3d_calibration import CalibratedFusionGate

def check_tensor(name, tensor, prefix=""):
    """Check tensor for NaN/inf."""
    if tensor is None:
        return False
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan:
        print(f"{prefix}*** NaN in {name}! shape={list(tensor.shape)} ***")
    if has_inf:
        print(f"{prefix}*** Inf in {name}! shape={list(tensor.shape)} ***")
    return has_nan or has_inf

def get_grad_stats(module):
    """Get gradient stats for all parameters."""
    stats = {}
    for name, param in module.named_parameters():
        if param.grad is not None:
            stats[name] = {
                "has_nan": torch.isnan(param.grad).any().item(),
                "has_inf": torch.isinf(param.grad).any().item(),
                "norm": param.grad.norm().item(),
                "max": param.grad.abs().max().item(),
            }
    return stats

def main():
    print("=" * 60)
    print("Dense-calibrated Training Step NaN Debug")
    print("=" * 60)
    
    # Create sample data
    B, N, D_obj, D_lang = 4, 20, 320, 256
    
    print(f"\nCreating sample data: B={B}, N={N}")
    
    torch.manual_seed(42)
    object_embeddings = torch.randn(B, N, D_obj)
    lang_features = torch.randn(B, D_lang)
    base_logits = torch.randn(B, N)
    object_mask = torch.ones(B, N, dtype=torch.bool)
    target_index = torch.randint(0, N, (B,))
    
    # Mask some positions
    object_mask[:, 15:] = False
    object_embeddings[:, 15:] = 0.0
    base_logits[:, 15:] = 0.0
    
    # Modules
    dense_module = DenseRelationModule(
        object_dim=D_obj,
        language_dim=D_lang,
        geometry_dim=6,
        hidden_dim=256,
        chunk_size=16,
        use_geometry=False,
    )
    
    calibration = CalibratedFusionGate(
        signal_dim=4,
        hidden_dim=32,
        min_gate=0.1,
        max_gate=0.9,
        init_bias=0.3,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    dense_module = dense_module.to(device)
    calibration = calibration.to(device)
    object_embeddings = object_embeddings.to(device)
    lang_features = lang_features.to(device)
    base_logits = base_logits.to(device)
    object_mask = object_mask.to(device)
    target_index = target_index.to(device)
    
    # Optimizer - only optimize calibration (dense-no-cal works, so issue is here)
    optimizer = torch.optim.AdamW(
        list(dense_module.parameters()) + list(calibration.parameters()),
        lr=1e-4
    )
    
    print("\n" + "=" * 60)
    print("Running 5 training steps...")
    print("=" * 60)
    
    for step in range(5):
        print(f"\n--- Step {step + 1} ---")
        
        # Forward
        dense_result = dense_module(
            object_embeddings=object_embeddings,
            utterance_features=lang_features,
            candidate_mask=object_mask,
        )
        relation_scores = dense_result["relation_scores"]
        
        # Check relation_scores before calibration
        if check_tensor("relation_scores (pre-cal)", relation_scores, "  "):
            print("  >> NaN in relation_scores BEFORE calibration!")
        
        # Calibration signals
        base_sorted = base_logits.sort(dim=-1, descending=True).values
        base_margin = base_sorted[:, 0] - base_sorted[:, 1]
        
        relation_sorted = relation_scores.sort(dim=-1, descending=True).values
        relation_margin = relation_sorted[:, 0] - relation_sorted[:, 1]
        
        relation_probs = F.softmax(relation_scores.masked_fill(~object_mask, float("-inf")), dim=-1)
        anchor_entropy = -(relation_probs * relation_probs.clamp(min=1e-8).log()).sum(dim=-1)
        
        # Calibration forward
        cal_result = calibration(
            base_logits=base_logits,
            relation_scores=relation_scores,
            anchor_posterior=relation_probs,
            anchor_entropy=anchor_entropy,
            base_margin=base_margin,
            relation_margin=relation_margin,
            candidate_mask=object_mask,
        )
        
        gate_values = cal_result["gate_values"]
        fused_logits = cal_result["fused_logits"]
        
        if check_tensor("gate_values", gate_values, "  "):
            print("  >> NaN in gate_values!")
        if check_tensor("fused_logits", fused_logits, "  "):
            print("  >> NaN in fused_logits!")
        
        # Loss
        loss = F.cross_entropy(fused_logits, target_index)
        
        if not torch.isfinite(loss):
            print(f"  >> Loss is not finite: {loss.item()}")
            break
        
        print(f"  Loss: {loss.item():.4f}")
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        print("  Gradient stats:")
        grad_stats = get_grad_stats(dense_module)
        dense_has_nan = any(s["has_nan"] or s["has_inf"] for s in grad_stats.values())
        cal_grad_stats = get_grad_stats(calibration)
        cal_has_nan = any(s["has_nan"] or s["has_inf"] for s in cal_grad_stats.values())
        
        if dense_has_nan:
            print("    DenseRelationModule gradients have NaN/Inf!")
            for name, s in grad_stats.items():
                if s["has_nan"] or s["has_inf"]:
                    print(f"      {name}: nan={s['has_nan']}, inf={s['inf']}, norm={s['norm']}")
        
        if cal_has_nan:
            print("    CalibratedFusionGate gradients have NaN/Inf!")
            for name, s in cal_grad_stats.items():
                if s["has_nan"] or s["has_inf"]:
                    print(f"      {name}: nan={s['has_nan']}, inf={s['inf']}, norm={s['norm']}")
        
        # Check individual gradient norms
        all_params = list(dense_module.named_parameters()) + list(calibration.named_parameters())
        max_grad_norm = 0
        max_grad_param = ""
        for name, param in all_params:
            if param.grad is not None:
                norm = param.grad.norm().item()
                if norm > max_grad_norm:
                    max_grad_norm = norm
                    max_grad_param = name
        
        print(f"    Max grad norm: {max_grad_norm:.4f} ({max_grad_param})")
        
        # Step
        optimizer.step()
        
        # Check weights for NaN
        for name, param in list(dense_module.named_parameters()) + list(calibration.named_parameters()):
            if torch.isnan(param).any():
                print(f"  >> NaN in weights: {name}!")
            if torch.isinf(param).any():
                print(f"  >> Inf in weights: {name}!")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("If NaN appeared only in gradients, the issue is gradient explosion.")
    print("If NaN appeared in forward, the issue is numerical stability.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
