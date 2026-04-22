#!/usr/bin/env python3
"""Debug: verify fix for calibration NaN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.models.cover3d_dense_relation import DenseRelationModule
from rag3d.models.cover3d_calibration import CalibratedFusionGate

def main():
    print("=" * 60)
    print("Dense-calibrated NaN Fix Verification")
    print("=" * 60)
    
    B, N, D_obj, D_lang = 4, 20, 320, 256
    
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    object_embeddings = torch.randn(B, N, D_obj, device=device)
    lang_features = torch.randn(B, D_lang, device=device)
    base_logits = torch.randn(B, N, device=device)
    object_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    
    object_mask[:, 15:] = False
    object_embeddings[:, 15:] = 0.0
    base_logits[:, 15:] = 0.0
    
    target_index = torch.randint(0, 15, (B,), device=device)
    
    dense_module = DenseRelationModule(
        object_dim=D_obj, language_dim=D_lang, geometry_dim=6,
        hidden_dim=256, chunk_size=16, use_geometry=False,
    ).to(device)
    
    calibration = CalibratedFusionGate(
        signal_dim=4, hidden_dim=32, min_gate=0.1, max_gate=0.9, init_bias=0.3,
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        list(dense_module.parameters()) + list(calibration.parameters()),
        lr=1e-4
    )
    
    print("\nRunning 5 training steps with FIX...")
    
    all_ok = True
    for step in range(5):
        optimizer.zero_grad()
        
        dense_result = dense_module(
            object_embeddings=object_embeddings,
            utterance_features=lang_features,
            candidate_mask=object_mask,
        )
        relation_scores = dense_result["relation_scores"]
        
        base_sorted = base_logits.sort(dim=-1, descending=True).values
        base_margin = base_sorted[:, 0] - base_sorted[:, 1]
        relation_sorted = relation_scores.sort(dim=-1, descending=True).values
        relation_margin = relation_sorted[:, 0] - relation_sorted[:, 1]
        relation_probs = F.softmax(relation_scores.masked_fill(~object_mask, float("-inf")), dim=-1)
        anchor_entropy = -(relation_probs * relation_probs.clamp(min=1e-8).log()).sum(dim=-1)
        
        # === FIX: Use clamped relation for fusion ===
        safe_relation = relation_scores.clamp(min=-1e6)
        
        cal_result = calibration(
            base_logits=base_logits,
            relation_scores=safe_relation,  # Use safe relation
            anchor_posterior=relation_probs,
            anchor_entropy=anchor_entropy,
            base_margin=base_margin,
            relation_margin=relation_margin,
            candidate_mask=object_mask,
        )
        
        gate_values = cal_result["gate_values"]
        fused_logits = cal_result["fused_logits"]
        fused_logits = fused_logits.masked_fill(~object_mask, float("-inf"))
        
        loss = F.cross_entropy(fused_logits, target_index)
        
        if not torch.isfinite(loss):
            print(f"Step {step+1}: LOSS NOT FINITE: {loss.item()}")
            all_ok = False
            break
        
        loss.backward()
        
        # Check gradients
        has_nan_grad = False
        for name, param in list(dense_module.named_parameters()) + list(calibration.named_parameters()):
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                print(f"Step {step+1}: NaN/Inf grad in {name}")
                has_nan_grad = True
                all_ok = False
        
        optimizer.step()
        
        # Check weights
        has_nan_weight = False
        for name, param in list(dense_module.named_parameters()) + list(calibration.named_parameters()):
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Step {step+1}: NaN/Inf weight in {name}")
                has_nan_weight = True
                all_ok = False
        
        print(f"Step {step+1}: Loss={loss.item():.4f}, gate_mean={gate_values.mean().item():.4f}, ok={not has_nan_grad and not has_nan_weight}")
    
    print("\n" + "=" * 60)
    if all_ok:
        print("SUCCESS: All 5 steps completed without NaN!")
    else:
        print("FAILURE: NaN detected")
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
