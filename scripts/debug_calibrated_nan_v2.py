#!/usr/bin/env python3
"""Debug: isolate calibration NaN with valid targets."""

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
    print("Dense-calibrated NaN Debug v2 (valid targets)")
    print("=" * 60)
    
    B, N, D_obj, D_lang = 4, 20, 320, 256
    
    torch.manual_seed(42)
    object_embeddings = torch.randn(B, N, D_obj)
    lang_features = torch.randn(B, D_lang)
    base_logits = torch.randn(B, N)
    object_mask = torch.ones(B, N, dtype=torch.bool)
    
    # Mask positions 15-19
    object_mask[:, 15:] = False
    object_embeddings[:, 15:] = 0.0
    base_logits[:, 15:] = 0.0
    
    # CRITICAL: Target must be in VALID range
    target_index = torch.randint(0, 15, (B,))  # Only valid positions
    
    print(f"Targets: {target_index.tolist()}")
    print(f"Valid positions per sample: {object_mask.sum(dim=-1).tolist()}")
    
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
    
    # Test 1: Forward only (no grad)
    print("\n" + "=" * 60)
    print("Test 1: Forward only (no grad)")
    print("=" * 60)
    
    with torch.no_grad():
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
        
        loss = F.cross_entropy(fused_logits, target_index)
        print(f"Loss (no grad): {loss.item():.4f}")
        print(f"Loss finite: {torch.isfinite(loss).item()}")
        print(f"Gate mean: {gate_values.mean().item():.4f}")
    
    # Test 2: With grad on calibration ONLY
    print("\n" + "=" * 60)
    print("Test 2: Grad on calibration ONLY")
    print("=" * 60)
    
    # Freeze dense module
    for param in dense_module.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.AdamW(calibration.parameters(), lr=1e-4)
    
    for step in range(3):
        optimizer.zero_grad()
        
        with torch.no_grad():
            dense_result = dense_module(
                object_embeddings=object_embeddings,
                utterance_features=lang_features,
                candidate_mask=object_mask,
            )
            relation_scores = dense_result["relation_scores"].clone()
        
        relation_scores.requires_grad = False  # No grad through relation
        
        base_sorted = base_logits.sort(dim=-1, descending=True).values
        base_margin = base_sorted[:, 0] - base_sorted[:, 1]
        relation_sorted = relation_scores.sort(dim=-1, descending=True).values
        relation_margin = relation_sorted[:, 0] - relation_sorted[:, 1]
        relation_probs = F.softmax(relation_scores.masked_fill(~object_mask, float("-inf")), dim=-1)
        anchor_entropy = -(relation_probs * relation_probs.clamp(min=1e-8).log()).sum(dim=-1)
        
        cal_result = calibration(
            base_logits=base_logits.detach(),
            relation_scores=relation_scores.detach(),
            anchor_posterior=relation_probs.detach(),
            anchor_entropy=anchor_entropy.detach(),
            base_margin=base_margin.detach(),
            relation_margin=relation_margin.detach(),
            candidate_mask=object_mask,
        )
        
        fused_logits = cal_result["fused_logits"]
        loss = F.cross_entropy(fused_logits, target_index)
        loss.backward()
        
        print(f"Step {step+1}: Loss={loss.item():.4f}, finite={torch.isfinite(loss).item()}")
        
        # Check cal gradients
        for name, param in calibration.named_parameters():
            if param.grad is not None:
                has_nan = torch.isnan(param.grad).any().item()
                has_inf = torch.isinf(param.grad).any().item()
                norm = param.grad.norm().item()
                if has_nan or has_inf:
                    print(f"  GRAD ISSUE: {name}: nan={has_nan}, inf={has_inf}, norm={norm}")
                else:
                    print(f"  {name}: norm={norm:.4f}")
        
        optimizer.step()
    
    # Test 3: Full training (both modules)
    print("\n" + "=" * 60)
    print("Test 3: Full training (both modules)")
    print("=" * 60)
    
    # Unfreeze dense module
    for param in dense_module.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        list(dense_module.parameters()) + list(calibration.parameters()),
        lr=1e-4
    )
    
    for step in range(3):
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
        
        cal_result = calibration(
            base_logits=base_logits,
            relation_scores=relation_scores,
            anchor_posterior=relation_probs,
            anchor_entropy=anchor_entropy,
            base_margin=base_margin,
            relation_margin=relation_margin,
            candidate_mask=object_mask,
        )
        
        fused_logits = cal_result["fused_logits"]
        loss = F.cross_entropy(fused_logits, target_index)
        
        if not torch.isfinite(loss):
            print(f"Step {step+1}: Loss NOT FINITE: {loss.item()}")
            break
        
        loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}: Loss={loss.item():.4f}")
        
        # Check for NaN in weights
        for name, param in list(dense_module.named_parameters()) + list(calibration.named_parameters()):
            if torch.isnan(param).any():
                print(f"  NaN in weights: {name}")
            if torch.isinf(param).any():
                print(f"  Inf in weights: {name}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("If Test 2 (cal-only) works but Test 3 (full) fails:")
    print("  -> Issue is gradient flow through DenseRelationModule + calibration interaction")
    print("If Test 2 fails:")
    print("  -> Issue is in CalibratedFusionGate itself")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
