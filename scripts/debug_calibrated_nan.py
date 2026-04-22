#!/usr/bin/env python3
"""Debug script to locate Dense-calibrated NaN source."""

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

def check_tensor(name, tensor):
    """Check tensor for NaN/inf and print stats."""
    if tensor is None:
        return f"{name}: None"
    
    stats = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "has_nan": torch.isnan(tensor).any().item(),
        "has_inf": torch.isinf(tensor).any().item(),
        "min": tensor.min().item() if tensor.numel() > 0 else None,
        "max": tensor.max().item() if tensor.numel() > 0 else None,
        "mean": tensor.mean().item() if tensor.numel() > 0 else None,
        "std": tensor.std().item() if tensor.numel() > 0 else None,
    }
    return json.dumps(stats, indent=2)

def main():
    print("=" * 60)
    print("Dense-calibrated NaN Debug Script")
    print("=" * 60)
    
    # Create sample data matching training
    B, N, D_obj, D_lang = 4, 20, 320, 256
    
    print(f"\nCreating sample data: B={B}, N={N}, D_obj={D_obj}, D_lang={D_lang}")
    
    # Random inputs
    object_embeddings = torch.randn(B, N, D_obj)
    lang_features = torch.randn(B, D_lang)
    base_logits = torch.randn(B, N)
    object_mask = torch.ones(B, N, dtype=torch.bool)
    
    # Mask some positions (simulate padding)
    object_mask[:, 15:] = False
    object_embeddings[:, 15:] = 0.0
    base_logits[:, 15:] = 0.0
    
    print(f"Object mask: {object_mask.sum(dim=-1)} valid objects per sample")
    
    # Create modules
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
    
    # Move to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dense_module = dense_module.to(device)
    calibration = calibration.to(device)
    object_embeddings = object_embeddings.to(device)
    lang_features = lang_features.to(device)
    base_logits = base_logits.to(device)
    object_mask = object_mask.to(device)
    
    # Step 1: Dense relation module
    print("\n" + "=" * 60)
    print("Step 1: DenseRelationModule forward")
    print("=" * 60)
    
    dense_result = dense_module(
        object_embeddings=object_embeddings,
        utterance_features=lang_features,
        candidate_mask=object_mask,
    )
    
    relation_scores = dense_result["relation_scores"]
    print(f"relation_scores stats:\n{check_tensor('relation_scores', relation_scores)}")
    
    # Check intermediate values
    if "relation_evidence" in dense_result:
        print(f"relation_evidence stats:\n{check_tensor('relation_evidence', dense_result['relation_evidence'])}")
    
    # Step 2: Compute calibration signals
    print("\n" + "=" * 60)
    print("Step 2: Compute calibration signals")
    print("=" * 60)
    
    base_sorted = base_logits.sort(dim=-1, descending=True).values
    base_margin = base_sorted[:, 0] - base_sorted[:, 1]
    print(f"base_margin stats:\n{check_tensor('base_margin', base_margin)}")
    
    relation_sorted = relation_scores.sort(dim=-1, descending=True).values
    relation_margin = relation_sorted[:, 0] - relation_sorted[:, 1]
    print(f"relation_margin stats:\n{check_tensor('relation_margin', relation_margin)}")
    
    # Softmax with mask
    masked_relation = relation_scores.masked_fill(~object_mask, float("-inf"))
    print(f"masked_relation stats (before softmax):\n{check_tensor('masked_relation', masked_relation)}")
    
    relation_probs = F.softmax(masked_relation, dim=-1)
    print(f"relation_probs stats:\n{check_tensor('relation_probs', relation_probs)}")
    
    anchor_entropy = -(relation_probs * relation_probs.clamp(min=1e-8).log()).sum(dim=-1)
    print(f"anchor_entropy stats:\n{check_tensor('anchor_entropy', anchor_entropy)}")
    
    # Step 3: Calibration gate
    print("\n" + "=" * 60)
    print("Step 3: CalibratedFusionGate forward")
    print("=" * 60)
    
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
    
    print(f"gate_values stats:\n{check_tensor('gate_values', gate_values)}")
    print(f"fused_logits stats:\n{check_tensor('fused_logits', fused_logits)}")
    
    # Check diagnostics
    print(f"\nCalibration diagnostics: {cal_result.get('diagnostics', {})}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    has_nan_relation = torch.isnan(relation_scores).any().item()
    has_nan_gate = torch.isnan(gate_values).any().item()
    has_nan_fused = torch.isnan(fused_logits).any().item()
    
    print(f"NaN in relation_scores: {has_nan_relation}")
    print(f"NaN in gate_values: {has_nan_gate}")
    print(f"NaN in fused_logits: {has_nan_fused}")
    
    if has_nan_relation:
        print("\n>> NaN originates in DenseRelationModule!")
    elif has_nan_gate:
        print("\n>> NaN originates in CalibratedFusionGate (gate computation)!")
    elif has_nan_fused:
        print("\n>> NaN originates in fusion formula!")
    else:
        print("\n>> No NaN detected in single forward pass!")
        print(">> NaN may only appear during/after backward pass (gradient issue)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
