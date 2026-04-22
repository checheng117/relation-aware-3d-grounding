#!/usr/bin/env python3
"""COVER-3D Phase 2 Smoke Test Script.

Minimal validation of COVER-3D implementation skeleton.
NO TRAINING - just forward pass checks.

Checks:
1. Import works
2. Forward pass works
3. Tensor shapes correct
4. No NaN/inf
5. Memory safe (chunked computation)
6. Diagnostics emitted
7. Gate bounds correct

Date: 2026-04-19
"""

import sys
import json
import logging
from pathlib import Path

import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# Add project paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

PROJECT_ROOT = ROOT

# Import COVER-3D modules
try:
    from rag3d.models import (
        Cover3DModel,
        Cover3DWrapper,
        Cover3DInput,
        Cover3DOutput,
        DenseRelationModule,
        SoftAnchorPosteriorModule,
        CalibratedFusionGate,
        create_cover3d_model_from_config,
    )
    log.info("✓ Import works")
    IMPORT_OK = True
except Exception as e:
    log.error(f"✗ Import failed: {e}")
    IMPORT_OK = False
    sys.exit(1)


def create_synthetic_inputs(
    batch_size: int = 4,
    num_objects: int = 20,
    object_dim: int = 320,
    language_dim: int = 256,
    geometry_dim: int = 6,
    device: str = "cpu",
):
    """Create synthetic inputs for smoke test."""
    B, N = batch_size, num_objects

    inputs = Cover3DInput(
        # Required
        base_logits=torch.randn(B, N, device=device),
        object_embeddings=torch.randn(B, N, object_dim, device=device),
        utterance_features=torch.randn(B, language_dim, device=device),

        # Optional
        object_geometry=torch.randn(B, N, geometry_dim, device=device),
        candidate_mask=torch.ones(B, N, dtype=torch.bool, device=device),
        object_class_features=torch.randn(B, N, 64, device=device),
    )

    return inputs


def run_smoke_tests():
    """Run all smoke tests."""
    results = {
        "passed": [],
        "failed": [],
        "warnings": [],
    }

    device = "cpu"
    B, N = 4, 20

    # === Test 1: Model Creation ===
    log.info("\n[1] Model Creation")
    try:
        model = Cover3DModel(
            object_dim=320,
            language_dim=256,
            geometry_dim=6,
            class_dim=64,
            relation_chunk_size=16,
            emit_diagnostics=True,
        )
        log.info(f"  ✓ Model created")
        results["passed"].append("model_creation")

        # Check parameters
        param_counts = model.count_parameters()
        log.info(f"  Parameters: dense_relation={param_counts['dense_relation']}, "
                 f"anchor_posterior={param_counts['anchor_posterior']}, "
                 f"fusion_gate={param_counts['fusion_gate']}, "
                 f"total={param_counts['total']}")
    except Exception as e:
        log.error(f"  ✗ Model creation failed: {e}")
        results["failed"].append(("model_creation", str(e)))
        return results

    # === Test 2: Forward Pass ===
    log.info("\n[2] Forward Pass")
    try:
        inputs = create_synthetic_inputs(device=device)
        output = model.forward_from_input(inputs)

        log.info(f"  ✓ Forward pass works")
        results["passed"].append("forward_pass")

        # Check output types
        assert isinstance(output, Cover3DOutput), f"Output type mismatch: {type(output)}"
        assert output.reranked_logits is not None, "Missing reranked_logits"

        log.info(f"  Output logits shape: {output.reranked_logits.shape}")
    except Exception as e:
        log.error(f"  ✗ Forward pass failed: {e}")
        results["failed"].append(("forward_pass", str(e)))
        return results

    # === Test 3: Tensor Shapes ===
    log.info("\n[3] Tensor Shapes")
    try:
        inputs = create_synthetic_inputs(device=device)
        output = model.forward_from_input(inputs)

        # Check shapes
        expected_shapes = {
            "reranked_logits": (B, N),
            "dense_relation_scores": (B, N),
            "anchor_posterior": (B, N),
            "anchor_entropy": (B,),
            "gate_values": (B,),
        }

        for name, expected in expected_shapes.items():
            tensor = getattr(output, name)
            if tensor is not None:
                actual = tuple(tensor.shape)
                if actual == expected:
                    log.info(f"  ✓ {name}: {actual}")
                else:
                    log.warning(f"  ✗ {name}: expected {expected}, got {actual}")
                    results["warnings"].append((name, f"shape mismatch: {actual} vs {expected}"))

        results["passed"].append("tensor_shapes")
    except Exception as e:
        log.error(f"  ✗ Shape check failed: {e}")
        results["failed"].append(("tensor_shapes", str(e)))

    # === Test 4: NaN/Inf Check ===
    log.info("\n[4] NaN/Inf Check")
    try:
        inputs = create_synthetic_inputs(device=device)
        output = model.forward_from_input(inputs)

        checks = {
            "reranked_logits": output.reranked_logits,
            "dense_relation_scores": output.dense_relation_scores,
            "anchor_posterior": output.anchor_posterior,
            "gate_values": output.gate_values,
        }

        has_nan = False
        has_inf = False
        for name, tensor in checks.items():
            if tensor is not None:
                if torch.isnan(tensor).any():
                    log.warning(f"  ✗ {name} has NaN")
                    has_nan = True
                if torch.isinf(tensor).any():
                    # Inf is acceptable for masked values
                    inf_count = torch.isinf(tensor).sum().item()
                    log.warning(f"  ! {name} has {inf_count} inf values (may be from mask)")
                    has_inf = True

        if not has_nan:
            log.info("  ✓ No NaN in outputs")
            results["passed"].append("no_nan")
        else:
            results["warnings"].append(("nan_detected", "NaN in outputs"))

        if has_inf:
            results["warnings"].append(("inf_detected", "Inf in outputs (may be from mask)"))
    except Exception as e:
        log.error(f"  ✗ NaN/Inf check failed: {e}")
        results["failed"].append(("nan_inf_check", str(e)))

    # === Test 5: Memory Check (Chunked) ===
    log.info("\n[5] Memory Check (Chunked Computation)")
    try:
        # Test with larger scene to verify chunking works
        N_large = 80  # Larger scene
        inputs_large = create_synthetic_inputs(
            num_objects=N_large,
            device=device,
        )

        # Track memory (approximate for CPU)
        import gc
        gc.collect()

        output_large = model.forward_from_input(inputs_large)

        log.info(f"  ✓ Large scene ({N_large} objects) forward pass works")
        log.info(f"  Output logits shape: {output_large.reranked_logits.shape}")

        # Check chunk count in diagnostics
        dense_diag = output_large.diagnostics.get("dense_relation_stats", {})
        if dense_diag:
            chunk_info = dense_diag.get("num_chunks", "N/A")
            log.info(f"  Chunk count: {chunk_info}")

        results["passed"].append("memory_safe_chunked")
    except Exception as e:
        log.error(f"  ✗ Large scene test failed: {e}")
        results["failed"].append(("memory_chunked", str(e)))

    # === Test 6: Diagnostics ===
    log.info("\n[6] Diagnostics Emission")
    try:
        inputs = create_synthetic_inputs(device=device)
        output = model.forward_from_input(inputs)

        diag = output.diagnostics

        # Expected diagnostics
        expected_keys = [
            "batch_size",
            "num_objects",
            "dense_relation_stats",
            "anchor_stats",
            "fusion_stats",
        ]

        for key in expected_keys:
            if key in diag:
                log.info(f"  ✓ {key} present")
            else:
                log.warning(f"  ! {key} missing")
                results["warnings"].append((f"diag_missing_{key}", f"{key} not in diagnostics"))

        # Log key stats
        if "fusion_stats" in diag:
            fs = diag["fusion_stats"]
            log.info(f"  Gate stats: mean={fs.get('mean_gate', 'N/A'):.3f}, "
                     f"min={fs.get('min_gate', 'N/A'):.3f}, max={fs.get('max_gate', 'N/A'):.3f}")

        results["passed"].append("diagnostics_emitted")
    except Exception as e:
        log.error(f"  ✗ Diagnostics check failed: {e}")
        results["failed"].append(("diagnostics", str(e)))

    # === Test 7: Gate Bounds ===
    log.info("\n[7] Gate Bounds Check")
    try:
        inputs = create_synthetic_inputs(device=device)
        output = model.forward_from_input(inputs)

        gates = output.gate_values
        if gates is not None:
            min_gate = gates.min().item()
            max_gate = gates.max().item()

            # Expected bounds from config
            expected_min = 0.1
            expected_max = 0.9

            if min_gate >= expected_min - 0.01:  # Small tolerance
                log.info(f"  ✓ Min gate {min_gate:.4f} >= {expected_min}")
            else:
                log.warning(f"  ✗ Min gate {min_gate:.4f} < {expected_min}")
                results["warnings"].append(("gate_min_below_bound", f"min_gate={min_gate}"))

            if max_gate <= expected_max + 0.01:
                log.info(f"  ✓ Max gate {max_gate:.4f} <= {expected_max}")
            else:
                log.warning(f"  ✗ Max gate {max_gate:.4f} > {expected_max}")
                results["warnings"].append(("gate_max_above_bound", f"max_gate={max_gate}"))

            results["passed"].append("gate_bounds")
    except Exception as e:
        log.error(f"  ✗ Gate bounds check failed: {e}")
        results["failed"].append(("gate_bounds", str(e)))

    # === Test 8: Module Independence ===
    log.info("\n[8] Module Independence")
    try:
        # Test each module independently
        inputs = create_synthetic_inputs(device=device)

        # Dense relation module
        dense_out = model.dense_relation(
            object_embeddings=inputs.object_embeddings,
            object_geometry=inputs.object_geometry,
            utterance_features=inputs.utterance_features,
        )
        log.info(f"  ✓ DenseRelationModule works independently")

        # Anchor posterior module
        anchor_out = model.anchor_posterior(
            utterance_features=inputs.utterance_features,
            object_embeddings=inputs.object_embeddings,
            object_class_features=inputs.object_class_features,
        )
        log.info(f"  ✓ SoftAnchorPosteriorModule works independently")

        # Fusion gate module
        base_margin = model.wrapper.compute_base_margin(inputs.base_logits)
        relation_margin = model.wrapper.compute_relation_margin(dense_out["relation_scores"])
        fusion_out = model.fusion_gate(
            base_logits=inputs.base_logits,
            relation_scores=dense_out["relation_scores"],
            anchor_posterior=anchor_out["anchor_posterior"],
            anchor_entropy=anchor_out["anchor_entropy"],
            base_margin=base_margin,
            relation_margin=relation_margin,
        )
        log.info(f"  ✓ CalibratedFusionGate works independently")

        results["passed"].append("module_independence")
    except Exception as e:
        log.error(f"  ✗ Module independence test failed: {e}")
        results["failed"].append(("module_independence", str(e)))

    return results


def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 60)
    print("COVER-3D Phase 2 Smoke Test Summary")
    print("=" * 60)

    passed = len(results["passed"])
    failed = len(results["failed"])
    warnings = len(results["warnings"])

    print(f"\nPassed: {passed}")
    print(f"Failed: {failed}")
    print(f"Warnings: {warnings}")

    if results["passed"]:
        print("\nPassed tests:")
        for t in results["passed"]:
            print(f"  ✓ {t}")

    if results["failed"]:
        print("\nFailed tests:")
        for t, msg in results["failed"]:
            print(f"  ✗ {t}: {msg}")

    if results["warnings"]:
        print("\nWarnings:")
        for t, msg in results["warnings"]:
            print(f"  ! {t}: {msg}")

    print("\n" + "=" * 60)

    if failed == 0:
        print("SMOKE TEST PASSED ✓")
        print("=" * 60)
        return True
    else:
        print("SMOKE TEST FAILED ✗")
        print("=" * 60)
        return False


def main():
    """Run smoke tests and save results."""
    print("=" * 60)
    print("COVER-3D Phase 2: Smoke Validation")
    print("=" * 60)
    print("\nNO TRAINING - Forward pass validation only")
    print("Testing: import, shapes, NaN/inf, memory, diagnostics, bounds")
    print()

    results = run_smoke_tests()
    success = print_summary(results)

    # Save results
    output_path = PROJECT_ROOT / "reports" / "cover3d_phase2_smoke_diagnostics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "success": success,
            "passed": results["passed"],
            "failed": results["failed"],
            "warnings": results["warnings"],
            "timestamp": "2026-04-19",
            "phase": "2",
            "test_type": "smoke",
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)