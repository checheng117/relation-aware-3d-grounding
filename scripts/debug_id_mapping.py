#!/usr/bin/env python3
"""Debug the specific ID/index mapping issue."""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.datasets.collate import collate_grounding_samples


def debug_id_mapping_issue():
    """Debug the ID/index mapping issue."""

    print("=== Debugging ID/Index Mapping Issue ===\n")

    # Load the validation dataset
    eval_split = "val"
    manifest_path = Path("data/processed") / f"{eval_split}_manifest.jsonl"

    dataset = ReferIt3DManifestDataset(manifest_path=manifest_path)
    print(f"Loaded {len(dataset)} samples from {eval_split} split\n")

    # Examine the relationship between target_index and target_object_id
    print("Examining first 10 samples for ID/index mapping:")
    print("Format: [sample_idx] scene_id | target_index (numeric) | target_object_id (string) | num_objects | target_in_candidate_position")
    print("-" * 120)

    mismatch_count = 0

    for i in range(min(10, len(dataset))):
        sample = dataset[i]

        # Check if target_index refers to position in objects list
        target_idx = sample.target_index  # This is numeric
        target_id = sample.target_object_id  # This is string

        # Check if objects list is 0-indexed
        if target_idx < len(sample.objects):
            obj_at_target_idx = sample.objects[target_idx]
            obj_id_at_target_idx = obj_at_target_idx.object_id

            # Check if the object ID at target_idx matches the target_object_id
            id_matches = obj_id_at_target_idx == target_id
            if not id_matches:
                mismatch = "MISMATCH!"
                mismatch_count += 1
            else:
                mismatch = "OK"

            print(f"[{i:2d}] {sample.scene_id[:15]:15} | target_idx={target_idx:2d} | target_id={target_id:>3s} | n_objs={len(sample.objects):2d} | pos[{target_idx}]={obj_id_at_target_idx:>3s} | {mismatch}")
        else:
            print(f"[{i:2d}] {sample.scene_id[:15]:15} | target_idx={target_idx:2d} | target_id={target_id:>3s} | n_objs={len(sample.objects):2d} | IDX_OUT_OF_BOUNDS")

    print(f"\nID/Index mismatches found: {mismatch_count}/10 samples")

    # Test the collation to see what gets passed to models
    print("\n=== Testing Collation Process ===")

    # Create small dataloader
    eval_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_grounding_samples
    )

    print("Examining batch tensors and targets...")

    for batch_idx, batch in enumerate(eval_loader):
        if batch_idx > 1:  # Just look at first 2 batches
            break

        tensors = batch.to_tensors(feat_dim=256, device="cpu")

        print(f"\nBatch {batch_idx}:")
        print(f"  Raw samples in batch: {len(batch.samples)}")
        print(f"  Object features shape: {tensors['object_features'].shape}")
        print(f"  Target indices (from dataset): {tensors['target_index'].tolist()}")
        print(f"  Object mask shape: {tensors['object_mask'].shape}")

        for i, sample in enumerate(batch.samples):
            print(f"  Sample {i} (scene: {sample.scene_id}):")
            print(f"    Dataset target_index: {sample.target_index}")
            print(f"    Dataset target_object_id: {sample.target_object_id}")
            print(f"    Tensor target_index: {tensors['target_index'][i].item()}")
            print(f"    Number of objects: {len(sample.objects)}")

            # Check if target_index points to correct object_id
            target_idx = sample.target_index
            if target_idx < len(sample.objects):
                obj_at_idx = sample.objects[target_idx]
                print(f"    Object at target_idx {target_idx}: {obj_at_idx.object_id}")
                print(f"    Matches target_object_id? {obj_at_idx.object_id == sample.target_object_id}")
            else:
                print(f"    ERROR: target_idx {target_idx} out of bounds for {len(sample.objects)} objects")

    print("\n=== Root Cause Analysis ===")
    print("The issue may be in how the models interpret target indices vs target object IDs.")
    print("Most likely, the models are using the target_index as a direct tensor index,")
    print("but if target_index doesn't correspond to the correct position in the object list,")
    print("then all predictions will be misaligned.")

    # Test if this is the actual issue
    print(f"\nTesting if the issue is ID/index misalignment...")

    problematic_samples = []
    for i in range(min(20, len(dataset))):
        sample = dataset[i]
        target_idx = sample.target_index
        target_id = sample.target_object_id

        if target_idx < len(sample.objects):
            obj_at_idx = sample.objects[target_idx]
            if obj_at_idx.object_id != target_id:
                problematic_samples.append((i, sample.scene_id, target_idx, target_id, obj_at_idx.object_id))

    print(f"Found {len(problematic_samples)} samples with ID/index misalignment out of 20 checked:")
    for scene_idx, scene_id, tgt_idx, tgt_id, obj_id in problematic_samples[:5]:  # Show first 5
        print(f"  Sample {scene_idx} ({scene_id}): target_index={tgt_idx} points to obj_id={obj_id}, but target_object_id={tgt_id}")


if __name__ == "__main__":
    debug_id_mapping_issue()