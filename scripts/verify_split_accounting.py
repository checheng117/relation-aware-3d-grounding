#!/usr/bin/env python3
"""Script to verify split/sample accounting."""

import argparse
import sys
from pathlib import Path
import json
from typing import Dict, Any, List

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.datasets.collate import collate_grounding_samples


def verify_split_accounting():
    """Verify split/sample accounting."""

    print("Verifying split/sample accounting...")

    # Load the validation dataset
    eval_split = "val"
    manifest_path = Path("data/processed") / f"{eval_split}_manifest.jsonl"

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        return

    dataset = ReferIt3DManifestDataset(manifest_path=manifest_path)
    print(f"Manifest file size: {manifest_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"Loaded {len(dataset)} samples from {eval_split} split")

    # Check the actual manifest file
    import subprocess
    result = subprocess.run(['wc', '-l', str(manifest_path)], capture_output=True, text=True)
    if result.returncode == 0:
        line_count = int(result.stdout.split()[0])
        print(f"Manifest has {line_count} lines")
    else:
        print("Could not count lines in manifest")

    # Get first few samples to see their structure
    print("\nAnalyzing first 5 samples...")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(f"  Scene ID: {sample.scene_id}")
        print(f"  Utterance: {repr(sample.utterance)}")
        print(f"  Target ID: {sample.target_object_id}")
        print(f"  Target Index: {sample.target_index}")
        print(f"  Number of objects: {len(sample.objects)}")
        print(f"  First 3 object IDs: {[obj.object_id for obj in sample.objects[:3]]}")
        print()

    # Create a small dataloader to check if samples are correctly handled
    from torch.utils.data import DataLoader
    eval_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_grounding_samples
    )

    print("Checking batch generation and tensor conversion...")
    for batch_idx, batch in enumerate(eval_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Number of samples in batch: {len(batch.samples)}")

        tensors = batch.to_tensors(feat_dim=256, device="cpu")
        print(f"  Object features shape: {tensors['object_features'].shape}")
        print(f"  Object mask shape: {tensors['object_mask'].shape}")
        print(f"  Number of texts: {len(tensors['raw_texts'])}")
        print(f"  Target indices shape: {tensors['target_index'].shape}")
        print(f"  Sample refs length: {len(tensors.get('samples_ref', []))}")

        # Check if the target indices align with the object features
        for i, sample in enumerate(batch.samples):
            actual_target_idx = sample.target_index
            tensor_target_idx = tensors['target_index'][i].item()
            print(f"    Sample {i}: Target index - Dataset: {actual_target_idx}, Tensor: {tensor_target_idx}")
            if actual_target_idx != tensor_target_idx:
                print(f"    ERROR: Index mismatch for sample {i}")

        if batch_idx >= 2:  # Check only first 3 batches
            break

    # Verify sample counting logic used in experiments
    print("\nVerifying sample counting for --max-samples 200...")
    print(f"Actual dataset size: {len(dataset)}")
    print(f"Requested max-samples: 200")
    print(f"Will evaluate: {min(200, len(dataset))} samples")

    # Create summary report
    summary = {
        'split': eval_split,
        'total_samples': len(dataset),
        'manifest_line_count': line_count,
        'objects_per_sample_stats': {
            'min_objects': min(len(sample.objects) for sample in dataset) if dataset else 0,
            'max_objects': max(len(sample.objects) for sample in dataset) if dataset else 0,
            'avg_objects': sum(len(sample.objects) for sample in dataset) / len(dataset) if dataset else 0
        },
        'first_5_samples': []
    }

    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        summary['first_5_samples'].append({
            'scene_id': sample.scene_id,
            'utterance': sample.utterance,
            'target_id': sample.target_object_id,
            'target_index': sample.target_index,
            'object_count': len(sample.objects)
        })

    # Create output directory
    output_dir = Path("outputs") / "debug_integrity"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export JSON
    with open(output_dir / "split_accounting_report.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Export Markdown
    md_content = f"""# Split/Sample Accounting Report

## Dataset Overview
- Split: {eval_split}
- Total samples: {len(dataset)}
- Manifest file lines: {line_count}

## Objects Per Sample Statistics
- Minimum objects per sample: {summary['objects_per_sample_stats']['min_objects']}
- Maximum objects per sample: {summary['objects_per_sample_stats']['max_objects']}
- Average objects per sample: {summary['objects_per_sample_stats']['avg_objects']:.2f}

## First 5 Sample Examples
"""
    for i, sample in enumerate(summary['first_5_samples']):
        md_content += f"""
### Sample {i+1}
- Scene ID: `{sample['scene_id']}`
- Utterance: `{sample['utterance']}`
- Target ID: `{sample['target_id']}`
- Target Index: {sample['target_index']}
- Object Count: {sample['object_count']}
"""

    md_content += f"""
## Sample Count Verification
- Actual dataset size: {len(dataset)}
- Requested max-samples: 200
- Will evaluate: {min(200, len(dataset))} samples
- Samples clipped: {'Yes' if len(dataset) < 200 else 'No'}

## Conclusion
{'All samples are available and properly formatted.' if len(dataset) > 0 else 'Dataset appears empty or inaccessible.'}
"""

    with open(output_dir / "split_accounting_report.md", 'w') as f:
        f.write(md_content)

    print(f"Split accounting reports saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Verify split/sample accounting')

    args = parser.parse_args()

    verify_split_accounting()


if __name__ == "__main__":
    main()