#!/usr/bin/env python3
"""Script to verify candidate/target integrity in the evaluation."""

import argparse
import sys
from pathlib import Path
import json
import torch
from typing import Dict, Any, List

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.datasets.collate import collate_grounding_samples
from rag3d.datasets.schema import GroundingSample


def check_candidate_target_integrity(max_samples=50):
    """Check integrity of target/candidate relationships in dataset."""

    print(f"Checking candidate/target integrity for first {max_samples} samples...")

    # Load the validation dataset
    eval_split = "val"
    manifest_path = Path("data/processed") / f"{eval_split}_manifest.jsonl"

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        return

    dataset = ReferIt3DManifestDataset(manifest_path=manifest_path)
    print(f"Loaded {len(dataset)} samples from {eval_split} split")

    # Create data loader to get batches
    from torch.utils.data import DataLoader
    eval_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_grounding_samples
    )

    integrity_issues = []
    total_samples = 0
    gt_found_in_candidates = 0

    for batch_idx, batch in enumerate(eval_loader):
        for i, sample in enumerate(batch.samples):
            if total_samples >= max_samples:
                break

            target_id = sample.target_object_id  # Use the correct field
            candidate_ids = [obj.object_id for obj in sample.objects]

            # Check if GT target is in candidate set
            target_in_candidates = target_id in candidate_ids

            if not target_in_candidates:
                issue = {
                    'scene_id': sample.scene_id,
                    'sample_idx_in_batch': i,
                    'target_id': target_id,
                    'candidate_ids': candidate_ids,
                    'target_present': False,
                    'message': f"GT target '{target_id}' not found in candidate set of {len(candidate_ids)} objects"
                }
                integrity_issues.append(issue)
            else:
                gt_found_in_candidates += 1

            # Additional checks
            if len(candidate_ids) != len(set(candidate_ids)):
                issue = {
                    'scene_id': sample.scene_id,
                    'sample_idx_in_batch': i,
                    'target_id': target_id,
                    'candidate_ids': candidate_ids,
                    'target_present': target_in_candidates,
                    'message': 'Duplicate candidate IDs found in candidate set'
                }
                integrity_issues.append(issue)

            total_samples += 1

        if total_samples >= max_samples:
            break

    print(f"\nChecked {total_samples} samples")
    print(f"GT target found in candidates for {gt_found_in_candidates}/{total_samples} samples")
    print(f"Issues found: {len(integrity_issues)}")

    # Summary report
    results = {
        'total_samples_checked': total_samples,
        'gt_found_in_candidates': gt_found_in_candidates,
        'gt_not_found': total_samples - gt_found_in_candidates,
        'issues_found': len(integrity_issues),
        'integrity_percentage': gt_found_in_candidates / total_samples if total_samples > 0 else 0,
        'issues': integrity_issues
    }

    # Create output directory
    output_dir = Path("outputs") / "debug_integrity"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export JSON
    with open(output_dir / "target_candidate_integrity_report.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Export Markdown
    md_content = f"""# Target-Candidate Integrity Report

## Summary
- Total samples checked: {results['total_samples_checked']}
- GT targets found in candidates: {results['gt_found_in_candidates']}
- GT targets NOT found in candidates: {results['gt_not_found']}
- Integrity percentage: {results['integrity_percentage']:.4f}
- Issues found: {results['issues_found']}

## Integrity Check Results
The dataset appears {'' if results['integrity_percentage'] > 0.95 else 'PROBLEMATIC - '}to have {'good' if results['integrity_percentage'] > 0.95 else 'POOR'} target-candidate alignment.

### Issue Details
"""
    if results['issues']:
        for i, issue in enumerate(results['issues'][:10]):  # Show first 10 issues
            md_content += f"\n{i+1}. **Scene**: {issue['scene_id']}\n"
            md_content += f"   **Message**: {issue['message']}\n"
            md_content += f"   **Target ID**: {issue['target_id']}\n"
            md_content += f"   **Candidate count**: {len(issue['candidate_ids'])}\n"

        if len(results['issues']) > 10:
            md_content += f"\n(... and {len(results['issues']) - 10} more issues)\n"
    else:
        md_content += "\nNo integrity issues found!\n"

    md_content += f"""
## Conclusion
{f"{'GOOD:' if results['integrity_percentage'] > 0.95 else 'ISSUE:'} {results['integrity_percentage']*100:.1f}% of targets were found in candidate sets. This {'suggests' if results['integrity_percentage'] > 0.95 else 'indicates potential'} proper dataset integrity for evaluation."}
"""

    with open(output_dir / "target_candidate_integrity_report.md", 'w') as f:
        f.write(md_content)

    print(f"Reports saved to {output_dir}/")

    return results


def main():
    parser = argparse.ArgumentParser(description='Verify candidate/target integrity in evaluation')
    parser.add_argument('--max-samples', type=int, default=50, help='Maximum samples to check')

    args = parser.parse_args()

    results = check_candidate_target_integrity(args.max_samples)

    # Exit with error if integrity issues found
    if results['integrity_percentage'] < 0.95:
        print(f"\n⚠️  WARNING: Only {results['integrity_percentage']*100:.1f}% of targets found in candidates.")
        print("This could explain the poor model performance.")
        return 1
    else:
        print(f"\n✓ Integrity looks good: {results['integrity_percentage']*100:.1f}% of targets found in candidates.")
        return 0


if __name__ == "__main__":
    exit(main())