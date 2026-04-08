#!/usr/bin/env python3
"""Run the complete Phase 2 experiment suite."""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]

def run_command(cmd, description, output_dir):
    """Run a command and save its output."""
    print(f"\n{description}...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True
        )

        print(f"✓ {description} completed successfully")

        # Save output
        log_file = output_dir / f"command_log.txt"
        log_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"STDOUT:\n{result.stdout}\n")
            f.write(f"STDERR:\n{result.stderr}\n")

        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with return code {e.returncode}")
        print(f"Error: {e.stderr}")

        # Log error
        error_file = output_dir / f"command_error.txt"
        error_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(error_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {e.returncode}\n")
            f.write(f"STDOUT:\n{e.stdout}\n")
            f.write(f"STDERR:\n{e.stderr}\n")

        return False


def main():
    parser = argparse.ArgumentParser(description='Run complete Phase 2 experiment suite')
    parser.add_argument('--output-base-dir', type=Path,
                       default=Path("outputs"),
                       help='Base directory for experiment outputs')

    args = parser.parse_args()

    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = args.output_base_dir / f"{timestamp}_experiment_suite"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting Phase 2 experiment suite in: {experiment_dir}")

    # Create component directories
    overall_dir = experiment_dir / "overall_comparison"
    relation_dir = experiment_dir / "relation_stratified"
    hard_case_dir = experiment_dir / "hard_case_comparison"
    ablation_dir = experiment_dir / "soft_anchor_ablation"
    diag_dir = experiment_dir / "diagnostic_analysis"

    overall_dir.mkdir(exist_ok=True)
    relation_dir.mkdir(exist_ok=True)
    hard_case_dir.mkdir(exist_ok=True)
    ablation_dir.mkdir(exist_ok=True)
    diag_dir.mkdir(exist_ok=True)

    # Define commands for each component
    commands = [
        (
            [sys.executable, "scripts/run_overall_comparison.py", "--output-dir", str(overall_dir), "--max-samples", "200"],
            "Overall model comparison",
            overall_dir
        ),
        (
            [sys.executable, "scripts/run_relation_stratified_comparison.py", "--output-dir", str(relation_dir), "--max-samples", "200"],
            "Relation-stratified comparison",
            relation_dir
        ),
        (
            [sys.executable, "scripts/run_hard_case_comparison.py", "--output-dir", str(hard_case_dir), "--max-samples", "200"],
            "Hard-case comparison",
            hard_case_dir
        ),
        (
            [sys.executable, "scripts/run_soft_anchor_ablation.py", "--output-dir", str(ablation_dir), "--max-samples", "200"],
            "Soft anchor ablation study",
            ablation_dir
        ),
        (
            [sys.executable, "scripts/run_diagnostic_analysis.py", "--output-dir", str(diag_dir), "--max-samples", "200"],
            "Diagnostic analysis",
            diag_dir
        )
    ]

    # Execute commands
    results = []
    for cmd, description, output_dir in commands:
        success = run_command(cmd, description, output_dir)
        results.append((description, success))

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUITE EXECUTION SUMMARY")
    print("="*60)

    all_passed = True
    for description, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:<8} {description}")
        if not success:
            all_passed = False

    print("="*60)
    if all_passed:
        print("✓ All experiment components completed successfully!")
        print(f"Results saved to: {experiment_dir}")
    else:
        print("✗ Some experiment components failed.")
        print(f"Partial results saved to: {experiment_dir}")

    # Create summary file
    summary_file = experiment_dir / "experiment_summary.md"
    with open(summary_file, 'w') as f:
        f.write(f"# Phase 2 Experiment Suite Summary\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Base directory**: {experiment_dir}\n\n")

        f.write("## Component Status\n\n")
        for description, success in results:
            status = "PASS" if success else "FAIL"
            icon = "✅" if success else "❌"
            f.write(f"- {icon} {description}: **{status}**\n")

        f.write(f"\n## Output Locations\n\n")
        f.write(f"- Overall comparison: `{overall_dir}`\n")
        f.write(f"- Relation-stratified: `{relation_dir}`\n")
        f.write(f"- Hard-case comparison: `{hard_case_dir}`\n")
        f.write(f"- Soft anchor ablation: `{ablation_dir}`\n")
        f.write(f"- Diagnostic analysis: `{diag_dir}`\n")

        f.write(f"\n## Status\n\n")
        f.write(f"{'✅ All components successful!' if all_passed else '❌ Some components failed.'}\n")

    print(f"\nDetailed summary saved to: {summary_file}")


if __name__ == "__main__":
    main()