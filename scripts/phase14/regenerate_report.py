#!/usr/bin/env python3
"""Regenerate Phase 14 report and visualizations from existing results."""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from ml_agents.core.phase14_comparison import ReasoningApproachComparison


def main():
    """Regenerate report and visualizations."""

    # Set up paths
    output_dir = Path("outputs/phase14/full_comparison")

    # Load existing CSV results
    csv_path = output_dir / "comparison_results.csv"
    if not csv_path.exists():
        print(f"Error: No results found at {csv_path}")
        print("Run the full comparison first!")
        sys.exit(1)

    print(f"Loading results from {csv_path}")
    df = pd.read_csv(csv_path)

    # Create comparison instance
    comparison = ReasoningApproachComparison(output_dir=output_dir)

    # Regenerate report
    print("Regenerating comparison report...")
    comparison._generate_summary_report(df)

    # Generate visualizations
    print("Generating visualizations...")
    comparison.generate_visualizations(df)

    print(f"\n✅ Report and visualizations regenerated in {output_dir}")

    # Check what files were created
    files = list(output_dir.glob("*"))
    print("\nGenerated files:")
    for f in sorted(files):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
