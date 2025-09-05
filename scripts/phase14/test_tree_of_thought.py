#!/usr/bin/env python3
"""Phase 14 test script for TreeOfThought reasoning approach."""

import sys
from pathlib import Path

# Add parent directory to path to import base_test
sys.path.insert(0, str(Path(__file__).parent))

from base_test import create_argument_parser, run_approach_test


def main():
    """Run TreeOfThought approach test."""
    parser = create_argument_parser(
        "Test TreeOfThought reasoning approach for Phase 14"
    )
    args = parser.parse_args()

    # Run test
    results = run_approach_test("TreeOfThought", args)

    # Exit with error if correctness is below threshold
    if results["metrics"]["correctness_rate"] < 0.5:
        sys.exit(1)


if __name__ == "__main__":
    main()
