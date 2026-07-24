#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from automl_market.improvement_experiments import run_improvement_experiments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    summary = run_improvement_experiments(args.output)
    print(f"Completed {len(summary['pricing'])} pricing-improvement schemes.")
    print(f"Completed {len(summary['discovery'])} discovery-improvement schemes.")
    print(f"Results: {args.output / 'improvement_experiments_summary.json'}")


if __name__ == "__main__":
    main()
