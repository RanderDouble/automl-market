#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from automl_market.experiments import run_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    summary = run_all(args.output)
    print(f"Completed {len(summary['pricing'])} pricing schemes.")
    print(f"Results: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()

