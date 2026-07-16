#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from automl_market.paper_experiments import run_paper_experiments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--rq2-repeats", type=int, default=10)
    parser.add_argument("--rq3-seeds", type=int, default=5)
    parser.add_argument("--rq3-rounds", type=int, default=1000)
    args = parser.parse_args()
    summary = run_paper_experiments(
        args.output,
        rq2_repeats=args.rq2_repeats,
        rq3_seeds=args.rq3_seeds,
        rq3_rounds=args.rq3_rounds,
    )
    print(f"Completed RQ2 ({summary['rq2']['repeats']} tasks) and RQ3 ({summary['rq3']['seeds']} seeds).")
    print(f"Results: {args.output / 'paper_experiments_summary.json'}")


if __name__ == "__main__":
    main()
