"""Command-line entry point for Task 2.

Examples
--------
    python main.py
    python main.py --alphas 0.1 1 10 100 --missing-fraction 0.05
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import DEFAULT_ALPHAS, RANDOM_STATE
from src.data import load_housing
from src.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ridge regression on California Housing")
    p.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS,
                   help="Ridge alpha values to search")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--missing-fraction", type=float, default=0.02,
                   help="Fraction of cells to blank out in 3 numeric columns (0 to disable)")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(out_dir / "run.log", mode="w"), logging.StreamHandler()],
    )

    X, y = load_housing(missing_fraction=args.missing_fraction, seed=args.seed)
    run_experiment(
        X, y,
        alphas=args.alphas, test_size=args.test_size, cv_folds=args.cv_folds,
        seed=args.seed, out_dir=out_dir, make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
