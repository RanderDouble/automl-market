from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from automl_market.discovery import (
    ScoreTable,
    calls_to_fraction,
    paired_bootstrap_mean_ci,
    pricing_guided_discovery,
    run_discovery_methods,
)


class DiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        validation = np.array(
            [
                [0.50, 0.52, 0.51],
                [0.60, 0.66, 0.62],
                [0.68, 0.74, 0.70],
                [0.65, 0.71, 0.69],
            ]
        )
        self.table = ScoreTable(
            validation=validation,
            test=validation - 0.01,
            model_names=("a", "b", "c"),
            augmentation_names=("none", "x", "y", "z"),
        )

    def test_all_methods_respect_budget_and_are_monotone_incumbents(self) -> None:
        curves = run_discovery_methods(self.table, budget=20, seed=3)
        self.assertEqual(set(curves), {"Data-Bandit", "Data-All", "Data-Alt", "AutoML"})
        for curve in curves.values():
            self.assertEqual(len(curve), 20)
            self.assertTrue(np.all(np.diff(curve) >= -1e-12))

    def test_pricing_signal_adds_guided_discovery_method(self) -> None:
        signal = np.array([0.0, 0.0, 1.0, 0.0])
        curves = run_discovery_methods(self.table, budget=4, seed=3, price_signal=signal)
        self.assertIn("Pricing-guided Data-Bandit", curves)
        self.assertEqual(len(curves["Pricing-guided Data-Bandit"]), 4)

    def test_pricing_guided_discovery_prioritizes_high_signal_augmentation(self) -> None:
        signal = np.array([0.0, 0.0, 1.0, 0.0])
        curve = pricing_guided_discovery(
            self.table,
            budget=1,
            seed=3,
            price_signal=signal,
            price_weight=1.0,
        )
        self.assertGreaterEqual(curve[0], 0.67)

    def test_data_all_reaches_validation_selected_oracle(self) -> None:
        curve = run_discovery_methods(self.table, budget=20, seed=3)["Data-All"]
        self.assertAlmostEqual(curve[-1], self.table.oracle_test_utility)

    def test_calls_to_fraction_uses_relative_oracle_gain(self) -> None:
        curve = np.array([0.50, 0.55, 0.60, 0.70])
        self.assertEqual(calls_to_fraction(curve, base=0.50, oracle=0.70, fraction=0.5), 3)

    def test_paired_bootstrap_interval_is_deterministic(self) -> None:
        differences = np.array([0.01, 0.02, 0.03, 0.04])
        first = paired_bootstrap_mean_ci(differences, seed=7, draws=1000)
        second = paired_bootstrap_mean_ci(differences, seed=7, draws=1000)
        self.assertEqual(first, second)
        self.assertGreater(first[0], 0.0)


if __name__ == "__main__":
    unittest.main()
