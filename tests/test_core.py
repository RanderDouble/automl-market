from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from automl_market.market import optimal_stopping_dp, simulate_markov_trajectories
from automl_market.pricing import expected_revenue, optimize_price_grid


class MarketTests(unittest.TestCase):
    def test_markov_sampler_is_reproducible_and_valid(self) -> None:
        initial = np.array([1.0, 0.0])
        transition = np.array([[0.5, 0.5], [0.0, 1.0]])
        a = simulate_markov_trajectories(initial, transition, 4, 20, np.random.default_rng(7))
        b = simulate_markov_trajectories(initial, transition, 4, 20, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all((a >= 0) & (a < 2)))

    def test_dp_prefers_continue_when_future_gain_exceeds_cost(self) -> None:
        transition = np.array([[0.0, 1.0], [0.0, 1.0]])
        values, policy = optimal_stopping_dp(
            np.array([0.1, 1.0]), np.zeros(2), transition, horizon=2, discovery_cost=0.1
        )
        self.assertTrue(policy[0, 0, 0])
        self.assertAlmostEqual(values[0, 0, 0], 0.8)

    def test_outside_option_removes_unrealizable_revenue(self) -> None:
        valuations = np.array([[1.0, 1.2], [0.2, 0.3]])
        prior = np.array([0.5, 0.5])
        paths = np.array([[0, 1], [0, 0]])
        prices = np.array([1.0, 1.2])
        forced = expected_revenue(prices, valuations, prior, paths, force_purchase=True)
        realized = expected_revenue(prices, valuations, prior, paths, force_purchase=False)
        self.assertGreater(forced, realized)

    def test_ir_grid_optimizer_matches_simple_monopoly_solution(self) -> None:
        valuations = np.array([[0.2], [1.0]])
        prior = np.array([0.8, 0.2])
        paths = np.zeros((5, 2), dtype=int)
        result = optimize_price_grid(valuations, prior, paths, force_purchase=False)
        self.assertAlmostEqual(result.prices[0], 0.2)
        self.assertAlmostEqual(result.objective, 0.2)


if __name__ == "__main__":
    unittest.main()

