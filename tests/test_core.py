from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from automl_market.market import (
    expected_payment_dp,
    optimal_stopping_dp,
    simulate_markov_trajectories,
    stopping_distribution_dp,
    stopping_time_for_trajectory,
)
from automl_market.dynamic_pricing import expected_markov_revenue, optimize_cost_aware_price_grid
from automl_market.learning import smoothed_bayesian_learning
from automl_market.milp import solve_pricing_milp
from automl_market.pricing import (
    expected_revenue,
    independent_prices,
    optimize_jiggle_prices,
    optimize_price_grid,
    optimize_shift_prices,
)

HAS_HIGHSPY = importlib.util.find_spec("highspy") is not None


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

    def test_exact_markov_payment_follows_stopping_policy_and_outside_option(self) -> None:
        initial = np.array([1.0, 0.0])
        transition = np.array([[0.0, 1.0], [0.0, 1.0]])
        payment = expected_payment_dp(
            np.array([0.1, 1.0]),
            np.array([0.0, 0.5]),
            initial,
            transition,
            horizon=2,
            discovery_cost=0.1,
            allow_no_purchase=True,
        )
        self.assertAlmostEqual(payment, 0.5)
        no_sale = expected_payment_dp(
            np.array([0.1, 1.0]),
            np.array([2.0, 2.0]),
            initial,
            transition,
            horizon=2,
            discovery_cost=0.1,
            allow_no_purchase=True,
        )
        self.assertAlmostEqual(no_sale, 0.0)

    def test_exact_markov_payment_supports_time_indexed_transitions(self) -> None:
        initial = np.array([1.0, 0.0])
        transitions = np.array(
            [
                [[0.5, 0.5], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]
        )
        payment = expected_payment_dp(
            np.array([0.2, 1.0]),
            np.array([0.1, 0.6]),
            initial,
            transitions,
            horizon=3,
            discovery_cost=0.05,
            allow_no_purchase=True,
        )
        self.assertAlmostEqual(payment, 0.6)

    def test_exact_stopping_distribution_matches_deterministic_path_case(self) -> None:
        initial = np.array([1.0, 0.0])
        transition = np.array([[0.0, 1.0], [0.0, 1.0]])
        valuations = np.array([0.1, 1.0])
        prices = np.zeros(2)
        distribution = stopping_distribution_dp(
            valuations,
            prices,
            initial,
            transition,
            horizon=2,
            discovery_cost=0.1,
        )
        np.testing.assert_allclose(distribution, np.array([0.0, 1.0]))
        _, policy = optimal_stopping_dp(
            valuations, prices, transition, horizon=2, discovery_cost=0.1
        )
        self.assertEqual(
            stopping_time_for_trajectory(np.array([0, 1]), valuations, prices, policy),
            2,
        )

    def test_outside_option_removes_unrealizable_revenue(self) -> None:
        valuations = np.array([[1.0, 1.2], [0.2, 0.3]])
        prior = np.array([0.5, 0.5])
        paths = np.array([[0, 1], [0, 0]])
        prices = np.array([1.0, 1.2])
        forced = expected_revenue(prices, valuations, prior, paths, force_purchase=True)
        realized = expected_revenue(prices, valuations, prior, paths, force_purchase=False)
        self.assertGreater(forced, realized)

    def test_availability_bitmask_rejects_large_quality_spaces(self) -> None:
        valuations = np.ones((1, 61))
        prior = np.array([1.0])
        paths = np.arange(61, dtype=np.int64)[None, :]
        prices = np.zeros(61)
        with self.assertRaisesRegex(ValueError, "bit-mask compression"):
            expected_revenue(prices, valuations, prior, paths)

    def test_ir_grid_optimizer_matches_simple_monopoly_solution(self) -> None:
        valuations = np.array([[0.2], [1.0]])
        prior = np.array([0.8, 0.2])
        paths = np.zeros((5, 2), dtype=int)
        result = optimize_price_grid(valuations, prior, paths, force_purchase=False)
        self.assertAlmostEqual(result.prices[0], 0.2)
        self.assertAlmostEqual(result.objective, 0.2)

    @unittest.skipUnless(HAS_HIGHSPY, "optional highspy dependency is not installed")
    def test_forced_choice_milp_matches_small_grid_solution(self) -> None:
        valuations = np.array([[0.2, 0.5], [0.4, 0.9]])
        prior = np.array([0.6, 0.4])
        paths = np.array([[0, 1], [0, 0], [1, 1]])
        grid = optimize_price_grid(valuations, prior, paths, force_purchase=True)
        milp = solve_pricing_milp(valuations, prior, paths, force_purchase=True)
        self.assertEqual(milp.status, "Optimal")
        self.assertAlmostEqual(milp.objective, grid.objective, places=7)
        self.assertAlmostEqual(milp.objective, milp.solver_objective, places=7)

    @unittest.skipUnless(HAS_HIGHSPY, "optional highspy dependency is not installed")
    def test_ir_milp_matches_monopoly_grid_and_allows_no_purchase(self) -> None:
        valuations = np.array([[0.2], [1.0]])
        prior = np.array([0.8, 0.2])
        paths = np.zeros((5, 2), dtype=int)
        grid = optimize_price_grid(valuations, prior, paths, force_purchase=False)
        milp = solve_pricing_milp(valuations, prior, paths, force_purchase=False)
        self.assertEqual(milp.status, "Optimal")
        self.assertAlmostEqual(milp.objective, grid.objective, places=7)
        self.assertAlmostEqual(milp.objective, milp.solver_objective, places=7)
        if milp.prices[0] > 0.2 + 1e-7:
            self.assertAlmostEqual(milp.revenue_by_type[0], 0.0, places=7)

        expensive = solve_pricing_milp(
            np.array([[0.2], [1.0]]),
            np.array([0.5, 0.5]),
            paths,
            force_purchase=False,
            price_upper_bounds=2.0,
        )
        self.assertLessEqual(expensive.prices[0], 1.0 + 1e-7)
        self.assertAlmostEqual(expensive.objective, 0.5, places=7)

    @unittest.skipUnless(HAS_HIGHSPY, "optional highspy dependency is not installed")
    def test_milp_compresses_duplicate_availability_sets(self) -> None:
        valuations = np.array([[0.3, 0.7]])
        paths = np.array([[0, 1], [1, 0], [0, 0], [0, 0]])
        result = solve_pricing_milp(valuations, np.array([1.0]), paths)
        self.assertEqual(result.unique_availability_sets, 2)
        self.assertEqual(result.trajectories, 4)

    def test_shift_and_jiggle_dominate_their_initial_baselines(self) -> None:
        valuations = np.array([[0.2, 0.3], [0.4, 0.8], [0.9, 1.0]])
        prior = np.array([0.5, 0.3, 0.2])
        paths = np.array([[0, 1], [0, 0], [1, 1]])
        independent = expected_revenue(
            independent_prices(valuations, prior), valuations, prior, paths, True
        )
        shifted = optimize_shift_prices(valuations, prior, paths)
        jiggled = optimize_jiggle_prices(valuations, prior, paths)
        self.assertGreaterEqual(shifted.objective + 1e-12, independent)
        self.assertGreaterEqual(jiggled.objective + 1e-12, shifted.objective)

    def test_learning_supports_all_paper_rate_schedules(self) -> None:
        likelihoods = np.array([[0.9, 0.1], [0.2, 0.8]])
        prior = np.array([0.6, 0.4])
        for rate in ("sqrt", "harmonic", 0.5):
            history, kl = smoothed_bayesian_learning(
                likelihoods,
                prior,
                rounds=5,
                batch_size=10,
                rng=np.random.default_rng(8),
                learning_rate=rate,
            )
            self.assertEqual(history.shape, (6, 2))
            self.assertTrue(np.all(np.isfinite(kl)))

    def test_cost_aware_grid_reports_worst_case_markov_revenue(self) -> None:
        valuations = np.array([[0.3, 1.0], [0.7, 1.0]])
        prior = np.array([0.7, 0.3])
        initial = np.array([1.0, 0.0])
        transition = np.array([[0.0, 1.0], [0.0, 1.0]])
        price_grids = [np.array([0.3, 0.7]), np.array([0.5, 1.0])]
        costs = [0.0, 0.45]

        result = optimize_cost_aware_price_grid(
            valuations,
            prior,
            initial,
            transition,
            horizon=2,
            discovery_costs=costs,
            price_grids=price_grids,
            robust=True,
        )

        scenario_revenues = [
            expected_markov_revenue(
                result.prices,
                valuations,
                prior,
                initial,
                transition,
                horizon=2,
                discovery_cost=cost,
            )
            for cost in costs
        ]
        self.assertAlmostEqual(result.objective, min(scenario_revenues))
        self.assertEqual(result.evaluated_curves, 4)


if __name__ == "__main__":
    unittest.main()
