"""Price-grid evaluation with nonzero discovery cost and exact Markov dynamics."""

from __future__ import annotations

import itertools

import numpy as np

from .market import expected_payment_dp
from .pricing import PricingResult, candidate_price_grid


def markov_revenue_by_type(
    prices: np.ndarray,
    valuations: np.ndarray,
    initial: np.ndarray,
    transitions: np.ndarray,
    horizon: int,
    discovery_cost: float,
    allow_no_purchase: bool = True,
) -> np.ndarray:
    """Return exact expected payment for each type under optimal stopping."""
    values = np.asarray(valuations, dtype=float)
    if values.ndim != 2:
        raise ValueError("valuations must have shape (types, qualities)")
    return np.asarray(
        [
            expected_payment_dp(
                type_values,
                prices,
                initial,
                transitions,
                horizon,
                discovery_cost,
                allow_no_purchase,
            )
            for type_values in values
        ]
    )


def expected_markov_revenue(
    prices: np.ndarray,
    valuations: np.ndarray,
    prior: np.ndarray,
    initial: np.ndarray,
    transitions: np.ndarray,
    horizon: int,
    discovery_cost: float,
    allow_no_purchase: bool = True,
) -> float:
    """Return exact expected seller payment under a prior and Markov process."""
    weights = np.asarray(prior, dtype=float)
    by_type = markov_revenue_by_type(
        prices,
        valuations,
        initial,
        transitions,
        horizon,
        discovery_cost,
        allow_no_purchase,
    )
    if weights.shape != by_type.shape or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("prior must be a distribution over buyer types")
    return float(weights @ by_type)


def optimize_markov_price_grid(
    valuations: np.ndarray,
    prior: np.ndarray,
    initial: np.ndarray,
    transitions: np.ndarray,
    horizon: int,
    discovery_cost: float,
    allow_no_purchase: bool = True,
    price_grids: list[np.ndarray] | None = None,
    prior_scenarios: list[np.ndarray] | None = None,
    discovery_cost_scenarios: list[float] | None = None,
    robust: bool = False,
) -> PricingResult:
    """Optimize exact dynamic revenue over a declared finite price grid.

    The optimizer accounts for order, transition probabilities, discovery cost,
    and optimal stopping.  Its optimality claim is deliberately limited to the
    supplied grid.  With ``robust=True``, it maximizes the minimum exact Markov
    revenue over the Cartesian product of ``prior_scenarios`` and
    ``discovery_cost_scenarios``.  The returned ``revenue_by_type`` is audited
    at the nominal ``discovery_cost``.
    """
    values = np.asarray(valuations, dtype=float)
    weights = np.asarray(prior, dtype=float)
    grids = candidate_price_grid(values) if price_grids is None else price_grids
    if len(grids) != values.shape[1]:
        raise ValueError("provide one price grid per quality")
    if weights.shape != (values.shape[0],) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("prior must be a distribution over buyer types")
    scenario_priors = _prior_scenarios(weights, prior_scenarios)
    scenario_costs = _discovery_cost_scenarios(discovery_cost, discovery_cost_scenarios)

    best_prices: np.ndarray | None = None
    best_by_type: np.ndarray | None = None
    best_objective = -np.inf
    evaluated = 0
    for curve in itertools.product(*grids):
        prices = np.asarray(curve, dtype=float)
        nominal_by_type = markov_revenue_by_type(
            prices,
            values,
            initial,
            transitions,
            horizon,
            discovery_cost,
            allow_no_purchase,
        )
        if robust:
            scenario_revenues = []
            for scenario_cost in scenario_costs:
                by_type = (
                    nominal_by_type
                    if np.isclose(scenario_cost, discovery_cost)
                    else markov_revenue_by_type(
                        prices,
                        values,
                        initial,
                        transitions,
                        horizon,
                        scenario_cost,
                        allow_no_purchase,
                    )
                )
                scenario_revenues.extend(float(p @ by_type) for p in scenario_priors)
            objective = float(np.min(scenario_revenues))
        else:
            objective = float(weights @ nominal_by_type)
        evaluated += 1
        if objective > best_objective + 1e-12:
            best_prices = prices.copy()
            best_by_type = nominal_by_type.copy()
            best_objective = objective
    assert best_prices is not None and best_by_type is not None
    return PricingResult(best_prices, best_objective, best_by_type, evaluated)


def optimize_cost_aware_price_grid(
    valuations: np.ndarray,
    prior: np.ndarray,
    initial: np.ndarray,
    transitions: np.ndarray,
    horizon: int,
    discovery_costs: list[float],
    allow_no_purchase: bool = True,
    price_grids: list[np.ndarray] | None = None,
    robust: bool = True,
) -> PricingResult:
    """Optimize prices for one or more nonzero search-cost scenarios.

    By default this returns the curve with the best worst-case exact Markov
    revenue across ``discovery_costs``.  Set ``robust=False`` to optimize only
    the first listed cost while keeping an explicit cost-aware call site.
    """
    if not discovery_costs:
        raise ValueError("discovery_costs must be non-empty")
    return optimize_markov_price_grid(
        valuations,
        prior,
        initial,
        transitions,
        horizon,
        float(discovery_costs[0]),
        allow_no_purchase=allow_no_purchase,
        price_grids=price_grids,
        discovery_cost_scenarios=discovery_costs,
        robust=robust,
    )


def _prior_scenarios(
    nominal: np.ndarray,
    prior_scenarios: list[np.ndarray] | None,
) -> list[np.ndarray]:
    scenarios = [nominal] if prior_scenarios is None else [np.asarray(p, dtype=float) for p in prior_scenarios]
    for p in scenarios:
        if p.shape != nominal.shape or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
            raise ValueError("invalid prior scenario")
    return scenarios


def _discovery_cost_scenarios(
    nominal: float,
    discovery_cost_scenarios: list[float] | None,
) -> list[float]:
    scenarios = [nominal] if discovery_cost_scenarios is None else [float(c) for c in discovery_cost_scenarios]
    if any(c < 0 or not np.isfinite(c) for c in scenarios):
        raise ValueError("discovery costs must be finite and nonnegative")
    return scenarios
