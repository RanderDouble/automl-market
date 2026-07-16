"""Empirical price-curve objectives and small-instance exact grid solvers."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PricingResult:
    prices: np.ndarray
    objective: float
    revenue_by_type: np.ndarray
    evaluated_curves: int


def revenue_by_type(
    prices: np.ndarray,
    valuations: np.ndarray,
    trajectories: np.ndarray,
    force_purchase: bool = False,
    seller_favorable_ties: bool = True,
) -> np.ndarray:
    """Return empirical mean payment for every buyer type.

    ``force_purchase=True`` reproduces Eq. (7)/(8), where every type-trajectory
    pair selects a quality.  ``False`` adds the zero-utility outside option.
    """
    prices = np.asarray(prices, dtype=float)
    values = np.asarray(valuations, dtype=float)
    paths = np.asarray(trajectories, dtype=np.int64)
    if values.ndim != 2 or prices.shape != (values.shape[1],):
        raise ValueError("valuations must be (types, qualities)")
    if paths.ndim != 2 or np.any(paths < 0) or np.any(paths >= prices.size):
        raise ValueError("trajectories contain invalid quality states")

    # Revenue depends only on the set of qualities encountered, not their order.
    # Grouping equal availability masks makes exhaustive grid search fast while
    # preserving exactly the empirical objective in Eq. (7).
    bit_masks = np.bitwise_or.reduce(1 << paths, axis=1)
    masks, counts = np.unique(bit_masks, return_counts=True)
    result = np.zeros(values.shape[0], dtype=float)
    for type_id, type_values in enumerate(values):
        total = 0.0
        net = type_values - prices
        for mask, count in zip(masks, counts):
            available = np.flatnonzero(mask & (1 << np.arange(prices.size)))
            available_net = net[available]
            best_utility = available_net.max()
            if force_purchase or best_utility > 1e-12 or (
                seller_favorable_ties and best_utility >= -1e-12
            ):
                tied = available[np.isclose(available_net, best_utility)]
                # Revenue-favorable tie breaking is also used by the MILP.
                selected = tied[np.argmax(prices[tied])]
                total += count * prices[selected]
        result[type_id] = total / len(paths)
    return result


def expected_revenue(
    prices: np.ndarray,
    valuations: np.ndarray,
    prior: np.ndarray,
    trajectories: np.ndarray,
    force_purchase: bool = False,
) -> float:
    prior = np.asarray(prior, dtype=float)
    by_type = revenue_by_type(prices, valuations, trajectories, force_purchase)
    if prior.shape != by_type.shape or not np.isclose(prior.sum(), 1.0):
        raise ValueError("prior must be a distribution over buyer types")
    return float(prior @ by_type)


def candidate_price_grid(valuations: np.ndarray) -> list[np.ndarray]:
    """Finite exact candidates: zero and each observed valuation per quality."""
    values = np.asarray(valuations, dtype=float)
    return [np.unique(np.r_[0.0, values[:, q]]) for q in range(values.shape[1])]


def optimize_price_grid(
    valuations: np.ndarray,
    prior: np.ndarray,
    trajectories: np.ndarray,
    force_purchase: bool = False,
    prior_scenarios: list[np.ndarray] | None = None,
    robust: bool = False,
) -> PricingResult:
    """Exactly optimize the empirical objective on the valuation-induced grid.

    This solver is intended for transparent small-scale reproduction.  With
    ``robust=True``, it maximizes the minimum revenue over ``prior_scenarios``.
    """
    values = np.asarray(valuations, dtype=float)
    prior = np.asarray(prior, dtype=float)
    scenarios = [prior] if prior_scenarios is None else [np.asarray(p) for p in prior_scenarios]
    for p in scenarios:
        if p.shape != prior.shape or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
            raise ValueError("invalid prior scenario")

    best_prices: np.ndarray | None = None
    best_objective = -np.inf
    best_by_type: np.ndarray | None = None
    evaluated = 0
    for curve in itertools.product(*candidate_price_grid(values)):
        prices = np.asarray(curve, dtype=float)
        by_type = revenue_by_type(prices, values, trajectories, force_purchase)
        scenario_revenues = np.asarray([p @ by_type for p in scenarios])
        objective = float(scenario_revenues.min() if robust else prior @ by_type)
        evaluated += 1
        if objective > best_objective + 1e-12:
            best_prices = prices.copy()
            best_objective = objective
            best_by_type = by_type.copy()
    assert best_prices is not None and best_by_type is not None
    return PricingResult(best_prices, best_objective, best_by_type, evaluated)


def independent_prices(valuations: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Appendix D.1 independent monopoly price for every quality."""
    values = np.asarray(valuations, dtype=float)
    prior = np.asarray(prior, dtype=float)
    prices = np.zeros(values.shape[1])
    for q in range(values.shape[1]):
        candidates = np.unique(np.r_[0.0, values[:, q]])
        revenues = [p * prior[values[:, q] >= p - 1e-12].sum() for p in candidates]
        prices[q] = candidates[int(np.argmax(revenues))]
    return prices
