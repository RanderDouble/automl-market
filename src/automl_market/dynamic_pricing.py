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
) -> PricingResult:
    """Optimize exact dynamic revenue over a declared finite price grid.

    The optimizer accounts for order, transition probabilities, discovery cost,
    and optimal stopping.  Its optimality claim is deliberately limited to the
    supplied grid.
    """
    values = np.asarray(valuations, dtype=float)
    weights = np.asarray(prior, dtype=float)
    grids = candidate_price_grid(values) if price_grids is None else price_grids
    if len(grids) != values.shape[1]:
        raise ValueError("provide one price grid per quality")

    best_prices: np.ndarray | None = None
    best_by_type: np.ndarray | None = None
    best_objective = -np.inf
    evaluated = 0
    for curve in itertools.product(*grids):
        prices = np.asarray(curve, dtype=float)
        by_type = markov_revenue_by_type(
            prices,
            values,
            initial,
            transitions,
            horizon,
            discovery_cost,
            allow_no_purchase,
        )
        if weights.shape != by_type.shape or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
            raise ValueError("prior must be a distribution over buyer types")
        objective = float(weights @ by_type)
        evaluated += 1
        if objective > best_objective + 1e-12:
            best_prices = prices.copy()
            best_by_type = by_type.copy()
            best_objective = objective
    assert best_prices is not None and best_by_type is not None
    return PricingResult(best_prices, best_objective, best_by_type, evaluated)
