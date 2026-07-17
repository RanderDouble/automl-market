"""Markov metric discovery and the buyer optimal-stopping dynamic program.

The state follows Proposition 4.1 / Algorithm 3 of Han et al.: at time t it
records the best net-value quality seen so far and the current quality.  The
implementation also exposes an optional outside option, used by our improved
individually-rational mechanism.
"""

from __future__ import annotations

import numpy as np


def validate_transition_matrix(transitions: np.ndarray) -> np.ndarray:
    """Return a time-indexed transition tensor after validating probabilities."""
    p = np.asarray(transitions, dtype=float)
    if p.ndim == 2:
        p = p[None, :, :]
    if p.ndim != 3 or p.shape[1] != p.shape[2]:
        raise ValueError("transitions must have shape (Q,Q) or (T-1,Q,Q)")
    if np.any(p < -1e-12) or not np.allclose(p.sum(axis=2), 1.0):
        raise ValueError("each transition row must be a probability distribution")
    return p


def simulate_markov_trajectories(
    initial: np.ndarray,
    transitions: np.ndarray,
    horizon: int,
    n_trajectories: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample integer-valued quality trajectories from a finite Markov chain."""
    initial = np.asarray(initial, dtype=float)
    p = validate_transition_matrix(transitions)
    if horizon < 1 or n_trajectories < 1:
        raise ValueError("horizon and n_trajectories must be positive")
    if initial.ndim != 1 or initial.size != p.shape[1]:
        raise ValueError("initial distribution and transition states disagree")
    if np.any(initial < 0) or not np.isclose(initial.sum(), 1.0):
        raise ValueError("initial must be a probability distribution")
    if p.shape[0] not in (1, horizon - 1) and horizon > 1:
        raise ValueError("provide one stationary matrix or one matrix per transition")

    paths = np.empty((n_trajectories, horizon), dtype=np.int64)
    paths[:, 0] = rng.choice(initial.size, size=n_trajectories, p=initial)
    for t in range(1, horizon):
        pt = p[0] if p.shape[0] == 1 else p[t - 1]
        uniforms = rng.random(n_trajectories)
        for state in range(initial.size):
            mask = paths[:, t - 1] == state
            if mask.any():
                paths[mask, t] = np.searchsorted(
                    np.cumsum(pt[state]), uniforms[mask], side="right"
                )
    return paths


def optimal_stopping_dp(
    valuations: np.ndarray,
    prices: np.ndarray,
    transitions: np.ndarray,
    horizon: int,
    discovery_cost: float = 0.0,
    allow_no_purchase: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Algorithm 3's value and continuation tables.

    Returns
    -------
    values:
        Array ``(T,Q,Q)``. ``values[t,best,current]`` is optimal expected
        utility after observing round ``t`` (zero-indexed).
    continues:
        Boolean array of the same shape. True means continue discovery.
    """
    v = np.asarray(valuations, dtype=float)
    x = np.asarray(prices, dtype=float)
    p = validate_transition_matrix(transitions)
    if v.ndim != 1 or x.shape != v.shape or p.shape[1] != v.size:
        raise ValueError("valuations, prices, and transition states must agree")
    if horizon < 1 or discovery_cost < 0:
        raise ValueError("invalid horizon or discovery cost")
    if p.shape[0] not in (1, horizon - 1) and horizon > 1:
        raise ValueError("provide one stationary matrix or one matrix per transition")

    q_count = v.size
    net = v - x
    values = np.empty((horizon, q_count, q_count), dtype=float)
    continues = np.zeros_like(values, dtype=bool)

    terminal_cost = discovery_cost * horizon
    for best in range(q_count):
        stop = net[best] - terminal_cost
        if allow_no_purchase:
            stop = max(stop, -terminal_cost)
        values[-1, best, :] = stop

    for t in range(horizon - 2, -1, -1):
        pt = p[0] if p.shape[0] == 1 else p[t]
        elapsed_cost = discovery_cost * (t + 1)
        for best in range(q_count):
            stop = net[best] - elapsed_cost
            if allow_no_purchase:
                stop = max(stop, -elapsed_cost)
            for current in range(q_count):
                future = 0.0
                for nxt in range(q_count):
                    next_best = best if net[best] >= net[nxt] else nxt
                    future += pt[current, nxt] * values[t + 1, next_best, nxt]
                values[t, best, current] = max(stop, future)
                continues[t, best, current] = future > stop + 1e-12
    return values, continues


def stopping_time_for_trajectory(
    trajectory: np.ndarray,
    valuations: np.ndarray,
    prices: np.ndarray,
    continues: np.ndarray,
) -> int:
    """Apply a precomputed policy and return the one-indexed stopping round."""
    path = np.asarray(trajectory, dtype=np.int64)
    net = np.asarray(valuations, dtype=float) - np.asarray(prices, dtype=float)
    best = int(path[0])
    for t, current in enumerate(path):
        current = int(current)
        if net[current] > net[best]:
            best = current
        if t == len(path) - 1 or not continues[t, best, current]:
            return t + 1
    return len(path)


def stopping_distribution_dp(
    valuations: np.ndarray,
    prices: np.ndarray,
    initial: np.ndarray,
    transitions: np.ndarray,
    horizon: int,
    discovery_cost: float = 0.0,
    allow_no_purchase: bool = False,
) -> np.ndarray:
    """Compute the exact stopping-round distribution under Algorithm 3.

    The state distribution is propagated over ``(best_quality, current_quality)``
    rather than sampled trajectories. This is the exact discrete likelihood
    used by the prior-learning stage for the supplied finite environment.
    """
    v = np.asarray(valuations, dtype=float)
    x = np.asarray(prices, dtype=float)
    p0 = np.asarray(initial, dtype=float)
    p = validate_transition_matrix(transitions)
    if v.ndim != 1 or x.shape != v.shape or p.shape[1] != v.size:
        raise ValueError("valuations, prices, and transition states must agree")
    if horizon < 1 or discovery_cost < 0:
        raise ValueError("invalid horizon or discovery cost")
    if p.shape[0] not in (1, horizon - 1) and horizon > 1:
        raise ValueError("provide one stationary matrix or one matrix per transition")
    if (
        p0.shape != (v.size,)
        or np.any(p0 < 0)
        or not np.all(np.isfinite(p0))
        or not np.isclose(p0.sum(), 1.0)
    ):
        raise ValueError("initial must be a distribution over quality states")

    _, continues = optimal_stopping_dp(
        v, x, p, horizon, discovery_cost=discovery_cost, allow_no_purchase=allow_no_purchase
    )
    net = v - x
    distribution = np.zeros((v.size, v.size), dtype=float)
    states = np.arange(v.size)
    distribution[states, states] = p0
    stopping = np.zeros(horizon, dtype=float)

    for t in range(horizon):
        next_distribution = np.zeros_like(distribution)
        transition = None if t == horizon - 1 else (p[0] if p.shape[0] == 1 else p[t])
        for best in range(v.size):
            for current in range(v.size):
                probability = distribution[best, current]
                if probability == 0:
                    continue
                stop = t == horizon - 1 or not continues[t, best, current]
                if stop:
                    stopping[t] += probability
                    continue
                assert transition is not None
                for nxt, transition_probability in enumerate(transition[current]):
                    if transition_probability == 0:
                        continue
                    next_best = best if net[best] >= net[nxt] else nxt
                    next_distribution[next_best, nxt] += probability * transition_probability
        distribution = next_distribution

    if not np.isclose(stopping.sum(), 1.0, atol=1e-10):
        raise RuntimeError("stopping probability mass was not conserved")
    return stopping / stopping.sum()


def expected_payment_dp(
    valuations: np.ndarray,
    prices: np.ndarray,
    initial: np.ndarray,
    transitions: np.ndarray,
    horizon: int,
    discovery_cost: float = 0.0,
    allow_no_purchase: bool = False,
) -> float:
    """Return exact expected seller payment under the buyer's stopping policy.

    This propagates the Markov state distribution through Algorithm 3 rather
    than sampling trajectories.  Discovery costs affect the stopping policy
    but are not seller revenue.  Ties between equal-net-utility models retain
    the incumbent model, consistently with ``optimal_stopping_dp``.
    """
    v = np.asarray(valuations, dtype=float)
    x = np.asarray(prices, dtype=float)
    p0 = np.asarray(initial, dtype=float)
    p = validate_transition_matrix(transitions)
    if v.ndim != 1 or x.shape != v.shape or p.shape[1] != v.size:
        raise ValueError("valuations, prices, and transition states must agree")
    if (
        p0.shape != (v.size,)
        or np.any(p0 < 0)
        or not np.all(np.isfinite(p0))
        or not np.isclose(p0.sum(), 1.0)
    ):
        raise ValueError("initial must be a distribution over quality states")

    _, continues = optimal_stopping_dp(
        v,
        x,
        p,
        horizon,
        discovery_cost=discovery_cost,
        allow_no_purchase=allow_no_purchase,
    )
    net = v - x
    distribution = np.zeros((v.size, v.size), dtype=float)
    states = np.arange(v.size)
    distribution[states, states] = p0
    revenue = 0.0

    for t in range(horizon):
        next_distribution = np.zeros_like(distribution)
        transition = p[0] if p.shape[0] == 1 else p[t]
        for best in range(v.size):
            for current in range(v.size):
                probability = distribution[best, current]
                if probability == 0:
                    continue
                stop = t == horizon - 1 or not continues[t, best, current]
                if stop:
                    if not allow_no_purchase or net[best] >= -1e-12:
                        revenue += probability * x[best]
                    continue
                for nxt, transition_probability in enumerate(transition[current]):
                    if transition_probability == 0:
                        continue
                    next_best = best if net[best] >= net[nxt] else nxt
                    next_distribution[next_best, nxt] += probability * transition_probability
        distribution = next_distribution
    return float(revenue)
