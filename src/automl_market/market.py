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

