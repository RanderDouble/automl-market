"""Bayesian learning of the buyer-type prior from stopping-round observations."""

from __future__ import annotations

import numpy as np


def smoothed_bayesian_learning(
    likelihoods: np.ndarray,
    true_prior: np.ndarray,
    rounds: int,
    batch_size: int,
    rng: np.random.Generator,
    learning_rate: str = "sqrt",
) -> tuple[np.ndarray, np.ndarray]:
    """Run Algorithm 1's posterior smoothing with categorical observations.

    ``likelihoods[type, observation]`` is the precomputed stopping-round
    distribution.  A batch posterior uses the mean of per-observation Bayes
    posteriors, then applies the paper's convex smoothing update.
    """
    likelihoods = np.asarray(likelihoods, dtype=float)
    true_prior = np.asarray(true_prior, dtype=float)
    if likelihoods.ndim != 2 or true_prior.shape != (likelihoods.shape[0],):
        raise ValueError("likelihood and prior shapes disagree")
    if not np.allclose(likelihoods.sum(axis=1), 1.0):
        raise ValueError("each likelihood row must sum to one")

    belief = np.full_like(true_prior, 1.0 / len(true_prior))
    history = [belief.copy()]
    kl_history = [_kl(true_prior, belief)]
    for t in range(rounds):
        posteriors = []
        sampled_types = rng.choice(len(true_prior), size=batch_size, p=true_prior)
        for buyer_type in sampled_types:
            observation = rng.choice(likelihoods.shape[1], p=likelihoods[buyer_type])
            unnormalized = likelihoods[:, observation] * belief
            if unnormalized.sum() <= 1e-15:
                posterior = belief
            else:
                posterior = unnormalized / unnormalized.sum()
            posteriors.append(posterior)
        batch_posterior = np.mean(posteriors, axis=0)
        eta = 1.0 / np.sqrt(t + 1) if learning_rate == "sqrt" else 1.0 / (t + 1)
        belief = (1.0 - eta) * belief + eta * batch_posterior
        belief /= belief.sum()
        history.append(belief.copy())
        kl_history.append(_kl(true_prior, belief))
    return np.asarray(history), np.asarray(kl_history)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-15))))

