"""Bayesian learning of the buyer-type prior from stopping-round observations."""

from __future__ import annotations

import numpy as np


def smoothed_bayesian_learning(
    likelihoods: np.ndarray,
    true_prior: np.ndarray,
    rounds: int,
    batch_size: int,
    rng: np.random.Generator,
    learning_rate: str | float = "sqrt",
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
    if isinstance(learning_rate, str) and learning_rate not in {"sqrt", "harmonic"}:
        raise ValueError("learning_rate must be 'sqrt', 'harmonic', or a number in (0, 1]")
    if not isinstance(learning_rate, str) and not 0 < float(learning_rate) <= 1:
        raise ValueError("constant learning rate must lie in (0, 1]")

    belief = np.full_like(true_prior, 1.0 / len(true_prior))
    history = [belief.copy()]
    kl_history = [_kl(true_prior, belief)]
    for t in range(rounds):
        sampled_types = rng.choice(len(true_prior), size=batch_size, p=true_prior)
        observations = np.empty(batch_size, dtype=np.int64)
        for buyer_type in range(len(true_prior)):
            mask = sampled_types == buyer_type
            observations[mask] = rng.choice(
                likelihoods.shape[1], size=int(mask.sum()), p=likelihoods[buyer_type]
            )
        # One row per sampled stopping observation.  Vectorizing this batch
        # update keeps the Algorithm 1 semantics while making the 10k-round
        # Figure 5 settings practical to reproduce.
        unnormalized = likelihoods[:, observations].T * belief
        normalizers = unnormalized.sum(axis=1, keepdims=True)
        posteriors = np.divide(
            unnormalized,
            normalizers,
            out=np.broadcast_to(belief, unnormalized.shape).copy(),
            where=normalizers > 1e-15,
        )
        batch_posterior = posteriors.mean(axis=0)
        if learning_rate == "sqrt":
            eta = 1.0 / np.sqrt(t + 1)
        elif learning_rate == "harmonic":
            eta = 1.0 / (t + 1)
        else:
            eta = float(learning_rate)
        belief = (1.0 - eta) * belief + eta * batch_posterior
        belief /= belief.sum()
        history.append(belief.copy())
        kl_history.append(_kl(true_prior, belief))
    return np.asarray(history), np.asarray(kl_history)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-15))))
