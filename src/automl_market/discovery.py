"""Reduced-scale reproduction of Algorithm 2 and the RQ1 baselines.

The original 69K-table NYC market and Metam implementation are unavailable.
This module preserves the experiment's essential query-budget semantics on a
public join-augmentation task: Data-Bandit trains one sampled model per
candidate, Data-All trains all models, Data-Alt uses a cheap fixed model with
periodic model search, and AutoML searches models without external data.
"""

from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ScoreTable:
    validation: np.ndarray
    test: np.ndarray
    model_names: tuple[str, ...]
    augmentation_names: tuple[str, ...]

    @property
    def oracle_pair(self) -> tuple[int, int]:
        flat = int(np.argmax(self.validation))
        return tuple(int(x) for x in np.unravel_index(flat, self.validation.shape))

    @property
    def oracle_test_utility(self) -> float:
        return float(self.test[self.oracle_pair])


MODEL_SPECS: tuple[tuple[str, str, float], ...] = (
    ("Linear-0.01", "linear", 0.01),
    ("Linear-0.1", "linear", 0.1),
    ("Linear-1", "linear", 1.0),
    ("Linear-10", "linear", 10.0),
    ("Quadratic", "quadratic", 1.0),
    ("RandomFeatures", "relu", 1.0),
)


def load_wine_archive(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...]]]:
    """Read red and white Wine Quality CSV files without extracting the ZIP."""
    result: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = {}
    with zipfile.ZipFile(path) as archive:
        for color, member in (
            ("red", "winequality-red.csv"),
            ("white", "winequality-white.csv"),
        ):
            with archive.open(member) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8")
                table = np.genfromtxt(text, delimiter=";", names=True, dtype=float)
            names = tuple(table.dtype.names or ())
            matrix = np.column_stack([table[name] for name in names])
            result[color] = (matrix[:, :-1], matrix[:, -1], names[:-1])
    return result


def make_score_table(
    features: np.ndarray,
    quality: np.ndarray,
    feature_names: tuple[str, ...],
    task: str,
    seed: int,
    sample_size: int = 1200,
) -> ScoreTable:
    """Create one repeated-split augmentation/model score table.

    Candidate ordering uses only the training partition. Model selection uses
    validation scores; test scores are touched only when reporting the current
    validation-selected incumbent.
    """
    rng = np.random.default_rng(seed)
    n = min(sample_size, len(quality))
    selected = rng.choice(len(quality), size=n, replace=False)
    selected = selected[rng.permutation(n)]
    n_train, n_val = int(0.60 * n), int(0.20 * n)
    train_idx = selected[:n_train]
    val_idx = selected[n_train : n_train + n_val]
    test_idx = selected[n_train + n_val :]

    if task == "classification":
        target = (quality >= 6.0).astype(float)
    elif task == "regression":
        target = quality.astype(float)
    else:
        raise ValueError("task must be classification or regression")

    base = [0, 1]
    external = list(range(2, features.shape[1]))
    correlations = []
    y_train = target[train_idx]
    for column in external:
        x = features[train_idx, column]
        if np.std(x) < 1e-12 or np.std(y_train) < 1e-12:
            corr = 0.0
        else:
            corr = abs(float(np.corrcoef(x, y_train)[0, 1]))
        correlations.append((corr, column))
    ranked = [column for _, column in sorted(correlations, reverse=True)]

    feature_sets: list[list[int]] = [base.copy()]
    augmentation_names = ["No augmentation"]
    for column in ranked:
        feature_sets.append(feature_sets[-1] + [column])
        augmentation_names.append("+ " + feature_names[column])

    validation = np.zeros((len(feature_sets), len(MODEL_SPECS)), dtype=float)
    test = np.zeros_like(validation)
    for step, columns in enumerate(feature_sets):
        x_train = features[train_idx][:, columns]
        x_val = features[val_idx][:, columns]
        x_test = features[test_idx][:, columns]
        for model_id, (_, transform, penalty) in enumerate(MODEL_SPECS):
            pred_val, pred_test = _fit_predict_ridge_family(
                x_train,
                y_train,
                x_val,
                x_test,
                transform=transform,
                penalty=penalty,
                classification=task == "classification",
                seed=seed * 101 + model_id * 17 + step,
            )
            validation[step, model_id] = _utility(y_train=None, truth=target[val_idx], prediction=pred_val, task=task)
            test[step, model_id] = _utility(y_train=None, truth=target[test_idx], prediction=pred_test, task=task)

    return ScoreTable(
        validation=validation,
        test=test,
        model_names=tuple(spec[0] for spec in MODEL_SPECS),
        augmentation_names=tuple(augmentation_names),
    )


def run_discovery_methods(
    table: ScoreTable,
    budget: int,
    seed: int,
    gamma: float = 0.1,
) -> dict[str, np.ndarray]:
    """Run Data-Bandit, Data-All, Data-Alt, and AutoML on one score table."""
    if budget < 1:
        raise ValueError("budget must be positive")
    return {
        "Data-Bandit": _data_bandit(table, budget, seed, gamma),
        "Data-All": _data_all(table, budget),
        "Data-Alt": _data_alt(table, budget),
        "AutoML": _automl(table, budget),
    }


def calls_to_fraction(curve: np.ndarray, base: float, oracle: float, fraction: float = 0.95) -> int:
    """Training calls needed to recover a fraction of oracle improvement."""
    target = base + fraction * max(0.0, oracle - base)
    reached = np.flatnonzero(np.asarray(curve) >= target - 1e-12)
    return int(reached[0] + 1) if reached.size else int(len(curve) + 1)


def paired_bootstrap_mean_ci(
    differences: np.ndarray,
    seed: int,
    draws: int = 10000,
) -> tuple[float, float]:
    """Deterministic percentile CI for a paired mean difference."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("differences must be a non-empty one-dimensional array")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _fit_predict_ridge_family(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    transform: str,
    penalty: float,
    classification: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-10] = 1.0
    train = (x_train - mean) / scale
    val = (x_val - mean) / scale
    test = (x_test - mean) / scale
    train, val, test = _transform(train, val, test, transform, seed)

    train = np.column_stack([np.ones(len(train)), train])
    val = np.column_stack([np.ones(len(val)), val])
    test = np.column_stack([np.ones(len(test)), test])
    fit_target = 2.0 * y_train - 1.0 if classification else y_train
    regularizer = np.eye(train.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    gram = train.T @ train + regularizer
    rhs = train.T @ fit_target
    try:
        weights = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(gram) @ rhs
    pred_val = val @ weights
    pred_test = test @ weights
    if classification:
        pred_val = (pred_val >= 0.0).astype(float)
        pred_test = (pred_test >= 0.0).astype(float)
    return pred_val, pred_test


def _transform(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    kind: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if kind == "linear":
        return train, val, test
    if kind == "quadratic":
        pairs = [(i, j) for i in range(train.shape[1]) for j in range(i, train.shape[1])]
        def expand(x: np.ndarray) -> np.ndarray:
            interactions = np.column_stack([x[:, i] * x[:, j] for i, j in pairs])
            return np.column_stack([x, interactions])
        return expand(train), expand(val), expand(test)
    if kind == "relu":
        rng = np.random.default_rng(seed)
        width = 32
        weights = rng.normal(scale=1.0 / math.sqrt(train.shape[1]), size=(train.shape[1], width))
        bias = rng.normal(scale=0.25, size=width)
        def expand(x: np.ndarray) -> np.ndarray:
            return np.column_stack([x, np.maximum(0.0, x @ weights + bias)])
        return expand(train), expand(val), expand(test)
    raise ValueError(f"unknown transform: {kind}")


def _utility(
    y_train: np.ndarray | None,
    truth: np.ndarray,
    prediction: np.ndarray,
    task: str,
) -> float:
    del y_train
    if task == "classification":
        return float(np.mean(truth == prediction))
    rmse = float(np.sqrt(np.mean((truth - prediction) ** 2)))
    scale = float(np.std(truth))
    return float(1.0 / (1.0 + rmse / max(scale, 1e-12)))


def _curve_recorder(table: ScoreTable, budget: int):
    best_val = -np.inf
    best_test = 0.0
    best_pair = (0, 0)
    curve: list[float] = []

    def query(step: int, model: int) -> float:
        nonlocal best_val, best_test, best_pair
        value = float(table.validation[step, model])
        if value > best_val + 1e-12:
            best_val = value
            best_test = float(table.test[step, model])
            best_pair = (step, model)
        curve.append(best_test)
        return value

    def finish() -> np.ndarray:
        if not curve:
            curve.append(float(table.test[0, 0]))
        curve.extend([curve[-1]] * max(0, budget - len(curve)))
        return np.asarray(curve[:budget], dtype=float)

    def pair() -> tuple[int, int]:
        return best_pair

    return query, finish, pair, curve


def _data_bandit(table: ScoreTable, budget: int, seed: int, gamma: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, finish, incumbent, curve = _curve_recorder(table, budget)
    models = table.validation.shape[1]
    weights = np.ones(models, dtype=float)
    observed: list[tuple[float, int]] = []
    for step in range(table.validation.shape[0]):
        if len(curve) >= budget:
            return finish()
        probabilities = (1.0 - gamma) * weights / weights.sum() + gamma / models
        arm = int(rng.choice(models, p=probabilities))
        reward = q(step, arm)
        observed.append((reward, step))
        exponent = gamma * reward / (models * probabilities[arm])
        weights[arm] *= math.exp(min(exponent, 20.0))

    top_t = max(1, math.ceil(math.log2(table.validation.shape[0])))
    best_arm = int(np.argmax(weights))
    for _, step in sorted(observed, reverse=True)[:top_t]:
        if len(curve) >= budget:
            return finish()
        q(step, best_arm)

    best_step, _ = incumbent()
    for arm in range(models):
        if len(curve) >= budget:
            break
        q(best_step, arm)
    return finish()


def _data_all(table: ScoreTable, budget: int) -> np.ndarray:
    q, finish, _, curve = _curve_recorder(table, budget)
    for step in range(table.validation.shape[0]):
        for arm in range(table.validation.shape[1]):
            if len(curve) >= budget:
                return finish()
            q(step, arm)
    return finish()


def _data_alt(table: ScoreTable, budget: int) -> np.ndarray:
    q, finish, _, curve = _curve_recorder(table, budget)
    fixed_arm = 2  # efficient linear ridge model
    seen: list[int] = []
    for step in range(table.validation.shape[0]):
        if len(curve) >= budget:
            return finish()
        q(step, fixed_arm)
        seen.append(step)
        if (step + 1) % 3 == 0 or step == table.validation.shape[0] - 1:
            best_step = max(seen, key=lambda s: table.validation[s, fixed_arm])
            for arm in range(table.validation.shape[1]):
                if len(curve) >= budget:
                    return finish()
                q(best_step, arm)
    return finish()


def _automl(table: ScoreTable, budget: int) -> np.ndarray:
    q, finish, _, curve = _curve_recorder(table, budget)
    for arm in range(table.validation.shape[1]):
        if len(curve) >= budget:
            break
        q(0, arm)
    return finish()
