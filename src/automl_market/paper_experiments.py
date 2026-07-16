"""Reduced, transparent reproductions of the paper's RQ2 and RQ3 experiments."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .experiments import synthetic_instance
from .learning import smoothed_bayesian_learning
from .market import optimal_stopping_dp, simulate_markov_trajectories, stopping_time_for_trajectory
from .milp import solve_pricing_milp
from .pricing import (
    expected_revenue,
    independent_prices,
    optimize_jiggle_prices,
    optimize_shift_prices,
)


def run_paper_experiments(
    output_dir: Path,
    *,
    rq2_repeats: int = 10,
    rq3_seeds: int = 5,
    rq3_rounds: int = 1000,
    seed: int = 20260716,
) -> dict[str, object]:
    """Run reduced RQ2/RQ3 reproductions and write tables, figures, and JSON."""
    output = Path(output_dir)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    rq2 = run_rq2_reproduction(rq2_repeats, seed)
    rq3 = run_rq3_reproduction(rq3_seeds, rq3_rounds, seed + 10_000)
    _write_csv(tables / "rq2_paper_reproduction.csv", rq2["rows"])
    _write_csv(tables / "rq2_paper_summary.csv", rq2["summary"])
    _write_csv(tables / "rq3_paper_reproduction.csv", rq3["rows"])
    _write_csv(tables / "rq3_paper_summary.csv", rq3["summary"])
    _plot_rq2(rq2["summary"], figures / "rq2_paper_reproduction.pdf", figures / "rq2_paper_reproduction.png")
    _plot_rq3(rq3["curves"], figures / "rq3_paper_reproduction.pdf", figures / "rq3_paper_reproduction.png")

    result = {
        "scope": (
            "Reduced synthetic reproduction: public paper text does not include the School-data "
            "tasks, sampled valuations, or original implementation needed for pointwise Figure 4/5 replication."
        ),
        "seed": seed,
        "rq2": {key: value for key, value in rq2.items() if key != "rows"},
        "rq3": {key: value for key, value in rq3.items() if key not in {"rows", "curves"}},
    }
    (output / "paper_experiments_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_rq2_reproduction(repeats: int, seed: int) -> dict[str, object]:
    """Compare MILP, Independent, Shift, and Jiggle under IS/OOS availability."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        data = _rq2_instance(seed + repeat)
        values = data["valuations"]
        prior = data["prior"]
        train_is = data["train_is"]
        train_oos = data["train_oos"]
        test_oos = data["test_oos"]
        welfare_is = _expected_welfare(values, prior, train_is)
        welfare_oos = _expected_welfare(values, prior, test_oos)

        milp = solve_pricing_milp(values, prior, train_is, force_purchase=True)
        shifted = optimize_shift_prices(values, prior, train_is, force_purchase=True)
        jiggled = optimize_jiggle_prices(values, prior, train_is, force_purchase=True)
        oos_oracle = solve_pricing_milp(
            values,
            prior,
            train_oos,
            force_purchase=True,
            time_limit=20.0,
            mip_rel_gap=1e-4,
        )
        schemes = {
            "MILP (IS)": milp.prices,
            "Independent": independent_prices(values, prior),
            "Shift": shifted.prices,
            "Jiggle": jiggled.prices,
            "OOS-informed MILP": oos_oracle.prices,
        }
        for method, prices in schemes.items():
            forced_is = expected_revenue(prices, values, prior, train_is, True)
            forced_oos = expected_revenue(prices, values, prior, test_oos, True)
            realized_is = expected_revenue(prices, values, prior, train_is, False)
            realized_oos = expected_revenue(prices, values, prior, test_oos, False)
            rows.append(
                {
                    "repeat": repeat,
                    "seed": seed + repeat,
                    "method": method,
                    "forced_is_normalized": forced_is / welfare_is,
                    "forced_oos_normalized": forced_oos / welfare_oos,
                    "realized_is_normalized": realized_is / welfare_is,
                    "realized_oos_normalized": realized_oos / welfare_oos,
                    "welfare_is": welfare_is,
                    "welfare_oos": welfare_oos,
                    "prices": " ".join(f"{price:.6f}" for price in prices),
                }
            )
    summary = _aggregate_rows(
        rows,
        group_keys=("method",),
        metrics=(
            "forced_is_normalized",
            "forced_oos_normalized",
            "realized_is_normalized",
            "realized_oos_normalized",
        ),
    )
    return {
        "repeats": repeats,
        "quality_states": 4,
        "buyer_types": 6,
        "train_trajectories": 100,
        "oos_test_trajectories": 4000,
        "rows": rows,
        "summary": summary,
    }


def run_rq3_reproduction(seeds: int, rounds: int, seed: int) -> dict[str, object]:
    """Reproduce prior learning across distributions, rates, and batch sizes."""
    if seeds < 1 or rounds < 1:
        raise ValueError("seeds and rounds must be positive")
    rows: list[dict[str, object]] = []
    selected_curves: dict[str, list[np.ndarray]] = defaultdict(list)
    for seed_offset in range(seeds):
        rng = np.random.default_rng(seed + seed_offset)
        likelihoods = _rq3_likelihoods(rng)
        priors = _rq3_priors(rng)
        for prior_name, prior in priors.items():
            for rate_name, rate in (("1/(t+1)", "harmonic"), ("1/sqrt(t)", "sqrt"), ("1/2", 0.5)):
                for batch_size in (10, 100):
                    _, kl = smoothed_bayesian_learning(
                        likelihoods,
                        prior,
                        rounds=rounds,
                        batch_size=batch_size,
                        rng=np.random.default_rng(seed + seed_offset * 10_000 + batch_size),
                        learning_rate=rate,
                    )
                    rows.append(
                        {
                            "seed": seed + seed_offset,
                            "prior": prior_name,
                            "learning_rate": rate_name,
                            "batch_size": batch_size,
                            "rounds": rounds,
                            "initial_kl": float(kl[0]),
                            "final_kl": float(kl[-1]),
                            "tail_mean_kl": float(np.mean(kl[-min(100, len(kl)) :])),
                            "tail_std_kl": float(np.std(kl[-min(100, len(kl)) :])),
                        }
                    )
                    if prior_name == "random" and batch_size == 100:
                        selected_curves[rate_name].append(kl)
    summary = _aggregate_rows(
        rows,
        group_keys=("prior", "learning_rate", "batch_size"),
        metrics=("final_kl", "tail_mean_kl", "tail_std_kl"),
    )
    curves = {
        rate: np.mean(np.vstack(rate_curves), axis=0)
        for rate, rate_curves in selected_curves.items()
    }
    return {
        "seeds": seeds,
        "rounds": rounds,
        "quality_states": 10,
        "buyer_types": 5,
        "horizon": 15,
        "prior_families": ["random", "uniform", "slightly_skewed", "highly_skewed", "extremely_skewed"],
        "learning_rates": ["1/(t+1)", "1/sqrt(t)", "1/2"],
        "batch_sizes": [10, 100],
        "rows": rows,
        "summary": summary,
        "curves": curves,
    }


def _rq2_instance(seed: int) -> dict[str, np.ndarray]:
    data = synthetic_instance(seed)
    rng = np.random.default_rng(seed + 97)
    values = np.asarray(data["valuations"]) * rng.lognormal(0.0, 0.04, size=(6, 4))
    values = np.maximum.accumulate(values, axis=1)
    prior = rng.dirichlet(80.0 * np.asarray(data["prior"]))
    train_is = np.tile(np.arange(values.shape[1]), (100, 1))
    return {
        "valuations": values,
        "prior": prior,
        "train_is": train_is,
        "train_oos": np.asarray(data["train_paths"])[:100],
        "test_oos": np.asarray(data["test_paths"]),
    }


def _expected_welfare(
    valuations: np.ndarray, prior: np.ndarray, trajectories: np.ndarray
) -> float:
    paths = np.asarray(trajectories, dtype=np.int64)
    masks = np.bitwise_or.reduce(1 << paths, axis=1)
    unique_masks, counts = np.unique(masks, return_counts=True)
    by_type = np.zeros(len(prior), dtype=float)
    for type_id, type_values in enumerate(valuations):
        total = 0.0
        for mask, count in zip(unique_masks, counts):
            available = np.flatnonzero(mask & (1 << np.arange(valuations.shape[1])))
            total += count * float(np.max(type_values[available]))
        by_type[type_id] = total / len(paths)
    return float(np.asarray(prior) @ by_type)


def _rq3_likelihoods(rng: np.random.Generator) -> np.ndarray:
    qualities = np.linspace(0.1, 1.0, 10)
    # Calibrated to produce five distinct stopping distributions (mean stopping
    # rounds are approximately 1.0, 4.0, 7.2, 10.2, and 12.6).  This makes the
    # identifiability assumption behind Section 5 explicit rather than hiding
    # indistinguishable buyer types in the experiment.
    caps = np.array([0.20, 0.30, 0.40, 0.50, 0.55])
    powers = np.full(5, 0.5)
    values = np.asarray([cap * qualities**power for cap, power in zip(caps, powers)])
    initial = np.zeros(10)
    initial[:3] = [0.72, 0.23, 0.05]
    transition = np.zeros((10, 10))
    for quality in range(10):
        transition[quality, quality] += 0.52
        transition[quality, min(quality + 1, 9)] += 0.34
        transition[quality, min(quality + 2, 9)] += 0.14
    horizon = 15
    paths = simulate_markov_trajectories(initial, transition, horizon, 12_000, rng)
    likelihoods = np.zeros((len(values), horizon))
    for type_id, type_values in enumerate(values):
        _, policy = optimal_stopping_dp(
            type_values,
            np.zeros(10),
            transition,
            horizon,
            discovery_cost=0.018,
        )
        stops = [
            stopping_time_for_trajectory(path, type_values, np.zeros(10), policy)
            for path in paths
        ]
        likelihoods[type_id] = np.bincount(stops, minlength=horizon + 1)[1:] / len(stops)
    return likelihoods


def _rq3_priors(rng: np.random.Generator) -> dict[str, np.ndarray]:
    random_prior = rng.dirichlet(np.ones(5))
    slight = rng.beta(2, 3, size=5)
    high = rng.beta(1, 5, size=5)
    extreme = np.geomspace(1.0, 0.02, 5)
    return {
        "random": random_prior,
        "uniform": np.full(5, 0.2),
        "slightly_skewed": slight / slight.sum(),
        "highly_skewed": high / high.sum(),
        "extremely_skewed": extreme / extreme.sum(),
    }


def _aggregate_rows(
    rows: list[dict[str, object]],
    group_keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    summary: list[dict[str, object]] = []
    for group, group_rows in grouped.items():
        item = {key: value for key, value in zip(group_keys, group)}
        item["n"] = len(group_rows)
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group_rows])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_se"] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot_rq2(summary: list[dict[str, object]], pdf: Path, png: Path) -> None:
    order = ["MILP (IS)", "Independent", "Shift", "Jiggle", "OOS-informed MILP"]
    indexed = {str(row["method"]): row for row in summary}
    x = np.arange(len(order))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    for axis, prefix, title in (
        (axes[0], "forced", "Paper forced-choice objective"),
        (axes[1], "realized", "Realizable revenue with outside option"),
    ):
        is_values = [float(indexed[name][f"{prefix}_is_normalized_mean"]) for name in order]
        oos_values = [float(indexed[name][f"{prefix}_oos_normalized_mean"]) for name in order]
        axis.bar(x - width / 2, is_values, width, label="IS", color="#4c72b0")
        axis.bar(x + width / 2, oos_values, width, label="OOS", color="#dd8452")
        axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_xticks(x, order, rotation=18, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Revenue / zero-price welfare")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_rq3(curves: dict[str, np.ndarray], pdf: Path, png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    colors = {"1/(t+1)": "#4c72b0", "1/sqrt(t)": "#55a868", "1/2": "#c44e52"}
    for rate in ("1/(t+1)", "1/sqrt(t)", "1/2"):
        ax.plot(curves[rate], label=rate, color=colors[rate], linewidth=1.7)
    ax.set(
        xlabel="Learning round",
        ylabel="KL(true prior || belief)",
        title="RQ3 reduced reproduction: random prior, batch size 100",
    )
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(title="Learning rate", frameon=False)
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
