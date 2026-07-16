"""Deterministic synthetic reproduction experiments for RQ2/RQ3 and our fix."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .learning import smoothed_bayesian_learning
from .market import optimal_stopping_dp, simulate_markov_trajectories, stopping_time_for_trajectory
from .pricing import expected_revenue, independent_prices, optimize_price_grid, revenue_by_type


def synthetic_instance(seed: int = 20260716) -> dict[str, np.ndarray | int]:
    rng = np.random.default_rng(seed)
    qualities = np.array([0.45, 0.60, 0.75, 0.90])
    # Monotone but heterogeneous value curves: some buyers saturate quickly,
    # while others value only top-quality models.  This creates the
    # identifiable stopping behavior required by Theorem 5.2.
    valuations = np.array(
        [
            [0.20, 0.21, 0.22, 0.23],
            [0.18, 0.25, 0.31, 0.35],
            [0.15, 0.27, 0.40, 0.50],
            [0.12, 0.30, 0.52, 0.72],
            [0.22, 0.39, 0.70, 1.02],
            [0.08, 0.28, 0.78, 1.30],
        ]
    )
    prior = np.array([0.10, 0.14, 0.19, 0.22, 0.20, 0.15])
    oos_prior = np.array([0.28, 0.25, 0.19, 0.14, 0.09, 0.05])
    uniform_prior = np.full(len(prior), 1 / len(prior))
    high_prior = oos_prior[::-1].copy()
    initial = np.array([0.72, 0.20, 0.07, 0.01])
    transition = np.array(
        [
            [0.50, 0.38, 0.10, 0.02],
            [0.08, 0.49, 0.36, 0.07],
            [0.02, 0.10, 0.50, 0.38],
            [0.01, 0.03, 0.16, 0.80],
        ]
    )
    horizon = 6
    train_paths = simulate_markov_trajectories(initial, transition, horizon, 240, rng)
    test_paths = simulate_markov_trajectories(initial, transition, horizon, 4000, rng)
    return {
        "qualities": qualities,
        "valuations": valuations,
        "prior": prior,
        "oos_prior": oos_prior,
        "uniform_prior": uniform_prior,
        "high_prior": high_prior,
        "initial": initial,
        "transition": transition,
        "horizon": horizon,
        "train_paths": train_paths,
        "test_paths": test_paths,
        "seed": seed,
    }


def run_all(output_dir: Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    data = synthetic_instance()
    values = data["valuations"]
    prior = data["prior"]
    paths_train = data["train_paths"]
    paths_test = data["test_paths"]

    paper = optimize_price_grid(values, prior, paths_train, force_purchase=True)
    ir = optimize_price_grid(values, prior, paths_train, force_purchase=False)
    robust = optimize_price_grid(
        values,
        prior,
        paths_train,
        force_purchase=False,
        prior_scenarios=[prior, data["oos_prior"], data["uniform_prior"], data["high_prior"]],
        robust=True,
    )
    independent = independent_prices(values, prior)
    schemes = {
        "Paper forced-choice": paper.prices,
        "Independent": independent,
        "IR-aware (ours)": ir.prices,
        "Robust IR (ours)": robust.prices,
    }

    rows: list[dict[str, object]] = []
    for name, prices in schemes.items():
        forced_prediction = expected_revenue(prices, values, prior, paths_train, True)
        train_realized = expected_revenue(prices, values, prior, paths_train, False)
        test_is = expected_revenue(prices, values, prior, paths_test, False)
        test_oos = expected_revenue(prices, values, data["oos_prior"], paths_test, False)
        forced_by_type = revenue_by_type(prices, values, paths_test, True)
        ir_by_type = revenue_by_type(prices, values, paths_test, False)
        violation_mass = float(prior @ (forced_by_type > ir_by_type + 1e-12))
        rows.append(
            {
                "scheme": name,
                "prices": " ".join(f"{p:.3f}" for p in prices),
                "forced_train_objective": forced_prediction,
                "realized_train_revenue": train_realized,
                "realized_is_revenue": test_is,
                "realized_oos_revenue": test_oos,
                "ir_violation_type_mass": violation_mass,
            }
        )

    with (tables / "pricing_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot_pricing(rows, figures / "pricing_comparison.pdf", figures / "pricing_comparison.png")

    learning_summary = _run_learning(data, figures)
    summary = {
        "seed": int(data["seed"]),
        "quality_states": len(data["qualities"]),
        "buyer_types": len(prior),
        "horizon": int(data["horizon"]),
        "train_trajectories": len(paths_train),
        "test_trajectories": len(paths_test),
        "grid_curves": paper.evaluated_curves,
        "pricing": rows,
        "learning": learning_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _run_learning(data: dict[str, np.ndarray | int], figures: Path) -> dict[str, object]:
    values = np.asarray(data["valuations"])
    prior = np.asarray(data["prior"])
    transition = np.asarray(data["transition"])
    horizon = int(data["horizon"])
    paths = np.asarray(data["test_paths"])
    # A zero-price exploration phase isolates how willingness-to-wait reveals
    # type information; charging begins after the prior-estimation phase.
    prices = np.zeros(values.shape[1])
    likelihoods = np.zeros((len(prior), horizon), dtype=float)
    for buyer_type in range(len(prior)):
        _, policy = optimal_stopping_dp(
            values[buyer_type], prices, transition, horizon, discovery_cost=0.03,
            allow_no_purchase=True,
        )
        stops = [stopping_time_for_trajectory(p, values[buyer_type], prices, policy) for p in paths]
        counts = np.bincount(stops, minlength=horizon + 1)[1:]
        likelihoods[buyer_type] = (counts + 1.0) / (counts.sum() + horizon)

    history, kl = smoothed_bayesian_learning(
        likelihoods, prior, rounds=500, batch_size=100,
        rng=np.random.default_rng(int(data["seed"]) + 1), learning_rate="sqrt",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    axes[0].plot(kl, color="#3b6ea8", linewidth=1.8)
    axes[0].set(xlabel="Learning round", ylabel="KL(true || belief)", title="Prior-learning convergence")
    axes[0].grid(alpha=0.25)
    x = np.arange(len(prior))
    axes[1].bar(x - 0.18, prior, 0.36, label="True prior", color="#3b6ea8")
    axes[1].bar(x + 0.18, history[-1], 0.36, label="Learned prior", color="#dd8452")
    axes[1].set(xlabel="Buyer type", ylabel="Probability", title="True vs. learned prior")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "prior_learning.pdf", bbox_inches="tight")
    fig.savefig(figures / "prior_learning.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "initial_kl": float(kl[0]),
        "final_kl": float(kl[-1]),
        "true_prior": prior.tolist(),
        "learned_prior": history[-1].tolist(),
    }


def _plot_pricing(rows: list[dict[str, object]], pdf: Path, png: Path) -> None:
    labels = [str(r["scheme"]) for r in rows]
    train = [float(r["realized_train_revenue"]) for r in rows]
    is_test = [float(r["realized_is_revenue"]) for r in rows]
    oos = [float(r["realized_oos_revenue"]) for r in rows]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.bar(x - width, train, width, label="Train (realized)", color="#4c72b0")
    ax.bar(x, is_test, width, label="IS test", color="#55a868")
    ax.bar(x + width, oos, width, label="OOS test", color="#c44e52")
    ax.set_ylabel("Expected model-payment revenue")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_title("Realizable revenue after adding the no-purchase outside option")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
