"""Supplemental experiments for the mechanism-improvement proposals."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .discovery import (
    ScoreTable,
    calls_to_fraction,
    pricing_guided_discovery,
    run_discovery_methods,
)
from .dynamic_pricing import (
    expected_markov_revenue,
    optimize_cost_aware_price_grid,
    optimize_markov_price_grid,
)
from .experiments import synthetic_instance
from .market import simulate_markov_trajectories
from .pricing import (
    expected_revenue,
    independent_prices,
    optimize_price_grid,
)

SYNTHETIC_MODEL_GENERALIZATION_GAP = np.array([0.012, 0.014, 0.013, 0.015])
"""Validation-to-test discounts for the synthetic discovery table.

The gaps represent mild validation optimism for each model family and preserve
the intended augmentation ordering.  They are declared constants because they
are part of the synthetic design, not estimates from hidden paper data.
"""

WARM_START_PRICE_SIGNAL = np.array([0.05, 0.10, 0.16, 0.20, 0.78, 1.00, 0.62, 0.90])
"""Warm-start revenue-potential prior for the eight synthetic augmentations.

Large values are attached to the named high-value augmentations.  The signal
models information from historical transactions or metadata; it is not assumed
to be available in a cold-start first transaction.
"""


def run_improvement_experiments(output_dir: Path) -> dict[str, object]:
    """Run all supplemental mechanism-improvement experiments."""
    output = Path(output_dir)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    pricing_rows, pricing_multi_rows, scenario_rows = _run_pricing_improvements(tables, figures)
    discovery_rows, discovery_ablation_rows, discovery_curves = _run_discovery_improvements(tables, figures)
    summary = {
        "scope": (
            "Supplemental mechanism improvements: IR-aware pricing, robust priors, "
            "nonzero search-cost-aware Markov pricing, and pricing-guided discovery."
        ),
        "prior_cost_scenarios": scenario_rows,
        "pricing": pricing_rows,
        "pricing_multi_instance": pricing_multi_rows,
        "discovery": discovery_rows,
        "discovery_signal_ablation": discovery_ablation_rows,
    }
    (output / "improvement_experiments_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary_table(tables / "improvement_summary.csv", pricing_rows, discovery_rows)
    del discovery_curves
    return summary


def _run_pricing_improvements(
    tables: Path,
    figures: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    data = synthetic_instance()
    rows = _pricing_rows_for_instance(data)
    scenario_rows = _scenario_rows(data)

    multi_rows = _aggregate_pricing_improvements()
    _write_csv(tables / "improvement_pricing_results.csv", rows)
    _write_csv(tables / "improvement_pricing_multi_instance.csv", multi_rows)
    _write_csv(tables / "improvement_pricing_scenarios.csv", scenario_rows)
    _plot_pricing_improvements(
        rows,
        figures / "improvement_pricing_revenue.pdf",
        figures / "improvement_pricing_revenue.png",
    )
    _plot_pricing_multi_instance(
        multi_rows,
        figures / "improvement_pricing_multi_instance.pdf",
        figures / "improvement_pricing_multi_instance.png",
    )
    return rows, multi_rows, scenario_rows

def _pricing_rows_for_instance(data: dict[str, object]) -> list[dict[str, object]]:
    values = np.asarray(data["valuations"])
    prior = np.asarray(data["prior"])
    priors = _prior_scenarios(data)
    initial = np.asarray(data["initial"])
    transition = np.asarray(data["transition"])
    horizon = int(data["horizon"])
    train_paths = np.asarray(data["train_paths"])
    test_paths = np.asarray(data["test_paths"])
    costs = _cost_scenarios()
    paper = optimize_price_grid(values, prior, train_paths, force_purchase=True)
    ir = optimize_price_grid(values, prior, train_paths, force_purchase=False)
    robust_ir = optimize_price_grid(
        values,
        prior,
        train_paths,
        force_purchase=False,
        prior_scenarios=priors,
        robust=True,
    )
    cost_aware = optimize_cost_aware_price_grid(
        values,
        prior,
        initial,
        transition,
        horizon,
        discovery_costs=[0.03],
        robust=False,
    )
    robust_cost = optimize_markov_price_grid(
        values,
        prior,
        initial,
        transition,
        horizon,
        discovery_cost=0.03,
        prior_scenarios=priors,
        discovery_cost_scenarios=costs,
        robust=True,
    )

    schemes = {
        "Paper forced-choice": paper.prices,
        "Independent": independent_prices(values, prior),
        "IR-aware": ir.prices,
        "Robust IR": robust_ir.prices,
        "Cost-aware Markov": cost_aware.prices,
        "Robust cost-aware Markov": robust_cost.prices,
    }
    rows: list[dict[str, object]] = []
    for method, prices in schemes.items():
        nominal_c0 = expected_markov_revenue(
            prices, values, prior, initial, transition, horizon, 0.0
        )
        nominal_c03 = expected_markov_revenue(
            prices, values, prior, initial, transition, horizon, 0.03
        )
        nominal_c08 = expected_markov_revenue(
            prices, values, prior, initial, transition, horizon, 0.08
        )
        oos_c03 = expected_markov_revenue(
            prices, values, priors[1], initial, transition, horizon, 0.03
        )
        worst_case = min(
            expected_markov_revenue(prices, values, p, initial, transition, horizon, cost)
            for p in priors
            for cost in costs
        )
        rows.append(
            {
                "method": method,
                "prices": _format_prices(prices),
                "empirical_train_realized": expected_revenue(
                    prices, values, prior, train_paths, force_purchase=False
                ),
                "empirical_test_realized": expected_revenue(
                    prices, values, prior, test_paths, force_purchase=False
                ),
                "markov_nominal_cost_0": nominal_c0,
                "markov_nominal_cost_003": nominal_c03,
                "markov_nominal_cost_008": nominal_c08,
                "markov_oos_prior_cost_003": oos_c03,
                "markov_worst_prior_cost": worst_case,
            }
        )
    return rows


def _aggregate_pricing_improvements(repeats: int = 5, seed: int = 20260716) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset in range(repeats):
        data = _perturbed_pricing_instance(seed + offset)
        rows.extend(_pricing_rows_for_instance(data))
    metrics = (
        "markov_nominal_cost_003",
        "markov_oos_prior_cost_003",
        "markov_worst_prior_cost",
    )
    return _aggregate_rows(rows, ("method",), metrics)


def _perturbed_pricing_instance(seed: int) -> dict[str, object]:
    base = synthetic_instance(seed)
    rng = np.random.default_rng(seed + 71)
    values = np.asarray(base["valuations"]) * rng.lognormal(0.0, 0.06, size=(6, 4))
    values = np.maximum.accumulate(values, axis=1)
    prior = rng.dirichlet(70.0 * np.asarray(base["prior"]))
    oos_prior = rng.dirichlet(55.0 * np.asarray(base["oos_prior"]))
    initial = rng.dirichlet(120.0 * np.asarray(base["initial"]))
    transition = np.vstack(
        [rng.dirichlet(160.0 * row) for row in np.asarray(base["transition"])]
    )
    horizon = int(base["horizon"])
    train_paths = simulate_markov_trajectories(initial, transition, horizon, 240, rng)
    test_paths = simulate_markov_trajectories(initial, transition, horizon, 4000, rng)
    return {
        **base,
        "valuations": values,
        "prior": prior,
        "oos_prior": oos_prior,
        "uniform_prior": np.full(len(prior), 1 / len(prior)),
        "high_prior": oos_prior[::-1].copy(),
        "initial": initial,
        "transition": transition,
        "train_paths": train_paths,
        "test_paths": test_paths,
        "seed": seed,
    }


def _run_discovery_improvements(
    tables: Path,
    figures: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    table, price_signal = _synthetic_guided_discovery_table()
    budget = 16
    curves = run_discovery_methods(
        table,
        budget=budget,
        seed=20260716,
        gamma=0.1,
        price_signal=price_signal,
        price_weight=0.7,
    )
    base = float(table.test[0, 0])
    oracle = table.oracle_test_utility
    rows: list[dict[str, object]] = []
    for method, curve in curves.items():
        rows.append(
            {
                "method": method,
                "final_test_utility": float(curve[-1]),
                "normalized_auc": float(np.mean(curve)),
                "calls_to_95pct_oracle_gain": calls_to_fraction(curve, base, oracle),
                "first_four_call_mean": float(np.mean(curve[:4])),
            }
        )
    ablation_rows = _run_discovery_signal_ablation()
    _write_csv(tables / "improvement_discovery_results.csv", rows)
    _write_csv(tables / "improvement_discovery_signal_ablation.csv", ablation_rows)
    _plot_discovery_improvements(
        curves,
        figures / "improvement_discovery_curves.pdf",
        figures / "improvement_discovery_curves.png",
    )
    _plot_discovery_signal_ablation(
        ablation_rows,
        figures / "improvement_discovery_signal_ablation.pdf",
        figures / "improvement_discovery_signal_ablation.png",
    )
    return rows, ablation_rows, curves


def _synthetic_guided_discovery_table(seed: int | None = None) -> tuple[ScoreTable, np.ndarray]:
    validation = np.array(
        [
            [0.500, 0.520, 0.510, 0.505],
            [0.530, 0.550, 0.540, 0.535],
            [0.560, 0.585, 0.570, 0.565],
            [0.585, 0.610, 0.600, 0.590],
            [0.700, 0.760, 0.725, 0.710],
            [0.735, 0.820, 0.765, 0.750],
            [0.680, 0.735, 0.705, 0.695],
            [0.720, 0.800, 0.755, 0.740],
        ],
        dtype=float,
    )
    test = validation - SYNTHETIC_MODEL_GENERALIZATION_GAP[None, :]
    if seed is not None:
        rng = np.random.default_rng(seed)
        perturbation = rng.normal(0.0, 0.008, size=validation.shape)
        perturbation[0] = 0.0
        validation = np.clip(validation + perturbation, 0.0, 1.0)
        test = np.clip(test + 0.85 * perturbation + rng.normal(0.0, 0.003, size=test.shape), 0.0, 1.0)
    table = ScoreTable(
        validation=validation,
        test=test,
        model_names=("Linear", "Tree", "Kernel", "Ensemble"),
        augmentation_names=(
            "Base",
            "+ cheap profile",
            "+ weak metadata",
            "+ noisy context",
            "+ buyer-rich feature",
            "+ premium external table",
            "+ niche segment",
            "+ high-value interaction",
        ),
    )
    price_signal = WARM_START_PRICE_SIGNAL.copy()
    if seed is not None:
        rng = np.random.default_rng(seed + 17)
        price_signal = np.clip(price_signal + rng.normal(0.0, 0.035, size=price_signal.shape), 0.0, None)
    return table, price_signal


def _run_discovery_signal_ablation(repeats: int = 12, seed: int = 20260716) -> list[dict[str, object]]:
    raw_rows: list[dict[str, object]] = []
    budget = 16
    for offset in range(repeats):
        table, price_signal = _synthetic_guided_discovery_table(seed + offset)
        rng = np.random.default_rng(seed + 10_000 + offset)
        noisy_signal = np.clip(price_signal + rng.normal(0.0, 0.22, size=price_signal.shape), 0.0, None)
        random_signal = rng.random(price_signal.shape)
        curves = run_discovery_methods(table, budget=budget, seed=seed + offset, gamma=0.1)
        curves["Random-signal guided"] = pricing_guided_discovery(
            table, budget, seed + offset, random_signal, gamma=0.1, price_weight=0.7
        )
        curves["Noisy-pricing guided"] = pricing_guided_discovery(
            table, budget, seed + offset, noisy_signal, gamma=0.1, price_weight=0.7
        )
        curves["Pricing-guided Data-Bandit"] = pricing_guided_discovery(
            table, budget, seed + offset, price_signal, gamma=0.1, price_weight=0.7
        )
        base = float(table.test[0, 0])
        oracle = table.oracle_test_utility
        for method in (
            "Data-Bandit",
            "Random-signal guided",
            "Noisy-pricing guided",
            "Pricing-guided Data-Bandit",
        ):
            curve = curves[method]
            raw_rows.append(
                {
                    "repeat": offset,
                    "method": method,
                    "final_test_utility": float(curve[-1]),
                    "normalized_auc": float(np.mean(curve)),
                    "calls_to_95pct_oracle_gain": calls_to_fraction(curve, base, oracle),
                    "first_four_call_mean": float(np.mean(curve[:4])),
                }
            )
    return _aggregate_rows(
        raw_rows,
        ("method",),
        (
            "final_test_utility",
            "normalized_auc",
            "calls_to_95pct_oracle_gain",
            "first_four_call_mean",
        ),
    )


def _plot_pricing_improvements(rows: list[dict[str, object]], pdf: Path, png: Path) -> None:
    methods = [str(row["method"]) for row in rows]
    metrics = [
        ("markov_nominal_cost_003", "Nominal, c=0.03"),
        ("markov_oos_prior_cost_003", "OOS prior, c=0.03"),
        ("markov_worst_prior_cost", "Worst prior/cost"),
    ]
    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    for offset, (key, label) in enumerate(metrics):
        values = [float(row[key]) for row in rows]
        ax.bar(x + (offset - 1) * width, values, width, label=label, color=colors[offset])
    ax.set_ylabel("Exact expected model-payment revenue")
    ax.set_xticks(x, methods, rotation=15, ha="right")
    ax.set_title("Mechanism improvements under exact Markov buyer response")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_pricing_multi_instance(rows: list[dict[str, object]], pdf: Path, png: Path) -> None:
    methods = [str(row["method"]) for row in rows]
    metrics = [
        ("markov_nominal_cost_003", "Nominal, c=0.03"),
        ("markov_oos_prior_cost_003", "OOS prior, c=0.03"),
        ("markov_worst_prior_cost", "Worst prior/cost"),
    ]
    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    for offset, (key, label) in enumerate(metrics):
        means = [float(row[f"{key}_mean"]) for row in rows]
        ses = [float(row[f"{key}_se"]) for row in rows]
        ax.bar(
            x + (offset - 1) * width,
            means,
            width,
            yerr=ses,
            capsize=3,
            label=label,
            color=colors[offset],
        )
    ax.set_ylabel("Exact expected model-payment revenue")
    ax.set_xticks(x, methods, rotation=15, ha="right")
    ax.set_title("Mechanism improvements across perturbed market instances")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_discovery_improvements(
    curves: dict[str, np.ndarray],
    pdf: Path,
    png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    calls = np.arange(1, len(next(iter(curves.values()))) + 1)
    palette = {
        "Data-Bandit": "#4c72b0",
        "Pricing-guided Data-Bandit": "#c44e52",
        "Data-All": "#55a868",
        "Data-Alt": "#8172b2",
        "AutoML": "#777777",
    }
    for method, curve in curves.items():
        ax.plot(calls, curve, label=method, linewidth=2.0, color=palette.get(method))
    ax.set_xlabel("Model-training calls")
    ax.set_ylabel("Incumbent test utility")
    ax.set_title("Pricing-guided discovery reaches high-revenue augmentations earlier")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_discovery_signal_ablation(rows: list[dict[str, object]], pdf: Path, png: Path) -> None:
    methods = [str(row["method"]) for row in rows]
    metrics = [
        ("normalized_auc", "Utility AUC"),
        ("first_four_call_mean", "First 4-call utility"),
    ]
    x = np.arange(len(methods))
    width = 0.32
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    colors = ["#4c72b0", "#c44e52"]
    for offset, (key, label) in enumerate(metrics):
        means = [float(row[f"{key}_mean"]) for row in rows]
        ses = [float(row[f"{key}_se"]) for row in rows]
        ax.bar(
            x + (offset - 1) * width,
            means,
            width,
            yerr=ses,
            capsize=3,
            label=label,
            color=colors[offset],
        )
    ax.set_ylabel("Mean over perturbed score tables")
    ax.set_xticks(x, methods, rotation=15, ha="right")
    ax.set_title("Pricing signal ablation for guided discovery")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_summary_table(
    path: Path,
    pricing_rows: list[dict[str, object]],
    discovery_rows: list[dict[str, object]],
) -> None:
    best_worst = max(pricing_rows, key=lambda row: float(row["markov_worst_prior_cost"]))
    best_auc = max(discovery_rows, key=lambda row: float(row["normalized_auc"]))
    rows = [
        {
            "experiment": "pricing",
            "best_method": best_worst["method"],
            "primary_metric": "markov_worst_prior_cost",
            "value": best_worst["markov_worst_prior_cost"],
        },
        {
            "experiment": "discovery",
            "best_method": best_auc["method"],
            "primary_metric": "normalized_auc",
            "value": best_auc["normalized_auc"],
        },
    ]
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _format_prices(prices: np.ndarray) -> str:
    return " ".join(f"{float(price):.3f}" for price in prices)


def _prior_scenarios(data: dict[str, object]) -> list[np.ndarray]:
    return [
        np.asarray(data["prior"]),
        np.asarray(data["oos_prior"]),
        np.asarray(data["uniform_prior"]),
        np.asarray(data["high_prior"]),
    ]


def _cost_scenarios() -> list[float]:
    return [0.0, 0.03, 0.08]


def _scenario_rows(data: dict[str, object]) -> list[dict[str, object]]:
    names = [
        ("nominal", "training prior used by the base pricing objective"),
        ("oos_prior", "buyer mix shifted toward lower-valuation early stoppers"),
        ("uniform_prior", "uninformative cold-start prior"),
        ("high_prior", "buyer mix shifted toward higher-valuation patient buyers"),
    ]
    rows: list[dict[str, object]] = []
    for prior_name, description in names:
        prior = np.asarray(data[prior_name if prior_name != "nominal" else "prior"])
        rows.append(
            {
                "scenario": prior_name,
                "description": description,
                "prior": " ".join(f"{float(p):.3f}" for p in prior),
                "costs": " ".join(f"{cost:.2f}" for cost in _cost_scenarios()),
            }
        )
    return rows


def _aggregate_rows(
    rows: list[dict[str, object]],
    group_keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, object]] = []
    for key, group in grouped.items():
        out = {k: v for k, v in zip(group_keys, key)}
        out["n"] = len(group)
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            out[f"{metric}_mean"] = float(np.mean(values))
            out[f"{metric}_se"] = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        result.append(out)
    return result
