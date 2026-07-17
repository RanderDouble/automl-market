#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from automl_market.discovery import (
    calls_to_fraction,
    load_wine_archive,
    make_score_table,
    paired_bootstrap_mean_ci,
    run_discovery_methods,
)


def run(output: Path, repeats_per_color: int = 15, budget: int = 60) -> dict[str, object]:
    output = Path(output)
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    datasets = load_wine_archive(ROOT / "data/raw/wine-quality.zip")
    methods = ("Data-Bandit", "Data-All", "Data-Alt", "AutoML")
    tasks = ("classification", "regression")
    curves = {task: {method: [] for method in methods} for task in tasks}
    calls = {task: {method: [] for method in methods} for task in tasks}
    oracle = {task: [] for task in tasks}

    long_rows: list[dict[str, object]] = []
    for task_id, task in enumerate(tasks):
        for color_id, (color, (x, y, names)) in enumerate(datasets.items()):
            for repeat in range(repeats_per_color):
                seed = 31000 + task_id * 10000 + color_id * 1000 + repeat
                table = make_score_table(x, y, names, task, seed)
                method_curves = run_discovery_methods(table, budget, seed + 77)
                base_utility = float(np.max(table.test[0]))
                oracle_utility = table.oracle_test_utility
                oracle[task].append(oracle_utility)
                for method, curve in method_curves.items():
                    curves[task][method].append(curve)
                    n_calls = calls_to_fraction(curve, base_utility, oracle_utility)
                    normalized_auc = float(np.mean(curve))
                    calls[task][method].append(n_calls)
                    long_rows.append(
                        {
                            "task": task,
                            "dataset": color,
                            "repeat": repeat,
                            "seed": seed,
                            "method": method,
                            "final_utility": float(curve[-1]),
                            "normalized_auc": normalized_auc,
                            "utility_at_call_10": _utility_at_call(curve, 10),
                            "utility_at_call_20": _utility_at_call(curve, 20),
                            "utility_at_call_60": _utility_at_call(curve, 60),
                            "calls_to_95pct_oracle_gain": n_calls,
                            "reached_95pct_oracle_gain": int(n_calls <= budget),
                            "final_utility_matches_oracle": int(
                                abs(float(curve[-1]) - oracle_utility) <= 1e-12
                            ),
                            "base_utility": base_utility,
                            "oracle_utility": oracle_utility,
                            "final_gap_to_oracle": float(oracle_utility - curve[-1]),
                        }
                    )

    with (tables / "rq1_task_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(long_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(long_rows)

    summary_rows: list[dict[str, object]] = []
    for task in tasks:
        for method in methods:
            matrix = np.asarray(curves[task][method])
            normalized_auc = matrix.mean(axis=1)
            method_calls = np.asarray(calls[task][method])
            summary_rows.append(
                {
                    "task": task,
                    "method": method,
                    "tasks": len(matrix),
                    "final_utility_mean": float(matrix[:, -1].mean()),
                    "final_utility_se": float(matrix[:, -1].std(ddof=1) / np.sqrt(len(matrix))),
                    "normalized_auc_mean": float(normalized_auc.mean()),
                    "normalized_auc_se": float(normalized_auc.std(ddof=1) / np.sqrt(len(matrix))),
                    "utility_at_call_10_mean": _mean_at_call(matrix, 10),
                    "utility_at_call_20_mean": _mean_at_call(matrix, 20),
                    "utility_at_call_60_mean": _mean_at_call(matrix, 60),
                    "calls_to_95pct_mean": float(method_calls.mean()),
                    "calls_to_95pct_se": float(method_calls.std(ddof=1) / np.sqrt(len(matrix))),
                    "reached_95pct_count": int(np.sum(method_calls <= budget)),
                    "final_utility_matches_oracle_count": int(
                        np.sum(np.abs(matrix[:, -1] - np.asarray(oracle[task])) <= 1e-12)
                    ),
                    "mean_final_gap_to_oracle": float(
                        (np.asarray(oracle[task]) - matrix[:, -1]).mean()
                    ),
                }
            )
    with (tables / "rq1_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    paired_rows: list[dict[str, object]] = []
    for task_id, task in enumerate(tasks):
        bandit_auc = np.asarray(curves[task]["Data-Bandit"]).mean(axis=1)
        for baseline_id, baseline in enumerate(methods[1:]):
            baseline_auc = np.asarray(curves[task][baseline]).mean(axis=1)
            differences = bandit_auc - baseline_auc
            ci_low, ci_high = paired_bootstrap_mean_ci(
                differences,
                seed=51000 + task_id * 100 + baseline_id,
            )
            paired_rows.append(
                {
                    "task": task,
                    "comparison": f"Data-Bandit - {baseline}",
                    "paired_tasks": len(differences),
                    "normalized_auc_difference_mean": float(differences.mean()),
                    "bootstrap_95pct_ci_low": ci_low,
                    "bootstrap_95pct_ci_high": ci_high,
                    "bandit_wins": int(np.sum(differences > 1e-12)),
                    "ties": int(np.sum(np.abs(differences) <= 1e-12)),
                    "bandit_losses": int(np.sum(differences < -1e-12)),
                }
            )
    with (tables / "rq1_paired_comparisons.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(paired_rows)

    _plot(curves, calls, budget, figures / "rq1_discovery.pdf", figures / "rq1_discovery.png")
    result = {
        "dataset": "UCI Wine Quality (red and white)",
        "repeats_per_color": repeats_per_color,
        "tasks_per_problem": repeats_per_color * len(datasets),
        "budget_model_trainings": budget,
        "models": 6,
        "base_features": 2,
        "external_augmentations": 9,
        "normalized_auc_definition": (
            "mean incumbent test utility over budgets 1..B; curves are right-padded "
            "after a method terminates"
        ),
        "curve_padding": "final utility is carried forward after a method exhausts its prescribed search",
        "unreached_calls_sentinel": budget + 1,
        "summary": summary_rows,
        "paired_auc_comparisons": paired_rows,
    }
    (output / "rq1_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Completed RQ1 on {2 * repeats_per_color * len(datasets)} repeated tasks.")
    print(f"Results: {output / 'rq1_summary.json'}")
    return result


def _utility_at_call(curve: np.ndarray, call: int) -> float | None:
    if call < 1 or call > len(curve):
        return None
    return float(curve[call - 1])


def _mean_at_call(matrix: np.ndarray, call: int) -> float | None:
    if call < 1 or call > matrix.shape[1]:
        return None
    return float(matrix[:, call - 1].mean())


def _plot(
    curves: dict[str, dict[str, list[np.ndarray]]],
    calls: dict[str, dict[str, list[int]]],
    budget: int,
    pdf: Path,
    png: Path,
) -> None:
    colors = {
        "Data-Bandit": "#4c72b0",
        "Data-All": "#55a868",
        "Data-Alt": "#c44e52",
        "AutoML": "#8172b3",
    }
    # The main RQ1 figure focuses on the informative early budget.  The
    # supplementary oracle-efficiency panel is intentionally omitted.
    display_budget = min(30, budget)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    x = np.arange(1, display_budget + 1)
    for axis, task in zip(axes, ("classification", "regression")):
        for method, task_curves in curves[task].items():
            matrix = np.asarray(task_curves)
            mean = matrix[:, :display_budget].mean(axis=0)
            axis.plot(x, mean, label=method, color=colors[method], linewidth=1.8)
        axis.set(
            xlabel="Model-training budget",
            ylabel="Test utility",
            title=task.capitalize(),
            xlim=(1, display_budget),
        )
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--repeats-per-color", type=int, default=15)
    parser.add_argument("--budget", type=int, default=60)
    args = parser.parse_args()
    run(args.output, args.repeats_per_color, args.budget)


if __name__ == "__main__":
    main()
