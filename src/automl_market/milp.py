"""Optional HiGHS backend for the empirical pricing MILP in Appendix B.1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pricing import independent_prices, revenue_by_type


@dataclass(frozen=True)
class MilpPricingResult:
    """Solution and audit metadata for an empirical pricing MILP."""

    prices: np.ndarray
    objective: float
    solver_objective: float
    objective_audit_difference: float
    revenue_by_type: np.ndarray
    status: str
    mip_gap: float
    runtime_seconds: float
    variables: int
    constraints: int
    unique_availability_sets: int
    trajectories: int
    price_upper_bounds: np.ndarray
    force_purchase: bool


def solve_pricing_milp(
    valuations: np.ndarray,
    prior: np.ndarray,
    trajectories: np.ndarray,
    *,
    force_purchase: bool = False,
    price_upper_bounds: np.ndarray | float | None = None,
    time_limit: float | None = None,
    mip_rel_gap: float = 1e-8,
    random_seed: int = 0,
    output_flag: bool = False,
    warm_start_prices: np.ndarray | None = None,
) -> MilpPricingResult:
    """Solve the sample-average pricing problem with the open-source HiGHS MILP solver.

    Equal trajectory availability sets are compressed before constructing the
    model.  ``force_purchase=False`` adds a zero-price, zero-value outside
    option and therefore enforces individual rationality.

    A finite price domain is essential: the forced-choice formulation is
    unbounded without it.  By default each quality is capped at its largest
    observed valuation, matching the domain implicit in ``candidate_price_grid``.
    This bound is without loss for the IR-aware formulation but is a modelling
    assumption for forced choice.

    ``highspy`` is an optional dependency.  Install it with
    ``conda env create -f environment.yml`` or ``pip install -r requirements-milp.txt``.
    """
    try:
        import highspy
    except ImportError as exc:  # pragma: no cover - exercised without optional dependency
        raise RuntimeError(
            "HiGHS is required for solve_pricing_milp; install environment.yml "
            "or requirements-milp.txt"
        ) from exc

    values, weights, paths, upper = _validate_inputs(
        valuations, prior, trajectories, price_upper_bounds
    )
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if mip_rel_gap < 0:
        raise ValueError("mip_rel_gap must be nonnegative")
    if warm_start_prices is None:
        warm_start = np.minimum(independent_prices(values, weights), upper)
    else:
        warm_start = np.asarray(warm_start_prices, dtype=float)
        if (
            warm_start.shape != upper.shape
            or not np.all(np.isfinite(warm_start))
            or np.any(warm_start < 0)
            or np.any(warm_start > upper + 1e-12)
        ):
            raise ValueError("warm_start_prices must lie inside the price bounds")

    availability, counts = _compress_availability(paths, values.shape[1])
    sample_count = len(paths)
    model = highspy.Highs()
    model.setOptionValue("output_flag", output_flag)
    model.setOptionValue("threads", 1)
    model.setOptionValue("random_seed", random_seed)
    model.setOptionValue("mip_rel_gap", mip_rel_gap)
    if time_limit is not None:
        model.setOptionValue("time_limit", time_limit)

    prices = [
        model.addVariable(lb=0.0, ub=float(upper[q]), name=f"x_{q}")
        for q in range(values.shape[1])
    ]
    binary = highspy.HighsVarType.kInteger

    for type_id, type_values in enumerate(values):
        for mask_id, mask in enumerate(availability):
            available = np.flatnonzero(mask)
            a_upper = float(np.max(type_values[available]))
            utility_floor = float(np.min(type_values[available] - upper[available]))
            a_lower = 0.0 if not force_purchase else utility_floor
            best_utility = model.addVariable(
                lb=a_lower, ub=a_upper, name=f"a_{type_id}_{mask_id}"
            )
            selections = []

            for quality in available:
                quality = int(quality)
                selected = model.addVariable(
                    lb=0.0,
                    ub=1.0,
                    type=binary,
                    name=f"z_{type_id}_{mask_id}_{quality}",
                )
                payment_weight = float(weights[type_id] * counts[mask_id] / sample_count)
                payment = model.addVariable(
                    lb=0.0,
                    ub=float(upper[quality]),
                    obj=payment_weight,
                    name=f"y_{type_id}_{mask_id}_{quality}",
                )
                selections.append(selected)

                utility_gap = best_utility - float(type_values[quality]) + prices[quality]
                utility_big_m = a_upper - float(type_values[quality]) + float(upper[quality])
                model.addConstr(utility_gap >= 0.0)
                model.addConstr(utility_gap <= utility_big_m * (1.0 - selected))

                # Exact linearization of y = x z on 0 <= x <= upper[quality].
                model.addConstr(payment <= prices[quality])
                model.addConstr(payment <= float(upper[quality]) * selected)
                model.addConstr(
                    payment >= prices[quality] - float(upper[quality]) * (1.0 - selected)
                )

            if force_purchase:
                model.addConstr(model.qsum(selections) == 1.0)
            else:
                outside = model.addVariable(
                    lb=0.0,
                    ub=1.0,
                    type=binary,
                    name=f"z_out_{type_id}_{mask_id}",
                )
                # The outside option has utility and payment zero.  The revenue
                # objective selects a paid model when utility is tied at zero,
                # matching seller-favourable tie breaking in revenue_by_type.
                model.addConstr(best_utility <= a_upper * (1.0 - outside))
                model.addConstr(model.qsum([*selections, outside]) == 1.0)

    # A partial continuous-price start lets HiGHS repair the associated buyer
    # choices.  It prevents short time limits from returning an incumbent worse
    # than the transparent independent-pricing baseline.
    start_indices = np.asarray([int(variable) for variable in prices], dtype=np.int32)
    model.setSolution(len(prices), start_indices, warm_start.astype(np.float64))
    model.maximize()
    solution = model.getSolution()
    status = model.modelStatusToString(model.getModelStatus())
    if not solution.value_valid:
        raise RuntimeError(f"HiGHS returned no feasible pricing solution ({status})")

    solved_prices = np.asarray([model.val(variable) for variable in prices], dtype=float)
    solved_prices[np.abs(solved_prices) < 1e-10] = 0.0
    by_type = revenue_by_type(
        solved_prices,
        values,
        paths,
        force_purchase=force_purchase,
        seller_favorable_ties=True,
    )
    realized = float(weights @ by_type)
    solver_objective = float(model.getObjectiveValue())
    tolerance = 2e-6 * max(1.0, abs(solver_objective))
    audit_difference = realized - solver_objective
    if status == "Optimal" and abs(audit_difference) > tolerance:
        raise RuntimeError(
            "MILP objective does not match direct buyer-response evaluation: "
            f"solver={solver_objective:.9g}, direct={realized:.9g}"
        )

    info = model.getInfo()
    return MilpPricingResult(
        prices=solved_prices,
        objective=realized,
        solver_objective=solver_objective,
        objective_audit_difference=audit_difference,
        revenue_by_type=by_type,
        status=status,
        mip_gap=float(info.mip_gap),
        runtime_seconds=float(model.getRunTime()),
        variables=int(model.getNumCol()),
        constraints=int(model.getNumRow()),
        unique_availability_sets=len(availability),
        trajectories=sample_count,
        price_upper_bounds=upper,
        force_purchase=force_purchase,
    )


def _validate_inputs(
    valuations: np.ndarray,
    prior: np.ndarray,
    trajectories: np.ndarray,
    price_upper_bounds: np.ndarray | float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(valuations, dtype=float)
    weights = np.asarray(prior, dtype=float)
    paths = np.asarray(trajectories, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("valuations must have shape (types, qualities)")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("valuations must be finite and nonnegative")
    if (
        weights.shape != (values.shape[0],)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError("prior must be a distribution over buyer types")
    if (
        paths.ndim != 2
        or paths.shape[0] == 0
        or paths.shape[1] == 0
        or np.any(paths < 0)
        or np.any(paths >= values.shape[1])
    ):
        raise ValueError("trajectories contain invalid quality states")

    if price_upper_bounds is None:
        upper = np.max(values, axis=0)
    else:
        upper = np.asarray(price_upper_bounds, dtype=float)
        if upper.ndim == 0:
            upper = np.full(values.shape[1], float(upper))
    if upper.shape != (values.shape[1],) or not np.all(np.isfinite(upper)) or np.any(upper < 0):
        raise ValueError("price_upper_bounds must be finite and nonnegative")
    return values, weights, paths, upper


def _compress_availability(trajectories: np.ndarray, qualities: int) -> tuple[np.ndarray, np.ndarray]:
    availability = np.zeros((len(trajectories), qualities), dtype=np.uint8)
    availability[np.arange(len(trajectories))[:, None], trajectories] = 1
    masks, counts = np.unique(availability, axis=0, return_counts=True)
    return masks.astype(bool), counts
