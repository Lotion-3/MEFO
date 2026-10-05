"""FVA-based point prediction: midpoint of each target's feasible range at near-optimal growth."""

from __future__ import annotations

import cobra
from cobra.util.solver import fix_objective_as_constraint

from metabench.core.registry import register_predictor
from metabench.predictors.fba.base import ConditionResult, MechanisticPredictor


@register_predictor("fva_summary")
class FVASummary(MechanisticPredictor):
    """For each target (a linear combination of fluxes), min and max it subject to
    growth >= fraction_of_optimum * max growth. Prediction = midpoint, uncertainty = half-range.

    Ranges are computed on the target expression itself, so lumped/signed targets get their true
    range rather than a combination of per-reaction FVA bounds. No flux vector is returned: the
    per-reaction midpoints are not a single feasible flux distribution.
    """

    returns_uncertainty = True

    def solve(self, model: cobra.Model, targets: dict[str, dict[str, float]]) -> ConditionResult:
        fraction = float(self.params.get("fraction_of_optimum", 1.0))
        opt = model.slim_optimize(error_value=float("nan"))
        n_lp = 1
        if opt != opt:  # NaN -> infeasible
            return ConditionResult(status=model.solver.status, n_lp_calls=n_lp)
        mid: dict[str, float] = {}
        half: dict[str, float] = {}
        with model:
            fix_objective_as_constraint(model, fraction=fraction)
            for t, coefs in targets.items():
                expr = sum(
                    c * model.reactions.get_by_id(r).flux_expression for r, c in coefs.items()
                )
                bounds = []
                for direction in ("min", "max"):
                    model.objective = model.problem.Objective(expr, direction=direction)
                    bounds.append(model.slim_optimize(error_value=float("nan")))
                    n_lp += 1
                lo, hi = bounds
                mid[t] = (lo + hi) / 2
                half[t] = (hi - lo) / 2
        return ConditionResult(status="optimal", targets=mid, uncertainty=half, n_lp_calls=n_lp)
