"""Plain FBA and parsimonious FBA."""

from __future__ import annotations

import cobra
from cobra.exceptions import OptimizationError

from metabench.core.registry import register_predictor
from metabench.predictors.fba.base import ConditionResult, MechanisticPredictor, evaluate_target


@register_predictor("fba")
class FBA(MechanisticPredictor):
    """Maximise the model objective (biomass). Alternative optima are resolved by the solver."""

    def solve(self, model: cobra.Model, targets: dict[str, dict[str, float]]) -> ConditionResult:
        sol = model.optimize()
        if sol.status != "optimal":
            return ConditionResult(status=sol.status, n_lp_calls=1)
        return ConditionResult(
            status="optimal",
            targets={t: evaluate_target(sol.fluxes, c) for t, c in targets.items()},
            fluxes=sol.fluxes,
            n_lp_calls=1,
        )


@register_predictor("pfba")
class PFBA(MechanisticPredictor):
    """Optimal growth, then minimise total absolute flux (Lewis et al. 2010)."""

    def solve(self, model: cobra.Model, targets: dict[str, dict[str, float]]) -> ConditionResult:
        fraction = float(self.params.get("fraction_of_optimum", 1.0))
        try:
            sol = cobra.flux_analysis.pfba(model, fraction_of_optimum=fraction)
        except OptimizationError as exc:
            return ConditionResult(status=f"infeasible: {exc}", n_lp_calls=1)
        if sol.status != "optimal":
            return ConditionResult(status=sol.status, n_lp_calls=2)
        return ConditionResult(
            status="optimal",
            targets={t: evaluate_target(sol.fluxes, c) for t, c in targets.items()},
            fluxes=sol.fluxes,
            n_lp_calls=2,
        )
