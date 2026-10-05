"""Enzyme-constrained FBA (sMOMENT/ECMpy-style total enzyme budget) and its parsimonious variant.

Params:
  enzyme_params: registered parameter set (default: the set registered for the model)
  budget:        override the published enzyme budget (g/gDW); default = published value
Nothing is fitted to the benchmark data: kcats and budget are the published ECMpy values.
"""

from __future__ import annotations

import cobra
from cobra.exceptions import OptimizationError

from metabench.core.registry import register_predictor
from metabench.metabolic.enzyme import (
    add_enzyme_constraint,
    default_parameter_set,
    load_enzyme_parameters,
)
from metabench.predictors.fba.base import ConditionResult, MechanisticPredictor, evaluate_target


class _EnzymeConstrained(MechanisticPredictor):
    parsimonious = False

    def solve(self, model: cobra.Model, targets: dict[str, dict[str, float]]) -> ConditionResult:
        name = self.params.get("enzyme_params") or default_parameter_set(model.id)
        params = load_enzyme_parameters(str(name))
        budget = self.params.get("budget")
        with model:
            add_enzyme_constraint(model, params, None if budget is None else float(budget))
            try:
                if self.parsimonious:
                    sol = cobra.flux_analysis.pfba(model)
                    n_lp = 2
                else:
                    sol = model.optimize()
                    n_lp = 1
            except OptimizationError as exc:
                return ConditionResult(status=f"infeasible: {exc}", n_lp_calls=1)
        if sol.status != "optimal":
            return ConditionResult(status=sol.status, n_lp_calls=n_lp)
        return ConditionResult(
            status="optimal",
            targets={t: evaluate_target(sol.fluxes, c) for t, c in targets.items()},
            fluxes=sol.fluxes,
            n_lp_calls=n_lp,
        )


@register_predictor("ecfba")
class ECFBA(_EnzymeConstrained):
    """Max growth subject to stoichiometry, bounds, and a total enzyme budget."""


@register_predictor("ecpfba")
class ECPFBA(_EnzymeConstrained):
    """ecFBA, then minimise total flux at optimal growth (resolves alternative optima)."""

    parsimonious = True
