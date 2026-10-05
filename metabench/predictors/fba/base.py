"""Shared machinery for predictors that solve a genome-scale model per condition."""

from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import cobra
import numpy as np
import pandas as pd

from metabench.core.interfaces import Predictions, Predictor, TaskData
from metabench.metabolic.models import (
    MODEL_UNITS,
    condition_context,
    load_model,
    target_to_reactions,
)


@dataclass
class ConditionResult:
    status: str
    targets: dict[str, float] = field(default_factory=dict)  # NaN-filled when not optimal
    uncertainty: dict[str, float] | None = None
    fluxes: pd.Series | None = None
    n_lp_calls: int = 0


def evaluate_target(fluxes: pd.Series, coefs: dict[str, float]) -> float:
    return float(sum(c * fluxes[r] for r, c in coefs.items()))


class MechanisticPredictor(Predictor):
    """Solves each test condition independently; `fit` is a no-op (no training data used)."""

    mechanistic = True
    returns_uncertainty = False
    # In-process memo: a mechanistic prediction depends only on the condition, never on the
    # training split, so it is computed once per (predictor, params, model, condition, targets).
    # Cached results keep their original n_lp_calls, so reported LP cost is per prediction.
    _memo: ClassVar[dict[str, ConditionResult]] = {}

    def check_units(self, test: TaskData) -> None:
        for t in test.target_ids:
            expected = MODEL_UNITS.get(test.target_types[t])
            if expected is None:
                raise ValueError(f"{self.name} cannot predict '{test.target_types[t]}' targets")
            if test.units[t] != expected:
                raise ValueError(
                    f"target '{t}' is in {test.units[t]}, model predicts {expected}; "
                    "convert units in the dataset loader"
                )

    @abstractmethod
    def solve(
        self, model: cobra.Model, targets: dict[str, dict[str, float]]
    ) -> ConditionResult: ...

    def predict(self, test: TaskData) -> Predictions:
        self.check_units(test)
        model = load_model(test.model_id)
        target_rxns = {t: target_to_reactions(model, t, test.target_map) for t in test.target_ids}
        values = pd.DataFrame(np.nan, index=test.targets.index, columns=test.targets.columns)
        unc = values.copy() if self.returns_uncertainty else None
        flux_rows: dict[str, pd.Series] = {}
        statuses: dict[str, str] = {}
        n_lp = 0
        for cid in test.condition_ids:
            key = json.dumps(
                [
                    self.name,
                    self.params,
                    test.model_id,
                    test.media[cid],
                    vars(test.perturbation(cid)),
                    target_rxns,
                ],
                sort_keys=True,
            )
            res = MechanisticPredictor._memo.get(key)
            if res is None:
                with condition_context(model, test.media[cid], test.perturbation(cid)) as m:
                    res = self.solve(m, target_rxns)
                MechanisticPredictor._memo[key] = res
            statuses[cid] = res.status
            n_lp += res.n_lp_calls
            for t, v in res.targets.items():
                values.loc[cid, t] = v
            if unc is not None and res.uncertainty is not None:
                for t, v in res.uncertainty.items():
                    unc.loc[cid, t] = v
            if res.fluxes is not None:
                flux_rows[cid] = res.fluxes
        fluxes = None
        if flux_rows:
            rxn_ids = [r.id for r in model.reactions]
            fluxes = pd.DataFrame(flux_rows).T.reindex(index=test.condition_ids, columns=rxn_ids)
        return Predictions(
            values=values,
            units=dict(test.units),
            uncertainty=unc,
            fluxes=fluxes,
            solver_status=statuses,
            n_lp_calls=n_lp,
        )
