"""Mechanistic-validity metrics on predicted full flux vectors.

A predictor that returns no flux vector gets NaN ("not applicable"), never 0.
"""

from __future__ import annotations

import numpy as np

from metabench.core.interfaces import ALL, Metric, Predictions, TaskData
from metabench.core.registry import register_metric
from metabench.metabolic.models import stoichiometric_matrix


@register_metric("mass_balance_violation")
class MassBalanceViolation(Metric):
    """max over test conditions of ||S v||_2 (mmol/gDW/h). ~0 for any LP solution."""

    kind = "validity"

    def evaluate(self, test: TaskData, pred: Predictions) -> dict[str, float]:
        if pred.fluxes is None:
            return {ALL: float("nan")}
        s, rxn_ids = stoichiometric_matrix(test.model_id)
        v = pred.fluxes.reindex(columns=rxn_ids).to_numpy(float)
        v = v[~np.isnan(v).any(axis=1)]  # conditions without a complete flux vector are skipped
        if len(v) == 0:
            return {ALL: float("nan")}
        norms = np.linalg.norm(v @ s.T, axis=1)
        return {ALL: float(np.max(norms))}
