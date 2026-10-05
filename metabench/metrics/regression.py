"""Continuous-target metrics, reported per target and pooled within measurement type.

Each metric has a scalar `score` and a vectorised `score_rows` (used for bootstrap CIs);
tests/test_metrics_and_data.py checks that both agree.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr

from metabench.core.interfaces import PointMetric
from metabench.core.registry import register_metric


@register_metric("rmse")
class RMSE(PointMetric):
    kind = "continuous"

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def score_rows(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))


@register_metric("mae")
class MAE(PointMetric):
    kind = "continuous"

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    def score_rows(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(y_true - y_pred), axis=1)


@register_metric("r2")
class R2(PointMetric):
    """Coefficient of determination. NaN for < 2 points or constant truth.

    Note: on leave-one-condition-out folds each fold has one point per target, so per-fold R^2
    and Spearman are NaN by design; the report computes them on pooled out-of-fold predictions.
    """

    kind = "continuous"
    min_points = 2

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        if ss_tot == 0:
            return float("nan")
        return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot

    def score_rows(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        ss_tot = np.sum((y_true - y_true.mean(axis=1, keepdims=True)) ** 2, axis=1)
        ss_res = np.sum((y_true - y_pred) ** 2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(ss_tot == 0, np.nan, 1.0 - ss_res / ss_tot)


@register_metric("spearman")
class Spearman(PointMetric):
    kind = "continuous"
    min_points = 3

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
            return float("nan")
        return float(spearmanr(y_true, y_pred).statistic)

    def score_rows(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Spearman = Pearson correlation of average ranks (same tie handling as scipy).
        rt = rankdata(y_true, axis=1)
        rp = rankdata(y_pred, axis=1)
        rt -= rt.mean(axis=1, keepdims=True)
        rp -= rp.mean(axis=1, keepdims=True)
        num = np.sum(rt * rp, axis=1)
        den = np.sqrt(np.sum(rt**2, axis=1) * np.sum(rp**2, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den == 0, np.nan, num / den)
