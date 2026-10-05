"""Classification metrics for essentiality-type targets.

Truth is binary (1 = essential / positive). Predictions are scores where higher means more
likely positive; threshold-based metrics binarise at `threshold` (0.5).
sklearn is imported lazily to keep CLI start-up fast.
"""

from __future__ import annotations

import numpy as np

from metabench.core.interfaces import PointMetric
from metabench.core.registry import register_metric


class _Binary(PointMetric):
    kind = "classification"
    min_points = 2
    threshold = 0.5
    needs_both_classes = True

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y = y_true.astype(int)
        if self.needs_both_classes and len(np.unique(y)) < 2:
            return float("nan")
        return self._score(y, y_pred)


@register_metric("auroc")
class AUROC(_Binary):
    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_pred))


@register_metric("auprc")
class AUPRC(_Binary):
    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_true, y_pred))


@register_metric("mcc")
class MCC(_Binary):
    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import matthews_corrcoef

        return float(matthews_corrcoef(y_true, (y_pred >= self.threshold).astype(int)))


@register_metric("balanced_accuracy")
class BalancedAccuracy(_Binary):
    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import balanced_accuracy_score

        return float(balanced_accuracy_score(y_true, (y_pred >= self.threshold).astype(int)))
