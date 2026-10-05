"""Baselines required by rule 3: per-target mean, ridge and gradient boosting on condition features.

None is hyperparameter-tuned (settings are fixed by config), matching the untuned FBA
predictors. If tuning is added later it must use nested CV inside the training split only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metabench.core.interfaces import Predictions, Predictor, TaskData
from metabench.core.registry import register_predictor


@register_predictor("mean")
class MeanPredictor(Predictor):
    """Predicts the training-set mean of each target for every test condition.

    Under leave-one-out splits its OOF predictions are exactly anti-correlated with the truth
    (holding out a high value lowers the training mean), so its Spearman/R^2 are pessimistic.
    Compare it on RMSE/MAE.
    """

    def fit(self, train: TaskData) -> None:
        self.means_ = train.targets.mean(axis=0, skipna=True)

    def predict(self, test: TaskData) -> Predictions:
        values = pd.DataFrame(
            np.tile(
                self.means_.reindex(test.target_ids).to_numpy(float), (len(test.condition_ids), 1)
            ),
            index=test.targets.index,
            columns=test.targets.columns,
        )
        return Predictions(values=values, units=dict(test.units))


@register_predictor("ridge")
class RidgePredictor(Predictor):
    """One standardized ridge regression per target on the dataset's condition features."""

    def fit(self, train: TaskData) -> None:
        # Imported here: sklearn adds ~2 s to every CLI start otherwise.
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if train.features.shape[1] == 0:
            raise ValueError("ridge needs condition features; dataset provides none")
        alpha = float(self.params.get("alpha", 1.0))
        self.feature_cols_ = list(train.features.columns)
        self.models_: dict[str, object] = {}
        x = train.features.to_numpy(float)
        for t in train.target_ids:
            y = train.targets[t].to_numpy(float)
            ok = ~np.isnan(y)
            if ok.sum() >= 2:
                self.models_[t] = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
                    x[ok], y[ok]
                )

    def predict(self, test: TaskData) -> Predictions:
        x = test.features[self.feature_cols_].to_numpy(float)
        values = pd.DataFrame(np.nan, index=test.targets.index, columns=test.targets.columns)
        for t, model in self.models_.items():
            if t in values.columns:
                values[t] = model.predict(x)  # type: ignore[attr-defined]
        return Predictions(values=values, units=dict(test.units))


@register_predictor("gbm")
class GBMPredictor(RidgePredictor):
    """One histogram gradient-boosting regressor per target (sklearn defaults, seeded)."""

    def fit(self, train: TaskData) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        if train.features.shape[1] == 0:
            raise ValueError("gbm needs condition features; dataset provides none")
        self.feature_cols_ = list(train.features.columns)
        self.models_ = {}
        x = train.features.to_numpy(float)
        for t in train.target_ids:
            y = train.targets[t].to_numpy(float)
            ok = ~np.isnan(y)
            if ok.sum() >= 2:
                model = HistGradientBoostingRegressor(random_state=self.seed, **self.params)
                self.models_[t] = model.fit(x[ok], y[ok])
