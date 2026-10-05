"""Every registered predictor must return predictions of the expected shape and units."""

from __future__ import annotations

import numpy as np
import pytest

from metabench.core import registry as reg
from metabench.core.runner import check_predictions

reg.load_plugins()


@pytest.mark.parametrize("name", reg.PREDICTORS.names())
def test_predictor_output_contract(name: str, toy_data, task) -> None:
    ids = list(toy_data.conditions["condition_id"])
    train = task.build(toy_data, ids[:5])
    test = task.build(toy_data, ids[5:])
    predictor = reg.PREDICTORS.get(name)(seed=0)
    predictor.fit(train)
    pred = predictor.predict(test)
    check_predictions(test, pred)  # rows, columns, units, uncertainty/flux shapes
    assert pred.values.shape == (len(ids) - 5, len(test.target_ids))
    assert np.isfinite(pred.values.to_numpy(float)).all()
    assert pred.n_lp_calls >= 0
    if reg.PREDICTORS.get(name).mechanistic:
        assert pred.n_lp_calls > 0
        assert set(pred.solver_status) == set(test.condition_ids)
