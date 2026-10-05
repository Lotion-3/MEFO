from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabench.core import registry as reg
from metabench.core.interfaces import ALL, Predictions, pooled_key


def _pred(test, values: pd.DataFrame, fluxes: pd.DataFrame | None = None) -> Predictions:
    return Predictions(values=values, units=dict(test.units), fluxes=fluxes)


def test_point_metrics_per_target_and_pooled_by_type(toy_data, task) -> None:
    test = task.build(toy_data, list(toy_data.conditions["condition_id"]))
    values = test.targets + 1.0
    out = reg.METRICS.get("rmse")().evaluate(test, _pred(test, values))
    assert out["growth"] == pytest.approx(1.0)
    assert out[pooled_key("flux")] == pytest.approx(1.0)
    assert pooled_key("growth") not in out  # only one growth target: nothing to pool
    assert reg.METRICS.get("mae")().evaluate(test, _pred(test, values))["R2"] == pytest.approx(1.0)
    assert reg.METRICS.get("spearman")().evaluate(test, _pred(test, values))[
        "growth"
    ] == pytest.approx(1.0)


def test_r2_nan_on_single_point(toy_data, task) -> None:
    test = task.build(toy_data, ["u1"])
    out = reg.METRICS.get("r2")().evaluate(test, _pred(test, test.targets))
    assert np.isnan(out["growth"])


def test_metrics_ignore_unmeasured_targets(toy_data, task) -> None:
    test = task.build(toy_data, ["u1", "u2", "u3"])
    test.targets.loc["u1", "growth"] = np.nan
    values = test.targets.fillna(999.0)
    assert reg.METRICS.get("rmse")().evaluate(test, _pred(test, values))["growth"] == 0.0


def test_classification_metrics() -> None:
    auroc = reg.METRICS.get("auroc")()
    assert auroc.score(np.array([0, 0, 1, 1.0]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert np.isnan(auroc._safe(np.array([1, 1.0]), np.array([0.2, 0.3])))
    mcc = reg.METRICS.get("mcc")()
    assert mcc.score(np.array([0, 1.0]), np.array([0.9, 0.1])) == -1.0


def test_mass_balance_violation(toy_data, task) -> None:
    test = task.build(toy_data, ["u2"])
    ok = pd.DataFrame(
        [{"EX_A": -2, "R1": 2, "R2": 2, "R3": 0, "BIO": 4}], index=["u2"], dtype=float
    )
    bad = ok.assign(BIO=5.0)  # C_c produced 4, consumed 5
    metric = reg.METRICS.get("mass_balance_violation")()
    assert metric.evaluate(test, _pred(test, test.targets, ok))[ALL] == pytest.approx(0.0)
    assert metric.evaluate(test, _pred(test, test.targets, bad))[ALL] == pytest.approx(1.0)
    assert np.isnan(metric.evaluate(test, _pred(test, test.targets))[ALL])


def test_validate_rejects_bad_data(toy_data) -> None:
    toy_data.measurements = pd.concat([toy_data.measurements, toy_data.measurements.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        toy_data.validate()


def test_content_hash_stable_and_sensitive(toy_data) -> None:
    from metabench.datasets.toy import ToyDataset

    h = toy_data.content_hash()
    assert h == ToyDataset().load().content_hash()
    toy_data.measurements.loc[0, "value"] += 1e-6
    assert toy_data.content_hash() != h


def test_task_build_is_aligned(toy_data, task) -> None:
    test = task.build(toy_data, ["u5", "u1"])
    assert list(test.targets.index) == ["u5", "u1"]
    assert test.target_ids == ["R2", "R3", "growth"]
    assert test.units == {"R2": "mmol/gDW/h", "R3": "mmol/gDW/h", "growth": "1/h"}
    assert list(test.features.index) == ["u5", "u1"]


@pytest.mark.parametrize("name", ["rmse", "mae", "r2", "spearman"])
def test_vectorised_scores_match_scalar(name: str) -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(50, 12))
    y_pred = y_true + rng.normal(scale=0.5, size=(50, 12))
    y_true[3, :] = 1.0  # constant row -> NaN for r2/spearman
    y_pred[7, ::3] = y_pred[7, 0]  # ties
    y_true[9, 2] = np.nan  # NaN row -> scalar fallback with NaN dropping
    metric = reg.METRICS.get(name)()
    expected = np.array([metric._safe(t, p) for t, p in zip(y_true, y_pred, strict=True)])
    np.testing.assert_allclose(metric.batch(y_true, y_pred), expected, rtol=1e-10, equal_nan=True)
