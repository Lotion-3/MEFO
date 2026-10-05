"""Toy end-to-end: config -> run -> parquet -> resume -> report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from metabench.analysis.report import build_report
from metabench.core import registry as reg
from metabench.core.config import ExperimentConfig, load_config
from metabench.core.interfaces import Predictions, Predictor, TaskData
from metabench.core.results import ExperimentPaths, collect
from metabench.core.runner import run_experiment

CONFIG = Path(__file__).parents[1] / "configs" / "toy_e2e.yaml"


def test_toy_end_to_end(tmp_path: Path) -> None:
    cfg = load_config(CONFIG)
    s = run_experiment(cfg, output_dir=tmp_path)
    n_cond, n_pred = 8, len(cfg.predictors)
    assert (s.total, s.run, s.failed) == (n_cond * n_pred * len(cfg.seeds),) * 2 + (0,)

    paths = ExperimentPaths.for_experiment(tmp_path, cfg.experiment)
    metrics = collect(paths)
    assert not metrics.empty and (metrics["status"] == "ok").all()
    for col in ("git_sha", "versions", "data_hash", "config_hash", "seed", "solver"):
        assert col in metrics.columns

    fba_rmse = metrics.query("predictor == 'fba' and metric == 'rmse'")["value"]
    assert fba_rmse.max() < 1e-7  # true by construction on synthetic data

    # resume: nothing recomputed
    s2 = run_experiment(cfg, output_dir=tmp_path)
    assert (s2.run, s2.skipped) == (0, s.total)

    rep = build_report(paths, cfg.metrics, n_boot=50)
    assert rep.synthetic
    summ = rep.summary.set_index(["predictor", "target", "metric"])["value"]
    assert summ[("fba", "growth", "rmse")] < 1e-7
    assert summ[("mean", "growth", "rmse")] > summ[("ridge", "growth", "rmse")] > 0
    assert (rep.failures["failed_cells"] == 0).all()
    mb = rep.validity.set_index("predictor")["worst_over_folds"]
    assert mb["fba"] < 1e-6 and pd.isna(mb["mean"])


class _Broken(Predictor):
    def predict(self, test: TaskData) -> Predictions:
        raise RuntimeError("boom")


def test_failures_are_recorded_and_retried(tmp_path: Path) -> None:
    reg.register_predictor("_broken")(_Broken)
    try:
        cfg = ExperimentConfig.model_validate(
            {
                "experiment": "broken",
                "task": "steady_state_flux",
                "datasets": ["toy"],
                "predictors": ["_broken", "mean"],
                "split": {"type": "leave_condition_out"},
                "metrics": ["rmse"],
            }
        )
        s = run_experiment(cfg, output_dir=tmp_path)
        assert s.failed == 8
        metrics = collect(ExperimentPaths.for_experiment(tmp_path, "broken"))
        errs = metrics[metrics["status"] == "error"]
        assert len(errs) == 8 and errs["error"].str.contains("boom").all()
        # failed cells are retried on the next run, ok cells are not
        s2 = run_experiment(cfg, output_dir=tmp_path)
        assert (s2.run, s2.failed, s2.skipped) == (8, 8, 8)
    finally:
        reg.PREDICTORS.unregister("_broken")


BASE = {
    "experiment": "x",
    "task": "steady_state_flux",
    "datasets": ["toy"],
    "predictors": ["fba"],
    "split": {"type": "leave_condition_out"},
    "metrics": ["rmse"],
}


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"predictors": ["nope"]}, "unknown predictor"),
        ({"metrics": ["auroc"]}, "not valid for task"),
        ({"split": {"type": "grouped_kfold"}, "task": "nope"}, "unknown task"),
        ({"predictors": ["fba", "fba"]}, "duplicate predictor"),
    ],
)
def test_config_validation(override: dict, match: str) -> None:
    with pytest.raises((ValidationError, KeyError), match=match):
        ExperimentConfig.model_validate({**BASE, **override})


def test_labels_allow_same_predictor_twice() -> None:
    cfg = ExperimentConfig.model_validate(
        {
            **BASE,
            "predictors": ["ridge", {"name": "ridge", "params": {"alpha": 10}, "label": "ridge10"}],
        }
    )
    assert [p.id for p in cfg.predictors] == ["ridge", "ridge10"]
