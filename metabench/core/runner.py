"""Runs the (dataset x seed x fold x predictor) grid with per-cell caching and resume."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from metabench.core import registry as reg
from metabench.core.config import ExperimentConfig, PluginSpec
from metabench.core.interfaces import ALL, Predictions, StandardizedData, Task, TaskData
from metabench.core.results import ExperimentPaths, cell_is_complete, provenance, write_cell

log = logging.getLogger(__name__)


class ContractError(ValueError):
    """A predictor returned predictions that do not match the task's targets or units."""


def check_predictions(test: TaskData, pred: Predictions) -> None:
    if list(pred.values.index) != list(test.targets.index):
        raise ContractError("prediction rows do not match test conditions")
    if list(pred.values.columns) != list(test.targets.columns):
        raise ContractError("prediction columns do not match test targets")
    if pred.units != test.units:
        raise ContractError(f"prediction units {pred.units} != target units {test.units}")
    if pred.uncertainty is not None and pred.uncertainty.shape != pred.values.shape:
        raise ContractError("uncertainty shape does not match values")
    if pred.fluxes is not None and list(pred.fluxes.index) != list(test.targets.index):
        raise ContractError("flux rows do not match test conditions")


@dataclass
class RunSummary:
    total: int = 0
    run: int = 0
    skipped: int = 0
    failed: int = 0


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _long(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    wide = frame.rename_axis(index="condition_id", columns="target").reset_index()
    return wide.melt(id_vars="condition_id", var_name="target", value_name=value_name)


def _long_predictions(test: TaskData, pred: Predictions, ids: dict[str, Any]) -> pd.DataFrame:
    keys = ["condition_id", "target"]
    df = _long(test.targets, "y_true").merge(_long(pred.values, "y_pred"), on=keys)
    if pred.uncertainty is not None:
        df = df.merge(_long(pred.uncertainty, "uncertainty"), on=keys)
    else:
        df["uncertainty"] = np.nan
    df["measurement_type"] = df["target"].map(test.target_types)
    df["units"] = df["target"].map(test.units)
    df["solver_status"] = df["condition_id"].map(pred.solver_status).fillna("")
    for k, v in ids.items():
        df[k] = v
    return df


def run_cell(
    task: Task,
    data: StandardizedData,
    spec: PluginSpec,
    train_ids: list[str],
    test_ids: list[str],
    seed: int,
    model_id: str | None,
) -> tuple[TaskData, Predictions, float, float]:
    train = task.build(data, train_ids, model_id)
    test = task.build(data, test_ids, model_id)
    predictor = reg.PREDICTORS.get(spec.name)(seed=seed, **spec.params)
    t0 = time.perf_counter()
    predictor.fit(train)
    t1 = time.perf_counter()
    pred = predictor.predict(test)
    t2 = time.perf_counter()
    pred.wall_time_s = t2 - t1
    check_predictions(test, pred)
    return test, pred, t1 - t0, t2 - t1


def run_experiment(
    cfg: ExperimentConfig, force: bool = False, output_dir: str | Path | None = None
) -> RunSummary:
    reg.load_plugins()
    paths = ExperimentPaths.for_experiment(output_dir or cfg.output_dir, cfg.experiment)
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "config.json").write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    task: Task = reg.TASKS.get(cfg.task)()
    summary = RunSummary()
    cfg_hash = cfg.config_hash()

    for ds_spec in cfg.datasets:
        data: StandardizedData = reg.DATASETS.get(ds_spec.name)(**ds_spec.params).load()
        data.validate()
        data_hash = data.content_hash()
        splitter = reg.SPLITTERS.get(cfg.split.type)(**cfg.split.params)
        for seed in cfg.seeds:
            folds = splitter.split(data, seed)
            seen: Counter[str] = Counter()
            for fold, (train_ids, test_ids) in enumerate(folds):
                # repeat r = r-th time these conditions are tested under this seed; one
                # (seed, repeat) pair gives exactly one out-of-fold prediction per condition.
                repeat = max(seen[c] for c in test_ids)
                seen.update(test_ids)
                if set(train_ids) & set(test_ids):  # defence in depth; splitters are tested too
                    raise RuntimeError(
                        f"split leakage in fold {fold}: {set(train_ids) & set(test_ids)}"
                    )
                for spec in cfg.predictors:
                    summary.total += 1
                    ids: dict[str, Any] = {
                        "experiment": cfg.experiment,
                        "task": cfg.task,
                        "dataset": ds_spec.name,
                        "predictor": spec.id,
                        "predictor_name": spec.name,
                        "predictor_params": json.dumps(spec.params, sort_keys=True),
                        "mechanistic": bool(reg.PREDICTORS.get(spec.name).mechanistic),
                        "split": json.dumps(cfg.split.model_dump(), sort_keys=True),
                        "fold": fold,
                        "repeat": repeat,
                        "seed": seed,
                        "synthetic_data": data.synthetic,
                    }
                    key = _hash(
                        {
                            **ids,
                            "data_hash": data_hash,
                            "dataset_params": ds_spec.params,
                            "model": cfg.model,
                            "metrics": cfg.metrics,
                            "test_ids": sorted(test_ids),
                        }
                    )
                    if not force and cell_is_complete(paths, key):
                        summary.skipped += 1
                        continue
                    summary.run += 1
                    prov = provenance()
                    common = {
                        **ids,
                        "n_train": len(train_ids),
                        "n_test": len(test_ids),
                        "config_hash": cfg_hash,
                        "data_hash": data_hash,
                        "cell_key": key,
                        **prov,
                    }
                    try:
                        test, pred, fit_s, predict_s = run_cell(
                            task, data, spec, train_ids, test_ids, seed, cfg.model
                        )
                        rows = []
                        for mname in cfg.metrics:
                            metric = reg.METRICS.get(mname)()
                            for target, value in metric.evaluate(test, pred).items():
                                rows.append({"metric": mname, "target": target, "value": value})
                        rows += [
                            {"metric": "fit_time_s", "target": ALL, "value": fit_s},
                            {"metric": "predict_time_s", "target": ALL, "value": predict_s},
                            {
                                "metric": "n_lp_calls",
                                "target": ALL,
                                "value": float(pred.n_lp_calls),
                            },
                        ]
                        metrics_df = pd.DataFrame(rows).assign(status="ok", error="", **common)
                        preds_df = _long_predictions(test, pred, common)
                        status = {"status": "ok", **common}
                    except Exception as exc:  # recorded, not swallowed: failures are results too
                        summary.failed += 1
                        err = f"{type(exc).__name__}: {exc}"
                        log.error(
                            "cell %s (%s, fold %d, seed %d) failed: %s",
                            key,
                            spec.id,
                            fold,
                            seed,
                            err,
                        )
                        metrics_df = pd.DataFrame(
                            [{"metric": "__error__", "target": ALL, "value": np.nan}]
                        ).assign(status="error", error=err, **common)
                        preds_df = None
                        status = {
                            "status": "error",
                            "error": err,
                            "traceback": traceback.format_exc(),
                            **common,
                        }
                    write_cell(paths, key, metrics_df, preds_df, status)
    return summary
