"""The small, stable interfaces every plugin implements.

Data flows as:  Dataset.load() -> StandardizedData --Task.build(ids)--> TaskData
                Predictor.fit(TaskData); Predictor.predict(TaskData) -> Predictions
                Metric.evaluate(TaskData, Predictions) -> {target: value}
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd

MeasurementType = Literal["growth", "flux", "essentiality", "abundance"]
MEASUREMENT_TYPES: tuple[str, ...] = ("growth", "flux", "essentiality", "abundance")

# Required columns of the two tidy tables in StandardizedData.
CONDITION_COLUMNS: tuple[str, ...] = ("condition_id", "organism", "model_id")
MEASUREMENT_COLUMNS: tuple[str, ...] = (
    "condition_id",
    "measurement_type",
    "target_id",
    "value",
    "units",
)

# Special target id for the growth rate; mechanistic predictors map it to the model objective.
GROWTH_TARGET = "growth"


@dataclass
class Perturbation:
    """Genetic perturbation of one condition relative to the reference model."""

    gene_knockouts: list[str] = field(default_factory=list)
    reaction_knockouts: list[str] = field(default_factory=list)

    def is_wild_type(self) -> bool:
        return not (self.gene_knockouts or self.reaction_knockouts)


@dataclass
class StandardizedData:
    """Everything a dataset loader returns, in a fixed schema.

    conditions:   one row per condition; columns CONDITION_COLUMNS plus any grouping columns
                  used by splitters (e.g. strain, carbon_source, species).
    measurements: tidy long table, columns MEASUREMENT_COLUMNS (+ optional 'std').
    media:        condition_id -> {exchange reaction id: max uptake rate (positive, model units)}.
                  Exchanges not listed are closed for uptake (COBRApy `model.medium` semantics).
    perturbations: condition_id -> Perturbation (missing = wild type).
    features:     numeric condition descriptors used by ML predictors (index = condition_id).
                  Must never contain target information.
    target_map:   target_id -> {reaction id: coefficient}; a target equals the linear
                  combination of model fluxes. Identity mapping when a target is absent.
    """

    name: str
    conditions: pd.DataFrame
    measurements: pd.DataFrame
    media: dict[str, dict[str, float]]
    features: pd.DataFrame
    source: str
    citation: str
    license: str
    perturbations: dict[str, Perturbation] = field(default_factory=dict)
    target_map: dict[str, dict[str, float]] = field(default_factory=dict)
    synthetic: bool = False

    def validate(self) -> None:
        missing = set(CONDITION_COLUMNS) - set(self.conditions.columns)
        if missing:
            raise ValueError(f"{self.name}: conditions missing columns {sorted(missing)}")
        missing = set(MEASUREMENT_COLUMNS) - set(self.measurements.columns)
        if missing:
            raise ValueError(f"{self.name}: measurements missing columns {sorted(missing)}")
        if self.conditions["condition_id"].duplicated().any():
            raise ValueError(f"{self.name}: duplicate condition_id")
        cond_ids = set(self.conditions["condition_id"])
        unknown = set(self.measurements["condition_id"]) - cond_ids
        if unknown:
            raise ValueError(f"{self.name}: measurements for unknown conditions {sorted(unknown)}")
        bad_types = set(self.measurements["measurement_type"]) - set(MEASUREMENT_TYPES)
        if bad_types:
            raise ValueError(f"{self.name}: unknown measurement types {sorted(bad_types)}")
        if self.measurements.duplicated(["condition_id", "target_id"]).any():
            raise ValueError(f"{self.name}: duplicate (condition_id, target_id) measurements")
        units = self.measurements.groupby("target_id")["units"].nunique()
        if (units > 1).any():
            raise ValueError(f"{self.name}: mixed units for {list(units[units > 1].index)}")
        if not set(self.features.index) <= cond_ids:
            raise ValueError(f"{self.name}: features indexed by unknown conditions")
        if not set(self.media) <= cond_ids:
            raise ValueError(f"{self.name}: media for unknown conditions")

    def content_hash(self) -> str:
        """Stable hash of the data content (used for provenance and cache keys)."""
        h = hashlib.sha256()
        for df in (self.conditions, self.measurements, self.features):
            h.update(
                pd.util.hash_pandas_object(df.sort_index(axis=1), index=True).to_numpy().tobytes()
            )
        extra = {
            "media": self.media,
            "perturbations": {k: vars(v) for k, v in sorted(self.perturbations.items())},
            "target_map": self.target_map,
        }
        h.update(json.dumps(extra, sort_keys=True).encode())
        return h.hexdigest()[:16]


@dataclass
class TaskData:
    """The slice of a dataset one predictor sees (train or test side of a split)."""

    dataset: str
    model_id: str
    condition_ids: list[str]
    targets: pd.DataFrame  # index condition_id, columns target_id; NaN = not measured
    target_types: dict[str, str]  # target_id -> measurement type
    units: dict[str, str]  # target_id -> units
    features: pd.DataFrame  # index condition_id
    media: dict[str, dict[str, float]]
    perturbations: dict[str, Perturbation]
    target_map: dict[str, dict[str, float]]

    @property
    def target_ids(self) -> list[str]:
        return list(self.targets.columns)

    def perturbation(self, condition_id: str) -> Perturbation:
        return self.perturbations.get(condition_id, Perturbation())


@dataclass
class Predictions:
    values: pd.DataFrame  # same index/columns as TaskData.targets
    units: dict[str, str]
    uncertainty: pd.DataFrame | None = None
    fluxes: pd.DataFrame | None = None  # index condition_id, columns reaction ids
    solver_status: dict[str, str] = field(default_factory=dict)
    wall_time_s: float = 0.0
    n_lp_calls: int = 0


class Predictor(ABC):
    name: ClassVar[str] = "unnamed"
    # True for mechanistic predictors whose predictions come from a GEM rather than training.
    mechanistic: ClassVar[bool] = False

    def __init__(self, seed: int = 0, **params: Any) -> None:
        self.seed = seed
        self.params = params

    def fit(self, train: TaskData) -> None:  # noqa: B027 - intentional no-op default
        """Train on the training split. No-op for mechanistic solvers."""

    @abstractmethod
    def predict(self, test: TaskData) -> Predictions: ...


class Dataset(ABC):
    name: ClassVar[str] = "unnamed"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def load(self) -> StandardizedData: ...


class Task:
    """Concrete base: subclasses usually only set the class attributes below."""

    name: ClassVar[str] = "unnamed"
    target_types: ClassVar[tuple[str, ...]] = ()
    valid_metrics: ClassVar[tuple[str, ...]] = ()
    valid_splitters: ClassVar[tuple[str, ...]] = ()

    def select_targets(self, data: StandardizedData) -> list[str]:
        m = data.measurements
        return sorted(m.loc[m["measurement_type"].isin(self.target_types), "target_id"].unique())

    def build(
        self, data: StandardizedData, condition_ids: Sequence[str], model_id: str | None = None
    ) -> TaskData:
        ids = list(condition_ids)
        targets = self.select_targets(data)
        m = data.measurements[data.measurements["target_id"].isin(targets)]
        wide = m.pivot(index="condition_id", columns="target_id", values="value")
        wide = wide.reindex(index=ids, columns=targets)
        first = m.drop_duplicates("target_id").set_index("target_id")
        conds = data.conditions.set_index("condition_id")
        if model_id is None:
            model_ids = conds.loc[ids, "model_id"].unique()
            if len(model_ids) != 1:
                raise ValueError(f"conditions span several models: {list(model_ids)}")
            model_id = str(model_ids[0])
        return TaskData(
            dataset=data.name,
            model_id=model_id,
            condition_ids=ids,
            targets=wide,
            target_types={str(t): str(first.at[t, "measurement_type"]) for t in targets},
            units={str(t): str(first.at[t, "units"]) for t in targets},
            features=data.features.reindex(ids),
            media={c: data.media.get(c, {}) for c in ids},
            perturbations={c: data.perturbations[c] for c in ids if c in data.perturbations},
            target_map=data.target_map,
        )


class Splitter(ABC):
    """Splits *groups* of conditions, never individual rows (rule 2: no leakage)."""

    name: ClassVar[str] = "unnamed"

    def __init__(self, group_by: str = "condition_id", **params: Any) -> None:
        self.group_by = group_by
        self.params = params

    def groups(self, data: StandardizedData) -> pd.Series:
        conds = data.conditions
        if self.group_by not in conds.columns:
            raise ValueError(f"cannot group by '{self.group_by}': not a condition column")
        return pd.Series(conds[self.group_by].to_numpy(), index=conds["condition_id"])

    @abstractmethod
    def split_groups(self, groups: pd.Series, seed: int) -> Iterator[tuple[list[str], list[str]]]:
        """Yield (train condition ids, test condition ids) given condition_id -> group."""

    def split(self, data: StandardizedData, seed: int) -> list[tuple[list[str], list[str]]]:
        return list(self.split_groups(self.groups(data), seed))


class Metric(ABC):
    name: ClassVar[str] = "unnamed"
    kind: ClassVar[Literal["continuous", "classification", "validity", "efficiency"]]

    @abstractmethod
    def evaluate(self, test: TaskData, pred: Predictions) -> dict[str, float]:
        """Return {target_id | pooled_key(type) | ALL: value}."""


ALL = "__all__"


class PointMetric(Metric):
    """A metric on (y_true, y_pred) vectors, reported per target and pooled over targets."""

    min_points: ClassVar[int] = 1

    @abstractmethod
    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float: ...

    def _safe(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ok = ~(np.isnan(y_true) | np.isnan(y_pred))
        if ok.sum() < self.min_points:
            return float("nan")
        return float(self.score(y_true[ok], y_pred[ok]))

    def score_rows(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Score each row of (B, n) arrays (e.g. bootstrap resamples) -> (B,).

        Default: a loop over `_safe`. Subclasses override with a vectorised version that must
        give the same values on NaN-free rows; rows containing NaN always use the loop.
        """
        return np.array([self._safe(t, p) for t, p in zip(y_true, y_pred, strict=True)])

    def batch(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        nan_rows = np.isnan(y_true).any(axis=1) | np.isnan(y_pred).any(axis=1)
        if not nan_rows.any() and y_true.shape[1] >= self.min_points:
            return self.score_rows(y_true, y_pred)
        out = np.array([self._safe(t, p) for t, p in zip(y_true, y_pred, strict=True)])
        if (~nan_rows).any() and y_true.shape[1] >= self.min_points:
            out[~nan_rows] = self.score_rows(y_true[~nan_rows], y_pred[~nan_rows])
        return out

    def evaluate(self, test: TaskData, pred: Predictions) -> dict[str, float]:
        truth = test.targets
        values = pred.values.reindex(index=truth.index, columns=truth.columns)
        out = {
            t: self._safe(truth[t].to_numpy(float), values[t].to_numpy(float))
            for t in truth.columns
        }
        # Pool only within a measurement type so units are never mixed (growth vs flux).
        by_type: dict[str, list[str]] = {}
        for t in truth.columns:
            by_type.setdefault(test.target_types[t], []).append(t)
        for mtype, cols in by_type.items():
            if len(cols) > 1:
                out[pooled_key(mtype)] = self._safe(
                    truth[cols].to_numpy(float).ravel(), values[cols].to_numpy(float).ravel()
                )
        return out


def pooled_key(measurement_type: str) -> str:
    return f"__pooled_{measurement_type}__"
