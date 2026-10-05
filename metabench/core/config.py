"""Experiment configuration: pydantic schema + YAML loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from metabench.core import registry as reg


class PluginSpec(BaseModel):
    """A plugin reference: registered name, constructor params, and an optional display label."""

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    label: str | None = None

    @property
    def id(self) -> str:
        return self.label or self.name


def _as_spec(v: Any) -> Any:
    return {"name": v} if isinstance(v, str) else v


class SplitSpec(BaseModel):
    """Splitter name under `type`; every other key is passed to the splitter constructor."""

    model_config = ConfigDict(extra="allow")

    type: str

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: str
    task: str
    datasets: list[PluginSpec]
    model: str | None = None  # overrides each dataset's model_id when set
    predictors: list[PluginSpec]
    split: SplitSpec
    metrics: list[str]
    seeds: list[int] = Field(default_factory=lambda: [0])
    output_dir: str = "results"

    @field_validator("datasets", "predictors", mode="before")
    @classmethod
    def _coerce_specs(cls, v: Any) -> Any:
        return [_as_spec(x) for x in v]

    @model_validator(mode="after")
    def _check_names(self) -> ExperimentConfig:
        reg.load_plugins()
        task_cls = reg.TASKS.get(self.task)
        for d in self.datasets:
            reg.DATASETS.get(d.name)
        ids = [p.id for p in self.predictors]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate predictor ids {ids}; set `label` to disambiguate")
        for p in self.predictors:
            reg.PREDICTORS.get(p.name)
        reg.SPLITTERS.get(self.split.type)
        for m in self.metrics:
            reg.METRICS.get(m)
        if self.model is not None:
            reg.MODELS.get(self.model)
        bad = [m for m in self.metrics if m not in task_cls.valid_metrics]
        if bad:
            raise ValueError(f"metrics {bad} are not valid for task '{self.task}'")
        if self.split.type not in task_cls.valid_splitters:
            raise ValueError(f"splitter '{self.split.type}' is not valid for task '{self.task}'")
        return self

    def config_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)
