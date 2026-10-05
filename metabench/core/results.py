"""Tidy results storage (parquet) with provenance metadata."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

_VERSIONED = ("metabench", "cobra", "optlang", "numpy", "pandas", "scikit-learn", "scipy")


@cache
def git_info() -> tuple[str | None, bool | None]:
    """(commit SHA, dirty flag), or (None, None) when not inside a git repository."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


@cache
def library_versions() -> str:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for pkg in _VERSIONED:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return json.dumps(versions, sort_keys=True)


def solver_name() -> str:
    import cobra

    interface = cobra.Configuration().solver
    return (
        str(getattr(interface, "__name__", interface)).rsplit(".", 1)[-1].replace("_interface", "")
    )


def provenance() -> dict[str, Any]:
    sha, dirty = git_info()
    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "versions": library_versions(),
        "solver": solver_name(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path

    @classmethod
    def for_experiment(cls, output_dir: str | Path, experiment: str) -> ExperimentPaths:
        return cls(Path(output_dir) / experiment)

    @property
    def cells(self) -> Path:
        return self.root / "cells"

    @property
    def report(self) -> Path:
        return self.root / "report"

    def cell(self, key: str, kind: str) -> Path:
        suffix = "json" if kind == "status" else "parquet"
        return self.cells / f"{key}.{kind}.{suffix}"


def cell_is_complete(paths: ExperimentPaths, key: str) -> bool:
    status_file = paths.cell(key, "status")
    if not status_file.exists():
        return False
    return bool(json.loads(status_file.read_text(encoding="utf-8")).get("status") == "ok")


def write_cell(
    paths: ExperimentPaths,
    key: str,
    metrics: pd.DataFrame,
    preds: pd.DataFrame | None,
    status: dict[str, Any],
) -> None:
    paths.cells.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(paths.cell(key, "metrics"), index=False)
    if preds is not None:
        preds.to_parquet(paths.cell(key, "preds"), index=False)
    else:
        paths.cell(key, "preds").unlink(missing_ok=True)  # never leave stale predictions
    # Status is written last: its presence with status=ok marks the cell complete.
    paths.cell(key, "status").write_text(json.dumps(status, indent=2), encoding="utf-8")


def collect(paths: ExperimentPaths, kind: str = "metrics") -> pd.DataFrame:
    files = sorted(paths.cells.glob(f"*.{kind}.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
