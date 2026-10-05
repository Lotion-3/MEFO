"""Command line: `metabench run <config>`, `metabench report <experiment>`, `metabench list`."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from metabench.core import registry as reg
from metabench.core.config import ExperimentConfig, load_config
from metabench.core.results import ExperimentPaths

app = typer.Typer(add_completion=False, help="MetaBench: ML vs FBA-family benchmark.")


@app.command()
def run(
    config: Path,
    force: bool = typer.Option(False, help="Recompute cells even if cached."),
    output_dir: Path | None = typer.Option(None, help="Override the config's output_dir."),
) -> None:
    """Run every cell of an experiment grid (cached cells are skipped)."""
    from metabench.core.runner import run_experiment

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("cobra").setLevel(logging.WARNING)
    cfg = load_config(config)
    s = run_experiment(cfg, force=force, output_dir=output_dir)
    typer.echo(
        f"{cfg.experiment}: {s.total} cells, {s.run} run, {s.skipped} cached, {s.failed} failed"
    )
    if s.failed:
        raise typer.Exit(code=1)


@app.command()
def report(
    experiment: str,
    output_dir: Path = typer.Option(Path("results"), help="Directory holding experiment results."),
    n_boot: int = typer.Option(1000, help="Bootstrap resamples for CIs."),
) -> None:
    """Build summary tables for a finished (or partial) experiment."""
    from metabench.analysis.report import write_report

    paths = ExperimentPaths.for_experiment(output_dir, experiment)
    cfg = ExperimentConfig.model_validate(
        json.loads((paths.root / "config.json").read_text(encoding="utf-8"))
    )
    _, md_path = write_report(paths, cfg.metrics, experiment, n_boot=n_boot)
    typer.echo(md_path.read_text(encoding="utf-8"))
    typer.echo(f"written to {md_path}")


@app.command()
def figures(
    output_dir: Path = typer.Option(Path("results"), help="Directory holding experiment results."),
    out: Path = typer.Option(Path("docs/figures"), help="Where to write the PNGs."),
) -> None:
    """Render README figures (light + dark) from stored results; run `report` first."""
    from metabench.analysis.figures import make_all

    for p in make_all(output_dir, out):
        typer.echo(f"wrote {p}")


@app.command("list")
def list_plugins() -> None:
    """List registered plugins."""
    reg.load_plugins()
    for r in (reg.TASKS, reg.DATASETS, reg.MODELS, reg.PREDICTORS, reg.SPLITTERS, reg.METRICS):
        typer.echo(f"{r.kind}s: {', '.join(r.names())}")


if __name__ == "__main__":
    app()
