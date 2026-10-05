from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from metabench.analysis.report import build_report, holm, render_markdown
from metabench.core.config import load_config
from metabench.core.results import ExperimentPaths
from metabench.core.runner import run_experiment

CONFIG = Path(__file__).parents[1] / "configs" / "toy_e2e.yaml"


def test_holm_matches_hand_computation() -> None:
    # sorted p: 0.01, 0.02, 0.04 -> 3*0.01, max(0.03, 2*0.02), max(0.04, 1*0.04)
    adj = holm(np.array([0.04, 0.01, np.nan, 0.02]))
    assert adj[1] == pytest.approx(0.03)
    assert adj[3] == pytest.approx(0.04)
    assert adj[0] == pytest.approx(0.04)
    assert np.isnan(adj[2])


def test_holm_caps_at_one() -> None:
    assert holm(np.array([0.6, 0.7])).max() == 1.0


@pytest.fixture(scope="module")
def toy_report(tmp_path_factory):
    out = tmp_path_factory.mktemp("res")
    cfg = load_config(CONFIG)
    run_experiment(cfg, output_dir=out)
    paths = ExperimentPaths.for_experiment(out, cfg.experiment)
    return build_report(paths, cfg.metrics, n_boot=200, reference="pfba")


def test_paired_comparison_against_reference(toy_report) -> None:
    c = toy_report.comparisons.set_index(["predictor", "target"])
    assert "pfba" not in c.index.get_level_values("predictor")
    # fba and pfba both hit the analytic optimum: zero difference, not significant
    assert c.loc[("fba", "growth"), "delta"] == pytest.approx(0.0, abs=1e-9)
    # the mean baseline is clearly worse than pFBA on growth
    assert c.loc[("mean", "growth"), "delta"] > 0
    assert c.loc[("mean", "growth"), "p_holm"] < 0.05
    assert (c["p_holm"] >= c["p"] - 1e-12).all()


def test_markdown_contains_all_sections(toy_report) -> None:
    md = render_markdown(toy_report, "toy_e2e")
    for heading in (
        "Accuracy",
        "Paired comparison",
        "Mechanistic validity",
        "Failures",
        "Efficiency",
    ):
        assert heading in md
    assert "Synthetic data" in md
