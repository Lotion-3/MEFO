"""Figure generation smoke test (slow: needs the haverkorn2011 experiment results + reports)."""

from __future__ import annotations

from pathlib import Path

import pytest

from metabench.analysis.figures import make_all

RESULTS = Path("results")


@pytest.mark.slow
def test_figures_render_from_stored_results(tmp_path: Path) -> None:
    needed = [
        RESULTS / e / "report" / "summary.csv"
        for e in ("ss_flux_haverkorn_strain_out", "ss_flux_haverkorn_carbon_out")
    ]
    if not all(p.exists() for p in needed):
        pytest.skip("experiment reports not built")
    paths = make_all(RESULTS, tmp_path)
    assert len(paths) == 6
    assert all(p.stat().st_size > 20_000 for p in paths)
