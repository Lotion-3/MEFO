"""Loader tests for haverkorn2011 (needs `download_data.py --dataset haverkorn2011`: marked slow).

Expected values below were read from the supplementary tables (deep-well block) on 2026-09-30
and serve as regression checks on the parser, not as independent facts.
"""

from __future__ import annotations

import numpy as np
import pytest

from metabench.core import registry as reg
from metabench.core.interfaces import Perturbation
from metabench.datasets.haverkorn2011 import FLUX_MAP, Haverkorn2011, gene_name
from metabench.metabolic.models import load_model, resolve_genes

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def data():
    try:
        d = Haverkorn2011().load()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    d.validate()
    return d


def test_shape(data) -> None:
    assert len(data.conditions) == 190  # 95 strains x {glc, gal}
    counts = data.conditions.drop_duplicates("strain")["strain_class"].value_counts().to_dict()
    assert counts == {"transcription_factor": 81, "sigma_factor": 10, "enzyme": 3, "wild_type": 1}
    assert data.measurements["target_id"].nunique() == 2 + len(FLUX_MAP)
    assert not data.measurements["value"].isna().any()
    assert set(data.measurements["units"]) == {"1/h", "mmol/gDW/h"}


def test_regression_values(data) -> None:
    m = data.measurements.set_index(["condition_id", "target_id"])["value"]
    assert m[("WT_glc", "growth")] == pytest.approx(0.60)
    assert m[("WT_glc", "acetate_secretion")] == pytest.approx(4.95)
    assert m[("WT_glc", "zwf")] == pytest.approx(2.38)
    assert data.features.loc["WT_glc", "uptake"] == pytest.approx(8.14, abs=0.2)


def test_no_target_information_in_features(data) -> None:
    assert list(data.features.columns) == ["uptake", "is_galactose"]
    assert "uptake" not in set(data.measurements["target_id"])


def test_media_use_measured_uptake(data) -> None:
    for cid, medium in data.media.items():
        carbon = "EX_glc__D_e" if cid.endswith("_glc") else "EX_gal_e"
        assert medium[carbon] == data.features.loc[cid, "uptake"]
        assert ("EX_gal_e" if carbon == "EX_glc__D_e" else "EX_glc__D_e") not in medium


def test_target_map_reactions_exist_in_iml1515(data) -> None:
    model = load_model("iML1515")
    for tid, rxns in data.target_map.items():
        missing = [r for r in rxns if r not in model.reactions]
        assert not missing, f"{tid}: {missing}"


def test_only_enzyme_knockouts_are_in_the_model(data) -> None:
    model = load_model("iML1515")
    present = {
        s
        for s in data.conditions["strain"].unique()
        if s != "WT" and resolve_genes(model, [gene_name(s)])[0]
    }
    assert present == {"Pgi", "Zwf", "SdhC"}


def test_pgi_knockout_zeroes_pgi_flux(data) -> None:
    task = reg.TASKS.get("steady_state_flux")()
    test = task.build(data, ["Pgi_glc", "WT_glc"])
    pred = reg.PREDICTORS.get("fba")().predict(test)
    assert pred.values.loc["Pgi_glc", "pgi"] == pytest.approx(0.0, abs=1e-9)
    assert pred.values.loc["WT_glc", "pgi"] > 1.0
    assert np.isfinite(pred.values.to_numpy(float)).all()


def test_gene_name_rule() -> None:
    assert gene_name("ArcA") == "arcA"
    assert gene_name("IHF A") == "ihfA"
    assert gene_name("Hns") == "hns"
    assert Perturbation(gene_knockouts=[gene_name("SdhC")]).gene_knockouts == ["sdhC"]
