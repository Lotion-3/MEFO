"""Genome-scale model checks.

e_coli_core ships with COBRApy (fast). iML1515 needs `scripts/download_data.py --model iML1515`
(slow). The iML1515 expectations are model-behaviour checks computed on the BiGG file recorded in
data/checksums.json, not literature values; validation against measured growth rates and
essentiality needs verified datasets (milestones M2/M4).
"""

from __future__ import annotations

import pytest

from metabench.core.interfaces import Perturbation
from metabench.metabolic import media
from metabench.metabolic.models import (
    ModelNotAvailable,
    biomass_reaction_id,
    condition_context,
    load_model,
)


def test_e_coli_core_matches_cobrapy_reference() -> None:
    # 0.8739 is the growth rate COBRApy's own tests and docs report for its textbook model.
    model = load_model("e_coli_core")
    assert model.slim_optimize() == pytest.approx(0.8739215, abs=1e-6)


def test_resolve_genes_by_id_or_name() -> None:
    from metabench.metabolic.models import resolve_genes

    model = load_model("e_coli_core")
    present, absent = resolve_genes(model, ["b4025", "pgi", "notAGene"])
    assert present == ["b4025", "b4025"] and absent == ["notAGene"]


def test_absent_gene_knockout_has_no_effect() -> None:
    model = load_model("e_coli_core")
    wt = model.slim_optimize()
    with condition_context(model, {}, Perturbation(gene_knockouts=["arcA_not_in_core"])) as m:
        assert m.slim_optimize() == pytest.approx(wt)


@pytest.fixture(scope="module")
def iml1515():
    try:
        return load_model("iML1515")
    except ModelNotAvailable as exc:
        pytest.skip(str(exc))


@pytest.mark.slow
def test_iml1515_identity(iml1515) -> None:
    # Counts as reported by the BiGG API for iML1515 (checked 2026-09-30).
    assert (len(iml1515.reactions), len(iml1515.metabolites), len(iml1515.genes)) == (
        2712,
        1877,
        1516,
    )
    assert biomass_reaction_id(iml1515) == "BIOMASS_Ec_iML1515_core_75p37M"


@pytest.mark.slow
def test_explicit_medium_equals_model_default(iml1515) -> None:
    assert media.glucose_m9_aerobic("iML1515") == iml1515.medium


@pytest.mark.slow
def test_iml1515_growth_behaviour(iml1515) -> None:
    def growth(medium: dict[str, float], pert: Perturbation | None = None) -> float:
        with condition_context(iml1515, medium, pert or Perturbation()) as m:
            return m.slim_optimize(error_value=float("nan"))

    glc = media.glucose_m9_aerobic("iML1515")
    wt = growth(glc)
    assert wt == pytest.approx(0.877, abs=1e-3)
    # anaerobic growth is lower but positive
    anaer = growth(media.minimal_medium("iML1515", {"EX_glc__D_e": 10.0}, aerobic=False))
    assert 0 < anaer < wt
    # no carbon source: infeasible (the ATP maintenance requirement cannot be met)
    no_c = growth(media.minimal_medium("iML1515", {}, aerobic=True))
    assert no_c != no_c or no_c < 1e-6
    # gltA (b0720, citrate synthase) knockout is lethal in silico on glucose minimal medium
    assert growth(glc, Perturbation(gene_knockouts=["b0720"])) < 1e-6
    # the model is restored after every condition
    assert iml1515.slim_optimize() == pytest.approx(wt)


@pytest.mark.slow
def test_iml1515_pickle_cache_matches_sbml(iml1515) -> None:
    from metabench.metabolic.models import data_dir

    caches = list((data_dir() / "cache").glob("iML1515-*.pkl"))
    assert len(caches) == 1  # written by the first load, keyed by the SBML hash
    cached = load_model.__wrapped__("iML1515")  # bypass the in-process cache -> reads the pickle
    assert cached is not iml1515
    assert [r.id for r in cached.reactions] == [r.id for r in iml1515.reactions]
    assert [r.bounds for r in cached.reactions] == [r.bounds for r in iml1515.reactions]
    assert cached.slim_optimize() == pytest.approx(iml1515.slim_optimize())
