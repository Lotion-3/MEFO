"""Enzyme-constrained FBA: an analytic toy case (fast) and equivalence with ECMpy's own split
eciML1515 (slow; needs `download_data.py --model iML1515 --enzyme ecmpy_iML1515`)."""

from __future__ import annotations

import json
import logging

import pytest

from metabench.core.interfaces import Perturbation
from metabench.metabolic.enzyme import (
    EnzymeParameters,
    add_enzyme_constraint,
    load_enzyme_parameters,
    parse_ecmpy,
    split_id,
)
from metabench.metabolic.models import condition_context, data_dir, load_model
from metabench.metabolic.toy import build_toy_model


@pytest.mark.parametrize(
    ("rid", "expected"),
    [
        ("PGI", ("PGI", False)),
        ("PGI_reverse", ("PGI", True)),
        ("PFK_num2", ("PFK", False)),
        ("PTAr_reverse_num2", ("PTAr", True)),
        ("EX_glc__D_e_reverse", ("EX_glc__D_e", True)),
    ],
)
def test_split_id(rid: str, expected: tuple[str, bool]) -> None:
    assert split_id(rid) == expected


def test_parse_takes_cheapest_isozyme_per_direction(tmp_path) -> None:
    doc = {
        "enzyme_constraint": {
            "upperbound": 0.2,
            "total_protein_fraction": 0.5,
            "enzyme_mass_fraction": 0.4,
        },
        "reactions": [
            {"id": "R_num1", "kcat_MW": 10.0},
            {"id": "R_num2", "kcat_MW": 40.0},  # cheaper isozyme wins: cost 1/40
            {"id": "R_reverse_num1", "kcat_MW": 5.0},
            {"id": "S", "kcat_MW": ""},  # no kcat: unconstrained
        ],
    }
    path = tmp_path / "ec.json"
    path.write_text(json.dumps(doc))
    costs, budget, directions = parse_ecmpy(path)
    assert budget == pytest.approx(0.2)
    assert costs == {"R": (pytest.approx(1 / 40), pytest.approx(1 / 5))}
    # no bounds given -> COBRA defaults (0, 1000): forward only unless a reverse copy exists
    assert directions == {"R": (True, True), "S": (True, False)}


def test_parse_reads_kcat_table_and_directions(tmp_path) -> None:
    doc = {
        "enzyme_constraint": {
            "upperbound": 0.2,
            "total_protein_fraction": 0.5,
            "enzyme_mass_fraction": 0.4,
            "kcat_MW": {"R_num1": 10.0, "R_reverse": 20.0},
        },
        "reactions": [
            {"id": "R_num1", "lower_bound": 0, "upper_bound": 1000},
            {"id": "R_reverse", "lower_bound": 0, "upper_bound": 1000},
            {"id": "Q", "lower_bound": 0, "upper_bound": 1000},  # irreversible here
            {"id": "P", "lower_bound": -1000, "upper_bound": 1000},  # left reversible
        ],
    }
    path = tmp_path / "ec.json"
    path.write_text(json.dumps(doc))
    costs, _, directions = parse_ecmpy(path)
    assert costs == {"R": (pytest.approx(0.1), pytest.approx(0.05))}
    assert directions == {"R": (True, True), "Q": (True, False), "P": (True, True)}


def test_enzyme_budget_forces_low_yield_branch_on_toy() -> None:
    """Toy: v2 + v3 = u, v2 <= 4, growth = 2 v2 + v3. Costs R2 = 1.0, R3 = 0.1, budget 2.
    For u = 10: v2 + 0.1 (10 - v2) <= 2 -> v2 = 10/9, growth = 10 + 10/9 (vs 14 unconstrained)."""
    model = build_toy_model()
    params = EnzymeParameters("t", "toy", {"R2": (1.0, 0.0), "R3": (0.1, 0.0)}, 2.0, "test")
    with condition_context(model, {"EX_A": 10.0}, Perturbation()) as m:
        assert m.slim_optimize() == pytest.approx(14.0)
        with m:
            add_enzyme_constraint(m, params)
            sol = m.optimize()
            assert sol.objective_value == pytest.approx(10 + 10 / 9)
            assert sol.fluxes["R2"] == pytest.approx(10 / 9)
        assert m.slim_optimize() == pytest.approx(14.0)  # constraint removed on exit


def test_parameters_must_match_model() -> None:
    params = EnzymeParameters("t", "iML1515", {}, 1.0, "test")
    with pytest.raises(ValueError, match="enzyme parameters are for"):
        add_enzyme_constraint(build_toy_model(), params)


@pytest.fixture(scope="module")
def ecmpy_ref():
    path = data_dir() / "raw/ecmpy/eciML1515.json"
    if not path.exists():
        pytest.skip("ECMpy parameters not downloaded")
    import cobra

    logging.getLogger("cobra").setLevel(logging.ERROR)
    ref = cobra.io.load_json_model(str(path))
    doc = json.loads(path.read_text())
    cons = ref.problem.Constraint(0, lb=0, ub=doc["enzyme_constraint"]["upperbound"])
    ref.add_cons_vars(cons)
    ref.solver.update()
    kcat_mw = doc["enzyme_constraint"]["kcat_MW"]
    cons.set_linear_coefficients(
        {ref.reactions.get_by_id(r).forward_variable: 1 / k for r, k in kcat_mw.items() if k}
    )
    return ref


@pytest.mark.slow
@pytest.mark.parametrize("glc", [5.0, 10.0, 20.0])
def test_collapsed_constraint_matches_ecmpy_split_model(ecmpy_ref, glc: float) -> None:
    from metabench.metabolic import media

    with ecmpy_ref:
        ecmpy_ref.reactions.get_by_id("EX_glc__D_e_reverse").upper_bound = glc
        expected = ecmpy_ref.slim_optimize()
    model = load_model("iML1515")
    with condition_context(model, media.glucose_m9_aerobic("iML1515", glc), Perturbation()):
        add_enzyme_constraint(model, load_enzyme_parameters("ecmpy_iML1515"))
        assert model.slim_optimize() == pytest.approx(expected, rel=1e-6)


@pytest.mark.slow
def test_reproduces_published_calibrated_growth() -> None:
    """Mao et al. 2022 report 0.6802 1/h for the calibrated eciML1515 at glucose uptake 10."""
    from metabench.metabolic import media

    model = load_model("iML1515")
    with condition_context(model, media.glucose_m9_aerobic("iML1515", 10.0), Perturbation()):
        add_enzyme_constraint(model, load_enzyme_parameters("ecmpy_iML1515"))
        assert model.slim_optimize() == pytest.approx(0.6802, abs=5e-4)


@pytest.mark.parametrize("name", ["ecfba", "ecpfba"])
def test_ecfba_predictor_on_toy_matches_analytic(name: str, toy_data, task) -> None:
    """Builtin toy set: costs R2 = 1.0, R3 = 0.1, budget 2 ->
    v2 = min(u, 4, (2 - 0.1 u) / 0.9), v3 = u - v2, growth = 2 v2 + v3."""
    from metabench.core import registry as reg

    test = task.build(toy_data, list(toy_data.conditions["condition_id"]))
    pred = reg.PREDICTORS.get(name)().predict(test)
    for cid, u in toy_data.features["uptake_A"].items():
        v2 = min(u, 4.0, (2 - 0.1 * u) / 0.9)
        assert pred.values.loc[cid, "growth"] == pytest.approx(2 * v2 + (u - v2), abs=1e-7)
        assert pred.values.loc[cid, "R2"] == pytest.approx(v2, abs=1e-7)
