"""FBA-family predictors on the toy network, checked against the hand-derived optimum."""

from __future__ import annotations

import pytest

from metabench.core import registry as reg
from metabench.core.interfaces import Perturbation
from metabench.metabolic.models import condition_context, load_model
from metabench.metabolic.toy import R2_CAP, analytic_solution, build_toy_model


@pytest.mark.parametrize("u", [0.0, 1.5, R2_CAP, 7.0])
def test_toy_model_matches_analytic(u: float) -> None:
    model = build_toy_model()
    with condition_context(model, {"EX_A": u}, Perturbation()) as m:
        sol = m.optimize()
    expected = analytic_solution(u)
    assert sol.status == "optimal"
    for rid, v in expected.items():
        assert sol.fluxes[rid] == pytest.approx(v, abs=1e-9)


@pytest.mark.parametrize("name", ["fba", "pfba", "fva_summary"])
def test_predictors_recover_analytic_targets(name: str, toy_data, task) -> None:
    test = task.build(toy_data, list(toy_data.conditions["condition_id"]))
    pred = reg.PREDICTORS.get(name)().predict(test)
    assert (pred.values - test.targets).abs().to_numpy().max() < 1e-7
    assert set(pred.solver_status.values()) == {"optimal"}


def test_fva_ranges_collapse_at_unique_optimum(toy_data, task) -> None:
    test = task.build(toy_data, list(toy_data.conditions["condition_id"]))
    pred = reg.PREDICTORS.get("fva_summary")().predict(test)
    assert pred.uncertainty is not None
    assert pred.uncertainty.abs().to_numpy().max() < 1e-7
    assert pred.n_lp_calls == len(test.condition_ids) * (1 + 2 * len(test.target_ids))


def test_fva_fraction_widens_growth_range(toy_data, task) -> None:
    test = task.build(toy_data, ["u2"])  # growth optimum = 4
    pred = reg.PREDICTORS.get("fva_summary")(fraction_of_optimum=0.5).predict(test)
    # growth range is [2, 4] -> midpoint 3, half-range 1
    assert pred.values.loc["u2", "growth"] == pytest.approx(3.0)
    assert pred.uncertainty is not None
    assert pred.uncertainty.loc["u2", "growth"] == pytest.approx(1.0)


def test_reaction_knockout_applied_and_reverted(toy_data, task) -> None:
    toy_data.perturbations["u6"] = Perturbation(reaction_knockouts=["R2"])
    test = task.build(toy_data, ["u6", "u2"])
    pred = reg.PREDICTORS.get("fba")().predict(test)
    assert pred.values.loc["u6", "growth"] == pytest.approx(6.0)  # only R3 left: growth = u
    assert pred.values.loc["u2", "growth"] == pytest.approx(4.0)  # unaffected afterwards
    assert load_model("toy").reactions.R2.upper_bound == R2_CAP


def test_condition_context_reverts_medium() -> None:
    model = build_toy_model()
    with condition_context(model, {"EX_A": 3.0}, Perturbation()):
        assert model.reactions.EX_A.lower_bound == -3.0
    assert model.reactions.EX_A.lower_bound == -10.0


def test_unknown_medium_reaction_rejected() -> None:
    with (
        pytest.raises(KeyError),
        condition_context(build_toy_model(), {"EX_Z": 1.0}, Perturbation()),
    ):
        pass


def test_mechanistic_predictor_rejects_unit_mismatch(toy_data, task) -> None:
    test = task.build(toy_data, ["u1"])
    test.units["R2"] = "mmol/gDW/min"
    with pytest.raises(ValueError, match="convert units"):
        reg.PREDICTORS.get("fba")().predict(test)
