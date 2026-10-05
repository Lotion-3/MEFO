"""Enzyme-constrained FBA in the sMOMENT/ECMpy style: one total enzyme budget,

    sum_r  v_r_fwd * c_r_fwd + v_r_rev * c_r_rev  <=  budget      (g enzyme / gDW)

where c = MW / (sigma * kcat) is the enzyme cost of a unit flux (g/gDW per mmol/gDW/h).
Reactions without a kcat are left unconstrained (as in ECMpy).

Parameters come from ECMpy's published eciML1515 (Mao et al. 2022, Biomolecules 12:65;
kcats from Heckmann et al. 2018 ML predictions, calibrated by ECMpy). ECMpy stores an
irreversible model with isozyme copies (`R`, `R_reverse`, `R_num2`, `R_reverse_num1`, ...),
each with its own kcat/MW. Copies of one reaction share stoichiometry, so an LP always uses the
cheapest copy; we therefore put the minimum cost per (reaction, direction) on the original
reversible iML1515 reaction. tests/test_enzyme.py checks this against ECMpy's split model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import cobra

from metabench.metabolic.models import data_dir

_SPLIT_ID = re.compile(r"^(?P<base>.+?)(?P<rev>_reverse)?(?:_num\d+)?$")


@dataclass(frozen=True)
class EnzymeParameters:
    name: str
    model_id: str
    costs: dict[str, tuple[float, float]]  # reaction -> (forward cost, reverse cost); 0 = free
    budget: float  # g enzyme / gDW
    source: str
    # reaction -> (forward allowed, reverse allowed); empty = no direction changes
    directions: dict[str, tuple[bool, bool]] = field(default_factory=dict)


# Registered parameter sets: name -> (model id, path under data/, description).
# model/eciML1515.json reproduces the paper's calibrated growth (0.6802 1/h at glucose uptake
# 10, Mao et al. 2022); model/iML1515_irr_enz_constraint_adj.json in the same repo does not
# (0.36 1/h) and is not used.
PARAMETER_SETS = {
    "ecmpy_iML1515": (
        "iML1515",
        "raw/ecmpy/eciML1515.json",
        "ECMpy eciML1515 (github.com/tibbdc/ECMpy @467e040, model/eciML1515.json); "
        "budget = total_protein_fraction * enzyme_mass_fraction",
    ),
}


def split_id(rid: str) -> tuple[str, bool]:
    """'PTAr_reverse_num2' -> ('PTAr', True)."""
    m = _SPLIT_ID.match(rid)
    if m is None:  # pragma: no cover - the regex matches any non-empty id
        raise ValueError(rid)
    return m["base"], bool(m["rev"])


def parse_ecmpy(
    path: Path,
) -> tuple[dict[str, tuple[float, float]], float, dict[str, tuple[bool, bool]]]:
    """-> (costs, budget, directions).

    kcat/MW values are read from `enzyme_constraint.kcat_MW` (eciML1515.json format) or from
    per-reaction `kcat_MW` fields (older format). `directions` maps each base reaction to
    (forward allowed, reverse allowed) as encoded by ECMpy's irreversible copies.
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    ec = d["enzyme_constraint"]
    budget = float(ec["upperbound"])
    expected = float(ec["total_protein_fraction"]) * float(ec["enzyme_mass_fraction"])
    if abs(budget - expected) > 1e-3:
        raise ValueError(f"budget {budget} != ptot*f = {expected}")
    kcat_mw: dict[str, float] = {}
    if isinstance(ec.get("kcat_MW"), dict):
        kcat_mw = {k: float(v) for k, v in ec["kcat_MW"].items() if v}
    else:
        kcat_mw = {r["id"]: float(r["kcat_MW"]) for r in d["reactions"] if r.get("kcat_MW")}
    best: dict[str, list[float]] = {}  # base -> [max kcat_MW fwd, max kcat_MW rev]
    for rid, kmw in kcat_mw.items():
        base, rev = split_id(rid)
        slot = best.setdefault(base, [0.0, 0.0])
        slot[int(rev)] = max(slot[int(rev)], kmw)
    costs = {
        base: (1.0 / f if f > 0 else 0.0, 1.0 / b if b > 0 else 0.0)
        for base, (f, b) in best.items()
    }
    allowed: dict[str, list[bool]] = {}
    for r in d["reactions"]:
        base, rev = split_id(r["id"])
        slot2 = allowed.setdefault(base, [False, False])
        lb, ub = r.get("lower_bound", 0.0), r.get("upper_bound", 1000.0)  # COBRA defaults
        if rev:
            slot2[1] |= ub > 0
        else:
            slot2[0] |= ub > 0
            slot2[1] |= lb < 0
    directions = {b: (f, rv) for b, (f, rv) in allowed.items()}
    return costs, budget, directions


# Parameter sets defined in code (no file). The toy set gives the toy network's high-yield branch
# a high enzyme cost; with budget 2 it reproduces tests/test_enzyme.py's analytic case.
BUILTIN_SETS = {
    "toy": EnzymeParameters("toy", "toy", {"R2": (1.0, 0.0), "R3": (0.1, 0.0)}, 2.0, "synthetic"),
}

# Parameter set used when a predictor does not name one.
DEFAULT_SET_FOR_MODEL = {"iML1515": "ecmpy_iML1515", "toy": "toy"}


def default_parameter_set(model_id: str) -> str:
    try:
        return DEFAULT_SET_FOR_MODEL[model_id]
    except KeyError:
        raise KeyError(f"no enzyme parameter set registered for model '{model_id}'") from None


@cache
def load_enzyme_parameters(name: str) -> EnzymeParameters:
    if name in BUILTIN_SETS:
        return BUILTIN_SETS[name]
    if name not in PARAMETER_SETS:
        raise KeyError(f"unknown enzyme parameter set '{name}'; known: {sorted(PARAMETER_SETS)}")
    model_id, rel, source = PARAMETER_SETS[name]
    path = data_dir() / rel
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: uv run python scripts/download_data.py --enzyme {name}"
        )
    costs, budget, directions = parse_ecmpy(path)
    return EnzymeParameters(name, model_id, costs, budget, source, directions)


def add_enzyme_constraint(
    model: cobra.Model,
    params: EnzymeParameters,
    budget: float | None = None,
    apply_directions: bool = True,
) -> cobra.Model:
    """Add the total-enzyme constraint (and, by default, the parameter set's reaction
    directions for internal reactions; exchange bounds are left to the medium).
    Call inside a `with model:` block so everything is reverted."""
    if params.model_id != model.id:
        raise ValueError(f"enzyme parameters are for '{params.model_id}', model is '{model.id}'")
    if apply_directions:
        for rid, (fwd, rev) in params.directions.items():
            if rid not in model.reactions:
                continue
            rxn = model.reactions.get_by_id(rid)
            if rxn.boundary:
                continue
            if not fwd and rxn.upper_bound > 0:
                rxn.upper_bound = 0.0
            if not rev and rxn.lower_bound < 0:
                rxn.lower_bound = 0.0
    terms = {}
    for rid, (cf, cr) in params.costs.items():
        rxn = model.reactions.get_by_id(rid)
        if cf:
            terms[rxn.forward_variable] = cf
        if cr:
            terms[rxn.reverse_variable] = cr
    cons = model.problem.Constraint(
        0, lb=0, ub=params.budget if budget is None else budget, name="enzyme_budget"
    )
    model.add_cons_vars(cons)
    model.solver.update()
    cons.set_linear_coefficients(terms)
    return model
