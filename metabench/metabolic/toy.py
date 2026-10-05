"""A 5-reaction toy network whose FBA solution is known analytically.

    EX_A:  A_e <=>               (uptake bounded by the medium: v_EX_A >= -u)
    R1:    A_e -> B_c
    R2:    B_c -> 2 C_c          high-yield branch, capacity-limited: v_R2 <= R2_CAP
    R3:    B_c -> C_c            low-yield branch, unbounded
    BIO:   C_c ->                objective

For uptake u >= 0 the optimum is unique:
    v_R2 = min(u, R2_CAP),  v_R3 = max(u - R2_CAP, 0),  growth = 2 v_R2 + v_R3.
Because the optimum is unique, pFBA returns the same fluxes and FVA ranges collapse to points.
"""

from __future__ import annotations

import cobra

R2_CAP = 4.0


def build_toy_model() -> cobra.Model:
    model = cobra.Model("toy")
    a_e = cobra.Metabolite("A_e", compartment="e")
    b_c = cobra.Metabolite("B_c", compartment="c")
    c_c = cobra.Metabolite("C_c", compartment="c")

    def rxn(
        rid: str, stoich: dict[cobra.Metabolite, float], lb: float, ub: float
    ) -> cobra.Reaction:
        r = cobra.Reaction(rid, lower_bound=lb, upper_bound=ub)
        r.add_metabolites(stoich)
        return r

    model.add_reactions(
        [
            rxn("EX_A", {a_e: -1}, -10.0, 1000.0),
            rxn("R1", {a_e: -1, b_c: 1}, 0.0, 1000.0),
            rxn("R2", {b_c: -1, c_c: 2}, 0.0, R2_CAP),
            rxn("R3", {b_c: -1, c_c: 1}, 0.0, 1000.0),
            rxn("BIO", {c_c: -1}, 0.0, 1000.0),
        ]
    )
    model.objective = "BIO"
    return model


def analytic_solution(uptake: float) -> dict[str, float]:
    v_r2 = min(uptake, R2_CAP)
    v_r3 = max(uptake - R2_CAP, 0.0)
    return {
        "EX_A": -uptake,
        "R1": uptake,
        "R2": v_r2,
        "R3": v_r3,
        "BIO": 2 * v_r2 + v_r3,
    }
