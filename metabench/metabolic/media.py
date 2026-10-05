"""Explicit media definitions (exchange id -> max uptake, mmol/gDW/h).

Written out rather than read from the model at run time, so a changed model file cannot silently
change the medium. tests/test_iml1515.py checks these against the downloaded model's default
medium. Provenance: see docs/model_notes.md.
"""

from __future__ import annotations

# iML1515 (BiGG, sha256 in data/checksums.json): its default medium minus glucose and O2.
# These are unbounded (1000) inorganic salts/trace elements of an M9-like minimal medium.
IML1515_SALTS: dict[str, float] = {
    ex: 1000.0
    for ex in (
        "EX_pi_e",
        "EX_co2_e",
        "EX_fe3_e",
        "EX_h_e",
        "EX_mn2_e",
        "EX_fe2_e",
        "EX_zn2_e",
        "EX_mg2_e",
        "EX_ca2_e",
        "EX_ni2_e",
        "EX_cu2_e",
        "EX_sel_e",
        "EX_cobalt2_e",
        "EX_h2o_e",
        "EX_mobd_e",
        "EX_so4_e",
        "EX_nh4_e",
        "EX_k_e",
        "EX_na1_e",
        "EX_cl_e",
        "EX_tungs_e",
        "EX_slnt_e",
    )
}

SALTS: dict[str, dict[str, float]] = {"iML1515": IML1515_SALTS}
OXYGEN: dict[str, str] = {"iML1515": "EX_o2_e"}


def minimal_medium(
    model_id: str,
    carbon: dict[str, float],
    aerobic: bool = True,
    o2_uptake: float = 1000.0,
) -> dict[str, float]:
    """Salts + the given carbon uptake bounds (+ O2 if aerobic).

    The model default is glucose 10 mmol/gDW/h with unconstrained O2; that default is a modeling
    convention, not a measured uptake rate. Datasets with measured uptakes should pass them here.
    """
    if model_id not in SALTS:
        raise KeyError(f"no minimal medium defined for model '{model_id}'")
    medium = dict(SALTS[model_id])
    medium.update(carbon)
    if aerobic:
        medium[OXYGEN[model_id]] = o2_uptake
    return medium


def glucose_m9_aerobic(model_id: str = "iML1515", glucose_uptake: float = 10.0) -> dict[str, float]:
    return minimal_medium(model_id, {"EX_glc__D_e": glucose_uptake}, aerobic=True)
