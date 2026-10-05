"""Haverkorn van Rijsewijk et al. (2011) Mol Syst Biol 7:477 - 13C fluxes of 91 E. coli
transcriptional-regulator knockouts, wild type and 3 enzyme knockouts, on glucose and galactose.

Source: Supplementary Tables 2 (glucose) and 3 (galactose), deep-well plate section: averages of
2-3 independent measurements. See docs/dataset_notes.md for provenance, licence and quirks.

Per condition (strain x carbon source):
  inputs  - carbon source and the *measured* specific uptake rate (medium bound for FBA,
            feature for ML). Uptake is never a target.
  targets - growth rate (1/h), measured acetate secretion (mmol/gDW/h), and 22 of the 23
            13C-estimated absolute fluxes (mmol/gDW/h). The hexose->G6P flux is dropped as a
            target because it is the uptake rate reconciled by FiatFlux.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from metabench.core.interfaces import Dataset, Perturbation, StandardizedData
from metabench.core.registry import register_dataset
from metabench.metabolic.media import minimal_medium
from metabench.metabolic.models import data_dir

FLUX_UNITS = "mmol/gDW/h"  # paper: mmol gCDW^-1 h^-1

FILES = {"glc": "msb20119-s2.xls", "gal": "msb20119-s3.xls"}
UPTAKE_EXCHANGE = {"glc": "EX_glc__D_e", "gal": "EX_gal_e"}

# Paper reaction label -> (target id, iML1515 reactions with coefficients).
# Lumped paper reactions map to the sum of the model reactions they combine; see dataset_notes.
FLUX_MAP: dict[str, tuple[str, dict[str, float]]] = {
    "G6P>6PDG": ("zwf", {"G6PDH2r": 1.0}),
    "6PDG>R5P": ("gnd", {"GND": 1.0}),
    "G6P>F6P": ("pgi", {"PGI": 1.0}),
    "6PDG>PYR+G3P": ("edd_eda", {"EDD": 1.0}),
    "F6P>2 T3P": ("pfk_fba", {"PFK": 1.0, "FBP": -1.0}),  # net glycolytic direction
    "2 P5P > S7P + T3P": ("tktA", {"TKT1": 1.0}),
    "P5P + E4P + F6P + T3P": ("tktB", {"TKT2": 1.0}),
    "S7P + T3P = E4P + F6P": ("tal", {"TALA": 1.0}),
    "T3P > PGA": ("gap_pgk", {"GAPD": 1.0}),
    "PGA > PEP": ("eno", {"ENO": 1.0}),
    # The paper's PEP balance only closes if PEP>PYR includes PTS glucose uptake (PEP -> PYR).
    "PEP > PYR": ("pyk_pts", {"PYK": 1.0, "GLCptspp": 1.0}),
    "PYR > AcCoA": ("pdh", {"PDH": 1.0, "PFL": 1.0}),
    "OAA + AcCoA > ICT": ("gltA", {"CS": 1.0}),
    "ICT > OGA + CO2": ("icd", {"ICDHyr": 1.0}),
    "OGA > SUC + CO2": ("sucAB", {"AKGDH": 1.0}),
    "SUC > MAL": ("sdh_fum", {"SUCDi": 1.0}),
    "MAL > OAA": ("mdh", {"MDH": 1.0, "MOX": 1.0}),
    "MAL > PYR": ("mae", {"ME1": 1.0, "ME2": 1.0}),
    "OAA > PEP": ("pck", {"PPCK": 1.0}),
    "PEP > OAA": ("ppc", {"PPC": 1.0}),
    "AceCoA > Acetate": ("pta_ackA", {"PTAr": 1.0}),
    "AceCoA + ICT > SUC + MAL": ("aceAB", {"ICL": 1.0}),
}
HEXOSE_UPTAKE_LABELS = {"GLC>G6P", "GAL>G6P"}  # equal to uptake (an input): not a target

# Strain label in the table -> gene name of the deleted gene (Keio single-gene deletions).
SPECIAL_GENE_NAMES = {"IHF A": "ihfA", "IHF B": "ihfB"}
SECTION_CLASSES = {
    "Transcription factor mutants": "transcription_factor",
    "Sigma- anti sigma factor mutants": "sigma_factor",
    "Enzymes (Controls)": "enzyme",
}


def gene_name(label: str) -> str:
    """Table labels are protein names (e.g. 'ArcA', 'SdhC'); gene names start lower-case."""
    label = label.strip()
    return SPECIAL_GENE_NAMES.get(label, label[0].lower() + label[1:])


def _norm(v: Any) -> str:
    return " ".join(str(v).split()) if pd.notna(v) else ""


def _num(v: Any) -> float:
    return float(v)


def parse_table(path: Path) -> pd.DataFrame:
    """Parse the deep-well block of Supplementary Table 2 or 3.

    Returns one row per strain with columns: strain, strain_class, growth, uptake, acetate,
    and '<paper label>' / 'dev:<paper label>' for every absolute flux.
    """
    raw = pd.read_excel(path, sheet_name=0, header=None)
    col0 = raw[0].map(_norm)
    start = col0[col0 == "DEEP WELL PLATE"].index
    if len(start) != 1:
        raise ValueError(f"{path.name}: expected one 'DEEP WELL PLATE' section")
    hdr = start[0]
    # Header rows: block titles (hdr), gene names (hdr+1), reaction/quantity labels (hdr+2).
    titles = raw.loc[hdr].map(_norm)
    labels = raw.loc[hdr + 2].map(_norm)
    abs_start = next(c for c, t in titles.items() if t.startswith("Absolute fluxes"))
    dev_start = next(c for c, t in titles.items() if c > abs_start and t.startswith("Deviations"))
    n_flux = dev_start - abs_start
    if n_flux != 23:
        raise ValueError(f"{path.name}: expected 23 absolute flux columns, found {n_flux}")
    flux_cols = {labels[c]: c for c in range(abs_start, dev_start)}
    dev_cols = {labels[c]: c for c in range(dev_start, dev_start + n_flux)}
    if list(flux_cols) != list(dev_cols):
        raise ValueError(f"{path.name}: flux and deviation columns are not aligned")
    phys = {"growth": "Specific growth rate", "acetate": "Specific acetate secretion rate"}
    phys_cols = {k: labels[labels == v].index[0] for k, v in phys.items()}
    # Galactose table labels its uptake column 'glucose' as well; it is the hexose fed (col 2).
    uptake_col = labels[labels.str.startswith("Specific") & labels.str.contains("uptake")].index
    if len(uptake_col) != 1:
        raise ValueError(f"{path.name}: expected one uptake-rate column")

    rows = []
    strain_class = "wild_type"
    for r in range(hdr + 3, len(raw)):
        label = col0[r]
        values = pd.to_numeric(raw.iloc[r, 1:], errors="coerce")
        if values.notna().sum() == 0:
            for prefix, cls in SECTION_CLASSES.items():
                if label.startswith(prefix):
                    strain_class = cls
            continue
        if label == "WILD TYPE":
            strain_class = "wild_type"
        row: dict[str, object] = {
            "strain": "WT" if label == "WILD TYPE" else label,
            "strain_class": strain_class,
            "uptake": _num(raw.at[r, uptake_col[0]]),
            **{k: _num(raw.at[r, c]) for k, c in phys_cols.items()},
        }
        for lab, c in flux_cols.items():
            row[lab] = _num(raw.at[r, c])
            row[f"dev:{lab}"] = _num(raw.at[r, dev_cols[lab]])
        rows.append(row)
    df = pd.DataFrame(rows)
    if df["strain"].duplicated().any():
        raise ValueError(
            f"{path.name}: duplicate strains {df['strain'][df['strain'].duplicated()]}"
        )
    return df


@register_dataset("haverkorn2011")
class Haverkorn2011(Dataset):
    """Params: carbon_sources (default both), include_strain_classes (default all)."""

    def raw_dir(self) -> Path:
        return data_dir() / "raw" / "haverkorn2011"

    def load(self) -> StandardizedData:
        carbons = list(self.params.get("carbon_sources", ["glc", "gal"]))
        classes = self.params.get("include_strain_classes")
        missing = [
            f for c, f in FILES.items() if c in carbons and not (self.raw_dir() / f).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{missing} not in {self.raw_dir()}. "
                "Run: uv run python scripts/download_data.py --dataset haverkorn2011"
            )
        conds, meas, feats, media, perts = [], [], [], {}, {}
        for carbon in carbons:
            table = parse_table(self.raw_dir() / FILES[carbon])
            if classes is not None:
                table = table[table["strain_class"].isin(classes)]
            for rec in table.to_dict("records"):
                strain = str(rec["strain"])
                cid = f"{strain}_{carbon}"
                conds.append(
                    {
                        "condition_id": cid,
                        "organism": "Escherichia coli BW25113",
                        "model_id": "iML1515",
                        "strain": strain,
                        "strain_class": rec["strain_class"],
                        "carbon_source": carbon,
                    }
                )
                uptake = _num(rec["uptake"])
                feats.append(
                    {"condition_id": cid, "uptake": uptake, "is_galactose": float(carbon == "gal")}
                )
                media[cid] = minimal_medium("iML1515", {UPTAKE_EXCHANGE[carbon]: uptake})
                if strain != "WT":
                    perts[cid] = Perturbation(gene_knockouts=[gene_name(strain)])
                meas.append((cid, "growth", "growth", rec["growth"], "1/h", np.nan))
                meas.append((cid, "flux", "acetate_secretion", rec["acetate"], FLUX_UNITS, np.nan))
                for lab, (tid, _) in FLUX_MAP.items():
                    meas.append((cid, "flux", tid, rec[lab], FLUX_UNITS, rec[f"dev:{lab}"]))
                unmapped = {
                    k for k in rec if isinstance(k, str) and ">" in k and not k.startswith("dev:")
                }
                unmapped -= set(FLUX_MAP) | HEXOSE_UPTAKE_LABELS
                if unmapped:
                    raise ValueError(f"unmapped flux columns {sorted(unmapped)}")
        measurements = pd.DataFrame(
            meas, columns=["condition_id", "measurement_type", "target_id", "value", "units", "std"]
        )
        target_map = {tid: rxns for tid, rxns in FLUX_MAP.values()}
        target_map["acetate_secretion"] = {"EX_ac_e": 1.0}
        return StandardizedData(
            name="haverkorn2011",
            conditions=pd.DataFrame(conds),
            measurements=measurements,
            media=media,
            features=pd.DataFrame(feats).set_index("condition_id"),
            perturbations=perts,
            target_map=target_map,
            source=(
                "Supplementary Tables 2-3 (deep-well averages), Europe PMC PMC3094070; "
                "sha256 in data/checksums.json"
            ),
            citation=(
                "Haverkorn van Rijsewijk BRB, Nanchen A, Nallet S, Kleijn RJ, Sauer U (2011). "
                "Large-scale 13C-flux analysis reveals distinct transcriptional control of "
                "respiratory and fermentative metabolism in Escherichia coli. Mol Syst Biol 7:477. "
                "doi:10.1038/msb.2011.9, PMID 21451587"
            ),
            license="CC BY-NC-SA 3.0",
        )
