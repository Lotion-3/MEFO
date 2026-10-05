# Metabolic model notes

## iML1515 (*E. coli* K-12 MG1655)

- **Source:** BiGG Models, `https://bigg.ucsd.edu/static/models/iML1515.xml` (SBML).
- **Verified 2026-09-30** against the BiGG API (`/api/v2/models/iML1515`): organism
  *Escherichia coli* str. K-12 substr. MG1655, genome NC_000913.3, 2712 reactions, 1877
  metabolites, 1516 genes, reference PMID 29020004, last updated Oct 31 2019. The downloaded file
  loads in COBRApy 0.32.1 with the same counts (`tests/test_gems.py`).
- **Checksum:** SHA-256 recorded in `data/checksums.json` on first download.
- **Objective:** `BIOMASS_Ec_iML1515_core_75p37M`.
- **License:** not yet checked. TODO before redistributing anything derived from the file.

### Medium (`metabench/metabolic/media.py`)

The model's default medium, as shipped, is glucose minimal: 22 inorganic exchanges open at 1000
(`IML1515_SALTS`), `EX_glc__D_e` = 10 and `EX_o2_e` = 1000 mmol/gDW/h. `glucose_m9_aerobic()` reproduces
it exactly, and a test checks that. The glucose uptake of 10 is a modeling convention, not a measured rate.
Datasets with measured uptake rates should pass them via `minimal_medium()`.

Model behaviour on the downloaded file (GLPK, COBRApy 0.32.1). These are *model* outputs, not
validation against experiment:

| condition | FBA growth (1/h) |
|---|---|
| glucose minimal, aerobic | 0.877 |
| glucose minimal, anaerobic | 0.158 |
| no carbon source | infeasible (ATP maintenance lower bound cannot be met) |
| gltA (b0720) knockout, glucose aerobic | 0 |
| eno (b2779) knockout, glucose aerobic | 0.80 (**not** lethal in silico) |

Infeasible conditions are reported by predictors as NaN with the solver status, not as zero growth.

## Enzyme-constrained FBA (`ecfba`, `ecpfba`)

**Choice: sMOMENT/ECMpy-style total enzyme budget, not GECKO.** It was chosen with the Morgan
lab (Purdue ChE) in mind. Their work centres on 13C flux analysis of cyanobacteria and microalgae
and on engineering fast-growing cyanobacteria for aromatic amino acids. A single budget
constraint (Σ v·MW/kcat ≤ P·f) works with the sparse kcat coverage typical of cyanobacteria
(M5 transfer) and needs no proteomics. It also runs natively in our COBRApy stack; GECKO 3 is
MATLAB/RAVEN-based. Per-enzyme GECKO constraints can be added later as a separate predictor if
proteomics becomes available.

- **Implementation:** [metabench/metabolic/enzyme.py](../metabench/metabolic/enzyme.py). The
  constraint is added to the reversible iML1515, using the cheapest isozyme copy per reaction and
  direction. This is equivalent to ECMpy's irreversible split model: `tests/test_enzyme.py`
  matches its optimum to within 1e-6 at glucose uptake 5, 10 and 20.
- **Parameters:** `model/eciML1515.json` from github.com/tibbdc/ECMpy at commit `467e040`
  (SHA-256 in `data/checksums.json`).
  - **kcats:** Heckmann et al. 2018, Nat Commun 9, doi:10.1038/s41467-018-07652-6
    (ML-predicted).
  - **Calibration:** 14 reactions by an enzyme-usage rule, plus PDH and AKGDH against
    wild-type 13C data from ECMpy's ref. 34. Per the reference list in the paper XML, that is
    a *Metabolites* 2014, 4:408–420 paper on E. coli central carbon metabolism using
    intracellular free amino acids. Its authors were not checked. It was **not** calibrated on
    haverkorn2011, so there is no test-set leakage.
  - **Budget:** 0.56 g protein/gDW (measured composition) × 0.406 (enzyme mass fraction from
    PaxDB) = 0.227 g/gDW.
- **Citation:** Mao Z et al. (2022) ECMpy, a simplified workflow for constructing enzymatic
  constrained metabolic network model. Biomolecules 12:65. doi:10.3390/biom12010065 (CC BY).
- **Licence of the parameter file:** MIT + Commons Clause, so no commercial use. Fine for
  academic research; don't redistribute it in a sold product.
- **Validation:** reproduces the paper's calibrated growth of 0.6802 1/h at glucose uptake 10
  (regression test).
- **Pitfall found:** the same repo has `model/iML1515_irr_enz_constraint_adj.json`, which gives
  0.36 1/h and is *not* the paper's calibrated model. It is not used.
- **Reaction directions:** ECMpy's model makes 20 normally reversible iML1515 reactions
  irreversible (β-oxidation steps, SHK3Dr, E4PD, ...). These are applied together with the
  budget, because the calibration assumed them. Exchange bounds always come from the medium.
- Plain `ecfba` has many alternative optima: acetate secretion can jump between conditions.
  `ecpfba` (minimal total flux at optimal growth) is the variant to read.

## e_coli_core

COBRApy's bundled "textbook" model, used for fast tests (FBA growth 0.8739 on its default medium).

## toy

A 5-reaction synthetic network with an analytic optimum (`metabench/metabolic/toy.py`). Pipeline tests only.
