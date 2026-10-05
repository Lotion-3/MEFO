# Dataset notes

One section per dataset. Fill in **every** field from the primary source before the loader is
used in any experiment (CLAUDE.md rule 5). Unknown means "stop and ask", not "guess".

## Template

- **Registered name:**
- **Source (URL / DOI / supplementary table):**
- **Citation:**
- **License / terms of use:**
- **Organism / strain(s):**
- **Model used for mechanistic predictors / id mapping:**
- **Conditions (how condition_id is defined, what varies):**
- **Measurements and units (as published → as stored):**
- **Uptake/secretion rates available for medium bounds?**
- **Replicates / reported error:**
- **Known quirks (lumped fluxes, sign conventions, missing values):**
- **Grouping columns for leakage-safe splits:**
- **Download + checksum:**
- **Verified by / date:**

## haverkorn2011

- **Registered name:** `haverkorn2011` ([metabench/datasets/haverkorn2011.py](../metabench/datasets/haverkorn2011.py))
- **Source:** Supplementary Tables 1–3 (`msb20119-s1/s2/s3.xls`), downloaded from the Europe PMC REST
  supplementary archive for PMC3094070 (PMC's own site serves a bot challenge). SHA-256 values are in
  `data/checksums.json`. The files are the authors' Excel files (metadata author "Bart HvR", last
  saved 2011-01-28).
- **Citation:** Haverkorn van Rijsewijk BRB, Nanchen A, Nallet S, Kleijn RJ, Sauer U (2011).
  *Large-scale 13C-flux analysis reveals distinct transcriptional control of respiratory and
  fermentative metabolism in Escherichia coli.* Mol Syst Biol 7:477. doi:10.1038/msb.2011.9,
  PMID 21451587. Metadata verified against the article XML on 2026-09-30.
- **Licence:** CC BY-NC-SA 3.0 (from the article XML). Raw files stay out of git and are fetched
  by the download script. Anything derived from this data and redistributed must be
  non-commercial, attributed and share-alike.
- **Organism / strains:** *E. coli* BW25113 and single-gene Keio deletions. Two independent
  clones per mutant; for Crp and Cra one clone still carried the gene and was discarded (paper text).
- **Conditions:** 95 strains (wild type, 81 transcription factors, 10 sigma/anti-sigma factors,
  3 enzyme controls: Pgi, Zwf, SdhC) × {glucose, galactose} = 190 conditions. Aerobic batch
  growth in M9 in 96 deep-well plates at 37 °C. `condition_id` = `<strain>_<glc|gal>`.
- **Used block:** the "DEEP WELL PLATE" section: averages of measurements I, II (and III
  where applicable). The shake-flask wild-type rows at the top of each table are not used.
- **Measurements (as published → stored):**
  - growth rate (h⁻¹ → `1/h`, target `growth`)
  - acetate secretion (mmol/g/h → `mmol/gDW/h`, target `acetate_secretion`, mapped to `EX_ac_e`)
  - 23 absolute fluxes (mmol/g/h = mmol gCDW⁻¹ h⁻¹ → `mmol/gDW/h`); 22 are targets, with the
    replicate deviation in `std`
  - uptake rate: an **input** (the FBA medium bound and an ML feature), never a target
- **Flux → iML1515 mapping:** `FLUX_MAP` in the loader. Lumped reactions map to sums: PEP→PYR =
  PYK + GLCptspp (the paper's PEP balance only closes this way), PYR→AcCoA = PDH + PFL,
  MAL→OAA = MDH + MOX, MAL→PYR = ME1 + ME2, F6P→2 T3P = PFK − FBP. SUC→MAL uses SUCDi, and the
  glyoxylate shunt uses ICL. Alternative isozymes in iML1515 that bypass these reactions (e.g.
  PFK_3, F6PA) are *not* included, and that can cost FBA accuracy on the lumped targets.
- **Known quirks:**
  - The "absolute fluxes" are **estimates**, not direct measurements: FiatFlux fitted a 25-flux
    central-carbon model to 13C flux ratios plus measured uptake, growth and secretion. Scores
    against them measure agreement with that estimation pipeline.
  - The fitted hexose→G6P flux differs from the measured uptake by up to 1.7 mmol/g/h. It is
    dropped as a target because it duplicates an input.
  - Fitted acetate flux (pta/ackA) and measured acetate secretion differ by up to 2.1 mmol/g/h.
    Both are kept as separate targets.
  - The galactose table labels its uptake column "Specific glucose uptake rate". It is the
    galactose uptake.
  - Only 3 of the 94 deleted genes (pgi, zwf, sdhC) exist in iML1515. For the other 91 strains
    FBA differs from wild type **only through the measured uptake rate**. Knockouts of genes
    absent from a model have no effect in it (`resolve_genes`).
  - The strain label is a protein name; the gene name is derived by lower-casing the first letter
    (IHF A/B → ihfA/ihfB).
- **Grouping columns for leakage-safe splits:** `strain` (both carbon sources of a strain are held
  out together), `carbon_source`, `strain_class`.
- **Download:** `uv run python scripts/download_data.py --dataset haverkorn2011`. The server is
  slow; expect ~10 min.
- **Verified by / date:** Claude Code, 2026-09-30. Not yet reviewed by a human.

## Rejected / blocked candidates

- **Gerosa et al. 2015, Cell Systems** (doi:10.1016/j.cels.2015.09.008): subscription-only on
  Europe PMC (not open access, no licence stated), and the publisher blocks automated access.
  Blocked until someone with access can check its supplementary files and terms.

## toy (synthetic)

- Synthetic, generated from the analytic optimum of the toy network. No source, citation or
  license. FBA-family predictors score zero error on it by construction. It exists only to test
  the pipeline.

## Pending verification (none are used yet)

From CLAUDE.md §6: Gerosa 2015-style 13C fluxes, Schmidt 2016-style proteomics, Ishii 2007-style,
Toya 2010-style, Keio, Fitness Browser, cyanobacterial MFA/Tn-seq, community datasets.
