# Findings log

Each entry is traceable to a config in `configs/` and a report in `results/<experiment>/report/`.
Negative results carry the same weight as positive ones.

## 2026-09-30 — M2, first real dataset: haverkorn2011 (E. coli, 190 conditions)

Setup: FBA, pFBA and FVA-midpoint on iML1515, with each condition's **measured** uptake as the
medium bound. The mean, ridge and GBM baselines get the same uptake plus the carbon source as
features. Targets are growth, measured acetate secretion, and 22 13C-estimated central fluxes.
Nothing was tuned: FBA, ridge (α = 1) and GBM all use default settings. Seeds 0–2.

### A. Held-out strains (`ss_flux_haverkorn_strain_out`, 5-fold grouped by strain)

| target | pFBA RMSE | ridge RMSE | GBM RMSE | mean RMSE |
|---|---|---|---|---|
| growth (1/h) | 0.090 [0.082, 0.099] | 0.031 [0.026, 0.036] | 0.028 [0.024, 0.033] | 0.176 |
| fluxes pooled (mmol/gDW/h) | 1.84 [1.70, 2.01] | 0.36 [0.32, 0.41] | 0.36 [0.32, 0.41] | 1.76 |
| acetate secretion | 3.21 [2.96, 3.44] | 0.56 [0.46, 0.67] | 0.53 [0.40, 0.66] | 2.23 |

Ridge and GBM beat pFBA on 24 of 25 targets (paired bootstrap, Holm-corrected p < 0.05).
**This needs heavy qualification:**

1. **Within a carbon source the ML advantage is small.** The median within-carbon R² over fluxes is
   0.09 (ridge) / 0.19 (GBM) on glucose and 0.44 / 0.45 on galactose. ML explains fluxes that
   scale with uptake (glycolysis, PDH: R² 0.77–0.90). It explains ~none of the mutant-specific
   variation in TCA cycle and PPP fluxes (R² ≈ 0), because it gets no information about which
   gene was deleted.
2. **The targets are partly built from the input.** The "measured" fluxes are FiatFlux estimates
   fitted to the same uptake rate the predictors receive, and the paper reports near-invariant
   flux partitioning across mutants. So "flux ≈ uptake × a fixed ratio" is close to how the
   targets were made. This isn't leakage across splits, but it favours ML.
3. **pFBA's error is mostly systematic bias.** It predicts no acetate overflow (0 vs ~5 mmol/gDW/h
   measured on glucose) and a TCA cycle ~2.5× too high. Its ranking of growth is good (Spearman
   0.88). A fairer next comparator is FBA with an overflow or proteome constraint (ecFBA, M1
   follow-up). A linear calibration of FBA outputs would be a hybrid baseline.
4. Only 3 of the 94 deletions (pgi, zwf, sdhC) exist in iML1515, so FBA can separate the other
   91 strains only through their uptake rates.

### B. Held-out carbon source (`ss_flux_haverkorn_carbon_out`, train glc → test gal and vice versa)

| held-out | target | FBA/pFBA RMSE | ridge | GBM | mean |
|---|---|---|---|---|---|
| galactose | growth | **0.039** | 0.127 | 0.286 | 0.339 |
| glucose | growth | 0.122 | **0.073** | 0.291 | 0.339 |
| galactose | fluxes | 1.04 | **0.97** | 2.74 | 3.35 |
| glucose | fluxes | 2.39 | **1.61** | 2.77 | 3.35 |

- Across both directions **FBA predicts growth best** (RMSE 0.090 vs ridge 0.104, GBM 0.288).
  Training on galactose and predicting glucose is the one case where ridge wins on growth.
- **GBM collapses** when extrapolating to an unseen carbon source: trees cannot extrapolate
  uptake beyond the training range. Ridge extrapolates through a roughly linear uptake → flux
  relation.
- This is a weak transfer test: there are only two carbon sources, so each fold trains on one.
  More carbon sources (e.g. a Gerosa-style dataset, if its terms can be verified) are needed
  before drawing conclusions.

### What this does and doesn't show

It is not yet evidence that ML "beats FBA" in any general sense. Given the measured uptake rate
and data from the same lab and protocol, a linear model reproduces 13C-estimated central fluxes
better than untuned pFBA. The dominant reasons are pFBA's known overflow/TCA bias and the way
the flux targets were estimated. When extrapolating to a new carbon source, FBA's growth
predictions hold up better than the ML models'.

## 2026-10-01 — Enzyme-constrained FBA added (same two experiments)

`ecfba` / `ecpfba` use a total enzyme budget with ECMpy's published eciML1515 parameters
(see docs/model_notes.md). Nothing was fitted to this dataset. RMSE [95% CI]:

| experiment | target | pFBA | **ecpFBA** | ridge | GBM |
|---|---|---|---|---|---|
| held-out strains | growth | 0.090 [0.08, 0.10] | 0.046 [0.04, 0.05] | **0.031** | **0.028** |
| held-out strains | fluxes pooled | 1.84 [1.70, 2.01] | 1.35 [1.30, 1.41] | **0.36** | **0.36** |
| held-out strains | acetate | 3.21 [2.96, 3.44] | 1.81 [1.61, 1.99] | **0.56** | **0.53** |
| held-out carbon source | growth | 0.090 [0.08, 0.10] | **0.046 [0.04, 0.05]** | 0.104 [0.10, 0.11] | 0.288 |
| held-out carbon source | fluxes pooled | 1.84 [1.70, 2.01] | **1.35 [1.30, 1.41]** | **1.33 [1.28, 1.38]** | 2.75 |
| held-out carbon source | acetate | 3.21 [2.96, 3.44] | **1.81 [1.61, 1.99]** | 2.35 [2.28, 2.44] | 3.98 |

- **The enzyme budget roughly halves FBA's error.** ecpFBA beats pFBA on 11 of 25 targets and
  is worse on 2 (Holm-corrected paired bootstrap). The gain comes from glucose, where the budget
  binds: overflow acetate appears, and TCA flux drops toward the measured values (wild type on
  glucose: growth 0.61 vs 0.60 measured, acetate 3.9 vs 5.0). On galactose and in slow mutants
  the budget never binds, so ecpFBA equals pFBA.
- **When extrapolating to an unseen carbon source, ecpFBA is the best method overall:** best
  on growth and acetate, and statistically tied with ridge on pooled fluxes. GBM fails to
  extrapolate.
- **Within the training distribution (held-out strains), ML still wins**, with the caveats of
  the 2026-09-30 entry (targets estimated from the uptake input; ML explains little
  mutant-specific variation).
- Known ecpFBA errors: oxidative PPP too high on glucose (zwf 4.5 vs 2.4), and glyoxylate-shunt
  flux predicted where none was measured (1.4 vs 0).
- The numbers so far are consistent with a hybrid hypothesis: mechanistic structure generalises
  to new conditions, while ML corrects systematic within-distribution bias. That is the M3
  neural-mechanistic hybrid, which should be scored on both splits.
