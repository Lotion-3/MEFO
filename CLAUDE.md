# CLAUDE.md — MetaBench: an open-ended benchmark of ML vs. FBA-family methods

## 1. Project goal

Build a modular, reproducible research framework for testing **many predictors against many datasets and tasks** in constraint-based metabolic modeling. The question is open-ended:

> Where, if anywhere, do machine-learning and hybrid neural-mechanistic models outperform FBA and its extensions (pFBA, enzyme-constrained FBA, dFBA, community FBA, evolutionary strain design) when scored against **held-out experimental data**?

This is a benchmarking and exploration tool, not a single-result project. Adding a new solver, model, dataset, or task should take one small file and one config entry, and no changes to core code.

The organism focus is **E. coli** (data-rich, for method development) and **cyanobacteria** (*Synechocystis* PCC 6803, *Synechococcus elongatus* PCC 7942 / 11801) as the transfer test. Cyanobacterial flux data is small, so treat it as a transfer/evaluation set, not a training set.

## 2. Scientific ground rules (non-negotiable)

1. **"Beat FBA" means lower error against held-out experimental measurements.** A model trained on FBA output can at best match FBA. Never report agreement with FBA's own predictions as evidence of being better. Surrogate-vs-FBA comparisons are labeled **fidelity/speed** experiments, kept separate from **accuracy** experiments.
2. **No leakage.** Split by condition, strain, carbon source, or species, not by random row. Every split strategy lives in `metabench/splits/` and must be unit tested for overlap.
3. **Compare against the strongest baselines, not just vanilla FBA.** Always include pFBA and enzyme-constrained FBA (where proteomics/kcat data exists) plus trivial baselines (mean predictor, linear regression on conditions).
4. **Report uncertainty.** Datasets are small. Use repeated or nested cross-validation and bootstrap CIs. Report failures and negative results with the same prominence as wins.
5. **Never fabricate data, citations, or dataset properties.** If a dataset's provenance, units, or license is unclear, stop and write it up in `docs/dataset_notes.md` rather than guessing. Dataset names listed below come from memory and **must be verified** against the source before use.
6. **Reproducibility.** Every result is traceable to a config file, git commit, random seed, and data hash.

## 3. Architecture

Everything is a plugin behind a small interface, discovered through a registry (decorator-based, e.g. `@register_predictor("pfba")`).

```
metabench/
  core/
    registry.py          # generic registry + decorators
    interfaces.py        # Predictor, Dataset, Task, Metric, Splitter ABCs
    config.py            # pydantic config schemas, YAML loading
    runner.py            # runs (predictor x dataset x task x split x seed) grid
    results.py           # tidy results table (parquet), provenance metadata
  metabolic/             # wrappers around GEMs (COBRApy); model loading, medium setup, knockouts
  predictors/
    fba/                 # fba, pfba, fva-based, loopless, geometric FBA, ecFBA (GECKO/ECMpy-style), MOMA/ROOM, dFBA, community (MICOM)
    ml/                  # baselines: mean, ridge, RF/GBM, MLP, GNN on stoichiometric graph
    hybrid/              # neural-mechanistic: NN -> constraints/bounds -> differentiable or LP layer
    design/              # strain design: GA/OptKnock-style baseline vs ML-guided search
  datasets/              # one loader per dataset -> standardized schema
  tasks/                 # flux prediction, growth, essentiality, dynamics, community assembly, design
  splits/                # leave-condition-out, leave-strain-out, leave-species-out, grouped k-fold
  metrics/               # see section 5
  analysis/              # tables, plots, statistical comparisons
configs/                 # experiment YAMLs
tests/
docs/
notebooks/               # exploration only; nothing in the pipeline depends on notebooks
```

### Core interfaces (keep these small and stable)

- `Predictor.fit(train: TaskData) -> None` (may be a no-op for mechanistic solvers) and `Predictor.predict(test: TaskData) -> Predictions`.
- `Predictions` carries: predicted target values, optional uncertainty, optional full flux vector, wall-clock time, solver status.
- `Dataset.load() -> StandardizedData` with fixed schema: `organism, model_id, condition_id, medium/bounds, perturbation (KO/overexpression), measurements (typed: growth, flux, essentiality, abundance), units, source, citation, license`.
- `Task` defines inputs, targets, valid metrics, and valid splitters.

## 4. Tasks (implement in this order)

1. **Steady-state flux and growth prediction** across conditions. Targets: growth rate, measured central-carbon fluxes (13C-MFA).
2. **Gene essentiality / knockout growth phenotype.**
3. **Surrogate fidelity and speed** (ML approximating FBA/pFBA): accuracy vs FBA output, plus speedup. Labeled separately per rule 1.
4. **Dynamic prediction** (dFBA vs ML/hybrid on time-course data).
5. **Community assembly prediction** (pairwise to higher-order; community FBA vs ML).
6. **Strain design search** (evolutionary/OptKnock-style vs ML-guided search): solution quality vs number of solver calls.

Build tasks 1 and 2 fully, with a working end-to-end pipeline and reports, before starting 3–6. Tasks 4–6 are separate sub-projects and need their own data audits.

## 5. Metrics

- Continuous: RMSE, MAE, R², Spearman rho; per-flux and pooled; normalized error for fluxes (relative to uptake flux).
- Classification (essentiality): AUROC, AUPRC, MCC, balanced accuracy.
- Mechanistic validity for ML/hybrid models: mass-balance violation (||S·v||), thermodynamic/direction violations, bound violations. A model that is accurate but physically invalid must be flagged.
- Efficiency: wall-clock per prediction, calls to underlying LP, training cost.
- Statistics: bootstrap CIs, paired comparisons across identical splits (e.g., paired bootstrap or Wilcoxon), multiple-comparison correction when comparing many methods.

## 6. Candidate datasets (VERIFY each before use)

Record verified provenance in `docs/dataset_notes.md`. Do not assume any of the following are accessible, correctly labeled, or license-compatible until checked.

- *E. coli* 13C fluxes across carbon sources (Gerosa et al. 2015-style) and absolute proteomics across conditions (Schmidt et al. 2016-style).
- *E. coli* knockout multi-omics (Ishii et al. 2007-style) and knockout 13C fluxes (Toya et al. 2010-style).
- Keio knockout growth/essentiality; Fitness Browser (Price et al.) genome-wide fitness across bacteria, including cyanobacteria if present.
- Cyanobacterial fluxes: published 13C-MFA/INST-MFA for *Synechocystis* 6803 and *S. elongatus* (e.g., Young et al. 2011; Yu King Hing et al. 2019); Tn-seq/CRISPRi essentiality for 7942 and 6803.
- Diurnal omics for cyanobacteria (for dynamic tests).
- Community assembly data (Venturelli et al. 2018-style 12-species gut community; Friedman 2017; Clark 2021) with AGORA2 models for community-FBA baselines.
- Metabolic models: BiGG (iML1515 for *E. coli*; published *Synechocystis*/*S. elongatus* GEMs), AGORA2.

## 7. Experiment configuration

Experiments are declared in YAML and never hard-coded:

```yaml
experiment: ss_flux_ecoli_v1
task: steady_state_flux
datasets: [gerosa2015, schmidt2016]
model: iML1515
predictors: [fba, pfba, ecfba, ridge, gbm, mlp, hybrid_nn_lp]
split: {type: leave_condition_out, n_repeats: 5}
metrics: [rmse, spearman, mass_balance_violation]
seeds: [0, 1, 2, 3, 4]
```

`metabench run configs/<file>.yaml` executes the full grid, caches per-cell results, and resumes if interrupted. `metabench report <experiment>` builds tables and figures.

## 8. Engineering standards

- Python 3.11+, managed with `uv` or `pip-tools`; pin dependencies. Core stack: COBRApy, NumPy/pandas/polars, scikit-learn, PyTorch, PyTorch Geometric (optional), MICOM (community), optlang with a solver (GLPK baseline; Gurobi/CPLEX optional and never required).
- Type hints everywhere; `ruff` + `mypy` in CI; `pytest` with fast unit tests and marked slow tests.
- Required tests: split-leakage tests, registry tests, a tiny toy-network end-to-end test (a 5-reaction model where the FBA answer is known analytically), and a unit test that every predictor's output has the expected shape and units.
- Deterministic seeds; log library versions and git SHA into every results file.
- Results stored as tidy parquet (one row per predictor × dataset × split × seed × target). Plots are generated from the results table, never from ad hoc state.
- Keep large data out of git; use a `data/` directory with a download script that verifies checksums.

## 9. Milestones

1. **M0 – Skeleton:** registry, interfaces, config, runner, toy-network test passing.
2. **M1 – FBA family:** FBA, pFBA, FVA-based summary, and enzyme-constrained FBA on iML1515 with a verified medium setup. Validate against known growth rates and published essentiality results.
3. **M2 – First dataset + baselines:** one verified *E. coli* flux dataset, leave-condition-out split, mean/ridge/GBM baselines, first report.
4. **M3 – ML and hybrid models:** MLP, GNN, and a neural-mechanistic hybrid with an LP or differentiable mass-balance layer. Report accuracy plus mechanistic-validity metrics.
5. **M4 – Essentiality task and second dataset.**
6. **M5 – Cyanobacteria transfer:** load a cyanobacterial GEM, evaluate zero-shot and fine-tuned transfer on published flux data, with honest uncertainty given the tiny sample size.
7. **M6+ – Surrogate speed task, dFBA, community, strain design** as separate, individually scoped experiments.

## 10. Working agreements for Claude Code

- Before writing large amounts of code, propose the plan for the current milestone and wait for confirmation on anything ambiguous (data choices, model choices).
- Prefer small, reviewable commits per feature. Write tests with each new predictor, dataset, or split.
- When a dataset is added, also add a loader test and a `docs/dataset_notes.md` entry: source, units, condition definitions, known quirks, license.
- When results look too good, investigate leakage first (duplicate conditions, shared strains across splits, target information in features) before reporting.
- Flag any place where a comparison is unfair to a baseline (e.g., an ML model tuned with more effort than FBA's parameters) and fix or document it.
- Do not add features not listed here without asking; do keep interfaces general enough that new comparators can be plugged in.
