"""Summary tables built only from the stored results tables (never from in-memory state).

Point metrics are computed on pooled out-of-fold (OOF) predictions: for each
(dataset, predictor, seed, repeat) every condition has exactly one held-out prediction. Values
are averaged over (seed, repeat) units; 95% CIs come from bootstrapping conditions (resampled
identically for every predictor, so intervals are comparable across predictors).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from metabench.core import registry as reg
from metabench.core.interfaces import PointMetric, pooled_key
from metabench.core.results import ExperimentPaths, collect


@dataclass
class Report:
    summary: pd.DataFrame  # dataset, predictor, target, metric, value, ci_low, ci_high, ...
    validity: pd.DataFrame
    efficiency: pd.DataFrame
    failures: pd.DataFrame
    synthetic: bool
    comparisons: pd.DataFrame = field(default_factory=pd.DataFrame)


class _Unit:
    """OOF predictions of one (seed, repeat) unit for one target (or a pooled group of targets),
    stored as (n_conditions x n_targets) arrays aligned to a fixed condition order, so a bootstrap
    resample of conditions is a single fancy-indexing operation."""

    def __init__(self, df: pd.DataFrame, conditions: np.ndarray) -> None:
        def wide(col: str) -> np.ndarray:
            w = df.pivot_table(index="condition_id", columns="target", values=col, dropna=False)
            return w.reindex(conditions).to_numpy(float)

        self.y_true = wide("y_true")
        self.y_pred = wide("y_pred")

    def score(self, metric: PointMetric, idx: np.ndarray | None = None) -> float:
        if idx is None:
            return metric._safe(self.y_true.ravel(), self.y_pred.ravel())
        return metric._safe(self.y_true[idx].ravel(), self.y_pred[idx].ravel())

    def boot(self, metric: PointMetric, boots: np.ndarray) -> np.ndarray:
        """Metric on every bootstrap resample at once: boots (B, n) -> (B,)."""
        b = boots.shape[0]
        return metric.batch(self.y_true[boots].reshape(b, -1), self.y_pred[boots].reshape(b, -1))


def _nanmean(vals: list[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    return float("nan") if np.isnan(arr).all() else float(np.nanmean(arr))


def _nanmean_rows(arrs: list[np.ndarray]) -> np.ndarray:
    """Mean over units (axis 0) ignoring NaN; NaN where every unit is NaN."""
    stack = np.vstack(arrs)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stack, axis=0)


def _target_frames(preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-target frames plus pooled-within-measurement-type frames."""
    out = {str(t): g for t, g in preds.groupby("target")}
    for mtype, g in preds.groupby("measurement_type"):
        if g["target"].nunique() > 1:
            out[pooled_key(str(mtype))] = g
    return out


def _point_metrics(metric_names: list[str]) -> dict[str, PointMetric]:
    reg.load_plugins()
    return {
        m: reg.METRICS.get(m)() for m in metric_names if issubclass(reg.METRICS.get(m), PointMetric)
    }


def _units(
    dpreds: pd.DataFrame, conditions: np.ndarray
) -> dict[str, dict[str, dict[tuple[int, int], _Unit]]]:
    """predictor -> target -> (seed, repeat) -> _Unit."""
    out: dict[str, dict[str, dict[tuple[int, int], _Unit]]] = {}
    for (predictor,), ppreds in dpreds.groupby(["predictor"]):
        per_target: dict[str, dict[tuple[int, int], _Unit]] = {}
        for key, u in ppreds.groupby(["seed", "repeat"]):
            unit_key = (int(key[0]), int(key[1]))  # type: ignore[call-overload]
            for target, frame in _target_frames(u).items():
                per_target.setdefault(target, {})[unit_key] = _Unit(frame, conditions)
        out[str(predictor)] = per_target
    return out


def _paired_delta(
    pairs: list[tuple[_Unit, _Unit]], metric: PointMetric, idx: np.ndarray | None
) -> float:
    return _nanmean([a.score(metric, idx) - b.score(metric, idx) for a, b in pairs])


def _paired_delta_boot(
    pairs: list[tuple[_Unit, _Unit]], metric: PointMetric, boots: np.ndarray
) -> np.ndarray:
    return _nanmean_rows([a.boot(metric, boots) - b.boot(metric, boots) for a, b in pairs])


def _bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def oof_summary(
    preds: pd.DataFrame, metric_names: list[str], n_boot: int, seed: int
) -> pd.DataFrame:
    metrics = _point_metrics(metric_names)
    rows = []
    for dataset, dpreds in preds.groupby("dataset"):
        conditions = np.array(sorted(dpreds["condition_id"].unique()))
        # Same bootstrap resamples of conditions for every predictor/target in this dataset.
        boots = _bootstrap_indices(len(conditions), n_boot, seed)
        for predictor, per_target in _units(dpreds, conditions).items():
            for target, units in per_target.items():
                for mname, metric in metrics.items():
                    point = _nanmean([u.score(metric) for u in units.values()])
                    dist = _nanmean_rows([u.boot(metric, boots) for u in units.values()])
                    finite = dist[~np.isnan(dist)]
                    lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
                    rows.append(
                        {
                            "dataset": dataset,
                            "predictor": predictor,
                            "target": target,
                            "metric": mname,
                            "value": point,
                            "ci_low": lo,
                            "ci_high": hi,
                            "n_conditions": len(conditions),
                            "n_units": len(units),
                        }
                    )
    return pd.DataFrame(rows)


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaNs are left as NaN and not counted)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def paired_comparisons(
    preds: pd.DataFrame,
    reference: str,
    metric_name: str = "rmse",
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """delta = metric(predictor) - metric(reference) on identical conditions and resamples.

    For error metrics, delta < 0 means the predictor beats the reference. The p-value is the
    two-sided bootstrap p for delta = 0; `p_holm` is Holm-corrected over all comparisons.
    Deltas average over (seed, repeat) units present for both predictors.
    """
    metric = _point_metrics([metric_name])[metric_name]
    rows = []
    for dataset, dpreds in preds.groupby("dataset"):
        if reference not in set(dpreds["predictor"]):
            continue
        conditions = np.array(sorted(dpreds["condition_id"].unique()))
        boots = _bootstrap_indices(len(conditions), n_boot, seed)
        units = _units(dpreds, conditions)
        ref = units[reference]
        for predictor, per_target in units.items():
            if predictor == reference:
                continue
            for target, pu in per_target.items():
                keys = sorted(set(pu) & set(ref.get(target, {})))
                if not keys:
                    continue
                pairs = [(pu[k], ref[target][k]) for k in keys]
                point = _paired_delta(pairs, metric, None)
                dist = _paired_delta_boot(pairs, metric, boots)
                finite = dist[~np.isnan(dist)]
                if len(finite) == 0:
                    lo = hi = p = np.nan
                else:
                    lo, hi = np.percentile(finite, [2.5, 97.5])
                    p = min(1.0, 2 * min((finite <= 0).mean(), (finite >= 0).mean()))
                rows.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "predictor": predictor,
                        "reference": reference,
                        "metric": metric_name,
                        "delta": point,
                        "ci_low": lo,
                        "ci_high": hi,
                        "p": p,
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_holm"] = holm(df["p"].to_numpy())
    return df


def build_report(
    paths: ExperimentPaths,
    metric_names: list[str],
    n_boot: int = 1000,
    seed: int = 0,
    reference: str | None = "pfba",
) -> Report:
    metrics = collect(paths, "metrics")
    preds = collect(paths, "preds")
    if metrics.empty:
        raise FileNotFoundError(f"no results under {paths.cells}")
    ok = metrics[metrics["status"] == "ok"]
    summary = oof_summary(preds, metric_names, n_boot, seed) if not preds.empty else pd.DataFrame()

    validity = (
        ok[ok["metric"].isin([m for m in metric_names if reg.METRICS.get(m).kind == "validity"])]
        .groupby(["dataset", "predictor", "metric"])["value"]
        .max()
        .reset_index()
        .rename(columns={"value": "worst_over_folds"})
    )
    efficiency = (
        ok[ok["metric"].isin(["fit_time_s", "predict_time_s", "n_lp_calls"])]
        .groupby(["dataset", "predictor", "metric"])["value"]
        .sum()
        .unstack("metric")
        .reset_index()
    )
    n_pred = (
        preds.groupby(["dataset", "predictor"]).size().rename("n_predictions")
        if not preds.empty
        else None
    )
    non_optimal = (
        preds[(preds["solver_status"] != "") & (preds["solver_status"] != "optimal")]
        .drop_duplicates(["dataset", "predictor", "seed", "repeat", "condition_id"])
        .groupby(["dataset", "predictor"])
        .size()
        .rename("non_optimal_conditions")
        if not preds.empty
        else None
    )
    errors = (
        metrics[metrics["status"] == "error"]
        .groupby(["dataset", "predictor"])
        .size()
        .rename("failed_cells")
    )
    cells = (
        metrics.drop_duplicates("cell_key").groupby(["dataset", "predictor"]).size().rename("cells")
    )
    parts = [s for s in (cells, errors, non_optimal, n_pred) if s is not None]
    failures = pd.concat(parts, axis=1).fillna(0).astype(int).reset_index()
    comparisons = pd.DataFrame()
    if reference is not None and not preds.empty and reference in set(preds["predictor"]):
        comparisons = paired_comparisons(preds, reference, "rmse", n_boot, seed)
    return Report(
        summary,
        validity,
        efficiency,
        failures,
        synthetic=bool(metrics["synthetic_data"].any()),
        comparisons=comparisons,
    )


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(none)_\n"
    cols = list(df.columns)

    def fmt(v: object) -> str:
        if isinstance(v, float):
            return "nan" if np.isnan(v) else f"{v:.4g}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def render_markdown(rep: Report, experiment: str) -> str:
    out = [f"# Report: {experiment}\n"]
    if rep.synthetic:
        out.append(
            "> **Synthetic data.** At least one dataset is synthetic; these numbers test "
            "the pipeline and are not evidence about any method's real accuracy.\n"
        )
    out.append("## Accuracy on held-out conditions (OOF, 95% bootstrap CI over conditions)\n")
    if not rep.summary.empty:
        s = rep.summary.copy()
        s["value (95% CI)"] = [
            f"{v:.4g} [{lo:.4g}, {hi:.4g}]"
            for v, lo, hi in zip(s["value"], s["ci_low"], s["ci_high"], strict=True)
        ]
        table = s.pivot_table(
            index=["dataset", "target", "predictor"],
            columns="metric",
            values="value (95% CI)",
            aggfunc="first",
        ).reset_index()
        out.append(_md(table))
    else:
        out.append(_md(rep.summary))
    if not rep.comparisons.empty:
        c = rep.comparisons.copy()
        ref = c["reference"].iloc[0]
        out.append(
            f"\n## Paired comparison vs `{ref}` (RMSE difference; < 0 = better than {ref})\n\n"
            "Same conditions and bootstrap resamples for both predictors; p is two-sided and "
            "`p_holm` is Holm-corrected over every row of this table.\n"
        )
        c["delta (95% CI)"] = [
            f"{d:+.3g} [{lo:+.3g}, {hi:+.3g}]"
            for d, lo, hi in zip(c["delta"], c["ci_low"], c["ci_high"], strict=True)
        ]
        c["verdict"] = [
            "better" if (ph < 0.05 and d < 0) else "worse" if (ph < 0.05 and d > 0) else "n.s."
            for d, ph in zip(c["delta"], c["p_holm"], strict=True)
        ]
        cols = ["dataset", "target", "predictor", "delta (95% CI)", "p_holm", "verdict"]
        out.append(_md(c[cols]))
    out += [
        "\n## Mechanistic validity (worst fold; NaN = predictor returns no flux vector)\n",
        _md(rep.validity),
        "\n## Failures and solver status (reported with the same prominence as results)\n",
        _md(rep.failures),
        "\n## Efficiency (totals over all cells)\n",
        _md(rep.efficiency),
    ]
    return "\n".join(out)


def write_report(
    paths: ExperimentPaths, metric_names: list[str], experiment: str, n_boot: int = 1000
) -> tuple[Report, Path]:
    rep = build_report(paths, metric_names, n_boot=n_boot)
    paths.report.mkdir(parents=True, exist_ok=True)
    rep.summary.to_csv(paths.report / "summary.csv", index=False)
    rep.failures.to_csv(paths.report / "failures.csv", index=False)
    rep.comparisons.to_csv(paths.report / "comparisons.csv", index=False)
    md_path = paths.report / "report.md"
    md_path.write_text(render_markdown(rep, experiment), encoding="utf-8")
    return rep, md_path
