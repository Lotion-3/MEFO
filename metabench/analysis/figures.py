"""Publication figures, generated only from stored results (summary.csv and the predictions
parquet), never from in-memory state. Plain matplotlib defaults: default style, default colour
cycle (C0, C1, ...), one PNG per figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from metabench.core.results import ExperimentPaths, collect  # noqa: E402

# Method family -> default colour-cycle slot.
FAMILY_COLOR = {"mechanistic": "C0", "ml": "C1", "baseline": "C2"}
CARBON_COLOR = {"glc": "C0", "gal": "C1"}

# Predictors shown in figures, in display order (top to bottom), with labels and families.
PREDICTORS: dict[str, tuple[str, str]] = {
    "ecpfba": ("ecpFBA (enzyme-constrained)", "mechanistic"),
    "pfba": ("pFBA", "mechanistic"),
    "ridge": ("Ridge regression", "ml"),
    "gbm": ("Gradient boosting", "ml"),
    "mean": ("Mean baseline", "baseline"),
}
SPLITS = {
    "ss_flux_haverkorn_strain_out": "Held-out strains (5-fold, grouped by strain)",
    "ss_flux_haverkorn_carbon_out": "Held-out carbon source (glucose ↔ galactose)",
}
TARGETS = {
    "growth": "Growth rate RMSE (1/h)",
    "__pooled_flux__": "Pooled central-carbon flux RMSE (mmol/gDW/h)",
    "acetate_secretion": "Acetate secretion RMSE (mmol/gDW/h)",
}
CARBON_LABEL = {"glc": "glucose", "gal": "galactose"}
SHORT = {"pfba": "pFBA", "ecpfba": "ecpFBA", "ridge": "Ridge", "gbm": "GBM", "mean": "Mean"}


def _save(fig: plt.Figure, out: Path, stem: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _cell(s: pd.DataFrame, target: str, predictor: str, col: str) -> float:
    return float(s.loc[(target, predictor), col])  # type: ignore[arg-type]


def _summary(results: Path, experiment: str) -> pd.DataFrame:
    path = results / experiment / "report" / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `metabench report {experiment}` first")
    return pd.read_csv(path)


def headline_figure(results: Path, out: Path) -> Path:
    """Small multiples: rows = target, columns = split; dot = OOF RMSE, error bar = 95% CI."""
    plt.style.use("default")
    fig, axes = plt.subplots(len(TARGETS), len(SPLITS), figsize=(11, 9), sharey=True)
    order = list(PREDICTORS)
    ypos = np.arange(len(order))[::-1]
    for c, (exp, split_label) in enumerate(SPLITS.items()):
        s = _summary(results, exp)
        s = s[s["metric"] == "rmse"].set_index(["target", "predictor"])
        for r, (target, tlabel) in enumerate(TARGETS.items()):
            ax = axes[r, c]
            for y, p in zip(ypos, order, strict=True):
                v, lo, hi = (_cell(s, target, p, k) for k in ("value", "ci_low", "ci_high"))
                ax.errorbar(
                    v,
                    y,
                    xerr=[[v - lo], [hi - v]],
                    fmt="o",
                    color=FAMILY_COLOR[PREDICTORS[p][1]],
                    capsize=3,
                )
                ax.annotate(
                    f"{v:.3g}",
                    (hi, y),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                )
            ax.set_yticks(ypos, [PREDICTORS[p][0] for p in order])
            xmax = max(_cell(s, target, p, "ci_high") for p in order)
            ax.set_xlim(0, xmax * 1.25)
            ax.grid(True, axis="x", alpha=0.3)
            ax.set_xlabel(tlabel)
            if r == 0:
                ax.set_title(split_label)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=FAMILY_COLOR[f], label=lab)
        for f, lab in [
            ("mechanistic", "Mechanistic (FBA family, untrained)"),
            ("ml", "Machine learning (trained)"),
            ("baseline", "Trivial baseline"),
        ]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3)
    fig.suptitle("Out-of-fold RMSE on held-out E. coli conditions (lower is better; 95% CIs)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, "fig1_headline")


def overflow_figure(results: Path, out: Path) -> Path:
    """Measured vs predicted acetate secretion (held-out carbon source, seed 0)."""
    plt.style.use("default")
    paths = ExperimentPaths.for_experiment(results, "ss_flux_haverkorn_carbon_out")
    p = collect(paths, "preds")
    p = p[(p["target"] == "acetate_secretion") & (p["seed"] == 0)].copy()
    p["carbon"] = p["condition_id"].str.rsplit("_", n=1).str[-1]
    shown = ["pfba", "ecpfba", "ridge"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharex=True, sharey=True)
    lim = float(np.nanmax(p[["y_true", "y_pred"]].to_numpy())) * 1.08
    for ax, pred in zip(axes, shown, strict=True):
        d = p[p["predictor"] == pred]
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
        for carbon in ["glc", "gal"]:
            dc = d[d["carbon"] == carbon]
            ax.scatter(
                dc["y_true"],
                dc["y_pred"],
                s=20,
                color=CARBON_COLOR[carbon],
                alpha=0.7,
                label=f"tested on {CARBON_LABEL[carbon]}",
            )
        rmse = float(np.sqrt(np.mean((d["y_true"] - d["y_pred"]) ** 2)))
        ax.set_title(f"{PREDICTORS[pred][0]} (RMSE {rmse:.2f})")
        ax.set_xlim(-0.2, lim)
        ax.set_ylim(-0.2, lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Measured acetate (mmol/gDW/h)")
    axes[0].set_ylabel("Predicted acetate (mmol/gDW/h)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Acetate overflow, held-out carbon source split (out-of-fold, seed 0)")
    fig.tight_layout()
    return _save(fig, out, "fig2_overflow")


FLUX_LABELS = {
    "growth": "growth",
    "acetate_secretion": "acetate secretion",
    "zwf": "G6P dehydrogenase",
    "gnd": "6PG dehydrogenase",
    "pgi": "PGI",
    "edd_eda": "ED pathway",
    "pfk_fba": "PFK/FBA",
    "tktA": "transketolase A",
    "tktB": "transketolase B",
    "tal": "transaldolase",
    "gap_pgk": "GAPDH/PGK",
    "eno": "enolase",
    "pyk_pts": "pyruvate kinase + PTS",
    "pdh": "pyruvate dehydrogenase",
    "gltA": "citrate synthase",
    "icd": "isocitrate DH",
    "sucAB": "2-oxoglutarate DH",
    "sdh_fum": "SDH/fumarase",
    "mdh": "malate DH",
    "mae": "malic enzyme",
    "pck": "PEP carboxykinase",
    "ppc": "PEP carboxylase",
    "pta_ackA": "acetate pathway (Pta/AckA)",
    "aceAB": "glyoxylate shunt",
}


def skill_heatmap(results: Path, out: Path) -> Path:
    """Skill score 1 - RMSE/RMSE(mean baseline) per target; > 0 beats the trivial baseline."""
    plt.style.use("default")
    cols = ["pfba", "ecpfba", "ridge", "gbm"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 9.5), sharey=True)
    targets = list(FLUX_LABELS)
    for ax, (exp, split_label) in zip(axes, SPLITS.items(), strict=True):
        s = _summary(results, exp)
        r = s[s["metric"] == "rmse"].pivot_table(
            index="target", columns="predictor", values="value"
        )
        skill = 1 - r[cols].div(r["mean"], axis=0)
        vals = skill.reindex(targets).to_numpy(float)
        ax.imshow(np.clip(vals, -1, 1), cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                txt = f"{v:+.2f}" if v > -10 else "< -10"
                color = "white" if abs(np.clip(v, -1, 1)) > 0.75 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
        ax.set_xticks(range(len(cols)), [SHORT[c] for c in cols])
        ax.set_yticks(range(len(targets)), [FLUX_LABELS[t] for t in targets])
        ax.set_title(split_label, fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    cax = fig.add_axes((0.93, 0.2, 0.015, 0.6))
    fig.colorbar(
        axes[0].images[0],
        cax=cax,
        label="Skill = 1 − RMSE / RMSE(mean baseline), clipped to [−1, 1]",
    )
    fig.suptitle("Per-target skill relative to the training-mean baseline")
    return _save(fig, out, "fig3_skill")


# Pipeline diagram: node -> (x, y, title, subtitle, colour or None).
PIPELINE_NODES: dict[str, tuple[float, float, str, str, str | None]] = {
    "dataset": (0, 0, "Dataset loader", "standardised schema", None),
    "task": (2.4, 0, "Task", "targets + inputs", None),
    "split": (4.8, 0, "Group splitter", "no leakage", None),
    "config": (7.2, 1.3, "configs/*.yaml", "experiment grid", None),
    "runner": (7.2, 0, "Runner", "cached, resumable", None),
    "mech": (9.6, 0.65, "Mechanistic (FBA)", "COBRApy + GLPK", FAMILY_COLOR["mechanistic"]),
    "ml": (9.6, -0.65, "Machine learning", "scikit-learn", FAMILY_COLOR["ml"]),
    "cells": (12, 0, "Per-cell parquet", "+ provenance", None),
    "report": (14.4, 0.65, "Report", "OOF metrics, CIs, tests", None),
    "figs": (14.4, -0.65, "Figures", "from stored results", None),
}
PIPELINE_EDGES = [
    ("dataset", "task"),
    ("task", "split"),
    ("split", "runner"),
    ("config", "runner"),
    ("runner", "mech"),
    ("runner", "ml"),
    ("mech", "cells"),
    ("ml", "cells"),
    ("cells", "report"),
    ("cells", "figs"),
]


def pipeline_diagram(out: Path) -> Path:
    """Data flow from config and dataset to results."""
    plt.style.use("default")
    w, h = 2.25, 0.78
    fig, ax = plt.subplots(figsize=(14, 3.4))
    ax.set_xlim(-w / 2 - 0.1, 14.4 + w / 2 + 0.1)
    ax.set_ylim(-0.65 - h / 2 - 0.1, 1.3 + h / 2 + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    for x, y, title, sub, color in PIPELINE_NODES.values():
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.1",
                facecolor=color or "white",
                edgecolor="black",
                alpha=0.25 if color else 1.0,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.1",
                fill=False,
                edgecolor="black",
            )
        )
        ax.text(x, y + 0.1, title, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.text(x, y - 0.16, sub, ha="center", va="center", fontsize=7.5)
    for a, b in PIPELINE_EDGES:
        xa, ya = PIPELINE_NODES[a][:2]
        xb, yb = PIPELINE_NODES[b][:2]
        if xa == xb:  # vertical edge (config -> runner)
            start, end = (xa, ya - h / 2), (xb, yb + h / 2)
        else:
            start, end = (xa + w / 2, ya), (xb - w / 2, yb)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "shrinkA": 0, "shrinkB": 0},
        )
    return _save(fig, out, "fig0_pipeline")


def make_all(results: Path, out: Path) -> list[Path]:
    return [
        pipeline_diagram(out),
        headline_figure(results, out),
        overflow_figure(results, out),
        skill_heatmap(results, out),
    ]
