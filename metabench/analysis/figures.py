"""Publication figures, generated only from stored results (summary.csv and the predictions
parquet), never from in-memory state. Each figure is written in a light and a dark variant so
the README can serve the one matching the reader's theme.

Palette: three validated categorical slots (method family) and a blue<->red diverging ramp with a
gray midpoint, from the dataviz reference palette (all-pairs CVD dE >= 9.2 in both modes).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

from metabench.core.results import ExperimentPaths, collect  # noqa: E402


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    family: dict[str, str]  # method family -> categorical slot
    carbon: dict[str, str]
    diverging: tuple[str, str, str]  # better (blue), neutral, worse (red)


LIGHT = Theme(
    "light",
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    {"mechanistic": "#2a78d6", "ml": "#eb6834", "baseline": "#1baf7a"},
    {"glc": "#2a78d6", "gal": "#eb6834"},
    ("#2a78d6", "#f0efec", "#e34948"),
)
DARK = Theme(
    "dark",
    "#1a1a19",
    "#ffffff",
    "#c3c2b7",
    "#898781",
    "#2c2c2a",
    "#383835",
    {"mechanistic": "#3987e5", "ml": "#d95926", "baseline": "#199e70"},
    {"glc": "#3987e5", "gal": "#d95926"},
    ("#3987e5", "#383835", "#e66767"),
)

# Predictors shown in figures, in display order (top to bottom), with labels and families.
PREDICTORS: dict[str, tuple[str, str]] = {
    "ecpfba": ("ecpFBA (enzyme-constrained)", "mechanistic"),
    "pfba": ("pFBA", "mechanistic"),
    "ridge": ("Ridge regression", "ml"),
    "gbm": ("Gradient boosting", "ml"),
    "mean": ("Mean baseline", "baseline"),
}
SPLITS = {
    "ss_flux_haverkorn_strain_out": "Held-out strains\n(5-fold, grouped by strain)",
    "ss_flux_haverkorn_carbon_out": "Held-out carbon source\n(train glucose ↔ test galactose)",
}
TARGETS = {
    "growth": "Growth rate RMSE (1/h)",
    "__pooled_flux__": "Central-carbon fluxes, pooled RMSE (mmol/gDW/h)",
    "acetate_secretion": "Acetate secretion RMSE (mmol/gDW/h)",
}
CARBON_LABEL = {"glc": "glucose", "gal": "galactose"}
SHORT = {"pfba": "pFBA", "ecpfba": "ecpFBA", "ridge": "Ridge", "gbm": "GBM", "mean": "Mean"}


def _style(theme: Theme) -> None:
    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "axes.edgecolor": theme.axis,
            "axes.labelcolor": theme.ink2,
            "axes.titlecolor": theme.ink,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "xtick.labelcolor": theme.ink2,
            "ytick.labelcolor": theme.ink2,
            "axes.grid": False,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "legend.labelcolor": theme.ink2,
        }
    )


def _header(fig: plt.Figure, theme: Theme, title: str, subtitle: str) -> None:
    """Title + subtitle placed in points above the laid-out axes, so they never collide."""
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]  # Agg canvas
    boxes = [ax.get_tightbbox(renderer) for ax in fig.axes]
    top = max(b.y1 for b in boxes if b is not None)
    y = top / fig.bbox.height
    sub = offset_copy(fig.transFigure, fig=fig, y=10, units="points")
    ttl = offset_copy(fig.transFigure, fig=fig, y=30, units="points")
    fig.text(0.01, y, subtitle, transform=sub, fontsize=9, color=theme.ink2, va="bottom")
    fig.text(
        0.01,
        y,
        title,
        transform=ttl,
        fontsize=12.5,
        color=theme.ink,
        fontweight="bold",
        va="bottom",
    )


def _save(fig: plt.Figure, out: Path, stem: str, theme: Theme) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}-{theme.name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def _cell(s: pd.DataFrame, target: str, predictor: str, col: str) -> float:
    return float(s.loc[(target, predictor), col])  # type: ignore[arg-type]


def _summary(results: Path, experiment: str) -> pd.DataFrame:
    path = results / experiment / "report" / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `metabench report {experiment}` first")
    return pd.read_csv(path)


def headline_figure(results: Path, out: Path, theme: Theme) -> Path:
    """Small multiples: rows = target, columns = split; dot = OOF RMSE, whisker = 95% CI."""
    _style(theme)
    fig, axes = plt.subplots(len(TARGETS), len(SPLITS), figsize=(11, 8.6), sharey=True)
    order = list(PREDICTORS)
    ypos = np.arange(len(order))[::-1]
    for c, (exp, split_label) in enumerate(SPLITS.items()):
        s = _summary(results, exp)
        s = s[s["metric"] == "rmse"].set_index(["target", "predictor"])
        for r, (target, tlabel) in enumerate(TARGETS.items()):
            ax = axes[r, c]
            ax.grid(axis="x")
            ax.set_axisbelow(True)
            best = min(_cell(s, target, p, "value") for p in order)
            for y, p in zip(ypos, order, strict=True):
                v, lo, hi = (_cell(s, target, p, c) for c in ("value", "ci_low", "ci_high"))
                color = theme.family[PREDICTORS[p][1]]
                ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round")
                ax.scatter([v], [y], s=60, color=color, edgecolor=theme.surface, lw=2, zorder=3)
                is_best = bool(np.isclose(v, best))
                ax.annotate(
                    f"{v:.3g}",
                    (hi, y),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8.5,
                    color=theme.ink if is_best else theme.muted,
                    fontweight="bold" if is_best else "normal",
                )
            ax.set_yticks(ypos, [PREDICTORS[p][0] for p in order])
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
            ax.set_xlim(left=0)
            xmax = max(_cell(s, target, p, "ci_high") for p in order)
            ax.set_xlim(0, xmax * 1.22)
            if r == 0:
                ax.set_title(split_label, fontsize=10.5, loc="left", pad=10, fontweight="bold")
            ax.set_xlabel(tlabel, fontsize=9)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=theme.family[f], ms=8, label=lab)
        for f, lab in [
            ("mechanistic", "Mechanistic (FBA family, untrained)"),
            ("ml", "Machine learning (trained)"),
            ("baseline", "Trivial baseline"),
        ]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=2.2, w_pad=3)
    _header(
        fig,
        theme,
        "Prediction error on held-out E. coli conditions (lower is better; bold = best)",
        "Out-of-fold RMSE, mean over 3 seeds; whiskers are 95% bootstrap CIs over conditions. "
        "Data: Haverkorn van Rijsewijk et al. 2011 (190 conditions).",
    )
    return _save(fig, out, "fig1_headline", theme)


def overflow_figure(results: Path, out: Path, theme: Theme) -> Path:
    """Measured vs predicted acetate secretion (held-out carbon source, seed 0)."""
    _style(theme)
    paths = ExperimentPaths.for_experiment(results, "ss_flux_haverkorn_carbon_out")
    p = collect(paths, "preds")
    p = p[(p["target"] == "acetate_secretion") & (p["seed"] == 0)].copy()
    p["carbon"] = p["condition_id"].str.rsplit("_", n=1).str[-1]
    shown = ["pfba", "ecpfba", "ridge"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharex=True, sharey=True)
    lim = float(np.nanmax(p[["y_true", "y_pred"]].to_numpy())) * 1.08
    for ax, pred in zip(axes, shown, strict=True):
        d = p[p["predictor"] == pred]
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.plot([0, lim], [0, lim], color=theme.axis, lw=1, zorder=1)
        for carbon in ["glc", "gal"]:
            dc = d[d["carbon"] == carbon]
            ax.scatter(
                dc["y_true"],
                dc["y_pred"],
                s=34,
                color=theme.carbon[carbon],
                edgecolor=theme.surface,
                lw=1.2,
                alpha=0.9,
                zorder=3,
                label=f"tested on {CARBON_LABEL[carbon]}",
            )
        rmse = float(np.sqrt(np.mean((d["y_true"] - d["y_pred"]) ** 2)))
        ax.set_title(
            f"{PREDICTORS[pred][0]}\nRMSE {rmse:.2f}", loc="left", fontsize=10, fontweight="bold"
        )
        ax.set_xlim(-0.2, lim)
        ax.set_ylim(-0.2, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("Measured acetate (mmol/gDW/h)", fontsize=9)
    axes[0].set_ylabel("Predicted acetate (mmol/gDW/h)", fontsize=9)
    axes[0].annotate("y = x", (lim * 0.78, lim * 0.86), color=theme.muted, fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=2)
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    _header(
        fig,
        theme,
        "Overflow metabolism: pFBA predicts no acetate; an enzyme budget recovers part of it",
        "Held-out carbon source split; each point is one strain x carbon source condition "
        "(out-of-fold, seed 0).",
    )
    return _save(fig, out, "fig2_overflow", theme)


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


def skill_heatmap(results: Path, out: Path, theme: Theme) -> Path:
    """Skill score 1 - RMSE/RMSE(mean baseline) per target; > 0 beats the trivial baseline."""
    _style(theme)
    cols = ["pfba", "ecpfba", "ridge", "gbm"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 9.4), sharey=True)
    cmap = LinearSegmentedColormap.from_list(
        "div", [theme.diverging[2], theme.diverging[1], theme.diverging[0]]
    )
    targets = list(FLUX_LABELS)
    for ax, (exp, split_label) in zip(axes, SPLITS.items(), strict=True):
        s = _summary(results, exp)
        r = s[s["metric"] == "rmse"].pivot_table(
            index="target", columns="predictor", values="value"
        )
        skill = 1 - r[cols].div(r["mean"], axis=0)
        skill = skill.reindex(targets)
        vals = skill.to_numpy(float)
        ax.imshow(np.clip(vals, -1, 1), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                strong = abs(np.clip(v, -1, 1)) > 0.75
                txt = f"{v:+.2f}" if v > -10 else "< -10"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#ffffff" if strong else theme.ink,
                )
        ax.set_xticks(range(len(cols)), [SHORT[c] for c in cols])
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_yticks(range(len(targets)), [FLUX_LABELS[t] for t in targets])
        ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(targets)), minor=True)
        ax.grid(which="minor", color=theme.surface, lw=2)
        ax.tick_params(which="minor", length=0)
        ax.set_title(split_label, loc="left", fontsize=10.5, fontweight="bold", pad=28)
    fig.tight_layout(w_pad=2)
    _header(
        fig,
        theme,
        "Where does each method beat a trivial baseline?",
        "Skill = 1 - RMSE / RMSE(mean baseline). Blue > 0: better than predicting the training "
        "mean; red < 0: worse. Colour clipped to [-1, 1].",
    )
    return _save(fig, out, "fig3_skill", theme)


def make_all(results: Path, out: Path) -> list[Path]:
    paths = []
    for theme in (LIGHT, DARK):
        paths += [
            headline_figure(results, out, theme),
            overflow_figure(results, out, theme),
            skill_heatmap(results, out, theme),
        ]
    return paths
