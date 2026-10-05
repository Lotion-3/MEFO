"""Genome-scale model loading and per-condition setup (medium, knockouts).

Models are registered by id. Loaded models are cached; callers must only modify them inside
`condition_context`, which reverts every change on exit so conditions never leak into each other.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import cobra
import numpy as np
from cobra.util.array import create_stoichiometric_matrix

from metabench.core.interfaces import GROWTH_TARGET, Perturbation
from metabench.core.registry import MODELS, register_model
from metabench.metabolic.toy import build_toy_model

# Units every mechanistic predictor reports in (COBRA convention).
MODEL_UNITS = {"flux": "mmol/gDW/h", "growth": "1/h"}


def data_dir() -> Path:
    return Path(os.environ.get("METABENCH_DATA", "data"))


class ModelNotAvailable(FileNotFoundError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_downloaded_sbml(model_id: str) -> cobra.Model:
    """Parse the SBML once, then reuse a pickle keyed by the SBML's SHA-256.

    Parsing iML1515 takes ~10 s; unpickling ~1 s. The key guarantees a changed SBML file is never
    served from a stale cache. The pickle is written and read only by this function.
    """
    path = data_dir() / "models" / f"{model_id}.xml"
    if not path.exists():
        raise ModelNotAvailable(
            f"{path} not found. Run: uv run python scripts/download_data.py --model {model_id}"
        )
    cache_file = data_dir() / "cache" / f"{model_id}-{_sha256(path)[:16]}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            model: cobra.Model = pickle.load(f)  # noqa: S301 - our own cache file
        return model
    model = cobra.io.read_sbml_model(str(path))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_file)
    return model


@register_model("toy")
def _toy() -> cobra.Model:
    return build_toy_model()


@register_model("e_coli_core")
def _e_coli_core() -> cobra.Model:
    # Bundled with COBRApy (its "textbook" model); no download needed.
    return cobra.io.load_model("textbook")


@register_model("iML1515")
def _iml1515() -> cobra.Model:
    return _load_downloaded_sbml("iML1515")


@cache
def load_model(model_id: str) -> cobra.Model:
    return MODELS.get(model_id)()


@cache
def stoichiometric_matrix(model_id: str) -> tuple[np.ndarray, list[str]]:
    model = load_model(model_id)
    s = create_stoichiometric_matrix(model, array_type="dense")
    return np.asarray(s, dtype=float), [r.id for r in model.reactions]


def biomass_reaction_id(model: cobra.Model) -> str:
    rxns = [r.id for r in model.reactions if r.objective_coefficient != 0]
    if len(rxns) != 1:
        raise ValueError(f"expected exactly one objective reaction, found {rxns}")
    return rxns[0]


def resolve_genes(model: cobra.Model, genes: list[str]) -> tuple[list[str], list[str]]:
    """Map gene ids or gene names to model gene ids -> (present ids, genes absent from the model).

    Absent genes (e.g. transcription factors in a metabolic model) have no effect on the model;
    callers report them rather than failing, so the comparison stays explicit about it.
    """
    by_name: dict[str, list[str]] = {}
    for g in model.genes:
        by_name.setdefault(g.name, []).append(g.id)
    present, absent = [], []
    for g in genes:
        if g in model.genes:
            present.append(g)
        elif len(by_name.get(g, [])) == 1:
            present.append(by_name[g][0])
        elif g in by_name:
            raise ValueError(f"gene name '{g}' is ambiguous in model '{model.id}': {by_name[g]}")
        else:
            absent.append(g)
    return present, absent


@contextmanager
def condition_context(
    model: cobra.Model, medium: dict[str, float], perturbation: Perturbation
) -> Iterator[cobra.Model]:
    """Apply a medium and knockouts; everything is reverted on exit.

    Knocked-out genes may be given as model gene ids or names; genes absent from the model are
    ignored here (see `resolve_genes`).
    """
    with model:
        if medium:
            unknown = set(medium) - {r.id for r in model.reactions}
            if unknown:
                raise KeyError(f"medium references unknown reactions {sorted(unknown)}")
            model.medium = medium
        present, _ = resolve_genes(model, perturbation.gene_knockouts)
        for gid in present:
            model.genes.get_by_id(gid).knock_out()
        for rid in perturbation.reaction_knockouts:
            model.reactions.get_by_id(rid).knock_out()
        yield model


def target_to_reactions(
    model: cobra.Model, target_id: str, target_map: dict[str, dict[str, float]]
) -> dict[str, float]:
    """Linear combination of reactions that a target corresponds to."""
    if target_id in target_map:
        return target_map[target_id]
    if target_id == GROWTH_TARGET:
        return {biomass_reaction_id(model): 1.0}
    if target_id in model.reactions:
        return {target_id: 1.0}
    raise KeyError(f"target '{target_id}' has no mapping to model '{model.id}' reactions")
