"""Group-based splitters. All of them split *groups* (conditions, strains, carbon sources, ...),
so every condition of a group is on one side only."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from metabench.core.interfaces import Splitter
from metabench.core.registry import register_splitter


def _ids_where(groups: pd.Series, mask: np.ndarray) -> list[str]:
    return [str(c) for c in groups.index[mask]]


@register_splitter("leave_group_out")
class LeaveGroupOut(Splitter):
    """Each distinct group is held out once. Deterministic: seed does not change folds."""

    def split_groups(self, groups: pd.Series, seed: int) -> Iterator[tuple[list[str], list[str]]]:
        for g in sorted(groups.unique(), key=str):
            test = (groups == g).to_numpy()
            yield _ids_where(groups, ~test), _ids_where(groups, test)


@register_splitter("leave_condition_out")
class LeaveConditionOut(LeaveGroupOut):
    """Leave-one-condition-out (spec name). Equivalent to leave_group_out on condition_id."""

    def __init__(self, **params: Any) -> None:
        if params.pop("group_by", "condition_id") != "condition_id":
            raise ValueError("leave_condition_out always groups by condition_id")
        super().__init__(group_by="condition_id", **params)


@register_splitter("grouped_kfold")
class GroupedKFold(Splitter):
    """Shuffled group k-fold, repeated n_repeats times; the shuffle depends on the seed.

    Folds are balanced in number of groups (not conditions).
    """

    def split_groups(self, groups: pd.Series, seed: int) -> Iterator[tuple[list[str], list[str]]]:
        n_splits = int(self.params.get("n_splits", 5))
        n_repeats = int(self.params.get("n_repeats", 1))
        unique = np.array(sorted(groups.unique(), key=str), dtype=object)
        if n_splits < 2 or n_splits > len(unique):
            raise ValueError(f"n_splits={n_splits} invalid for {len(unique)} groups")
        rng = np.random.default_rng(seed)
        for _ in range(n_repeats):
            perm = rng.permutation(len(unique))
            for fold_groups in np.array_split(unique[perm], n_splits):
                test = groups.isin(set(fold_groups)).to_numpy()
                yield _ids_where(groups, ~test), _ids_where(groups, test)
