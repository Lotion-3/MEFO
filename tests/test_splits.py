"""Leakage tests: every splitter must keep each group entirely on one side of every fold."""

from __future__ import annotations

import pandas as pd
import pytest

from metabench.core import registry as reg

# Conditions with several per group, so group-level (not row-level) splitting is exercised.
GROUPS = pd.Series(
    ["glc", "glc", "ace", "ace", "ace", "gal", "fru", "fru", "pyr", "succ"],
    index=[f"c{i}" for i in range(10)],
)

CASES = [
    ("leave_group_out", {}),
    ("grouped_kfold", {"n_splits": 3}),
    ("grouped_kfold", {"n_splits": 2, "n_repeats": 3}),
]


@pytest.mark.parametrize(("name", "params"), CASES)
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_no_group_overlap_and_full_coverage(name: str, params: dict, seed: int) -> None:
    splitter = reg.SPLITTERS.get(name)(group_by="carbon_source", **params)
    folds = list(splitter.split_groups(GROUPS, seed))
    n_repeats = params.get("n_repeats", 1)
    tested: list[str] = []
    for train, test in folds:
        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == set(GROUPS.index)
        assert set(GROUPS[train]).isdisjoint(set(GROUPS[test])), "group appears on both sides"
        tested += test
    # every condition is tested exactly once per repeat
    assert sorted(tested) == sorted(list(GROUPS.index) * n_repeats)


@pytest.mark.parametrize(("name", "params"), CASES)
def test_deterministic_given_seed(name: str, params: dict) -> None:
    splitter = reg.SPLITTERS.get(name)(**params)
    assert list(splitter.split_groups(GROUPS, 3)) == list(splitter.split_groups(GROUPS, 3))


def test_kfold_shuffle_depends_on_seed() -> None:
    splitter = reg.SPLITTERS.get("grouped_kfold")(n_splits=3)
    splits = {tuple(map(tuple, (t for _, t in splitter.split_groups(GROUPS, s)))) for s in range(5)}
    assert len(splits) > 1


def test_leave_condition_out_on_dataset(toy_data) -> None:
    splitter = reg.SPLITTERS.get("leave_condition_out")()
    folds = splitter.split(toy_data, seed=0)
    assert len(folds) == len(toy_data.conditions)
    for train, test in folds:
        assert len(test) == 1 and test[0] not in train


def test_leave_condition_out_rejects_other_grouping() -> None:
    with pytest.raises(ValueError):
        reg.SPLITTERS.get("leave_condition_out")(group_by="carbon_source")


def test_group_by_unknown_column(toy_data) -> None:
    with pytest.raises(ValueError, match="cannot group"):
        reg.SPLITTERS.get("leave_group_out")(group_by="strain").split(toy_data, 0)


def test_leave_group_out_regime(toy_data) -> None:
    folds = reg.SPLITTERS.get("leave_group_out")(group_by="regime").split(toy_data, 0)
    regimes = toy_data.conditions.set_index("condition_id")["regime"]
    assert len(folds) == 2
    for train, test in folds:
        assert set(regimes[train]).isdisjoint(regimes[test])
