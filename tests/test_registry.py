from __future__ import annotations

import pytest

from metabench.core import registry as reg
from metabench.core.registry import Registry, RegistryError


def test_register_and_get() -> None:
    r: Registry[type] = Registry("thing")

    @r.register("a")
    class A:
        pass

    assert r.get("a") is A
    assert A.name == "a"  # type: ignore[attr-defined]
    assert "a" in r and r.names() == ["a"]


def test_duplicate_name_rejected() -> None:
    r: Registry[type] = Registry("thing")
    r.register("a")(type("A", (), {}))
    with pytest.raises(RegistryError, match="already registered"):
        r.register("a")(type("B", (), {}))


def test_unknown_name_lists_known() -> None:
    r: Registry[type] = Registry("thing")
    r.register("a")(type("A", (), {}))
    with pytest.raises(RegistryError, match="Registered: a"):
        r.get("zzz")


def test_plugins_discovered() -> None:
    reg.load_plugins()
    for name in ("fba", "pfba", "fva_summary", "mean", "ridge"):
        assert name in reg.PREDICTORS
    assert {"toy", "e_coli_core", "iML1515"} <= set(reg.MODELS.names())
    assert "steady_state_flux" in reg.TASKS
    assert {"leave_group_out", "leave_condition_out", "grouped_kfold"} <= set(reg.SPLITTERS.names())
