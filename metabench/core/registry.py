"""Generic name -> object registries and the decorators plugins use to register themselves."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class RegistryError(KeyError):
    pass


class Registry(Generic[T]):
    """Maps a unique string name to a plugin (a class or a factory function)."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            if name in self._items and self._items[name] is not obj:
                raise RegistryError(f"{self.kind} '{name}' is already registered")
            self._items[name] = obj
            # Let classes know their registered name (used in results tables).
            if isinstance(obj, type):
                obj.name = name  # type: ignore[attr-defined]
            return obj

        return decorator

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "<none>"
            raise RegistryError(f"unknown {self.kind} '{name}'. Registered: {known}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def names(self) -> list[str]:
        return sorted(self._items)

    def unregister(self, name: str) -> None:
        """Remove an entry (for tests)."""
        self._items.pop(name, None)


PREDICTORS: Registry[Any] = Registry("predictor")
DATASETS: Registry[Any] = Registry("dataset")
TASKS: Registry[Any] = Registry("task")
METRICS: Registry[Any] = Registry("metric")
SPLITTERS: Registry[Any] = Registry("splitter")
MODELS: Registry[Any] = Registry("metabolic model")

register_predictor = PREDICTORS.register
register_dataset = DATASETS.register
register_task = TASKS.register
register_metric = METRICS.register
register_splitter = SPLITTERS.register
register_model = MODELS.register

_PLUGIN_PACKAGES = (
    "metabench.metabolic",
    "metabench.predictors",
    "metabench.datasets",
    "metabench.tasks",
    "metabench.splits",
    "metabench.metrics",
)
_loaded = False


def load_plugins() -> None:
    """Import every module under the plugin packages so their decorators run."""
    global _loaded
    if _loaded:
        return
    for pkg_name in _PLUGIN_PACKAGES:
        pkg = importlib.import_module(pkg_name)
        for mod in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg_name}."):
            importlib.import_module(mod.name)
    _loaded = True
