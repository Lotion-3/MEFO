"""Task 1: steady-state growth rate and flux prediction across conditions."""

from __future__ import annotations

from metabench.core.interfaces import Task
from metabench.core.registry import register_task


@register_task("steady_state_flux")
class SteadyStateFlux(Task):
    target_types = ("growth", "flux")
    valid_metrics = ("rmse", "mae", "r2", "spearman", "mass_balance_violation")
    # Group-based splitters only; there is deliberately no random-row splitter.
    valid_splitters = ("leave_group_out", "leave_condition_out", "grouped_kfold")
