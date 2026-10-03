"""Temperature schedule for simulated annealing."""

from __future__ import annotations

from . import _array_core as np
from ._containers import SpatialResult


def temperature_schedule(
    T0: float = 1.0,
    cooling: float = 0.95,
    n_iter: int = 100,
) -> SpatialResult:
    r"""Geometric cooling schedule for simulated annealing: ``T0 * cooling**k`` for k = 0 .. n_iter - 1."""
    schedule = T0 * cooling ** np.arange(n_iter)
    return SpatialResult(
        name="temperature_schedule",
        statistic=float(schedule[-1]),
        p_value=None,
        extra={"schedule": schedule},
    )


sgtmp = temperature_schedule


def cheatsheet() -> str:
    return "temperature_schedule({}) -> Temperature schedule for simulated annealing."
