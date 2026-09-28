# morie.fn -- function file (rootcoder007/morie)
"""Gravity model of flows by Poisson pseudo-maximum likelihood."""

from __future__ import annotations

from . import _stats_core as stats
from ._containers import SpatialResult
from .gravpp import gravity_ppml


def igravps(flows, mass_o, mass_d, dist) -> SpatialResult:
    """PPML gravity model (Santos Silva and Tenreyro 2006).

    Delegates to :func:`~morie.fn.gravpp.gravity_ppml`; ``statistic`` is the
    distance elasticity with its robust (HC0) two-sided p-value.

    Examples
    --------
    >>> F = [12.0, 0.0, 30.0, 7.0, 55.0, 3.0]
    >>> round(igravps(F, [5, 5, 9, 9, 20, 20], [9, 20, 5, 20, 5, 9], [1.0, 3.0, 1.0, 2.0, 3.0, 2.0]).statistic, 6)
    0.884215
    """
    r = gravity_ppml(flows, mass_o, mass_d, dist)
    b, se = r["coefficients"][3], r["se_robust"][3]
    return SpatialResult(
        name="igravps", statistic=b, p_value=float(2.0 * stats.norm.sf(abs(b / se))) if se > 0 else None, extra=dict(r)
    )


igravps_fn = igravps


def cheatsheet() -> str:
    return "igravps(flows, mass_o, mass_d, dist) -> PPML gravity distance elasticity."
