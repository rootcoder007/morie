# morie.fn -- function file (rootcoder007/morie)
"""Space-time kriging variance."""

from __future__ import annotations

from ._containers import SpatialResult
from .stkrig import _st_krige


def st_kriging_var(values, coords, times, new_coords, new_times, model: dict, *, beta=None) -> SpatialResult:
    r"""Space-time kriging variance at target points (``gstat::krigeST`` ``var1.var``).

    Simple kriging variance ``C_00 - c_0' C^{-1} c_0`` with a known ``beta``;
    ordinary kriging adds ``(1 - 1'C^{-1}c_0)^2 / (1'C^{-1}1)``.
    ``local_values`` are the variances, ``statistic`` their mean.

    Examples
    --------
    >>> m = {"type": "metric", "stAni": 1.0, "joint": {"model": "Exp", "psill": 1.0, "range": 1.0}}
    >>> round(st_kriging_var([1.0, 2.0], [(0, 0), (1, 0)], [0, 0], [(0.5, 0)], [0], m).local_values[0], 6)
    0.470878
    """
    r = _st_krige(values, coords, times, new_coords, new_times, model, beta=beta)
    v = r["variance"]
    return SpatialResult(
        name="st_kriging_var",
        statistic=sum(v) / len(v),
        p_value=None,
        local_values=v,
        extra={"prediction": r["prediction"], "mean": r["mean"]},
    )


st_k = st_kriging_var
stkrigingvar = st_kriging_var


def cheatsheet() -> str:
    return "st_kriging_var(values, coords, times, new_coords, new_times, model) -> space-time kriging variance."
