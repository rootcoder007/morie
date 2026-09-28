# morie.fn -- function file (rootcoder007/morie)
"""Space-time kriging prediction."""

from __future__ import annotations

from ._containers import SpatialResult
from .stkrig import _st_krige


def st_kriging(values, coords, times, new_coords, new_times, model: dict, *, beta=None) -> SpatialResult:
    r"""Space-time kriging (simple with known ``beta``, else ordinary), as ``gstat::krigeST``.

    Uses the space-time covariance models of :func:`morie.fn.stkrig.st_covariance`
    (separable, product-sum, metric, sum-metric).  ``local_values`` are the
    predictions; ``extra`` carries the kriging variances and the mean.
    ``statistic`` is the mean prediction.

    Examples
    --------
    >>> m = {"type": "separable", "sill": 1.0, "space": {"model": "Exp", "range": 1.0}, "time": {"model": "Exp", "range": 2.0}}
    >>> r = st_kriging([1.0, 2.0, 1.5], [(0, 0), (1, 0), (0, 1)], [0, 0, 1], [(0.5, 0.5)], [0.5], m)
    >>> round(r.local_values[0], 6), round(r.extra["variance"][0], 6)
    (1.522804, 0.725957)
    """
    r = _st_krige(values, coords, times, new_coords, new_times, model, beta=beta)
    p = r["prediction"]
    return SpatialResult(
        name="st_kriging",
        statistic=sum(p) / len(p),
        p_value=None,
        local_values=p,
        extra={"variance": r["variance"], "mean": r["mean"]},
    )


st_k = st_kriging
stkriging = st_kriging


def cheatsheet() -> str:
    return "st_kriging(values, coords, times, new_coords, new_times, model) -> space-time kriging (gstat::krigeST)."
