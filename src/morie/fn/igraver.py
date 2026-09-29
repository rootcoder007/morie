# morie.fn -- function file (rootcoder007/morie)
"""Gravity model RMSE."""

from __future__ import annotations

import math

from . import _gravity as gv
from ._containers import SpatialResult


def igraver(flows, flows_hat):
    r"""Gravity model RMSE.

    Root mean squared error ``sqrt(mean((F - Fhat)^2))`` of predicted
    flows, and the standardised RMSE ``SRMSE = RMSE / mean(F)`` of Knudsen
    and Fotheringham (1986), which is comparable across flow systems (0 is a
    perfect fit).

    Parameters
    ----------
    flows, flows_hat : array-like, shape (n,)
        Observed and predicted flows.

    Returns
    -------
    SpatialResult
        ``statistic`` is the RMSE; ``extra["srmse"]``.

    References
    ----------
    Knudsen, D. C. and Fotheringham, A. S. (1986). Matrix comparison, goodness-of-fit, and spatial
    interaction modeling. *International Regional Science Review*, 10(2), 127-147.

    Examples
    --------
    >>> r = igraver([10.0, 20.0, 30.0], [12.0, 18.0, 33.0])
    >>> round(r.statistic, 12), round(r.extra["srmse"], 12)
    (2.380476142848, 0.119023807142)
    """
    F, H = gv.vec(flows), gv.vec(flows_hat)
    if len(F) != len(H):
        raise ValueError("flows and flows_hat must have the same length")
    n = len(F)
    rmse = math.sqrt(gv.ssum((a - b) ** 2 for a, b in zip(F, H)) / n)
    return SpatialResult(name="igraver", statistic=rmse, extra={"srmse": rmse / (gv.ssum(F) / n)})


igraver_fn = igraver


def cheatsheet() -> str:
    return "igraver(flows, flows_hat) -> RMSE and SRMSE"
