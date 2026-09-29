# morie.fn -- function file (rootcoder007/morie)
"""Getis-Ord G global statistic."""

from __future__ import annotations

from . import _lattice as lat
from ._containers import SpatialResult


def lacgetg(y, W):
    r"""Getis-Ord G global statistic.

    ``G = sum_{i != j} w_ij y_i y_j / sum_{i != j} y_i y_j`` for
    non-negative ``y`` (Getis and Ord 1992), with ``E(G) = S0 / (n (n -
    1))`` and the Getis-Ord variance from the power sums of ``y`` and the
    weight constants ``S0, S1, S2``; one-sided (greater, high-value
    clustering) normal p-value, exactly ``spdep::globalG.test`` (``B1correct
    = TRUE``). Binary or distance weights are the intended input.

    Parameters
    ----------
    y : array-like, shape (n,)
        Non-negative variable.
    W : array-like, shape (n, n)
        Spatial weights (zero diagonal).

    Returns
    -------
    SpatialResult
        ``statistic`` G, ``expected``, ``variance``, ``p_value``,
        ``extra["z"]``.

    References
    ----------
    Getis, A. and Ord, J. K. (1992). The analysis of spatial association by use of distance
    statistics. *Geographical Analysis*, 24(3), 189-206.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> r = lacgetg([1.0, 2.4, 1.3, 3.1, 1.9, 2.2], W)
    >>> round(r.statistic, 12), round(r.expected, 12)
    (0.199044309296, 0.2)
    """
    r = lat.getis_ord(lat.vec(y), lat.mat(W))
    return SpatialResult(
        name="lacgetg",
        statistic=r["statistic"],
        p_value=r["p_value"],
        expected=r["expected"],
        variance=r["variance"],
        extra={"z": r["z"]},
    )


lacgetg_fn = lacgetg


def cheatsheet() -> str:
    return "lacgetg(y, W) -> global Getis-Ord G test (spdep::globalG.test)"
