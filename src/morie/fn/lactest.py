# morie.fn -- function file (rootcoder007/morie)
"""General lattice spatial test battery."""

from __future__ import annotations

from . import _lattice as lat
from ._containers import SpatialResult


def lactest(y, W):
    r"""General lattice spatial test battery.

    The three global tests of spatial autocorrelation on one variable, each
    with its randomisation moments and one-sided normal p-value: Moran's I
    (``spdep::moran.test``; ``E(I) = -1/(n-1)``, Cliff and Ord 1981),
    Geary's C (``spdep::geary.test``) and, when ``y`` is non-negative, the
    Getis-Ord G (``spdep::globalG.test``).

    Parameters
    ----------
    y : array-like, shape (n,)
        Variable observed on the n lattice units.
    W : array-like, shape (n, n)
        Spatial weights (zero diagonal).

    Returns
    -------
    SpatialResult
        ``statistic``, ``expected``, ``variance`` and ``p_value`` of
        Moran's I; ``extra`` has dicts ``moran``, ``geary`` and ``getis_ord``
        (``None`` for negative data).

    References
    ----------
    Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and Applications*. Pion, London.

    Geary, R. C. (1954). The contiguity ratio and statistical mapping. *The Incorporated
    Statistician*, 5(3), 115-145.

    Getis, A. and Ord, J. K. (1992). The analysis of spatial association by use of distance
    statistics. *Geographical Analysis*, 24(3), 189-206.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> r = lactest([1.0, 2.4, 1.3, 3.1, 1.9, 2.2], W)
    >>> round(r.statistic, 12), round(r.extra["geary"]["statistic"], 12)
    (-0.612893982808, 1.284383954155)
    """
    yv, Wm = lat.vec(y), lat.mat(W)
    m = lat.moran(yv, Wm)
    g = lat.geary(yv, Wm)
    go = lat.getis_ord(yv, Wm) if min(yv) >= 0 else None
    return SpatialResult(
        name="lactest",
        statistic=m["statistic"],
        p_value=m["p_value"],
        expected=m["expected"],
        variance=m["variance"],
        extra={"moran": m, "geary": g, "getis_ord": go},
    )


lactest_fn = lactest


def cheatsheet() -> str:
    return "lactest(y, W) -> Moran, Geary and Getis-Ord global tests"
