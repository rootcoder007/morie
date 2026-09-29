# morie.fn -- function file (rootcoder007/morie)
"""Geary's C statistic."""

from __future__ import annotations

from . import _lattice as lat
from ._containers import SpatialResult


def lacgear(y, W, randomisation=True):
    r"""Geary's C statistic.

    ``C = (n - 1) sum_ij w_ij (y_i - y_j)^2 / (2 S0 sum_i (y_i - ybar)^2)``
    (Geary 1954), with ``E(C) = 1`` and the variance under randomisation
    (kurtosis ``K = n sum z^4 / (sum z^2)^2``) or normality (Cliff and Ord
    1981, sec. 2.4); ``z = (1 - C) / sqrt(Var C)`` and the one-sided p-value
    for positive autocorrelation (``C < 1``), exactly ``spdep::geary.test``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Variable observed on the n lattice units.
    W : array-like, shape (n, n)
        Spatial weights (zero diagonal).
    randomisation : bool
        Randomisation (default) or normality variance.

    Returns
    -------
    SpatialResult
        ``statistic`` C, ``expected``, ``variance``, ``p_value``;
        ``extra`` has ``z`` and ``K``.

    References
    ----------
    Geary, R. C. (1954). The contiguity ratio and statistical mapping. *The Incorporated
    Statistician*, 5(3), 115-145.

    Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and Applications*. Pion, London.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> r = lacgear([1.0, 2.4, 1.3, 3.1, 1.9, 2.2], W)
    >>> round(r.statistic, 12), round(r.variance, 12)
    (1.284383954155, 0.117829834457)
    """
    r = lat.geary(lat.vec(y), lat.mat(W), bool(randomisation))
    return SpatialResult(
        name="lacgear",
        statistic=r["statistic"],
        p_value=r["p_value"],
        expected=r["expected"],
        variance=r["variance"],
        extra={"z": r["z"], "K": r["K"]},
    )


lacgear_fn = lacgear


def cheatsheet() -> str:
    return "lacgear(y, W, randomisation) -> Geary's C test (spdep::geary.test)"
