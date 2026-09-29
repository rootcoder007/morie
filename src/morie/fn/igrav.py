# morie.fn -- function file (rootcoder007/morie)
"""Gravity model OLS estimation."""

from __future__ import annotations

import math

from . import _gravity as gv
from ._containers import SpatialResult


def igrav(flows, mass_o, mass_d, dist):
    r"""Gravity model OLS estimation.

    The log-linear gravity equation ``log F = b0 + b1 log M_o + b2 log M_d +
    b3 log d + e`` (Tinbergen 1962) by OLS on the pairs with positive flow
    (zero flows have no logarithm; see :func:`morie.fn.gravpp.gravity_ppml`
    for the PPML estimator that keeps them and is consistent under
    heteroskedasticity, Santos Silva and Tenreyro 2006). Standard errors
    ``s^2 (X'X)^{-1}``, ``s^2 = e'e / (n - 4)``.

    Parameters
    ----------
    flows : array-like, shape (n,)
        Observed flows of the n origin-destination pairs.
    mass_o, mass_d : array-like, shape (n,)
        Positive origin and destination masses of each pair.
    dist : array-like, shape (n,)
        Positive distances.

    Returns
    -------
    SpatialResult
        ``statistic`` is the distance elasticity ``b3``; ``extra`` has
        ``coefficients``, ``se``, ``r2``, ``sigma2`` and ``n_used``.

    References
    ----------
    Tinbergen, J. (1962). *Shaping the World Economy*. Twentieth Century Fund, New York.

    Santos Silva, J. M. C. and Tenreyro, S. (2006). The log of gravity. *Review of Economics and
    Statistics*, 88(4), 641-658.

    Examples
    --------
    >>> F = [12.0, 3.0, 30.0, 7.0, 55.0, 4.0, 9.0, 21.0]
    >>> mo = [5, 5, 9, 9, 20, 20, 7, 7]
    >>> md = [9, 20, 5, 20, 5, 9, 20, 9]
    >>> d = [1.0, 3.0, 1.0, 2.0, 3.0, 2.0, 2.5, 1.5]
    >>> r = igrav(F, mo, md, d)
    >>> [round(v, 10) for v in r.extra["coefficients"]]
    [7.464610248, -0.571542517, -1.7577699687, 0.5894245802]
    """
    F = gv.vec(flows)
    X = gv.design(mass_o, mass_d, dist)
    keep = [i for i, v in enumerate(F) if v > 0]
    r = gv.ols([X[i] for i in keep], [math.log(F[i]) for i in keep])
    return SpatialResult(
        name="igrav",
        statistic=r["coefficients"][3],
        extra={
            "coefficients": r["coefficients"],
            "se": r["se"],
            "r2": r["r2"],
            "sigma2": r["sigma2"],
            "n_used": len(keep),
        },
    )


igrav_fn = igrav


def cheatsheet() -> str:
    return "igrav(flows, mass_o, mass_d, dist) -> log-linear gravity OLS"
