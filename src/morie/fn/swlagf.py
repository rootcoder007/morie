# morie.fn -- function file (rootcoder007/morie)
"""Spatial lag filter (I - rho*W)^-1."""

from .swops import error_operator


def swlagf(W, rho):
    r"""Spatial multiplier ``(I - rho W)^{-1}`` (``spatialreg::invIrW``).

    Maps innovations to the autoregressive process ``u = rho W u + e`` and
    gives the reduced form of the spatial lag model. Thin front-end to
    :func:`morie.fn.swops.error_operator`; returns the matrix as lists.

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.

    Examples
    --------
    >>> [[round(v, 6) for v in r] for r in swlagf([[0, 1], [1, 0]], 0.5)]
    [[1.333333, 0.666667], [0.666667, 1.333333]]
    """
    return error_operator(W, float(rho))


swlagf_fn = swlagf


def cheatsheet() -> str:
    return "swlagf(W, rho) -> spatial multiplier (I - rho W)^-1 (spatialreg::invIrW)."
