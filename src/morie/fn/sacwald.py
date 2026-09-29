# morie.fn -- function file (rootcoder007/morie)
"""SAC Wald test on rho and lambda jointly."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult
from .spdurbin import spatial_wald_test


def sacwald(params, vcov):
    r"""SAC Wald test on rho and lambda jointly.

    ``W = b' V^{-1} b`` for ``b = (rho, lambda)`` (or any parameter vector)
    with covariance ``V``, against chi-square with ``len(b)`` degrees of
    freedom (Anselin 1988, sec. 6.3); exactly
    :func:`morie.fn.spdurbin.spatial_wald_test` (no ridge on ``V``).

    Parameters
    ----------
    params : array-like
        Estimates, e.g. ``(rho, lambda)``.
    vcov : array-like, square
        Their covariance matrix.

    Returns
    -------
    SpatialResult
        ``statistic``, ``p_value``, ``extra["df"]``.

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer, Dordrecht.

    Examples
    --------
    >>> r = sacwald([0.3, 0.2], [[0.01, 0.002], [0.002, 0.02]])
    >>> round(r.statistic, 12), round(r.p_value, 12)
    (10.0, 0.006737946999)
    """
    r = spatial_wald_test(sd._vec(params), sd._mat(vcov))
    return SpatialResult(name="sacwald", statistic=r.statistic, p_value=r.pvalue, extra={"df": r.df})


sacwald_fn = sacwald


def cheatsheet() -> str:
    return "sacwald(params, vcov) -> joint Wald b'V^-1 b, chi-square(k)"
