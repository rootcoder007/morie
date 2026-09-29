# morie.fn -- function file (rootcoder007/morie)
"""SAR score / LM test for spatial lag."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult
from .lmtests import lm_spatial_tests


def sarsc(y, X, W):
    r"""SAR score / LM test for spatial lag.

    The Rao score test of ``rho = 0`` after OLS: ``LM_lag = (e'Wy /
    sigma^2)^2 / (n J)``, ``n J = (W X b)' M (W X b) / sigma^2 + T``, ``T =
    tr(W'W + WW)``, ``sigma^2 = e'e / n``, against chi-square(1) (Anselin
    1988); the error-robust form ``adjRSlag`` of Anselin, Bera, Florax and
    Yoon (1996) is returned alongside. The score needs the fitted values, so
    the arguments are the response and design, not residuals; exactly
    ``spdep::lm.RStests`` via :func:`morie.fn.lmtests.lm_spatial_tests`.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p)
        Design matrix including any intercept column.
    W : array-like, shape (n, n)
        Spatial weights.

    Returns
    -------
    SpatialResult
        ``statistic`` and ``p_value`` of ``RSlag``; ``extra`` has
        ``robust_statistic`` and ``robust_p_value`` (``adjRSlag``).

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer, Dordrecht.

    Anselin, L., Bera, A. K., Florax, R. and Yoon, M. J. (1996). Simple diagnostic tests for spatial
    dependence. *Regional Science and Urban Economics*, 26(1), 77-104.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> r = sarsc([1.0, 2.4, 1.3, 3.1, 1.9, 2.2], [[1, .1], [1, .7], [1, .2], [1, .9], [1, .4], [1, .6]], W)
    >>> round(r.statistic, 12), round(r.p_value, 12)
    (4.997524073508, 0.025383605525)
    """
    r = lm_spatial_tests(sd._vec(y), sd._mat(X), sd._mat(W))
    lag, adj = r["RSlag"], r["adjRSlag"]
    return SpatialResult(
        name="sarsc",
        statistic=lag["statistic"],
        p_value=lag["p_value"],
        extra={"df": 1, "robust_statistic": adj["statistic"], "robust_p_value": adj["p_value"]},
    )


sarsc_fn = sarsc


def cheatsheet() -> str:
    return "sarsc(y, X, W) -> LM lag (RSlag) and robust adjRSlag"
