# morie.fn -- function file (rootcoder007/morie)
"""Lagrange multiplier (Rao score) tests for spatial dependence in OLS residuals (Anselin 1988; Anselin et al. 1996)."""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats
from ._qpcore import inverse, ssum
from ._richresult import RichResult

__all__ = ["lm_spatial_tests"]


def _mv(W, v):
    return [ssum(a * b for a, b in zip(r, v)) for r in W]


def lm_spatial_tests(y, X, W) -> RichResult:
    r"""LM tests for spatial error and lag dependence after OLS.

    With OLS residuals ``e``, ``sigma^2 = e'e/n``, ``T = tr(W'W + WW)``,
    ``yhat`` the fitted values and ``M = I - X(X'X)^{-1}X'``,
    ``J = [(W yhat)' M (W yhat) + T sigma^2] / (n sigma^2)``:

    - ``RSerr = (e'We / sigma^2)^2 / T`` (Burridge 1980);
    - ``RSlag = (e'Wy / sigma^2)^2 / (n J)`` (Anselin 1988);
    - ``adjRSerr = (e'We/s^2 - T (nJ)^{-1} e'Wy/s^2)^2 / (T (1 - T/(nJ)))``
      and ``adjRSlag = (e'Wy/s^2 - e'We/s^2)^2 / (nJ - T)``, robust to the
      other alternative (Anselin, Bera, Florax and Yoon 1996);
    - ``SARMA = adjRSlag + RSerr`` (2 df).

    Each is referred to chi-square (1 df, SARMA 2); a statistic whose
    denominator vanishes (``W X beta`` in the span of ``X``, e.g. ``W = cI``) is NaN.  These are exactly
    ``spdep::lm.RStests`` (formerly ``lm.LMtests``) with ``test = "all"``.

    :param y: Response (n,).
    :param X: Design matrix (n, p) including any intercept column.
    :param W: Spatial weights (n, n), normally row-standardised.
    :return: :class:`RichResult` with, for each of ``RSerr``, ``RSlag``,
        ``adjRSerr``, ``adjRSlag``, ``SARMA``, a dict ``statistic``, ``df``,
        ``p_value``.

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer,
    Dordrecht.
    Anselin, L., Bera, A. K., Florax, R. and Yoon, M. J. (1996). Simple
    diagnostic tests for spatial dependence. *Regional Science and Urban
    Economics*, 26(1), 77-104.

    Examples
    --------
    >>> n = 8
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(n)] for i in range(n)]
    >>> W = [[v / sum(r) for v in r] for r in W]
    >>> X = [[1.0, v] for v in (0.1, 0.6, 0.2, 0.9, 0.3, 0.5, 0.8, 0.4)]
    >>> r = lm_spatial_tests([1.0, 2.2, 1.4, 3.1, 0.9, 2.0, 2.6, 1.1], X, W)
    >>> round(r.RSerr["statistic"], 6), round(r.RSlag["statistic"], 6)
    (0.003087, 1.298358)
    """
    yv = [float(v) for v in np.asarray(y, dtype=float).tolist()]
    Xm = [[float(v) for v in r] for r in np.asarray(X, dtype=float).tolist()]
    Wm = [[float(v) for v in r] for r in np.asarray(W, dtype=float).tolist()]
    n, p = len(yv), len(Xm[0])
    Xc = [list(c) for c in zip(*Xm)]
    XtXi = [
        [float(v) for v in r]
        for r in inverse([[ssum(a * b for a, b in zip(Xc[i], Xc[j])) for j in range(p)] for i in range(p)])
    ]
    Xty = [ssum(a * b for a, b in zip(c, yv)) for c in Xc]
    beta = [ssum(XtXi[i][j] * Xty[j] for j in range(p)) for i in range(p)]
    yhat = [ssum(Xm[i][k] * beta[k] for k in range(p)) for i in range(n)]
    u = [yv[i] - yhat[i] for i in range(n)]
    s2 = ssum(v * v for v in u) / n
    T = ssum(Wm[i][j] * Wm[i][j] + Wm[i][j] * Wm[j][i] for i in range(n) for j in range(n))
    Wu, Wy, Wyh = _mv(Wm, u), _mv(Wm, yv), _mv(Wm, yhat)
    XtWyh = [ssum(a * b for a, b in zip(c, Wyh)) for c in Xc]
    quad = ssum(XtWyh[i] * XtXi[i][j] * XtWyh[j] for i in range(p) for j in range(p))
    J = (ssum(v * v for v in Wyh) - quad + T * s2) / (n * s2)
    dutWu = ssum(a * b for a, b in zip(u, Wu)) / s2
    dutWy = ssum(a * b for a, b in zip(u, Wy)) / s2
    nJ = n * J

    def ratio(num, den):
        # undefined when W X beta lies in the span of X (e.g. W = c I): nJ = T
        return num / den if abs(den) > 1e-12 * max(abs(nJ), abs(T), 1e-300) else float("nan")

    stat = {
        "RSerr": (ratio(dutWu**2, T), 1),
        "RSlag": (ratio(dutWy**2, nJ), 1),
        "adjRSerr": (ratio((dutWu - T / nJ * dutWy) ** 2, T * (1.0 - T / nJ)), 1),
        "adjRSlag": (ratio((dutWy - dutWu) ** 2, nJ - T), 1),
    }
    stat["SARMA"] = (stat["adjRSlag"][0] + stat["RSerr"][0], 2)
    out = {
        k: {"statistic": float(v), "df": d, "p_value": float(stats.chi2.sf(v, d)) if v == v else float("nan")}
        for k, (v, d) in stat.items()
    }
    return RichResult(payload=out)


def cheatsheet() -> str:
    return "lm_spatial_tests(y, X, W) -> RSerr, RSlag, robust variants and SARMA (spdep::lm.RStests)."
