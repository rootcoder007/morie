# morie.fn -- function file (rootcoder007/morie)
"""Lagrange multiplier (Rao score) diagnostics for spatial dependence in OLS regression.

Shared core of the ``lm*`` modules: the Anselin (1988) and Anselin, Bera,
Florax and Yoon (1996) tests of ``spdep::lm.RStests`` and the Koley-Bera
(2024) spatial Durbin tests of ``spdep::SD.RStests`` / PySAL ``spreg``.
"""

import math

from ._containers import SpatialResult
from ._qpcore import inverse, ssum
from ._rrng_core import pchisq


def _flat(v):
    return [float(a) for a in (v.tolist() if hasattr(v, "tolist") else v)]


def _mat(A):
    rows = A.tolist() if hasattr(A, "tolist") else A
    return [[float(v) for v in (r if hasattr(r, "__len__") else [r])] for r in rows]


def _design(X, n):
    """X as a list of rows; an intercept column is prepended when X has no constant column."""
    Xm = _mat(X) if X is not None else [[] for _ in range(n)]
    k = len(Xm[0])
    const = [c for c in range(k) if len({r[c] for r in Xm}) == 1 and Xm[0][c] != 0.0]
    if not const:
        Xm = [[1.0] + r for r in Xm]
        const = [0]
    return Xm, const


def _mv(A, v):
    return [ssum(a * b for a, b in zip(r, v)) for r in A]


def _xtv(X, v):
    return [ssum(X[i][a] * v[i] for i in range(len(v))) for a in range(len(X[0]))]


def _quad(g, A):
    return ssum(g[a] * ssum(A[a][b] * g[b] for b in range(len(g))) for a in range(len(g)))


def _rs_core(y, X, W):
    """All Rao score statistics for an OLS fit of y on X with weights W (spreg / spdep algebra)."""
    yv = _flat(y)
    n = len(yv)
    Xm, const = _design(X, n)
    Wm = _mat(W)
    k = len(Xm[0])
    xtxi = inverse([[ssum(r[a] * r[b] for r in Xm) for b in range(k)] for a in range(k)])
    beta = _mv(xtxi, _xtv(Xm, yv))
    predy = _mv(Xm, beta)
    u = [yv[i] - predy[i] for i in range(n)]
    sig2n = ssum(v * v for v in u) / n
    wu, wy, wxb = _mv(Wm, u), _mv(Wm, yv), _mv(Wm, predy)
    T = ssum(Wm[j][i] * Wm[j][i] + Wm[i][j] * Wm[j][i] for i in range(n) for j in range(n))
    xwxb = _xtv(Xm, wxb)
    j = (ssum(v * v for v in wxb) - _quad(xwxb, xtxi) + T * sig2n) / (n * sig2n)
    nj = n * j
    d_err = ssum(a * b for a, b in zip(u, wu)) / sig2n
    d_lag = ssum(a * b for a, b in zip(u, wy)) / sig2n
    out = {"n": n, "k": k, "T": T, "J": j, "sigma2": sig2n, "residuals": u, "coefficients": beta}
    out["RSerr"] = d_err * d_err / T
    out["RSlag"] = d_lag * d_lag / nj
    out["adjRSerr_z"] = (d_err - T * d_lag / nj) / math.sqrt(T * (1.0 - T / nj))
    out["adjRSerr"] = out["adjRSerr_z"] ** 2
    out["adjRSlag"] = (d_lag - d_err) ** 2 / (nj - T)
    out["SARMA"] = out["adjRSlag"] + out["RSerr"]
    # spatial Durbin (Koley and Bera 2024): lagged non-constant regressors
    xcols = [c for c in range(k) if c not in const]
    kx = len(xcols)
    if kx:
        wx = [_mv(Wm, [r[c] for r in Xm]) for c in xcols]  # kx columns
        WX = [[wx[a][i] for a in range(kx)] for i in range(n)]
        xtwx = [[ssum(Xm[i][a] * WX[i][b] for i in range(n)) for b in range(kx)] for a in range(k)]
        wxwx = [[ssum(WX[i][a] * WX[i][b] for i in range(n)) for b in range(kx)] for a in range(kx)]
        adj = [[_bil(xtwx, xtxi, a, b) for b in range(kx)] for a in range(kx)]
        xqxi = inverse([[wxwx[a][b] - adj[a][b] for b in range(kx)] for a in range(kx)])
        g = [ssum(WX[i][a] * u[i] for i in range(n)) for a in range(kx)]
        out["RSWX"] = _quad(g, xqxi) / sig2n
        # joint (rho, gamma): information block of [Wy-direction, WX] net of X
        C = [[xwxb[a]] + xtwx[a] for a in range(k)]  # k x (1 + kx)
        wxbwx = [ssum(wxb[i] * WX[i][b] for i in range(n)) for b in range(kx)]
        J22 = [[ssum(v * v for v in wxb) + T * sig2n] + wxbwx]
        J22 += [[wxbwx[a]] + wxwx[a] for a in range(kx)]
        m = 1 + kx
        CtC = [
            [ssum(C[r][a] * ssum(xtxi[r][s] * C[s][b] for s in range(k)) for r in range(k)) for b in range(m)]
            for a in range(m)
        ]
        J22i = inverse([[J22[a][b] - CtC[a][b] for b in range(m)] for a in range(m)])
        dd = [d_lag] + [v / sig2n for v in g]
        out["RSjoint_durbin"] = _quad(dd, J22i) * sig2n
        out["adjRSWX"] = out["RSjoint_durbin"] - out["RSlag"]
        out["adjRSlag_durbin"] = out["RSjoint_durbin"] - out["RSWX"]
        out["RSerr_WX"] = out["RSerr"] + out["RSWX"]
    out["kx"] = kx
    return out


def _bil(xtwx, xtxi, a, b):
    k = len(xtxi)
    return ssum(xtwx[r][a] * ssum(xtxi[r][s] * xtwx[s][b] for s in range(k)) for r in range(k))


_DF = {"RSerr": 1, "RSlag": 1, "adjRSerr": 1, "adjRSlag": 1, "SARMA": 2}


def _result(name, stat, df, extra=None):
    return SpatialResult(
        name=name, statistic=stat, p_value=float(pchisq(stat, df, lower_tail=False)), extra=dict(extra or {}, df=df)
    )


def lmdiag(y, X, W):
    r"""Rao score (LM) diagnostics for spatial dependence after OLS (``spdep::lm.RStests(test = "all")``).

    With OLS residuals ``u``, ``sigma2 = u'u/n``, ``T = tr((W' + W) W)`` and
    ``nJ = [(WXb)'M(WXb) + T sigma2] / sigma2``, ``d_err = u'Wu / sigma2`` and
    ``d_lag = u'Wy / sigma2``:

    * ``RSerr = d_err^2 / T`` (Burridge 1980; Anselin 1988),
    * ``RSlag = d_lag^2 / nJ`` (Anselin 1988),
    * ``adjRSerr = (d_err - T d_lag / nJ)^2 / (T (1 - T / nJ))`` and
      ``adjRSlag = (d_lag - d_err)^2 / (nJ - T)`` (Anselin, Bera, Florax and
      Yoon 1996),
    * ``SARMA = adjRSlag + RSerr`` (2 df).

    ``X`` is the regressor matrix; an intercept is prepended when it has no
    constant column. ``statistic`` is ``RSerr``; all five are in ``extra``
    with their p-values.

    References
    ----------
    Anselin, L. (1988). Lagrange multiplier test diagnostics for spatial
    dependence and spatial heterogeneity. *Geographical Analysis* 20, 1-17.
    Anselin, L., Bera, A. K., Florax, R. and Yoon, M. J. (1996). Simple
    diagnostic tests for spatial dependence. *Regional Science and Urban
    Economics* 26, 77-104.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0], [0.5, 0, 0.5, 0, 0], [0, 0.5, 0, 0.5, 0], [0, 0, 0.5, 0, 0.5], [0, 0, 0, 1, 0]]
    >>> r = lmdiag([1.0, 2.5, 2.0, 4.5, 4.0], [[0.0], [1.0], [2.0], [3.0], [4.0]], W)
    >>> round(r.extra["RSerr"], 10)
    3.4904112508
    """
    c = _rs_core(y, X, W)
    extra = {}
    for key, df in _DF.items():
        extra[key] = c[key]
        extra["p_" + key] = float(pchisq(c[key], df, lower_tail=False))
    return SpatialResult(name="lmdiag", statistic=c["RSerr"], p_value=extra["p_RSerr"], extra=extra)


lmdiag_fn = lmdiag


def cheatsheet() -> str:
    return "lmdiag(y, X, W) -> RSerr, RSlag, adjRSerr, adjRSlag, SARMA (spdep::lm.RStests)."
