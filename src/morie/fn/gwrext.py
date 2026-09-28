# morie.fn -- function file (rootcoder007/morie)
"""Geographically weighted regression extras: local collinearity diagnostics, prediction at new
locations with prediction variance, Leung's F tests of non-stationarity, and geographically weighted
summary statistics."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rrng_core import pchisq, pf
from .gwrbas import gwr_kernel_weights

__all__ = ["gwr_collinearity", "gwr_predict", "gwr_f_tests", "gw_summary"]


def _mat(A):
    return [[float(v) for v in r] for r in (A.tolist() if hasattr(A, "tolist") else A)]


def _inv(A):
    return [[float(v) for v in r] for r in inverse([list(r) for r in A])]


def _dists(P, q):
    return [math.dist(p, q) for p in P]


def _wcor(x, y, w):
    if all(v == x[0] for v in x) or all(v == y[0] for v in y):
        return math.nan
    mx, my = ssum(a * b for a, b in zip(w, x)), ssum(a * b for a, b in zip(w, y))
    sxy = ssum(a * (b - mx) * (c - my) for a, b, c in zip(w, x, y))
    sxx = ssum(a * (b - mx) ** 2 for a, b in zip(w, x))
    syy = ssum(a * (c - my) ** 2 for a, c in zip(w, y))
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else math.nan


def gwr_collinearity(X, coords, bw: float, *, kernel: str = "bisquare", adaptive: bool = False) -> RichResult:
    r"""Local collinearity diagnostics of GWR, as ``GWmodel::gwr.collin.diagno``.

    At each data location, with normalised kernel weights ``w``: weighted
    correlations of every pair of columns of ``X`` (``nan`` for pairs with
    the constant intercept), local variance inflation factors
    ``diag(R^-1)`` of the non-intercept columns' weighted correlation matrix
    ``R``, and from the SVD of the weighted design ``w X`` with unit-norm
    columns the local condition number ``d_1 / d_p`` and the
    variance-decomposition proportions of the smallest singular value
    (Belsley, Kuh and Welsch 1980; Wheeler 2007). ``X`` has the intercept
    in its first column.

    References
    ----------
    Wheeler, D. C. (2007). Diagnostic tools and a remedial method for
    collinearity in geographically weighted regression. *Environment and
    Planning A*, 39(10), 2464-2481.
    Belsley, D. A., Kuh, E. and Welsch, R. E. (1980). *Regression
    Diagnostics*. Wiley.

    Examples
    --------
    >>> X = [[1, 0, 1], [1, 1, 0], [1, 2, 2], [1, 3, 1], [1, 4, 3]]
    >>> r = gwr_collinearity(X, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], 100.0)
    >>> r.local_cn[0] >= 1
    True
    """
    Xm = _mat(X)
    P = [tuple(float(v) for v in r) for r in coords]
    n, k = len(Xm), len(Xm[0])
    cols = [[r[j] for r in Xm] for j in range(k)]
    corr, vif, cn, vdp = [], [], [], []
    for i in range(n):
        w = gwr_kernel_weights(_dists(P, P[i]), bw, kernel, adaptive)
        sw = ssum(w)
        w = [v / sw for v in w]
        corr.append([_wcor(cols[a], cols[b], w) for a in range(k - 1) for b in range(a + 1, k)])
        R = [[_wcor(cols[a], cols[b], w) for b in range(1, k)] for a in range(1, k)]
        Ri = _inv(R)
        vif.append([Ri[a][a] for a in range(k - 1)])
        xw = [[Xm[r][j] * w[r] for j in range(k)] for r in range(n)]
        nrm = [math.sqrt(ssum(xw[r][j] ** 2 for r in range(n))) for j in range(k)]
        xs = [[xw[r][j] / nrm[j] for j in range(k)] for r in range(n)]
        U, d, Vt = np.linalg.svd(np.asarray(xs, dtype=float))
        d = [float(v) for v in d.tolist()]
        V = [[float(v) for v in r] for r in Vt.tolist()]
        cn.append(d[0] / d[-1])
        phi = [[(V[m][j] / d[m]) ** 2 for m in range(k)] for j in range(k)]
        vdp.append([phi[j][k - 1] / ssum(phi[j]) for j in range(k)])
    return RichResult(payload={"corr": corr, "vif": vif, "local_cn": cn, "vdp": vdp})


def gwr_predict(
    y, X, coords, X0, coords0, bw: float, *, kernel: str = "bisquare", adaptive: bool = False
) -> RichResult:
    r"""GWR prediction at new locations with prediction variance, as ``GWmodel::gwr.predict``.

    ``beta(u0) = (X'W0X)^-1 X'W0 y`` with the kernel weights of the data
    around ``u0``; the prediction is ``x0'beta(u0)`` and its variance
    ``sigma^2 (1 + x0'(X'W0X)^-1 X'W0^2 X (X'W0X)^-1 x0)`` with ``sigma^2 =
    RSS / (n - 2 tr S + tr S'S)`` of the GWR fitted at the data (Harris,
    Fotheringham, Crespo and Charlton 2010).

    References
    ----------
    Harris, P., Fotheringham, A. S., Crespo, R. and Charlton, M. (2010). The
    use of geographically weighted regression for spatial prediction: an
    evaluation of models using simulated data sets. *Mathematical
    Geosciences*, 42(6), 657-680.

    Examples
    --------
    >>> X = [[1, 0.0], [1, 1.0], [1, 2.0], [1, 3.0], [1, 4.0]]
    >>> r = gwr_predict([1.0, 3.0, 5.0, 7.0, 9.0], X, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], [[1, 2.5]], [(2.5, 0)], 100.0)
    >>> round(r.prediction[0], 10)
    6.0
    """
    yv = [float(v) for v in y]
    Xm = _mat(X)
    P = [tuple(float(v) for v in r) for r in coords]
    n, k = len(Xm), len(Xm[0])
    S = []
    for i in range(n):
        w = gwr_kernel_weights(_dists(P, P[i]), bw, kernel, adaptive)
        A = _inv([[ssum(w[r] * Xm[r][a] * Xm[r][b] for r in range(n)) for b in range(k)] for a in range(k)])
        xa = [ssum(Xm[i][a] * A[a][b] for a in range(k)) for b in range(k)]
        S.append([ssum(xa[b] * Xm[r][b] for b in range(k)) * w[r] for r in range(n)])
    trS = ssum(S[i][i] for i in range(n))
    trStS = ssum(v * v for r in S for v in r)
    fitted = [ssum(S[i][r] * yv[r] for r in range(n)) for i in range(n)]
    rss = ssum((a - b) ** 2 for a, b in zip(yv, fitted))
    s2 = rss / (n - 2 * trS + trStS)
    betas, pred, var = [], [], []
    for x0, q in zip(_mat(X0), [tuple(float(v) for v in r) for r in coords0]):
        w = gwr_kernel_weights(_dists(P, q), bw, kernel, adaptive)
        A = _inv([[ssum(w[r] * Xm[r][a] * Xm[r][b] for r in range(n)) for b in range(k)] for a in range(k)])
        b = [ssum(A[a][c] * ssum(w[r] * Xm[r][c] * yv[r] for r in range(n)) for c in range(k)) for a in range(k)]
        B2 = [[ssum(w[r] ** 2 * Xm[r][a] * Xm[r][c] for r in range(n)) for c in range(k)] for a in range(k)]
        s0 = [
            [ssum(A[a][e] * B2[e][f] * A[f][c] for e in range(k) for f in range(k)) for c in range(k)] for a in range(k)
        ]
        betas.append(b)
        pred.append(ssum(x0[a] * b[a] for a in range(k)))
        var.append(s2 * (1 + ssum(x0[a] * s0[a][c] * x0[c] for a in range(k) for c in range(k))))
    return RichResult(payload={"betas": betas, "prediction": pred, "variance": var, "sigma2": s2})


def _pf(q, d1, d2, lower):
    if math.isinf(d1) and math.isinf(d2):
        p = 1.0 if q >= 1 else 0.0
    elif math.isinf(d1):
        p = float(pchisq(d2 / q, d2, lower_tail=False)) if q > 0 else 0.0
    elif math.isinf(d2):
        p = float(pchisq(d1 * q, d1))
    else:
        p = float(pf(q, d1, d2))
    return p if lower else 1 - p


def gwr_f_tests(
    y, X, coords, bw: float, *, kernel: str = "bisquare", adaptive: bool = False, method: str = "leung"
) -> RichResult:
    r"""Leung, Mei and Zhang's (2000) F tests of a GWR against OLS and of each coefficient's spatial variation.

    With the hat matrix ``S``, ``R = (I - S)'(I - S)``, ``delta_1 = tr R``,
    ``delta_2 = tr R^2`` and the OLS residual sum of squares ``RSS_o`` on
    ``DF_o = n - p``: ``F1 = (RSS_g / delta_1) / (RSS_o / DF_o)`` (lower
    tail, df ``delta_1^2 / delta_2``, ``DF_o``); ``F2 = ((RSS_o - RSS_g) /
    (DF_o - delta_1)) / (RSS_o / DF_o)``; ``F3`` per coefficient, the
    variance ``V_k^2 = beta_k'(I - J/n)beta_k / n`` of the local estimates
    over ``gamma_1 = tr B_k'(I - J/n)B_k / n`` scaled by ``RSS_g /
    delta_1`` (df ``gamma_1^2 / gamma_2``); ``F4 = RSS_g / RSS_o``.
    ``method="gwmodel"`` reproduces ``GWmodel`` 2.4 (``delta_2 = 0``, so
    ``F1`` has infinite numerator df, and ``gamma_2`` the sum of squared
    diagonal elements); the default uses the traces of the paper.

    References
    ----------
    Leung, Y., Mei, C.-L. and Zhang, W.-X. (2000). Statistical tests for
    spatial nonstationarity based on the geographically weighted regression
    model. *Environment and Planning A*, 32(1), 9-32.

    Examples
    --------
    >>> X = [[1, float(i)] for i in range(8)]
    >>> y = [0.5 * i + (i % 3) * 0.3 for i in range(8)]
    >>> r = gwr_f_tests(y, X, [(i, 0) for i in range(8)], 3.0)
    >>> r.F4 > 0
    True
    """
    if method not in ("leung", "gwmodel"):
        raise ValueError("method must be leung or gwmodel")
    yv = [float(v) for v in y]
    Xm = _mat(X)
    P = [tuple(float(v) for v in r) for r in coords]
    n, k = len(Xm), len(Xm[0])
    S, betas, Bs = [], [], []
    for i in range(n):
        w = gwr_kernel_weights(_dists(P, P[i]), bw, kernel, adaptive)
        A = _inv([[ssum(w[r] * Xm[r][a] * Xm[r][b] for r in range(n)) for b in range(k)] for a in range(k)])
        C = [[ssum(A[a][c] * Xm[r][c] for c in range(k)) * w[r] for r in range(n)] for a in range(k)]
        Bs.append(C)
        betas.append([ssum(C[a][r] * yv[r] for r in range(n)) for a in range(k)])
        S.append([ssum(Xm[i][a] * C[a][r] for a in range(k)) for r in range(n)])
    IS = [[(1.0 if i == j else 0.0) - S[i][j] for j in range(n)] for i in range(n)]
    R = [[ssum(IS[t][i] * IS[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
    d1 = ssum(R[i][i] for i in range(n))
    d2 = 0.0 if method == "gwmodel" else ssum(R[i][j] * R[j][i] for i in range(n) for j in range(n))
    fitted = [ssum(S[i][r] * yv[r] for r in range(n)) for i in range(n)]
    rss_g = ssum((a - b) ** 2 for a, b in zip(yv, fitted))
    XtX = _inv([[ssum(Xm[r][a] * Xm[r][b] for r in range(n)) for b in range(k)] for a in range(k)])
    bo = [ssum(XtX[a][c] * ssum(Xm[r][c] * yv[r] for r in range(n)) for c in range(k)) for a in range(k)]
    rss_o = ssum((yv[r] - ssum(Xm[r][a] * bo[a] for a in range(k))) ** 2 for r in range(n))
    dfo = n - k
    f1 = (rss_g / d1) / (rss_o / dfo)
    f1df = (d1 * d1 / d2 if d2 > 0 else math.inf, dfo)
    f2 = ((rss_o - rss_g) / (dfo - d1)) / (rss_o / dfo)
    f2df = ((dfo - d1) ** 2 / (dfo - 2 * d1 + d2), dfo)
    s2d = rss_g / d1
    f3, f3df, f3p = [], [], []
    for a in range(k):
        bk = [betas[i][a] for i in range(n)]
        mb = ssum(bk) / n
        vk2 = ssum((v - mb) ** 2 for v in bk) / n
        B = [[Bs[j][a][r] for r in range(n)] for j in range(n)]
        cm = [ssum(B[j][r] for j in range(n)) / n for r in range(n)]
        Bc = [[B[j][r] - cm[r] for r in range(n)] for j in range(n)]
        BJ = [[ssum(Bc[j][r] * Bc[j][c] for j in range(n)) / n for c in range(n)] for r in range(n)]
        g1 = ssum(BJ[r][r] for r in range(n))
        g2 = (
            ssum(BJ[r][r] ** 2 for r in range(n))
            if method == "gwmodel"
            else ssum(BJ[r][c] * BJ[c][r] for r in range(n) for c in range(n))
        )
        stat = (vk2 / g1) / s2d
        df = (g1 * g1 / g2, f1df[0])
        f3.append(stat)
        f3df.append(df)
        f3p.append(_pf(stat, df[0], df[1], False))
    f4 = rss_g / rss_o
    return RichResult(
        payload={
            "F1": f1,
            "F1_df": f1df,
            "F1_p": _pf(f1, f1df[0], f1df[1], True),
            "F2": f2,
            "F2_df": f2df,
            "F2_p": _pf(f2, f2df[0], f2df[1], False),
            "F3": f3,
            "F3_df": f3df,
            "F3_p": f3p,
            "F4": f4,
            "F4_df": (d1, dfo),
            "F4_p": _pf(f4, d1, dfo, True),
            "delta1": d1,
            "delta2": d2,
            "rss_gwr": rss_g,
            "rss_ols": rss_o,
            "betas": betas,
        }
    )


def _ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for t in range(i, j + 1):
            r[o[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def gw_summary(X, coords, bw: float, *, kernel: str = "bisquare", adaptive: bool = False) -> RichResult:
    r"""Geographically weighted summary statistics, as ``GWmodel::gwss``.

    With normalised kernel weights ``w`` at each location: local means,
    variances ``sum w (x - m)^2``, standard deviations, skewness ``sum w (x
    - m)^3 / sd^3`` and coefficients of variation ``sd / m`` of every column,
    and for every pair of columns the weighted covariance (``cov.wt``'s
    unbiased form, divided by ``1 - sum w^2``), the Pearson correlation and
    the Spearman correlation (weighted correlation of the global ranks)
    (Brunsdon, Fotheringham and Charlton 2002).

    References
    ----------
    Brunsdon, C., Fotheringham, A. S. and Charlton, M. (2002).
    Geographically weighted summary statistics - a framework for localised
    exploratory data analysis. *Computers, Environment and Urban Systems*,
    26(6), 501-524.

    Examples
    --------
    >>> r = gw_summary([[1, 2], [2, 4], [3, 6]], [(0, 0), (1, 0), (2, 0)], 100.0)
    >>> round(r.corr[0][0], 12)
    1.0
    """
    Xm = _mat(X)
    P = [tuple(float(v) for v in r) for r in coords]
    n, k = len(Xm), len(Xm[0])
    cols = [[r[j] for r in Xm] for j in range(k)]
    rk = [_ranks(c) for c in cols]
    out = {key: [] for key in ("mean", "sd", "var", "skewness", "cv", "cov", "corr", "spearman")}
    for i in range(n):
        w = gwr_kernel_weights(_dists(P, P[i]), bw, kernel, adaptive)
        sw = ssum(w)
        w = [v / sw for v in w]
        m = [ssum(a * b for a, b in zip(w, c)) for c in cols]
        var = [ssum(a * (b - mu) ** 2 for a, b in zip(w, c)) for c, mu in zip(cols, m)]
        sd = [math.sqrt(v) for v in var]
        out["mean"].append(m)
        out["var"].append(var)
        out["sd"].append(sd)
        out["skewness"].append(
            [
                ssum(a * (b - mu) ** 3 for a, b in zip(w, c)) / s**3 if s > 0 else math.nan
                for c, mu, s in zip(cols, m, sd)
            ]
        )
        out["cv"].append([s / mu if mu != 0 else math.nan for s, mu in zip(sd, m)])
        cw = 1 - ssum(v * v for v in w)
        cov, cor, sp = [], [], []
        for a in range(k - 1):
            for b in range(a + 1, k):
                cov.append(ssum(t * (x - m[a]) * (z - m[b]) for t, x, z in zip(w, cols[a], cols[b])) / cw)
                cor.append(_wcor(cols[a], cols[b], w))
                sp.append(_wcor(rk[a], rk[b], w))
        out["cov"].append(cov)
        out["corr"].append(cor)
        out["spearman"].append(sp)
    return RichResult(payload=out)


def cheatsheet() -> str:
    return "gwr_collinearity / gwr_predict / gwr_f_tests / gw_summary -> GWR diagnostics, prediction, F tests, GW summaries."
