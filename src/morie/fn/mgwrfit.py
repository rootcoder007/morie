# morie.fn -- function file (rootcoder007/morie)
"""MGWR model fit (Fotheringham et al. 2017)."""

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from .gwrbas import gwr_kernel_weights
from .gwrcoef import _flat, _rows, _with_intercept

_GOLD = (math.sqrt(5.0) - 1.0) / 2.0


def _prep(y, X, coords):
    yv = _flat(y)
    n = len(yv)
    Xm = _with_intercept(X, n)
    P = _rows(coords)
    D = [[math.sqrt(ssum((a - b) ** 2 for a, b in zip(P[i], P[j]))) for j in range(n)] for i in range(n)]
    return yv, Xm, D


def _smoother(x, D, bw, kernel, adaptive):
    """Rows c_i of the one-covariate local fit: beta_i = c_i . v with c_ij = w_ij x_j / sum_j w_ij x_j^2."""
    n = len(x)
    C = []
    for i in range(n):
        w = gwr_kernel_weights(D[i], bw, kernel, adaptive)
        den = ssum(w[j] * x[j] * x[j] for j in range(n))
        C.append([w[j] * x[j] / den for j in range(n)])
    return C


def _uni_aicc(x, v, D, bw, kernel, adaptive):
    """AICc of the one-covariate GWR of v on x (no intercept), the per-covariate selection criterion."""
    n = len(x)
    C = _smoother(x, D, bw, kernel, adaptive)
    fit = [x[i] * ssum(C[i][j] * v[j] for j in range(n)) for i in range(n)]
    rss = ssum((v[i] - fit[i]) ** 2 for i in range(n))
    tr = ssum(x[i] * C[i][i] for i in range(n))
    if tr >= n - 2.0 or rss <= 0.0:
        return math.inf
    return n * math.log(rss / n) + n * math.log(2.0 * math.pi) + n * (n + tr) / (n - 2.0 - tr)


def _golden(f, lo, hi, tol):
    a, b = lo, hi
    c, d = b - _GOLD * (b - a), a + _GOLD * (b - a)
    fc, fd = f(c), f(d)
    while abs(b - a) > tol * (abs(a) + abs(b)):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - _GOLD * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLD * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def _backfit(yv, Xm, D, bws, kernel, adaptive, threshold, max_iter, select, hat):
    n, p = len(yv), len(Xm[0])
    cols = [[r[k] for r in Xm] for k in range(p)]
    XtXi = inverse([[ssum(cols[a][i] * cols[b][i] for i in range(n)) for b in range(p)] for a in range(p)])
    b0 = [ssum(XtXi[a][b] * ssum(cols[b][i] * yv[i] for i in range(n)) for b in range(p)) for a in range(p)]
    beta = [[b0[k]] * n for k in range(p)]
    f = [[cols[k][i] * b0[k] for i in range(n)] for k in range(p)]
    resid = [yv[i] - ssum(f[k][i] for k in range(p)) for i in range(n)]
    R = None
    if hat:
        H = [[ssum(XtXi[k][b] * cols[b][j] for b in range(p)) for j in range(n)] for k in range(p)]
        R = [[list(H[k]) for _ in range(n)] for k in range(p)]  # coefficient maps: beta_k = R[k] y
        S = [[ssum(cols[k][i] * R[k][i][j] for k in range(p)) for j in range(n)] for i in range(n)]
    bws = list(bws) if bws is not None else [None] * p
    lo = min(v for r in D for v in r if v > 0)
    hi = max(v for r in D for v in r)
    rss0 = ssum(v * v for v in resid)
    it, crit = 0, math.inf
    while it < max_iter and crit > threshold:
        it += 1
        old_bws = list(bws)
        for k in range(p):
            yk = [resid[i] + f[k][i] for i in range(n)]
            x = cols[k]
            if select:
                bws[k] = _golden(lambda b, x=x, yk=yk: _uni_aicc(x, yk, D, b, kernel, adaptive), lo, hi, 1e-8)
            C = _smoother(x, D, bws[k], kernel, adaptive)
            bk = [ssum(C[i][j] * yk[j] for j in range(n)) for i in range(n)]
            beta[k] = bk
            f[k] = [x[i] * bk[i] for i in range(n)]
            resid = [yk[i] - f[k][i] for i in range(n)]
            if hat:
                # C_k <- C (I - sum_{m != k} diag(x_m) C_m); the hat block is R_k = diag(x_k) C_k
                M = [[(1.0 if i == j else 0.0) - S[i][j] + x[i] * R[k][i][j] for j in range(n)] for i in range(n)]
                new = [[ssum(C[i][m] * M[m][j] for m in range(n)) for j in range(n)] for i in range(n)]
                S = [[S[i][j] + x[i] * (new[i][j] - R[k][i][j]) for j in range(n)] for i in range(n)]
                R[k] = new
        rss1 = ssum(v * v for v in resid)
        crit = math.sqrt(abs(rss1 - rss0) / rss1) if rss1 > 0 else 0.0
        if select:
            crit = max(crit, max(abs(a - b) / b for a, b in zip(bws, old_bws)) if old_bws[0] is not None else math.inf)
        rss0 = rss1
    return beta, resid, rss0, it, bws, R, cols


def mgwrfit(y, X, coords, bandwidths=None, kernel="bisquare", adaptive=False, threshold=1e-10, max_iter=500):
    r"""Multiscale geographically weighted regression (Fotheringham, Yang and Kang 2017) with inference.

    ``y_i = sum_k beta_k(u_i) x_ik + e_i`` (an intercept is prepended when
    ``X`` has no constant column; predictors are not centred) is fitted by
    backfitting from the OLS solution: each covariate's smooth is refitted as
    a one-covariate GWR of its partial residual with its own bandwidth, until
    the change criterion ``sqrt(|RSS_new - RSS_old| / RSS_new)`` falls below
    ``threshold``. When ``bandwidths`` is None each bandwidth is re-selected
    at every pass by golden-section minimisation of the one-covariate AICc
    (see :func:`morie.fn.mgwrbw.mgwrbw`).

    Inference follows Yu, Fotheringham, Li, Oshan, Kang and Wolf (2020): the
    covariate-specific hat matrices ``R_k`` (``f_k = R_k y``) are updated
    alongside the smooths, ``R_k <- A_k (I - sum_{m != k} R_m)`` with ``A_k`` the one-covariate smoother, ``S = sum_k
    R_k``, ``sigma^2 = RSS / (n - tr S)``, the standard errors are
    ``sqrt(sigma^2 diag(C_k C_k'))`` with ``beta_k = C_k y`` (``R_k = diag(x_k) C_k``), ``ENP_k =
    tr R_k`` and ``AICc = n log(RSS/n) + n log(2 pi) + n (n + tr S) / (n - 2 -
    tr S)``. With fixed bandwidths this equals ``GWmodel::gwr.multiscale(...,
    predictor.centered = FALSE, hatmatrix = TRUE)``.

    References
    ----------
    Fotheringham, A. S., Yang, W. and Kang, W. (2017). Multiscale
    geographically weighted regression (MGWR). *Annals of the American
    Association of Geographers* 107, 1247-1265.
    Yu, H., Fotheringham, A. S., Li, Z., Oshan, T., Kang, W. and Wolf, L. J.
    (2020). Inference in multiscale geographically weighted regression.
    *Geographical Analysis* 52, 87-106.

    Examples
    --------
    >>> P = [(float(i % 5), float(i // 5)) for i in range(20)]
    >>> X = [[math.sin(i), (0.3 * i) % 1.1] for i in range(20)]
    >>> y = [1.0 + (1 + 0.2 * P[i][0]) * X[i][0] - X[i][1] + 0.1 * math.cos(3 * i) for i in range(20)]
    >>> r = mgwrfit(y, X, P, bandwidths=[6.0, 3.0, 8.0], kernel="gaussian")
    >>> round(r["trS"], 8)
    3.4050126
    """
    yv, Xm, D = _prep(y, X, coords)
    n, p = len(yv), len(Xm[0])
    select = bandwidths is None
    beta, resid, rss, it, bws, R, cols = _backfit(
        yv, Xm, D, bandwidths, kernel, adaptive, threshold, max_iter, select, True
    )
    trS = ssum(cols[k][i] * R[k][i][i] for k in range(p) for i in range(n))
    s2 = rss / (n - trS)
    se = [[math.sqrt(s2 * ssum(v * v for v in R[k][i])) for k in range(p)] for i in range(n)]
    aicc = n * math.log(rss / n) + n * math.log(2.0 * math.pi) + n * (n + trS) / (n - 2.0 - trS)
    return RichResult(
        payload={
            "betas": [[beta[k][i] for k in range(p)] for i in range(n)],
            "se": se,
            "fitted": [yv[i] - resid[i] for i in range(n)],
            "residuals": resid,
            "hat_diagonal": [ssum(cols[k][i] * R[k][i][i] for k in range(p)) for i in range(n)],
            "enp": [ssum(cols[k][i] * R[k][i][i] for i in range(n)) for k in range(p)],
            "trS": trS,
            "rss": rss,
            "sigma2": s2,
            "aicc": aicc,
            "bandwidths": bws,
            "iterations": it,
        }
    )


mgwrfit_fn = mgwrfit


def cheatsheet() -> str:
    return "mgwrfit(y, X, coords, bandwidths=None) -> MGWR backfitting with hat matrices, SEs, ENP, AICc."
