# morie.fn -- function file (rootcoder007/morie)
"""Shared kernels of the spatial-model diagnostic front ends (``sar*``, ``sem*``, ``sac*``, ``sdm*``, ``slx*``).

Log-Jacobians, parameter-space bounds, the analytic information matrix of
the lag / error / SAC models, residual bootstraps, simulated impacts, the
LM error test, residual Moran's I and the common-factor Wald test.  The R
twins live in ``R/SarDiagnostics.R`` (helpers ``.sxd_*``).
"""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, solve, ssum
from ._rng import random_uniform
from ._rrng_core import pchisq
from .sarreg import _brent, _logdet, _mv

__all__: list = []


def _mat(A):
    A = A.tolist() if hasattr(A, "tolist") else A
    return [[float(v) for v in r] for r in A]


def _vec(v):
    v = v.tolist() if hasattr(v, "tolist") else v
    if isinstance(v, (int, float)):
        return [float(v)]
    return [float(t) for t in v]


def _eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _mm(A, B):
    Bt = list(zip(*B))
    return [[ssum(a * b for a, b in zip(r, c)) for c in Bt] for r in A]


def _cols(X):
    return [list(c) for c in zip(*X)]


def _imw(W, a):
    n = len(W)
    return [[(1.0 if i == j else 0.0) - a * W[i][j] for j in range(n)] for i in range(n)]


def _upper_p(stat, df):
    return 1.0 - float(pchisq(stat, df))


def logdet(W, a):
    """``log|det(I - a W)|`` by LU (0 when ``a = 0``)."""
    return _logdet(_mat(W), float(a)) if a else 0.0


def bounds(W):
    """``(1 / min Re(ev), 1 / max Re(ev))`` over the eigenvalues of ``W`` (spatialreg's search interval)."""
    ev = [complex(v).real for v in np.linalg.eigvals(np.asarray(_mat(W), dtype=float)).tolist()]
    return 1.0 / min(ev), 1.0 / max(ev)


def information(X, W, beta, rho, lam, s2, lag, err):
    """Analytic information matrix, parameter order ``(beta, rho?, lambda?, sigma^2)``.

    With ``A = I - rho W``, ``B = I - lambda W``, ``W_A = W A^{-1}``,
    ``W_B = W B^{-1}``, ``C = B W_A B^{-1}`` and ``g = B W_A X beta``
    (Anselin 1988, ch. 6; Lee 2004): ``I_bb = (BX)'BX / s2``, ``I_b rho =
    (BX)'g / s2``, ``I_rho rho = tr(CC) + tr(C'C) + g'g / s2``,
    ``I_rho lam = tr(W_B C) + tr(W_B' C)``, ``I_lam lam = tr(W_B W_B) +
    tr(W_B' W_B)``, ``I_rho s2 = tr(W_A) / s2``, ``I_lam s2 = tr(W_B) / s2``,
    ``I_s2 s2 = n / (2 s2^2)``.
    """
    Xm, Wm = _mat(X), _mat(W)
    n, p = len(Xm), len(Xm[0])
    b = _vec(beta) if lag else [0.0] * p
    Bm = _imw(Wm, lam if err else 0.0)
    Binv = inverse(Bm) if err else _eye(n)
    BX = _mm(Bm, Xm)
    k = p + int(lag) + int(err) + 1
    info = [[0.0] * k for _ in range(k)]
    for u in range(p):
        for v in range(p):
            info[u][v] = ssum(r[u] * r[v] for r in BX) / s2
    ir = p
    il = p + int(lag)
    tr = lambda M1, M2: ssum(M1[i][m] * M2[m][i] for i in range(n) for m in range(n))  # noqa: E731
    trt = lambda M1, M2: ssum(M1[m][i] * M2[m][i] for i in range(n) for m in range(n))  # noqa: E731
    if lag:
        WA = _mm(Wm, inverse(_imw(Wm, rho)))
        C = _mm(_mm(Bm, WA), Binv) if err else WA
        Xb = [ssum(a * c for a, c in zip(r, b)) for r in Xm]
        g = _mv(Bm, _mv(WA, Xb))
        for u in range(p):
            info[u][ir] = info[ir][u] = ssum(r[u] * t for r, t in zip(BX, g)) / s2
        info[ir][ir] = tr(C, C) + trt(C, C) + ssum(t * t for t in g) / s2
        info[ir][k - 1] = info[k - 1][ir] = ssum(WA[i][i] for i in range(n)) / s2
    if err:
        WB = _mm(Wm, Binv)
        info[il][il] = tr(WB, WB) + trt(WB, WB)
        info[il][k - 1] = info[k - 1][il] = ssum(WB[i][i] for i in range(n)) / s2
        if lag:
            info[ir][il] = info[il][ir] = tr(WB, C) + trt(WB, C)
    info[k - 1][k - 1] = n / (2.0 * s2 * s2)
    return info


def covariance(X, W, beta, rho, lam, s2, lag, err):
    info = information(X, W, beta, rho, lam, s2, lag, err)
    V = [[float(v) for v in r] for r in inverse(info)]
    return info, V


def fit(y, X, W, model, interval=(-0.999, 0.999)):
    """Profile-likelihood estimates ``(beta, rho, lambda, sigma2)`` (no standard errors)."""
    from .sarreg import _profile

    lo, hi = interval
    n = len(y)

    def nll(rho, lam):
        _, s2 = _profile(y, X, W, rho, lam)
        v = 0.5 * n * math.log(2 * math.pi * s2) + 0.5 * n
        if rho != 0.0:
            v -= _logdet(W, rho)
        if lam != 0.0:
            v -= _logdet(W, lam)
        return v

    def score(rho, lam, which):
        # d(-log L)/d(which) by the envelope theorem: beta, sigma^2 profiled out
        b, _ = _profile(y, X, W, rho, lam)
        Wy = _mv(W, y)
        Ay = [a - rho * c for a, c in zip(y, Wy)]
        r = [a - ssum(u * v for u, v in zip(row, b)) for a, row in zip(Ay, X)]
        e = [a - lam * c for a, c in zip(r, _mv(W, r))]
        if which == 0:
            d = [a - lam * c for a, c in zip(Wy, _mv(W, Wy))]
            a_ = rho
        else:
            d = _mv(W, r)
            a_ = lam
        Mi = inverse(_imw(W, a_))
        tr = ssum(W[i][m] * Mi[m][i] for i in range(n) for m in range(n))
        return -n * ssum(u * v for u, v in zip(e, d)) / ssum(u * u for u in e) + tr

    def polish(x, other, which):
        # Newton steps on the analytic score: Brent's minimum is only accurate to ~sqrt(eps)
        for _ in range(4):
            args = (lambda t: (t, other)) if which == 0 else (lambda t: (other, t))
            s0 = score(*args(x), which)
            h = 1e-5
            ds = (score(*args(x + h), which) - score(*args(x - h), which)) / (2 * h)
            if ds <= 0.0:
                break
            x1 = x - s0 / ds
            if not lo < x1 < hi:
                break
            step, x = abs(x1 - x), x1
            if step < 1e-15:
                break
        return x

    rho = lam = 0.0
    if model == "lag":
        rho = polish(_brent(lambda a: nll(a, 0.0), lo, hi)[0], 0.0, 0)
    elif model == "error":
        lam = polish(_brent(lambda a: nll(0.0, a), lo, hi)[0], 0.0, 1)
    else:
        f = nll(0.0, 0.0)
        for _ in range(200):
            rho = polish(_brent(lambda a, lam=lam: nll(a, lam), lo, hi)[0], lam, 0)
            lam = polish(_brent(lambda a, rho=rho: nll(rho, a), lo, hi)[0], rho, 1)
            f_new = nll(rho, lam)
            if abs(f - f_new) < 1e-13 * max(1.0, abs(f_new)):
                break
            f = f_new
    beta, s2 = _profile(y, X, W, rho, lam)
    return beta, rho, lam, s2


def quantile7(x, q):
    s = sorted(x)
    h = (len(s) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def sd(x):
    m = ssum(x) / len(x)
    return math.sqrt(ssum((v - m) ** 2 for v in x) / (len(x) - 1))


def durbin(X, W):
    """``[X | W X_*]`` with ``X_*`` the non-constant columns, and their indices."""
    Xm, Wm = _mat(X), _mat(W)
    cols = _cols(Xm)
    lagc = [k for k in range(len(cols)) if max(cols[k]) - min(cols[k]) > 0.0]
    WX = [_mv(Wm, cols[k]) for k in lagc]
    return [list(r) + [WX[j][i] for j in range(len(lagc))] for i, r in enumerate(Xm)], lagc


def bootstrap(y, X, W, model, B, seed, level, interval=(-0.999, 0.999)):
    """Residual bootstrap of the autoregressive parameter of a lag / error / SAC model.

    Fit, take the innovations ``e = B(Ay - X beta)`` centred at their mean,
    resample them with replacement (Philox uniforms, index ``floor(u n)``),
    rebuild ``y* = A^{-1}(X beta + B^{-1} e*)`` and refit; the interval is the
    ``(1 - level)/2`` and ``(1 + level)/2`` type-7 quantiles of the refits.
    """
    yv, Xm, Wm = _vec(y), _mat(X), _mat(W)
    n = len(yv)
    beta, rho, lam, _s2 = fit(yv, Xm, Wm, model, interval)
    Xb = [ssum(a * c for a, c in zip(r, beta)) for r in Xm]
    Ay = [a - rho * c for a, c in zip(yv, _mv(Wm, yv))]
    r0 = [a - c for a, c in zip(Ay, Xb)]
    e = [a - lam * c for a, c in zip(r0, _mv(Wm, r0))]
    mu = ssum(e) / n
    e = [v - mu for v in e]
    Ainv = inverse(_imw(Wm, rho))
    Binv = inverse(_imw(Wm, lam))
    u = random_uniform(B * n, seed=seed)
    stats = []
    for b in range(B):
        es = [e[min(int(u[b * n + i] * n), n - 1)] for i in range(n)]
        v = [a + c for a, c in zip(Xb, _mv(Binv, es))]
        ys = _mv(Ainv, v)
        _bb, rb, lb, _sb = fit(ys, Xm, Wm, model, interval)
        stats.append((rb, lb))
    return (beta, rho, lam), stats, level


def ci_summary(est, draws, level):
    a = (1.0 - level) / 2.0
    return {
        "estimate": est,
        "ci_lower": quantile7(draws, a),
        "ci_upper": quantile7(draws, 1.0 - a),
        "se_boot": sd(draws),
        "draws": draws,
    }


def chol(V):
    """Lower Cholesky factor (plain loops, as the R twin)."""
    k = len(V)
    L = [[0.0] * k for _ in range(k)]
    for j in range(k):
        s = V[j][j] - ssum(L[j][m] * L[j][m] for m in range(j))
        if s <= 0.0:
            raise ValueError("vcov is not positive definite")
        L[j][j] = math.sqrt(s)
        for i in range(j + 1, k):
            L[i][j] = (V[i][j] - ssum(L[i][m] * L[j][m] for m in range(j))) / L[j][j]
    return L


def lm_error(resid, W):
    """``RSerr = (n e'We / e'e)^2 / tr(W'W + WW)`` (Burridge 1980; Anselin 1988)."""
    e, Wm = _vec(resid), _mat(W)
    n = len(e)
    T = ssum(Wm[i][j] * (Wm[i][j] + Wm[j][i]) for i in range(n) for j in range(n))
    s = n * ssum(a * b for a, b in zip(e, _mv(Wm, e))) / ssum(a * a for a in e)
    stat = s * s / T
    return stat, _upper_p(stat, 1)


def resid_moran(resid, W, X=None):
    """Moran's I of residuals; exact moments (``spdep::lm.morantest``) when ``X`` is given."""
    from .spdurbin import residual_moran

    e, Wm = _vec(resid), _mat(W)
    if X is not None:
        r = residual_moran(e, _mat(X), Wm)
        return {"I": r.I, "expected": r.expected, "variance": r.variance, "z": r.z, "p_value": r.pvalue}
    n = len(e)
    S0 = ssum(v for row in Wm for v in row)
    stat = n / S0 * ssum(a * b for a, b in zip(e, _mv(Wm, e))) / ssum(a * a for a in e)
    return {"I": stat, "expected": None, "variance": None, "z": None, "p_value": None}


def common_factor(beta, theta, vcov, rho):
    """Wald test of ``theta_r + rho beta_r = 0`` (Burridge 1981) by the delta method.

    ``vcov`` is the covariance of ``(rho, beta_1..k, theta_1..k)``; the
    Jacobian row ``r`` of ``g = theta + rho beta`` is ``beta_r`` in the
    ``rho`` column, ``rho`` in the ``beta_r`` column and 1 in ``theta_r``.
    """
    b, t, V = _vec(beta), _vec(theta), _mat(vcov)
    k = len(b)
    if len(t) != k or len(V) != 2 * k + 1:
        raise ValueError("need len(theta) == len(beta) and vcov of size 2k + 1 (rho, beta, theta)")
    g = [t[r] + rho * b[r] for r in range(k)]
    G = [[0.0] * (2 * k + 1) for _ in range(k)]
    for r in range(k):
        G[r][0] = b[r]
        G[r][1 + r] = rho
        G[r][1 + k + r] = 1.0
    GV = _mm(G, V)
    S = [[ssum(GV[i][m] * G[j][m] for m in range(2 * k + 1)) for j in range(k)] for i in range(k)]
    stat = ssum(a * c for a, c in zip(g, solve(S, g)))
    return stat, k, _upper_p(stat, k), g
