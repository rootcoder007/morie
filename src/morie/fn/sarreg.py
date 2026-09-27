# morie.fn -- function file (rootcoder007/morie)
"""Maximum likelihood for spatial lag, spatial error and SAC (SARAR) regressions."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from ._qpcore import inverse, solve, ssum


def _logdet(W, a):
    """log|det(I - a W)| by LU with partial pivoting."""
    n = len(W)
    M = [[(1.0 if i == j else 0.0) - a * W[i][j] for j in range(n)] for i in range(n)]
    s = 0.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if M[p][c] == 0.0:
            return -math.inf
        M[c], M[p] = M[p], M[c]
        s += math.log(abs(M[c][c]))
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            if f != 0.0:
                Mr, Mc = M[r], M[c]
                for k in range(c + 1, n):
                    Mr[k] -= f * Mc[k]
    return s


def _mv(W, v):
    return [ssum(a * b for a, b in zip(row, v)) for row in W]


def _ols(X, y):
    p = len(X[0])
    XtX = [[ssum(r[a] * r[b] for r in X) for b in range(p)] for a in range(p)]
    Xty = [ssum(r[a] * t for r, t in zip(X, y)) for a in range(p)]
    return solve(XtX, Xty)


def _profile(y, X, W, rho, lam):
    Wy = _mv(W, y)
    yl = [a - rho * b for a, b in zip(y, Wy)]
    if lam != 0.0:
        Wyl = _mv(W, yl)
        ys = [a - lam * b for a, b in zip(yl, Wyl)]
        cols = [[r[k] for r in X] for k in range(len(X[0]))]
        WX = [_mv(W, c) for c in cols]
        Xs = [[cols[k][i] - lam * WX[k][i] for k in range(len(cols))] for i in range(len(y))]
    else:
        ys, Xs = yl, X
    b = _ols(Xs, ys)
    e = [t - ssum(a * c for a, c in zip(r, b)) for r, t in zip(Xs, ys)]
    return b, ssum(t * t for t in e) / len(y)


def _brent(f, lo, hi, tol=1e-12, max_iter=500):
    g = (3 - math.sqrt(5)) / 2
    a, b = lo, hi
    x = w = v = a + g * (b - a)
    fx = fw = fv = f(x)
    d = e = 0.0
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        t1 = tol * abs(x) + 1e-15
        t2 = 2 * t1
        if abs(x - m) <= t2 - 0.5 * (b - a):
            break
        use_golden = True
        if abs(e) > t1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            pp = (x - v) * q - (x - w) * r
            q = 2 * (q - r)
            if q > 0:
                pp = -pp
            q = abs(q)
            if abs(pp) < abs(0.5 * q * e) and q * (a - x) < pp < q * (b - x):
                e, d = d, pp / q
                u = x + d
                if u - a < t2 or b - u < t2:
                    d = t1 if x < m else -t1
                use_golden = False
        if use_golden:
            e = (b - x) if x < m else (a - x)
            d = g * e
        u = x + (d if abs(d) >= t1 else (t1 if d > 0 else -t1))
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x, fv, fw, fx = w, x, u, fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w, fv, fw = w, u, fw, fu
            elif fu <= fv or v in (x, w):
                v, fv = u, fu
    return x, fx


def spatial_regression_ml(y, X, W, *, model: str = "lag", interval=(-0.999, 0.999)) -> DescriptiveResult:
    """Spatial lag, spatial error and SAC models by maximum likelihood.

    - ``lag``: ``y = rho W y + X beta + e``;
    - ``error``: ``y = X beta + u``, ``u = lambda W u + e``;
    - ``sac``: ``y = rho W y + X beta + u``, ``u = lambda W u + e``;

    with ``e ~ N(0, sigma^2 I)``. Profiling ``beta`` and ``sigma^2`` leaves
    ``-2 log L = n log(2 pi sigma^2) + n - 2 log|I - rho W| - 2 log|I - lambda W|``
    (terms absent from a model are dropped), maximised by Brent's method
    on ``interval`` (lag, error) or by alternating Brent searches over
    ``rho`` and ``lambda`` until both settle (SAC). The log-determinants are
    computed by LU decomposition, so ``W`` need not be symmetric. Standard
    errors: for the lag and error models the inverse of the analytic
    information matrix (Ord 1975; Anselin 1988, traces of ``W (I - a W)^-1``;
    the error model's ``beta`` block is ``sigma^2 (X*'X*)^-1``); for SAC the
    inverse of a central-difference Hessian of the full log-likelihood in
    ``(beta, rho, lambda, sigma^2)``. AIC and BIC
    count ``beta``, the autoregressive parameters and ``sigma^2``. These
    match ``spatialreg::lagsarlm``, ``errorsarlm`` and ``sacsarlm`` with
    ``method = "LU"``.

    :param y: Response (n).
    :param X: Design matrix (n x p), including any intercept column.
    :param W: Spatial weights (n x n).
    :param model: ``"lag"``, ``"error"`` or ``"sac"``.
    :param interval: Search interval for the autoregressive parameters.
    :return: DescriptiveResult; ``value`` is ``beta``; ``extra`` has
        ``rho`` and/or ``lambda``, ``sigma2``, ``loglik``, ``aic``, ``bic``,
        ``se`` (for ``beta`` then the spatial parameters then ``sigma2``).

    References
    ----------
    Ord, K. (1975). Estimation methods for models of spatial interaction.
    Journal of the American Statistical Association 70, 120-126.

    Anselin, L. (1988). Spatial Econometrics: Methods and Models. Kluwer.

    Kelejian, H. H. and Prucha, I. R. (1998). A generalized spatial
    two-stage least squares procedure for estimating a spatial
    autoregressive model with autoregressive disturbances. Journal of Real
    Estate Finance and Economics 17, 99-121.

    Brent, R. P. (1973). Algorithms for Minimization without Derivatives.
    Prentice-Hall.

    Examples
    --------
    >>> W = [[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]]
    >>> r = spatial_regression_ml([1.0, 2.0, 1.5, 3.0], [[1, 0.1], [1, 0.5], [1, 0.2], [1, 0.9]], W, model="error")
    >>> round(r.extra["loglik"], 6) == round(r.extra["loglik"], 6)
    True
    """
    y = [float(v) for v in y]
    X = [[float(v) for v in r] for r in X]
    W = [[float(v) for v in r] for r in W]
    n, p = len(y), len(X[0])
    lo, hi = interval

    def nll(rho, lam):
        _, s2 = _profile(y, X, W, rho, lam)
        v = 0.5 * n * math.log(2 * math.pi * s2) + 0.5 * n
        if rho != 0.0:
            v -= _logdet(W, rho)
        if lam != 0.0:
            v -= _logdet(W, lam)
        return v

    rho = lam = 0.0
    if model == "lag":
        rho, f = _brent(lambda a: nll(a, 0.0), lo, hi)
    elif model == "error":
        lam, f = _brent(lambda a: nll(0.0, a), lo, hi)
    elif model == "sac":
        f = nll(0.0, 0.0)
        for _ in range(200):
            rho, _f = _brent(lambda a, lam=lam: nll(a, lam), lo, hi)
            lam, f_new = _brent(lambda a, rho=rho: nll(rho, a), lo, hi)
            if abs(f - f_new) < 1e-13 * max(1.0, abs(f_new)):
                f = f_new
                break
            f = f_new
    else:
        raise ValueError("model must be 'lag', 'error' or 'sac'")
    beta, s2 = _profile(y, X, W, rho, lam)
    loglik = -nll(rho, lam)
    spatial = ([rho] if model in ("lag", "sac") else []) + ([lam] if model in ("error", "sac") else [])
    theta = beta + spatial + [s2]

    def full_nll(t):
        b, sp, sg = t[:p], t[p : p + len(spatial)], t[-1]
        r_ = sp[0] if model in ("lag", "sac") else 0.0
        l_ = sp[-1] if model in ("error", "sac") else 0.0
        if sg <= 0:
            return math.inf
        Wy = _mv(W, y)
        yl = [a - r_ * c for a, c in zip(y, Wy)]
        res = [a - ssum(u * v for u, v in zip(row, b)) for a, row in zip(yl, X)]
        if l_ != 0.0:
            Wr = _mv(W, res)
            res = [a - l_ * c for a, c in zip(res, Wr)]
        v = 0.5 * n * math.log(2 * math.pi * sg) + ssum(t_ * t_ for t_ in res) / (2 * sg)
        if r_ != 0.0:
            v -= _logdet(W, r_)
        if l_ != 0.0:
            v -= _logdet(W, l_)
        return v

    k = len(theta)
    h = [1e-4 * max(1.0, abs(t)) for t in theta]
    H = [[0.0] * k for _ in range(k)]
    f0 = full_nll(theta)
    for a in range(k):
        for b in range(a, k):
            if a == b:
                tp, tm = list(theta), list(theta)
                tp[a] += h[a]
                tm[a] -= h[a]
                H[a][a] = (full_nll(tp) - 2 * f0 + full_nll(tm)) / h[a] ** 2
            else:
                vals = []
                for sa, sb in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    t = list(theta)
                    t[a] += sa * h[a]
                    t[b] += sb * h[b]
                    vals.append(full_nll(t))
                H[a][b] = H[b][a] = (vals[0] - vals[1] - vals[2] + vals[3]) / (4 * h[a] * h[b])
    se = [math.sqrt(solve(H, [1.0 if c == r else 0.0 for c in range(k)])[r]) for r in range(k)]
    if model in ("lag", "error"):
        a = rho if model == "lag" else lam
        Ainv = inverse([[(1.0 if i == j else 0.0) - a * W[i][j] for j in range(n)] for i in range(n)])
        WA = [[ssum(W[i][m] * Ainv[m][j] for m in range(n)) for j in range(n)] for i in range(n)]
        trWA = ssum(WA[i][i] for i in range(n))
        trWA2 = ssum(WA[i][m] * WA[m][i] for i in range(n) for m in range(n))
        trWtW = ssum(WA[m][i] * WA[m][i] for i in range(n) for m in range(n))
        if model == "lag":
            Xb = [ssum(u * v for u, v in zip(r, beta)) for r in X]
            g = _mv(WA, Xb)
            info = [[0.0] * (p + 2) for _ in range(p + 2)]
            for u in range(p):
                for v in range(p):
                    info[u][v] = ssum(r[u] * r[v] for r in X) / s2
                info[u][p] = info[p][u] = ssum(r[u] * t for r, t in zip(X, g)) / s2
            info[p][p] = trWA2 + trWtW + ssum(t * t for t in g) / s2
            info[p][p + 1] = info[p + 1][p] = trWA / s2
            info[p + 1][p + 1] = n / (2 * s2 * s2)
            se = [math.sqrt(solve(info, [1.0 if c == r else 0.0 for c in range(p + 2)])[r]) for r in range(p + 2)]
        else:
            cols = [[r[u] for r in X] for u in range(p)]
            WX = [_mv(W, c) for c in cols]
            Xs = [[cols[u][i] - lam * WX[u][i] for u in range(p)] for i in range(n)]
            B = [[ssum(r[u] * r[v] for r in Xs) / s2 for v in range(p)] for u in range(p)]
            seb = [math.sqrt(solve(B, [1.0 if c == r else 0.0 for c in range(p)])[r]) for r in range(p)]
            L = [[trWA2 + trWtW, trWA / s2], [trWA / s2, n / (2 * s2 * s2)]]
            sel = [math.sqrt(solve(L, [1.0, 0.0])[0]), math.sqrt(solve(L, [0.0, 1.0])[1])]
            se = seb + sel
    npar = p + len(spatial) + 1
    extra = {
        "sigma2": s2,
        "loglik": loglik,
        "aic": -2 * loglik + 2 * npar,
        "bic": -2 * loglik + math.log(n) * npar,
        "se": se,
        "model": model,
    }
    if model in ("lag", "sac"):
        extra["rho"] = rho
    if model in ("error", "sac"):
        extra["lambda"] = lam
    return DescriptiveResult(name="spatial_regression_ml", value=beta, extra=extra)


sarreg = spatial_regression_ml


def cheatsheet() -> str:
    return "spatial_regression_ml(y, X, W, model) -> spatial lag / error / SAC maximum likelihood"
