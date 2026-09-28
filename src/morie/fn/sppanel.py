# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel data models by maximum likelihood (Elhorst 2003, 2014): the spatial lag, spatial
error and spatial Durbin models with individual, time or two-way fixed effects (or pooled), the
Lee-Yu bias correction of the variance, and a random-effects spatial lag model; plus information
criteria for fitted spatial models.

Rows are stacked by period: the first ``N`` rows are period 1 (units in the order of ``W``), the
next ``N`` period 2, and so on."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, solve, ssum
from ._richresult import RichResult

__all__ = ["spatial_panel_ml", "spatial_panel_re_lag", "information_criteria"]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _mat(X):
    a = np.asarray(X, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return [[float(v) for v in r] for r in a.tolist()]


def _logdet(A):
    n = len(A)
    M = [list(r) for r in A]
    ld = 0.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if p != c:
            M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        if piv == 0.0:
            return -math.inf
        ld += math.log(abs(piv))
        for r in range(c + 1, n):
            f = M[r][c] / piv
            if f != 0.0:
                for k in range(c, n):
                    M[r][k] -= f * M[c][k]
    return ld


def _lag(W, v, N, T):
    out = []
    for t in range(T):
        blk = v[t * N : (t + 1) * N]
        out += [ssum(W[i][j] * blk[j] for j in range(N)) for i in range(N)]
    return out


def _demean(v, N, T, effects):
    unit = [ssum(v[t * N + i] for t in range(T)) / T for i in range(N)]
    per = [ssum(v[t * N + i] for i in range(N)) / N for t in range(T)]
    tot = ssum(v) / (N * T)
    out = []
    for t in range(T):
        for i in range(N):
            x = v[t * N + i]
            if effects in ("individual", "twoways"):
                x -= unit[i]
            if effects in ("time", "twoways"):
                x -= per[t]
            if effects == "twoways":
                x += tot
            out.append(x)
    return out


def _ols(X, y):
    k = len(X[0])
    beta = solve(
        [[ssum(r[a] * r[b] for r in X) for b in range(k)] for a in range(k)],
        [ssum(r[a] * v for r, v in zip(X, y)) for a in range(k)],
    )
    return beta, [v - ssum(r[a] * beta[a] for a in range(k)) for r, v in zip(X, y)]


def _golden(f, lo, hi, tol=1e-12):
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol * (abs(c) + abs(d) + 1e-3):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def _trace_aw(W, r):
    # tr((I - r W)^{-1} W), the derivative of log|I - r W| is minus this
    n = len(W)
    Ainv = inverse([[(1.0 if i == j else 0.0) - r * W[i][j] for j in range(n)] for i in range(n)])
    return ssum(ssum(Ainv[i][k] * W[k][i] for k in range(n)) for i in range(n))


def _polish(df, x, lo, hi):
    # refine a golden-section maximiser by bisection on the sign of the derivative
    h = 1e-4 * max(1.0, abs(x))
    a, b = max(lo, x - h), min(hi, x + h)
    fa, fb = df(a), df(b)
    if not (fa > 0 > fb):
        return x
    for _ in range(200):
        m = (a + b) / 2.0
        if m <= a or m >= b:
            break
        if df(m) > 0:
            a = m
        else:
            b = m
    return (a + b) / 2.0


def _cols(X):
    return [list(c) for c in zip(*X)]


def spatial_panel_ml(y, X, W, n_units, model="lag", effects="individual", lee_yu=False, interval=(-0.99, 0.99)):
    r"""Maximum likelihood spatial panel models with fixed effects (Elhorst).

    The data are demeaned for ``effects`` (``"individual"``, ``"time"``,
    ``"twoways"`` or ``"pooled"`` with an intercept column added). ``model="lag"``:
    ``y = rho W y + X beta + e``, with the concentrated log-likelihood
    ``-NT/2 log((e0 - rho e1)'(e0 - rho e1)) + T log|I - rho W|`` from the OLS
    residuals ``e0`` of ``y`` and ``e1`` of ``W y`` on ``X``; ``"error"``: ``y = X
    beta + u``, ``u = lambda W u + e``, maximising ``-NT/2 log SSE(lambda) + T
    log|I - lambda W|`` with ``beta(lambda)`` by GLS; ``"durbin"``: the lag
    model with ``W X`` added. The parameter is found by golden-section search
    on ``interval`` and polished by bisection on the score (``d log|I - rho W| /
    d rho = -tr((I - rho W)^{-1} W)``); ``sigma2 = SSE / NT``, and ``lee_yu`` applies the Lee and Yu
    (2010) variance correction ``T/(T-1)`` (individual) or ``N/(N-1)`` (time).
    ``loglik`` is the full Gaussian log-likelihood at the optimum, as
    reported by ``splm::spml``.

    References
    ----------
    Elhorst, J. P. (2003). Specification and estimation of spatial panel data
    models. *International Regional Science Review* 26, 244-268.

    Lee, L.-F. and Yu, J. (2010). Estimation of spatial autoregressive panel
    data models with fixed effects. *Journal of Econometrics* 154, 165-185.

    Examples
    --------
    >>> W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    >>> y = [1.0, 2.0, 1.5, 1.2, 2.4, 1.1, 0.9, 2.2, 1.8]
    >>> X = [[0.5], [1.0], [0.2], [0.7], [1.3], [0.1], [0.4], [0.9], [0.6]]
    >>> r = spatial_panel_ml(y, X, W, 3)
    >>> round(r.rho, 6), [round(b, 6) for b in r.coefficients]
    (-0.167868, [0.999912])
    """
    yv, Xm, Wm = _vec(y), _mat(X), _mat(W)
    N = int(n_units)
    T = len(yv) // N
    if model == "durbin":
        wx = [_lag(Wm, c, N, T) for c in _cols(Xm)]
        Xm = [r + [c[i] for c in wx] for i, r in enumerate(Xm)]
    if effects == "pooled":
        Xm = [[1.0] + r for r in Xm]
        yt = yv
        Xt = Xm
    else:
        yt = _demean(yv, N, T, effects)
        Xt = _cols([_demean(c, N, T, effects) for c in _cols(Xm)])
    wyt = _lag(Wm, yt, N, T)
    NT = N * T

    def ldet(r):
        return _logdet([[(1.0 if i == j else 0.0) - r * Wm[i][j] for j in range(N)] for i in range(N)])

    if model in ("lag", "durbin"):
        _, e0 = _ols(Xt, yt)
        _, e1 = _ols(Xt, wyt)
        a, b, c = ssum(v * v for v in e0), ssum(u * v for u, v in zip(e0, e1)), ssum(v * v for v in e1)

        def f(r):
            return -NT / 2.0 * math.log(a - 2 * r * b + r * r * c) + T * ldet(r)

        rho = _golden(f, interval[0], interval[1])
        rho = _polish(lambda r: NT * (b - r * c) / (a - 2 * r * b + r * r * c) - T * _trace_aw(Wm, r), rho, *interval)
        beta, res = _ols(Xt, [u - rho * v for u, v in zip(yt, wyt)])
    elif model == "error":
        wxt = _cols([_lag(Wm, cc, N, T) for cc in _cols(Xt)])

        def fit(r):
            ys = [u - r * v for u, v in zip(yt, wyt)]
            xs = [[p - r * q for p, q in zip(u, v)] for u, v in zip(Xt, wxt)]
            return _ols(xs, ys)

        def f(r):
            return -NT / 2.0 * math.log(ssum(v * v for v in fit(r)[1])) + T * ldet(r)

        rho = _golden(f, interval[0], interval[1])

        def df(r):
            bt, e = fit(r)
            g = [u - ssum(q[k] * bt[k] for k in range(len(bt))) for u, q in zip(wyt, wxt)]
            return NT * ssum(p * q for p, q in zip(e, g)) / ssum(v * v for v in e) - T * _trace_aw(Wm, r)

        rho = _polish(df, rho, *interval)
        beta, res = fit(rho)
    else:
        raise ValueError("model must be 'lag', 'error' or 'durbin'")
    sse = ssum(v * v for v in res)
    s2 = sse / NT
    ll = T * ldet(rho) - NT / 2.0 * math.log(2 * math.pi) - NT / 2.0 * math.log(s2) - sse / (2 * s2)
    if lee_yu:
        if effects == "individual":
            s2 = T / (T - 1.0) * s2
        elif effects == "time":
            s2 = N / (N - 1.0) * s2
    return RichResult(
        payload={"rho": rho, "coefficients": beta, "sigma2": s2, "loglik": ll, "residuals": res, "n_obs": NT}
    )


def spatial_panel_re_lag(y, X, W, n_units, interval=(-0.99, 0.99), tol=1e-10, maxit=500):
    r"""Random-effects spatial lag panel model by maximum likelihood (Elhorst 2003).

    ``y = rho W y + X beta + mu + e`` with ``mu_i ~ N(0, sigma_mu^2)``. With
    ``phi^2 = sigma^2 / (T sigma_mu^2 + sigma^2)`` the data are quasi-demeaned
    ``z* = z - (1 - phi) zbar_i``; given ``phi``, ``(rho, beta, sigma^2)``
    follow the pooled lag model on the transformed data, and given those,
    ``phi`` maximises ``-NT/2 log e'e + N/2 log phi^2`` with ``e`` the
    transformed residuals, i.e. ``phi^2 = min(1, e'Qe / ((T - 1) e'Pe))``
    (Breusch 1987). The two steps alternate until ``phi`` changes by
    less than ``tol``. ``X`` gets an intercept.

    References
    ----------
    Elhorst, J. P. (2003). Specification and estimation of spatial panel data
    models. *International Regional Science Review* 26, 244-268.

    Examples
    --------
    >>> W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    >>> y = [1.0, 2.0, 1.5, 1.2, 2.4, 1.1, 0.9, 2.2, 1.8]
    >>> X = [[0.5], [1.0], [0.2], [0.7], [1.3], [0.1], [0.4], [0.9], [0.6]]
    >>> r = spatial_panel_re_lag(y, X, W, 3)
    >>> round(r.phi, 6)
    0.24825
    """
    yv, Xm, Wm = _vec(y), [[1.0] + r for r in _mat(X)], _mat(W)
    N = int(n_units)
    T = len(yv) // N
    NT = N * T
    wy = _lag(Wm, yv, N, T)

    def bar(v):
        return [ssum(v[t * N + i] for t in range(T)) / T for i in range(N)]

    def tr(v, phi):
        m = bar(v)
        return [v[t * N + i] - (1 - phi) * m[i] for t in range(T) for i in range(N)]

    def ldet(r):
        return _logdet([[(1.0 if i == j else 0.0) - r * Wm[i][j] for j in range(N)] for i in range(N)])

    phi = 1.0
    for _ in range(maxit):
        ys, wys = tr(yv, phi), tr(wy, phi)
        Xs = _cols([tr(c, phi) for c in _cols(Xm)])
        _, e0 = _ols(Xs, ys)
        _, e1 = _ols(Xs, wys)
        a, b, c = ssum(v * v for v in e0), ssum(u * v for u, v in zip(e0, e1)), ssum(v * v for v in e1)
        rho = _golden(
            lambda r, a=a, b=b, c=c: -NT / 2.0 * math.log(a - 2 * r * b + r * r * c) + T * ldet(r),
            interval[0],
            interval[1],
        )
        rho = _polish(
            lambda r, a=a, b=b, c=c: NT * (b - r * c) / (a - 2 * r * b + r * r * c) - T * _trace_aw(Wm, r),
            rho,
            *interval,
        )
        beta, _ = _ols(Xs, [u - rho * v for u, v in zip(ys, wys)])
        d = [u - rho * v - ssum(r[k] * beta[k] for k in range(len(beta))) for u, v, r in zip(yv, wy, Xm)]
        md = bar(d)
        q = ssum((d[t * N + i] - md[i]) ** 2 for t in range(T) for i in range(N))
        pq = T * ssum(v * v for v in md)

        new = min(1.0, math.sqrt(q / ((T - 1) * pq)))
        if abs(new - phi) < tol:
            phi = new
            break
        phi = new
    ys, wys = tr(yv, phi), tr(wy, phi)
    Xs = _cols([tr(c, phi) for c in _cols(Xm)])
    beta, res = _ols(Xs, [u - rho * v for u, v in zip(ys, wys)])
    s2 = ssum(v * v for v in res) / NT
    ll = (
        T * ldet(rho)
        + N / 2.0 * math.log(phi * phi)
        - NT / 2.0 * math.log(2 * math.pi * s2)
        - ssum(v * v for v in res) / (2 * s2)
    )
    return RichResult(
        payload={
            "rho": rho,
            "coefficients": beta,
            "phi": phi,
            "sigma2": s2,
            "sigma2_mu": (1 / (phi * phi) - 1) * s2 / T if phi > 0 else math.inf,
            "loglik": ll,
        }
    )


def information_criteria(loglik, k, n):
    r"""Akaike and Schwarz information criteria of a fitted model.

    ``AIC = -2 logL + 2k``, ``AICc = AIC + 2k(k + 1)/(n - k - 1)`` and ``BIC =
    -2 logL + k log n`` for ``k`` estimated parameters (count the spatial
    parameter and the variance) and ``n`` observations.

    References
    ----------
    Akaike, H. (1974). A new look at the statistical model identification.
    *IEEE Transactions on Automatic Control* 19, 716-723.

    Schwarz, G. (1978). Estimating the dimension of a model. *Annals of
    Statistics* 6, 461-464.

    Examples
    --------
    >>> r = information_criteria(-120.5, 4, 100)
    >>> r.aic, round(r.bic, 10)
    (249.0, 259.420680744)
    """
    aic = -2.0 * loglik + 2.0 * k
    return RichResult(
        payload={"aic": aic, "aicc": aic + 2.0 * k * (k + 1) / (n - k - 1), "bic": -2.0 * loglik + k * math.log(n)}
    )


def cheatsheet() -> str:
    return "spatial_panel_ml / spatial_panel_re_lag / information_criteria -> spatial panel models."
