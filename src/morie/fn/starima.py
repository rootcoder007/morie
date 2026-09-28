# morie.fn -- function file (rootcoder007/morie)
"""Space-time autoregressive moving-average (STARMA) models of Pfeifer and Deutsch (1980):
space-time autocorrelation and partial autocorrelation, STAR least-squares and STARMA
conditional-least-squares fits, forecasts and the space-time portmanteau test."""

from __future__ import annotations

import math

from ._qpcore import dot, inverse, matvec, solve, ssum
from ._regression_core import _chi2_sf
from ._richresult import RichResult
from .lbfgsb import lbfgsb_minimize

__all__ = ["st_lag", "st_acf", "st_pacf", "star_fit", "starma_fit", "starma_forecast", "st_portmanteau"]


def st_lag(z, weights, order: int) -> list:
    r"""Spatial lag of order ``order`` of a site vector: ``W^(order) z`` (order 0 is ``z`` itself).

    ``weights`` lists the spatial weight matrices of orders 1, 2, ... (row
    standardised in Pfeifer and Deutsch 1980).

    Examples
    --------
    >>> st_lag([1.0, 2.0, 4.0], [[[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]], 1)
    [2.0, 2.5, 2.0]
    """
    if order == 0:
        return [float(v) for v in z]
    return matvec(weights[order - 1], [float(v) for v in z])


def _center(x):
    T, N = len(x), len(x[0])
    m = ssum(float(v) for row in x for v in row) / (T * N)
    return [[float(v) - m for v in row] for row in x]


def _lagged(x, weights, L):
    return [[st_lag(z, weights, lag) for z in x] for lag in range(L + 1)]


def _stcov(lagged, L, S):
    # gamma[l][h][s] = (1 / (N (T - s))) sum_t (W^l z_t)' (W^h z_{t+s})
    T, N = len(lagged[0]), len(lagged[0][0])
    g = [[[0.0] * (S + 1) for _ in range(L + 1)] for _ in range(L + 1)]
    for lag in range(L + 1):
        for h in range(L + 1):
            for s in range(S + 1):
                g[lag][h][s] = ssum(dot(lagged[lag][t], lagged[h][t + s]) for t in range(T - s)) / (N * (T - s))
    return g


def st_acf(x, weights, *, max_lag_t: int = 10, max_lag_s: int | None = None, center: bool = True) -> RichResult:
    r"""Space-time autocorrelation function ``rho_l0(s)`` (Pfeifer and Deutsch 1980).

    ``x`` is the ``T x N`` field (rows are times). With
    ``gamma_lh(s) = (1 / (N (T - s))) sum_t (W^l z_t)' (W^h z_{t+s})``,
    ``rho_l0(s) = gamma_l0(s) / sqrt(gamma_ll(0) gamma_00(0))``. The field is
    centred by its grand mean when ``center``. Returns ``acf[s][l]`` for
    ``s = 0..max_lag_t`` and ``l = 0..max_lag_s`` (default all weight orders).

    References
    ----------
    Pfeifer, P. E. and Deutsch, S. J. (1980). A three-stage iterative
    procedure for space-time modeling. *Technometrics*, 22(1), 35-47.

    Examples
    --------
    >>> W = [[[0, 1], [1, 0]]]
    >>> r = st_acf([[1, 2], [2, 3], [4, 3], [3, 1]], W, max_lag_t=1)
    >>> [round(v, 12) for v in r.acf[0]], round(r.acf[1][0], 12)
    ([1.0, 0.111111111111], 0.037037037037)
    """
    z = _center(x) if center else [[float(v) for v in row] for row in x]
    L = len(weights) if max_lag_s is None else int(max_lag_s)
    lagged = _lagged(z, weights, L)
    g = _stcov(lagged, L, max_lag_t)
    acf = [
        [g[lag][0][s] / math.sqrt(g[lag][lag][0] * g[0][0][0]) for lag in range(L + 1)] for s in range(max_lag_t + 1)
    ]
    return RichResult(payload={"acf": acf, "gamma": g, "n_sites": len(x[0]), "n_times": len(x)})


def st_pacf(
    x, weights, *, max_lag_t: int = 5, max_lag_s: int | None = None, center: bool = True, method: str = "pfeifer"
) -> RichResult:
    r"""Space-time partial autocorrelation ``phi_kl`` by the space-time Yule-Walker equations (Pfeifer and Deutsch 1980).

    For each time order ``k`` the system
    ``sum_{j=1}^{k} sum_{h=0}^{L} phi_jh gamma_lh(s - j) = gamma_l0(s)``,
    ``s = 1..k``, ``l = 0..L`` (with ``gamma_lh(-m) = gamma_hl(m)``) is solved
    and the last-lag coefficients ``phi_kl`` are reported as ``pacf[k-1][l]``.
    ``method="starma"`` reproduces the starma package, which nests the
    systems lexicographically in ``(k, l)``: ``phi_kl`` comes from the system
    with all spatial orders at time lags below ``k`` but only orders ``0..l``
    at lag ``k`` (the two agree at ``l = L``).

    Examples
    --------
    >>> W = [[[0, 1], [1, 0]]]
    >>> r = st_pacf([[1, 2], [2, 3], [4, 3], [3, 1]], W, max_lag_t=1, max_lag_s=0)
    >>> round(r.pacf[0][0], 12)
    0.037037037037
    """
    z = _center(x) if center else [[float(v) for v in row] for row in x]
    L = len(weights) if max_lag_s is None else int(max_lag_s)
    lagged = _lagged(z, weights, L)
    g = _stcov(lagged, L, max_lag_t)

    def gam(lag, h, s):
        return g[lag][h][s] if s >= 0 else g[h][lag][-s]

    def last_coef(idx):
        # Yule-Walker equations indexed like the unknowns: rows (s, l), columns (j, h)
        A = [[gam(lag, h, s - j) for (j, h) in idx] for (s, lag) in idx]
        b = [gam(lag, 0, s) for (s, lag) in idx]
        return solve(A, b)[-1]

    pacf = []
    for k in range(1, max_lag_t + 1):
        row = []
        for h_last in range(L + 1):
            if method == "starma":
                idx = [(j, h) for j in range(1, k) for h in range(L + 1)] + [(k, h) for h in range(h_last + 1)]
            else:
                idx = [(j, h) for j in range(1, k + 1) for h in range(L + 1) if (j, h) != (k, h_last)] + [(k, h_last)]
            row.append(last_coef(idx))
        pacf.append(row)
    return RichResult(payload={"pacf": pacf, "gamma": g, "method": method})


def _design(lagged, ar, T, N, intercept):
    p = len(ar)
    cols = [(k, lag) for k in range(1, p + 1) for lag in ar[k - 1]]
    X, y = [], []
    for t in range(p, T):
        for i in range(N):
            row = [1.0] if intercept else []
            row.extend(lagged[lag][t - k][i] for (k, lag) in cols)
            X.append(row)
            y.append(lagged[0][t][i])
    return X, y, cols


def _ols(X, y):
    q = len(X[0])
    XtX = [[ssum(r[a] * r[b] for r in X) for b in range(q)] for a in range(q)]
    Xty = [ssum(r[a] * v for r, v in zip(X, y)) for a in range(q)]
    beta = solve(XtX, Xty)
    resid = [v - dot(r, beta) for r, v in zip(X, y)]
    rss = ssum(e * e for e in resid)
    n = len(y)
    inv = inverse(XtX)
    s2 = rss / (n - q)
    se = [math.sqrt(s2 * inv[a][a]) for a in range(q)]
    return beta, se, resid, rss, n


def star_fit(x, weights, ar, *, intercept: bool = False, center: bool = True) -> RichResult:
    r"""Space-time autoregressive STAR model by least squares (Pfeifer and Deutsch 1980).

    ``z_t = sum_{k=1}^{p} sum_{l in ar[k-1]} phi_kl W^(l) z_{t-k} + e_t`` with
    ``ar`` a list (one entry per time lag) of the spatial orders included,
    e.g. ``[[0, 1], [0]]`` for a STAR(2; 1, 0) model. The fit is ordinary
    least squares on the ``N (T - p)`` stacked observations; ``se`` uses
    ``sigma^2 (X'X)^{-1}``. Returns ``coefficients`` as ``[k, l, phi_kl]``
    rows, the residual matrix, ``sigma2``, ``loglik`` (Gaussian), ``aic`` and
    ``bic``.

    Examples
    --------
    >>> W = [[[0, 1], [1, 0]]]
    >>> r = star_fit([[1, 2], [2, 3], [4, 3], [3, 1], [2, 2]], W, [[0, 1]])
    >>> [round(c[2], 12) for c in r.coefficients]
    [0.099870717518, -0.185843568197]
    """
    z = _center(x) if center else [[float(v) for v in row] for row in x]
    T, N = len(z), len(z[0])
    L = max((max(o) for o in ar if o), default=0)
    lagged = _lagged(z, weights, L)
    X, y, cols = _design(lagged, ar, T, N, intercept)
    beta, se, resid, rss, n = _ols(X, y)
    off = 1 if intercept else 0
    q = len(beta)
    s2 = rss / n
    ll = -0.5 * n * (math.log(2 * math.pi * s2) + 1)
    p = len(ar)
    return RichResult(
        payload={
            "coefficients": [[k, lag, beta[off + j]] for j, (k, lag) in enumerate(cols)],
            "se": [[k, lag, se[off + j]] for j, (k, lag) in enumerate(cols)],
            "intercept": beta[0] if intercept else 0.0,
            "residuals": [resid[(t - p) * N : (t - p + 1) * N] for t in range(p, T)],
            "sigma2": rss / (n - q),
            "rss": rss,
            "loglik": ll,
            "aic": -2 * ll + 2 * q,
            "bic": -2 * ll + q * math.log(n),
            "n_obs": n,
            "ar": ar,
        }
    )


def _css_resid(z, lagged, weights, ar, ma, phi, theta, T, N, p):
    e = [[0.0] * N for _ in range(T)]
    ar_idx = [(kk, ll) for kk in range(1, len(ar) + 1) for ll in ar[kk - 1]]
    ma_idx = [(jj, ll) for jj in range(1, len(ma) + 1) for ll in ma[jj - 1]]
    for t in range(p, T):
        pred = [0.0] * N
        for (k, lag), c in zip(ar_idx, phi):
            for i in range(N):
                pred[i] += c * lagged[lag][t - k][i]
        for (j, lag), c in zip(ma_idx, theta):
            if t - j >= p:
                el = st_lag(e[t - j], weights, lag)
                for i in range(N):
                    pred[i] -= c * el[i]
        e[t] = [z[t][i] - pred[i] for i in range(N)]
    return e


def starma_fit(x, weights, ar, ma, *, center: bool = True, max_iter: int = 500) -> RichResult:
    r"""STARMA model by conditional least squares (Pfeifer and Deutsch 1980).

    ``z_t = sum_k sum_l phi_kl W^(l) z_{t-k} - sum_j sum_l theta_jl W^(l) e_{t-j} + e_t``;
    ``ar`` and ``ma`` list the spatial orders per time lag as in
    :func:`star_fit`. The conditional sum of squares of ``e_t`` (``e_t = 0``
    before the first usable time) is minimised by L-BFGS-B with central
    difference gradients from the STAR least-squares start (``theta = 0``).
    ``theta`` follows the sign convention above (the moving-average term is
    subtracted, as in Pfeifer and Deutsch).

    Examples
    --------
    >>> W = [[[0, 1], [1, 0]]]
    >>> x = [[1, 2], [2, 3], [4, 3], [3, 1], [2, 2], [1, 3], [3, 4], [2, 1]]
    >>> r = starma_fit(x, W, [[0]], [])
    >>> abs(r.phi[0][2] - star_fit(x, W, [[0]]).coefficients[0][2]) < 1e-6
    True
    """
    z = _center(x) if center else [[float(v) for v in row] for row in x]
    T, N = len(z), len(z[0])
    L = max([max(o) for o in ar if o] + [max(o) for o in ma if o] + [0])
    lagged = _lagged(z, weights, L)
    p = max(len(ar), len(ma))
    ar_idx = [(k, lag) for k in range(1, len(ar) + 1) for lag in ar[k - 1]]
    ma_idx = [(j, lag) for j in range(1, len(ma) + 1) for lag in ma[j - 1]]
    start = [c[2] for c in star_fit(x, weights, ar, center=center).coefficients] if ar_idx else []
    x0 = start + [0.0] * len(ma_idx)

    def css(v):
        e = _css_resid(z, lagged, weights, ar, ma, v[: len(ar_idx)], v[len(ar_idx) :], T, N, p)
        return ssum(ei * ei for t in range(p, T) for ei in e[t])

    def grad(v):
        out = []
        for a in range(len(v)):
            h = 1e-6 * max(1.0, abs(v[a]))
            vp, vm = list(v), list(v)
            vp[a] += h
            vm[a] -= h
            out.append((css(vp) - css(vm)) / (2 * h))
        return out

    if x0:
        opt = lbfgsb_minimize(css, x0, grad=grad, pgtol=1e-10, factr=10.0, max_iter=max_iter)
        v = [float(u) for u in opt["x"]]
    else:
        v = []
    e = _css_resid(z, lagged, weights, ar, ma, v[: len(ar_idx)], v[len(ar_idx) :], T, N, p)
    rss = ssum(ei * ei for t in range(p, T) for ei in e[t])
    n = N * (T - p)
    q = len(v)
    s2 = rss / n
    ll = -0.5 * n * (math.log(2 * math.pi * s2) + 1)
    return RichResult(
        payload={
            "phi": [[k, lag, v[a]] for a, (k, lag) in enumerate(ar_idx)],
            "theta": [[j, lag, v[len(ar_idx) + a]] for a, (j, lag) in enumerate(ma_idx)],
            "residuals": e[p:],
            "sigma2": rss / max(n - q, 1),
            "rss": rss,
            "loglik": ll,
            "aic": -2 * ll + 2 * q,
            "bic": -2 * ll + q * math.log(n),
            "ar": ar,
            "ma": ma,
            "mean": ssum(float(v_) for row in x for v_ in row) / (T * N) if center else 0.0,
            "n_obs": n,
        }
    )


def starma_forecast(fit, x, weights, h: int = 1) -> RichResult:
    r"""``h``-step forecasts from a :func:`starma_fit` (or :func:`star_fit`) result.

    Future shocks are set to zero; past shocks are the fitted residuals. The
    forecast is returned on the original scale (the grand mean is added back).

    Examples
    --------
    >>> W = [[[0, 1], [1, 0]]]
    >>> x = [[1, 2], [2, 3], [4, 3], [3, 1], [2, 2], [1, 3], [3, 4], [2, 1]]
    >>> f = starma_forecast(starma_fit(x, W, [[0, 1]], []), x, W, 2)
    >>> len(f.forecast), len(f.forecast[0])
    (2, 2)
    """
    phi = fit["phi"] if "phi" in fit else fit["coefficients"]
    theta = fit.get("theta", [])
    ar, ma = fit["ar"], fit.get("ma", [])
    mean = fit.get("mean", ssum(float(v) for row in x for v in row) / (len(x) * len(x[0])))
    z = [[float(v) - mean for v in row] for row in x]
    T, N = len(z), len(z[0])
    p = max(len(ar), len(ma))
    res = fit["residuals"]
    e = [[0.0] * N for _ in range(p)] + [list(r) for r in res]
    out = []
    for step in range(h):
        t = T + step
        pred = [0.0] * N
        for k, lag, c in phi:
            zl = st_lag(z[t - k], weights, lag)
            for i in range(N):
                pred[i] += c * zl[i]
        for j, lag, c in theta:
            if t - j < T:
                el = st_lag(e[t - j], weights, lag)
                for i in range(N):
                    pred[i] -= c * el[i]
        z.append(pred)
        e.append([0.0] * N)
        out.append([v + mean for v in pred])
    return RichResult(payload={"forecast": out, "h": h})


def st_portmanteau(
    residuals, weights, *, max_lag_t: int = 5, max_lag_s: int | None = None, n_params: int = 0
) -> RichResult:
    r"""Space-time portmanteau test of residual whiteness (Pfeifer and Deutsch 1980).

    ``Q = N T sum_{s=1}^{S} sum_{l=0}^{L} rho_l0(s)^2`` on the residual
    space-time ACF, compared with ``chi^2`` on ``S (L + 1) - n_params``
    degrees of freedom.

    Examples
    --------
    >>> W = [[[0, 1], [1, 0]]]
    >>> r = st_portmanteau([[1, -1], [-1, 1], [1, -1], [-1, 1]], W, max_lag_t=1)
    >>> round(r.statistic, 12), r.df
    (16.0, 2)
    """
    r = st_acf(residuals, weights, max_lag_t=max_lag_t, max_lag_s=max_lag_s, center=False)
    T, N = len(residuals), len(residuals[0])
    L = len(r.acf[0]) - 1
    Q = N * T * ssum(r.acf[s][lag] ** 2 for s in range(1, max_lag_t + 1) for lag in range(L + 1))
    df = max_lag_t * (L + 1) - n_params
    return RichResult(payload={"statistic": Q, "df": df, "p_value": _chi2_sf(Q, df) if df > 0 else float("nan")})


def cheatsheet() -> str:
    return (
        "st_lag / st_acf / st_pacf / star_fit / starma_fit / starma_forecast / st_portmanteau -> "
        "Pfeifer-Deutsch STARMA space-time models."
    )
