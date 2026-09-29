# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel SAC (lag+error) estimator."""

import math

from ._qpcore import ssum
from ._richresult import RichResult
from .sppanel import _cols, _demean, _golden, _lag, _logdet, _ols, _polish, _trace_aw
from .sppfe import _stack


def sppsac(y, X, W, time_id, unit_id, effects="individual", interval=(-0.99, 0.99), tol=1e-13, maxit=500):
    r"""Spatial panel SAC model y_t = rho W y_t + X_t beta + effects + u_t, u_t = lambda W u_t + e_t, by ML.

    After the effects transformation (Elhorst 2014, sec. 3.4), with A =
    I - rho W and B = I - lambda W applied period by period, beta and
    sigma^2 are concentrated out by OLS of B A y on B X and the
    concentrated log-likelihood -NT/2 log SSE(rho, lambda) + T log|A| + T
    log|B| is maximised by coordinate ascent: each coordinate by
    golden-section search on interval polished by bisection on its score
    (d/d rho = NT e'BWy / SSE - T tr(A^{-1}W), d/d lambda = NT e'Wu / SSE
    - T tr(B^{-1}W)), alternating until both change by less than tol
    (as splm::spml(lag = TRUE, spatial.error = "b")).

    References
    ----------
    Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data
    to Spatial Panels*. Springer.
    Lee, L.-F. and Yu, J. (2010). Estimation of spatial autoregressive panel
    data models with fixed effects. *Journal of Econometrics* 154, 165-185.

    Examples
    --------
    >>> import math
    >>> W = [[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]]
    >>> tid = [t for t in range(5) for _ in range(4)]
    >>> uid = [u for _ in range(5) for u in range(4)]
    >>> X = [[math.sin(1.3 * k) + 0.1 * k] for k in range(20)]
    >>> y = [1.0 + 0.8 * X[k][0] + 0.3 * math.cos(2.1 * k) + 0.2 * (k % 4) for k in range(20)]
    >>> r = sppsac(y, X, W, tid, uid)
    >>> round(r["rho"], 8), round(r["lambda"], 8)
    (-0.05473286, -0.10453391)
    """
    yv, Xm, N, T = _stack(y, X, time_id, unit_id)
    Wm = [[float(v) for v in r] for r in (W.tolist() if hasattr(W, "tolist") else W)]
    if effects == "pooled":
        yt, Xt = yv, [[1.0] + r for r in Xm]
    else:
        yt = _demean(yv, N, T, effects)
        Xt = _cols([_demean(c, N, T, effects) for c in _cols(Xm)])
    NT = N * T
    wy = _lag(Wm, yt, N, T)
    wx = _cols([_lag(Wm, c, N, T) for c in _cols(Xt)])

    def ldet(r):
        return _logdet([[(1.0 if i == j else 0.0) - r * Wm[i][j] for j in range(N)] for i in range(N)])

    def tf(v, lam):
        return [a - lam * b for a, b in zip(v, _lag(Wm, v, N, T))]

    def rho_step(lam):
        by, bwy = tf(yt, lam), tf(wy, lam)
        bx = [[p - lam * q for p, q in zip(u, v)] for u, v in zip(Xt, wx)]
        _, e0 = _ols(bx, by)
        _, e1 = _ols(bx, bwy)
        a, b, c = ssum(v * v for v in e0), ssum(u * v for u, v in zip(e0, e1)), ssum(v * v for v in e1)
        r = _golden(lambda r: -NT / 2.0 * math.log(a - 2 * r * b + r * r * c) + T * ldet(r), *interval)
        return _polish(lambda r: NT * (b - r * c) / (a - 2 * r * b + r * r * c) - T * _trace_aw(Wm, r), r, *interval)

    def lam_fit(rho, lam):
        z = [u - rho * v for u, v in zip(yt, wy)]
        wz = [u - rho * v for u, v in zip(wy, _lag(Wm, wy, N, T))]
        ys = [u - lam * v for u, v in zip(z, wz)]
        xs = [[p - lam * q for p, q in zip(u, v)] for u, v in zip(Xt, wx)]
        bt, e = _ols(xs, ys)
        return bt, e, z, wz

    def lam_step(rho):
        def f(lam):
            return -NT / 2.0 * math.log(ssum(v * v for v in lam_fit(rho, lam)[1])) + T * ldet(lam)

        def df(lam):
            bt, e, z, wz = lam_fit(rho, lam)
            wu = [p - ssum(q[k] * bt[k] for k in range(len(bt))) for p, q in zip(wz, wx)]
            return NT * ssum(p * q for p, q in zip(e, wu)) / ssum(v * v for v in e) - T * _trace_aw(Wm, lam)

        return _polish(df, _golden(f, *interval), *interval)

    rho, lam = rho_step(0.0), 0.0
    iters = 0
    for _ in range(maxit):
        iters += 1
        lam_new = lam_step(rho)
        rho_new = rho_step(lam_new)
        done = abs(rho_new - rho) + abs(lam_new - lam) < tol
        rho, lam = rho_new, lam_new
        if done:
            break
    beta, e, _, _ = lam_fit(rho, lam)
    sse = ssum(v * v for v in e)
    s2 = sse / NT
    ll = T * ldet(rho) + T * ldet(lam) - NT / 2.0 * (math.log(2 * math.pi) + math.log(s2) + 1.0)
    return RichResult(
        payload={
            "rho": rho,
            "lambda": lam,
            "coefficients": beta,
            "sigma2": s2,
            "loglik": ll,
            "residuals": e,
            "iterations": iters,
        }
    )


sppsac_fn = sppsac


def cheatsheet() -> str:
    return "sppsac(y, X, W, time_id, unit_id) -> SAC spatial panel (lag + error) by ML."
