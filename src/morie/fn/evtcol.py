# morie.fn -- function file (rootcoder007/morie)
"""Extreme-value entry points of the Coles shelf and the tail-index
estimators beside it.

Python arm of the thirteen ``morie_evt_*`` entry points in
``R/evt_coles.R`` and ``R/evt_native.R``.  Seven of them -- the Hill,
Pickands, Dekkers-Einmahl-de Haan and three extremal-index estimators
-- already had a Python implementation under a different name; those
are delegated to rather than restated, so there is one implementation
of each estimator in this package and not two that can drift apart.
The rest are implemented here against the R arm.

References
----------
Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme
    Values*. Springer.
"""

from __future__ import annotations

import math

from . import _evt_core as _ev
from ._richresult import RichResult

__all__ = [
    "evchiu", "evchibu", "evdedhm", "evextivl", "evextrun", "evextsld",
    "evgevlpd", "evgevtrd", "evtailhl", "evhillal", "evpickxi",
    "evrlvlci", "evrlvlpt",
]


_BIG = 1.0e35
_RELTOL = math.sqrt(2.220446049250313e-16)   # R's sqrt(.Machine$double.eps)


def _nmmin(fn, x0, maxit, reltol=_RELTOL, abstol=float("-inf"),
           alpha=1.0, bet=0.5, gamm=2.0):
    """R's ``optim(method = "Nelder-Mead")`` simplex, transcribed.

    The package's own :func:`morie.fn._sci_core.minimize` Nelder-Mead is
    a Scipy-shaped one: different initial simplex, different convergence
    test.  Both find the optimum, but they stop at points that differ by
    about 1e-6 -- enough to break parity with the R arm on the fitted
    parameters and every return level derived from them.  This is R's
    ``nmmin`` from ``optim.c``, so the two arms take the identical path
    and land on the identical point.
    """
    n = len(x0)
    bvec = [float(v) for v in x0]
    f = fn(bvec)
    if not math.isfinite(f):
        raise ValueError("function is not finite at the initial parameters")
    funcount = 1
    convtol = reltol * (abs(f) + reltol)
    n1 = n + 1
    # P[i][j]: vertex j's i-th coordinate; row n holds the function
    # values; column n + 1 is the centroid workspace.
    P = [[0.0] * (n + 2) for _ in range(n + 1)]
    P[n][0] = f
    for i in range(n):
        P[i][0] = bvec[i]
    L = 1
    size = 0.0
    step = 0.0
    for i in range(n):
        if 0.1 * abs(bvec[i]) > step:
            step = 0.1 * abs(bvec[i])
    if step == 0.0:
        step = 0.1
    for j in range(2, n1 + 1):
        for i in range(n):
            P[i][j - 1] = bvec[i]
        trystep = step
        while P[j - 2][j - 1] == bvec[j - 2]:
            P[j - 2][j - 1] = bvec[j - 2] + trystep
            trystep *= 10.0
        size += trystep
    oldsize = size
    calcvert = True
    while True:
        if calcvert:
            for j in range(n1):
                if j + 1 != L:
                    for i in range(n):
                        bvec[i] = P[i][j]
                    f = fn(bvec)
                    if not math.isfinite(f):
                        f = _BIG
                    funcount += 1
                    P[n][j] = f
            calcvert = False
        VL = P[n][L - 1]
        VH = VL
        H = L
        for j in range(1, n1 + 1):
            if j != L:
                f = P[n][j - 1]
                if f < VL:
                    L = j
                    VL = f
                if f > VH:
                    H = j
                    VH = f
        if VH <= VL + convtol or VL <= abstol:
            break
        for i in range(n):
            temp = -P[i][H - 1]
            for j in range(n1):
                temp += P[i][j]
            P[i][n + 1] = temp / n
        for i in range(n):
            bvec[i] = (1.0 + alpha) * P[i][n + 1] - alpha * P[i][H - 1]
        f = fn(bvec)
        if not math.isfinite(f):
            f = _BIG
        funcount += 1
        VR = f
        if VR < VL:
            P[n][n + 1] = f
            for i in range(n):
                f = gamm * bvec[i] + (1.0 - gamm) * P[i][n + 1]
                P[i][n + 1] = bvec[i]
                bvec[i] = f
            f = fn(bvec)
            if not math.isfinite(f):
                f = _BIG
            funcount += 1
            if f < VR:
                for i in range(n):
                    P[i][H - 1] = bvec[i]
                P[n][H - 1] = f
            else:
                for i in range(n):
                    P[i][H - 1] = P[i][n + 1]
                P[n][H - 1] = VR
        else:
            if VR < VH:
                for i in range(n):
                    P[i][H - 1] = bvec[i]
                P[n][H - 1] = VR
            for i in range(n):
                bvec[i] = (1.0 - bet) * P[i][H - 1] + bet * P[i][n + 1]
            f = fn(bvec)
            if not math.isfinite(f):
                f = _BIG
            funcount += 1
            if f < P[n][H - 1]:
                for i in range(n):
                    P[i][H - 1] = bvec[i]
                P[n][H - 1] = f
            elif VR >= VH:
                calcvert = True
                size = 0.0
                for j in range(n1):
                    if j + 1 != L:
                        for i in range(n):
                            P[i][j] = bet * (P[i][j] - P[i][L - 1]) \
                                + P[i][L - 1]
                            size += abs(P[i][j] - P[i][L - 1])
                if size < oldsize:
                    oldsize = size
                else:
                    break
        if funcount > maxit:
            break
    return [P[i][L - 1] for i in range(n)], P[n][L - 1], funcount <= maxit


def _solve_inv(H):
    """Plain LU-with-partial-pivoting inverse, matching R's ``solve``."""
    k = len(H)
    a = [list(map(float, row)) + [1.0 if i == j else 0.0
                                  for j in range(k)]
         for i, row in enumerate(H)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(a[r][c]))
        if abs(a[piv][c]) < 1e-300:
            raise ValueError("observed information is singular")
        a[c], a[piv] = a[piv], a[c]
        d = a[c][c]
        a[c] = [v / d for v in a[c]]
        for r in range(k):
            if r != c and a[r][c] != 0.0:
                m = a[r][c]
                a[r] = [v - m * w for v, w in zip(a[r], a[c])]
    return [row[k:] for row in a]


def _gevmle_r(xs):
    """GEV maximum likelihood by the R arm's exact recipe: Gumbel moment
    start values, R's Nelder-Mead on (mu, log sigma, xi), and the
    inverse of the central-difference observed information."""
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two observations")
    xbar = math.fsum(xs) / n
    s = math.sqrt(math.fsum((v - xbar) ** 2 for v in xs) / (n - 1))
    sigma0 = s * math.sqrt(6.0) / math.pi
    mu0 = xbar - 0.5772156649015329 * sigma0

    def nll(th):
        return -_ev.gev_loglik(xs, th[0], math.exp(th[1]), th[2])

    par, val, ok = _nmmin(nll, [mu0, math.log(sigma0), 0.1], 4000)
    mu, sigma, xi = par[0], math.exp(par[1]), par[2]

    def nll_nat(th):
        return -_ev.gev_loglik(xs, th[0], th[1], th[2])

    H = _ev._hessian(nll_nat, [mu, sigma, xi])
    return {"mu": mu, "sigma": sigma, "xi": xi, "loglik": -val,
            "cov": _solve_inv(H), "n": n, "converged": ok}


def _pair(x, y):
    xs = _ev._flat(x)
    ys = _ev._flat(y)
    if len(xs) != len(ys):
        raise ValueError(
            f"x has {len(xs)} entries and y has {len(ys)}")
    if len(xs) < 4:
        raise ValueError(f"need at least 4 pairs, got {len(xs)}")
    return xs, ys


def _rank_avg(v):
    """R's ``rank(v)`` with the default average handling of ties."""
    n = len(v)
    order = sorted(range(n), key=lambda i: v[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


# ------------------------------------------------------- tail dependence
def evchiu(x, y, u=0.95):
    r"""Empirical :math:`\chi(u) = 2 - \log C(u,u)/\log u` tail-dependence
    measure at the quantile level ``u``, on rank-transformed margins
    :math:`\hat F = \mathrm{rank}/(n+1)`.

    The joint probability is floored at :math:`1/(2n)` and capped at
    :math:`1 - 1/(2n)` before the logarithm, and the result is clipped
    into ``[0, 1]``.  ``chi = 0`` is asymptotic independence.

    The empirical copula here is evaluated on the **lower** orthant
    ``{F_X < u, F_Y < u}``, matching the R arm ``morie_evt_chi``.

    Parameters
    ----------
    x, y : sequence of float
        Equal-length series, at least four pairs.
    u : float
        Quantile level in ``(0, 1)``.

    Returns
    -------
    RichResult
        ``chi``, ``joint``, ``u``, ``n``, ``method``.
    """
    xs, ys = _pair(x, y)
    u = float(u)
    if not 0.0 < u < 1.0:
        raise ValueError("u must lie strictly in (0, 1)")
    n = len(xs)
    rx = [r / (n + 1.0) for r in _rank_avg(xs)]
    ry = [r / (n + 1.0) for r in _rank_avg(ys)]
    joint = sum(1 for a, b in zip(rx, ry) if a < u and b < u) / n
    joint = min(max(joint, 1.0 / (2 * n)), 1.0 - 1.0 / (2 * n))
    chi = min(max(2.0 - math.log(joint) / math.log(u), 0.0), 1.0)
    return RichResult(payload={
        "chi": chi, "joint": joint, "u": u, "n": n,
        "method": "empirical chi(u) tail dependence (Coles 2001 sec. 8.4)"})


def evchibu(x, y, ugrid=None):
    r"""Empirical :math:`\bar\chi(u) = 2\log(1-u)/\log\hat P(F_X>u, F_Y>u)
    - 1` over a grid of quantile levels, on rank-transformed margins.

    :math:`\bar\chi \to 1` signals asymptotic dependence; a limit
    strictly inside ``(-1, 1)`` signals asymptotic independence with the
    value measuring the strength of the dependence that survives.  Each
    point is clipped into ``[-1, 1]``.

    Parameters
    ----------
    x, y : sequence of float
        Equal-length series, at least four pairs.
    ugrid : sequence of float, optional
        Quantile levels; twenty points evenly spaced on ``[0.5, 0.95]``
        by default.

    Returns
    -------
    RichResult
        ``chibar`` (list), ``ugrid`` (list), ``n``, ``method``.
    """
    xs, ys = _pair(x, y)
    n = len(xs)
    if ugrid is None:
        ugrid = [0.5 + 0.45 * k / 19.0 for k in range(20)]
    else:
        ugrid = [float(v) for v in ugrid]
    if any(not 0.0 < v < 1.0 for v in ugrid):
        raise ValueError("every u must lie strictly in (0, 1)")
    rx = [r / (n + 1.0) for r in _rank_avg(xs)]
    ry = [r / (n + 1.0) for r in _rank_avg(ys)]
    curve = []
    for u in ugrid:
        joint = sum(1 for a, b in zip(rx, ry) if a > u and b > u) / n
        joint = min(max(joint, 1.0 / (2 * n)), 1.0 - 1.0 / (2 * n))
        cb = 2.0 * math.log(1.0 - u) / math.log(joint) - 1.0
        curve.append(min(max(cb, -1.0), 1.0))
    return RichResult(payload={
        "chibar": curve, "ugrid": ugrid, "n": n,
        "method": "empirical chibar(u) (Coles 2001 sec. 8.4)"})


# ---------------------------------------------------------------- GEV
def evgevlpd(x, mu, sigma, xi):
    r"""GEV log-density at each point of ``x``, ``-inf`` off the support
    :math:`1 + \xi(x-\mu)/\sigma > 0` and ``-inf`` everywhere when
    ``sigma <= 0``.  The Gumbel limit is used for ``|xi| < 1e-8``.

    Parameters
    ----------
    x : float or sequence of float
        Evaluation point(s).
    mu, sigma, xi : float
        GEV location, scale and shape.

    Returns
    -------
    RichResult
        ``logpdf`` (list), ``loglik`` (their sum), ``n_support``,
        ``method``.
    """
    xs = _ev._flat(x)
    if not xs:
        raise ValueError("x must not be empty")
    mu = float(mu)
    sigma = float(sigma)
    xi = float(xi)
    lp = [_ev.gev_logpdf(v, mu, sigma, xi) for v in xs]
    ok = [v for v in lp if v != float("-inf")]
    return RichResult(payload={
        "logpdf": lp,
        "loglik": math.fsum(ok) if len(ok) == len(lp)
                  else float("-inf"),
        "n_support": len(ok),
        "method": "GEV log-density (Coles 2001 sec. 3.3.2)"})


def evgevtrd(x, t=None):
    r"""Nonstationary GEV with a linear trend in the location parameter,
    :math:`\mu(t) = \beta_0 + \beta_1 t`, fitted by maximum likelihood
    on a standardised time index and reported on the original scale.

    ``lr_vs_stationary`` is the likelihood-ratio statistic against the
    stationary fit, to be read against a chi-squared with one degree of
    freedom.

    Parameters
    ----------
    x : sequence of float
        Series of block maxima.
    t : sequence of float, optional
        Time index; ``0, 1, ..., n-1`` by default.

    Returns
    -------
    RichResult
        ``beta0``, ``beta1``, ``sigma``, ``xi``, ``loglik``,
        ``lr_vs_stationary``, ``n``, ``method``.
    """
    xs = _ev._flat(x)
    n = len(xs)
    if n < 4:
        raise ValueError(f"need at least 4 maxima, got {n}")
    ts = [float(i) for i in range(n)] if t is None else _ev._flat(t)
    if len(ts) != n:
        raise ValueError(f"t has {len(ts)} entries and x has {n}")
    tbar = math.fsum(ts) / n
    # R: sd(t) * sqrt((n-1)/n) -- the population standard deviation
    tsd = max(math.sqrt(math.fsum((v - tbar) ** 2 for v in ts) / n), 1e-12)
    tz = [(v - tbar) / tsd for v in ts]
    f0 = _gevmle_r(xs)

    def nll(th):
        s = math.exp(th[2])
        return -sum(_ev.gev_logpdf(xs[i], th[0] + th[1] * tz[i],
                                   s, th[3]) for i in range(n))

    par, val, _ok = _nmmin(nll, [f0["mu"], 0.0, math.log(f0["sigma"]),
                                 f0["xi"]], 6000)
    ll = -val
    return RichResult(payload={
        "beta0": par[0], "beta1": par[1] / tsd,
        "sigma": math.exp(par[2]), "xi": par[3], "loglik": ll,
        "lr_vs_stationary": 2.0 * (ll - f0["loglik"]), "n": n,
        "method": "nonstationary GEV, linear trend in location "
                  "(Coles 2001 sec. 6.2)"})


# ------------------------------------------------------- return levels
def evrlvlci(x, T, alpha=0.05):
    r"""GEV ``T``-period return level with a delta-method confidence
    interval: :math:`\operatorname{Var}(z_T) = \nabla z_T^{\top} V
    \nabla z_T` with ``V`` the observed-information covariance of the
    maximum-likelihood fit to ``x``.

    Parameters
    ----------
    x : sequence of float
        Block maxima to fit.
    T : float
        Return period, greater than 1.
    alpha : float
        One minus the confidence level.

    Returns
    -------
    RichResult
        ``z_T``, ``ci_lo``, ``ci_hi``, ``se``, ``T``, ``method``.
    """
    T = float(T)
    if T <= 1:
        raise ValueError("return period T must exceed 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly in (0, 1)")
    from ._stats_core import norm as _norm
    f = _gevmle_r(_ev._flat(x))
    z = _ev.gev_return_level(T, f["mu"], f["sigma"], f["xi"])
    g = _ev.gev_return_level_grad(T, f["mu"], f["sigma"], f["xi"])
    V = f["cov"]
    var = math.fsum(g[i] * V[i][j] * g[j]
                    for i in range(3) for j in range(3))
    se = math.sqrt(max(var, 0.0))
    zc = float(_norm.ppf(1.0 - alpha / 2.0))
    return RichResult(payload={
        "z_T": float(z), "ci_lo": float(z - zc * se),
        "ci_hi": float(z + zc * se), "se": se, "T": T,
        "method": "delta-method GEV return-level CI "
                  "(Coles 2001 sec. 3.3.3)"})


def evrlvlpt(u, sigma, xi, zetau, m):
    r"""POT ``m``-observation return level
    :math:`x_m = u + (\sigma/\xi)[(m\zeta_u)^{\xi} - 1]`, with the
    logarithmic form as :math:`\xi \to 0`.

    Parameters
    ----------
    u : float
        Threshold.
    sigma : float
        GPD scale, positive.
    xi : float
        GPD shape.
    zetau : float
        Exceedance probability :math:`P(X > u)`, in ``(0, 1]``.
    m : float
        Number of observations; ``m * zetau`` must exceed 1.

    Returns
    -------
    RichResult
        ``z_T``, ``m``, ``u``, ``zeta_u``, ``method``.
    """
    u = float(u)
    sigma = float(sigma)
    xi = float(xi)
    zetau = float(zetau)
    m = float(m)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0.0 < zetau <= 1.0:
        raise ValueError("zetau must lie in (0, 1]")
    if m * zetau <= 1.0:
        raise ValueError("m * zetau must exceed 1 for a level above "
                         "the threshold")
    return RichResult(payload={
        "z_T": _ev.pot_return_level(m, u, sigma, xi, zetau),
        "m": m, "u": u, "zeta_u": zetau,
        "method": "POT m-observation return level "
                  "(Coles 2001 eq. 4.13)"})


# ------------------------------------ tail-index and extremal-index arms
def evtailhl(x, k=None):
    """Hill (1975) tail-index estimator; see :func:`morie.fn.evhill.ev_hill`,
    which this delegates to so the estimator has one implementation."""
    from .evhill import ev_hill
    return ev_hill(x, k=k)


def evhillal(x, k=None):
    """Hill estimator, alias entry point; see
    :func:`morie.fn.hillEst.hill_estimator`."""
    from .hillEst import hill_estimator
    return hill_estimator(x, k=k)


def evpickxi(x, k=None):
    """Pickands (1975) extreme-value index; see
    :func:`morie.fn.evpick.ev_pickands`."""
    from .evpick import ev_pickands
    return ev_pickands(x, k=k)


def evdedhm(x, k=None):
    """Dekkers-Einmahl-de Haan moment estimator; see
    :func:`morie.fn.evdedh.ev_dedh`."""
    from .evdedh import ev_dedh
    return ev_dedh(x, k=k)


def evextrun(x, threshold, runlength=1):
    """Runs estimator of the extremal index; see
    :func:`morie.fn.evextidx.ev_extremal_runs`."""
    from .evextidx import ev_extremal_runs
    return ev_extremal_runs(x, threshold, run_length=runlength)


def evextivl(x, threshold):
    """Ferro-Segers intervals estimator of the extremal index; see
    :func:`morie.fn.evextint.ev_extremal_intervals`."""
    from .evextint import ev_extremal_intervals
    return ev_extremal_intervals(x, threshold)


def evextsld(x, threshold=None, blocklength=None):
    """Northrop sliding-blocks estimator of the extremal index; see
    :func:`morie.fn.evextsl.ev_extremal_sliding`."""
    from .evextsl import ev_extremal_sliding
    return ev_extremal_sliding(x, threshold=threshold,
                               block_length=blocklength)


def cheatsheet():
    return ("evtcol: Coles-shelf entry points -- GEV log-density and "
            "trend, return levels with CIs, chi/chibar, and the tail- "
            "and extremal-index estimators")
