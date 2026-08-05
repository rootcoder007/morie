# morie.fn -- function file (rootcoder007/morie)
"""ETS state-space (error/trend/seasonal)."""

import math

from ._richresult import RichResult

__all__ = ["ets"]


def ets(y, error="A", trend=False, season=0, alpha=None, beta=None, gamma=None):
    """
    ETS state-space (error/trend/seasonal)

    Formula: the additive-error innovations state space form

        y_t = l_{t-1} + b_{t-1} + s_{t-m} + e_t
        l_t = l_{t-1} + b_{t-1} + alpha e_t
        b_t = b_{t-1}           + beta  e_t
        s_t = s_{t-m}           + gamma e_t

    i.e. y_t = w' x_{t-1} + e_t, x_t = F x_{t-1} + g e_t with
    w = (1, 1, 0, ..., 0, 1)', g = (alpha, beta, gamma, 0, ..., 0)'.
    Smoothing parameters are chosen on a deterministic 0.1-step grid that
    minimises the one-step sum of squared errors; they may also be fixed
    directly.

    Parameters
    ----------
    y : array-like
        Series.
    error : str
        Error type; only "A" (additive) is implemented.
    trend : bool
        Include the additive local trend b_t.
    season : int
        Seasonal period m; 0 or 1 means no seasonal component.
    alpha, beta, gamma : float, optional
        Fixed smoothing parameters (bypass the grid search).

    Returns
    -------
    result : dict
        Keys: estimate (one-step-ahead forecast), alpha, beta, gamma,
        sse, sigma2, aic, level, slope, forecast, n, method.

    References
    ----------
    Hyndman, Koehler, Snyder & Grose (2002), International Journal of
    Forecasting 18(3):439-454, doi:10.1016/S0169-2070(01)00110-8.
    """
    y = [float(v) for v in y]
    n = len(y)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    if str(error).upper() != "A":
        raise ValueError("only the additive error form 'A' is implemented")
    m = int(season)
    if m < 0:
        raise ValueError("season must be non-negative")
    if m <= 1:
        m = 0
    if m and n < 2 * m:
        raise ValueError("need at least two full seasons for season = %d" % m)
    use_b = bool(trend)

    def _init():
        if m:
            l0 = sum(y[:m]) / m
            b0 = ((sum(y[m:2 * m]) / m) - l0) / m if use_b else 0.0
            s0 = [y[j] - l0 for j in range(m)]
        else:
            l0 = y[0]
            b0 = (y[1] - y[0]) if (use_b and n > 1) else 0.0
            s0 = []
        return l0, b0, s0

    def _sse(a, b, g):
        l, bt, s0 = _init()
        s = list(s0)
        tot = 0.0
        for t in range(n):
            sea = s[t % m] if m else 0.0
            fit = l + bt + sea
            e = y[t] - fit
            tot += e * e
            lnew = l + bt + a * e
            if use_b:
                bt = bt + b * e
            if m:
                s[t % m] = sea + g * e
            l = lnew
        return tot, l, bt, s

    grid = [0.1 * k for k in range(1, 10)]
    if alpha is not None:
        A = [float(alpha)]
    else:
        A = grid
    best = None
    for a in A:
        Bs = [float(beta)] if beta is not None else ([b for b in grid if b <= a] if use_b else [0.0])
        for b in Bs:
            Gs = [float(gamma)] if gamma is not None else ([g for g in grid if g <= 1.0 - a] if m else [0.0])
            for g in Gs:
                tot = _sse(a, b, g)[0]
                if best is None or tot < best[0] - 1e-15:
                    best = (tot, a, b, g)
    sse, a, b, g = best
    _, level, slope, s = _sse(a, b, g)
    k = 1 + (1 if use_b else 0) + (1 if m else 0) + 1 + (1 if use_b else 0) + m
    sigma2 = sse / n
    aic = n * math.log(sse / n) + 2.0 * k if sse > 0.0 else float("-inf")
    fc = level + slope + (s[n % m] if m else 0.0)
    return RichResult(payload={
        "estimate": fc,
        "alpha": a,
        "beta": b,
        "gamma": g,
        "sse": sse,
        "sigma2": sigma2,
        "aic": aic,
        "level": level,
        "slope": slope,
        "forecast": fc,
        "n": n,
        "method": "ETS state-space (error/trend/seasonal)",
    })


def cheatsheet():
    return "etsmod: ETS state-space (error/trend/seasonal)"
