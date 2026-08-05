# morie.fn -- function file (rootcoder007/morie)
"""Epidemic curve smoothing."""

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["epicurve"]


def _tricube(u):
    a = abs(u)
    if a >= 1.0:
        return 0.0
    t = 1.0 - a * a * a
    return t * t * t


def _bisquare(u):
    a = abs(u)
    if a >= 1.0:
        return 0.0
    t = 1.0 - a * a
    return t * t


def epicurve(dates, cases, bandwidth, iterations=2):
    """
    Epidemic curve smoothing

    Formula: robust locally weighted regression of daily counts on time.
    At each day x0 the fit is the intercept of the weighted least-squares
    line of ``cases`` on ``dates - x0`` with tricube neighbourhood
    weights

        w_i = (1 - |x_i - x0|^3 / h^3)^3   for |x_i - x0| < h

    (Cleveland 1979 eq. 2), followed by ``iterations`` robustness passes
    in which each observation is additionally weighted by the bisquare of
    its residual scaled by six times the median absolute residual,

        delta_i = (1 - (r_i / (6 s))^2)^2  for |r_i| < 6 s

    (Cleveland 1979 eq. 5-7).  A fixed bandwidth in day units is used in
    place of Cleveland's nearest-neighbour span, which is the natural
    parameterisation for an epidemic curve on a calendar axis.

    Parameters
    ----------
    dates : array-like
        Day index of each observation (strictly increasing not required).
    cases : array-like
        Counts observed on those days.
    bandwidth : float
        Neighbourhood half-width h in the units of ``dates`` (> 0).
    iterations : int
        Number of robustness passes (0 gives the plain local linear fit).

    Returns
    -------
    result : dict
        Keys: estimate (smoothed peak height), fitted, peak_date,
        peak_value, total, n, method.

    References
    ----------
    Cleveland (1979), JASA 74(368):829-836,
    doi:10.1080/01621459.1979.10481038.
    """
    x = [float(v) for v in dates]
    y = [float(v) for v in cases]
    n = len(x)
    if n == 0:
        raise ValueError("empty input: dates has no observations")
    if len(y) != n:
        raise ValueError("dates and cases must have the same length")
    h = float(bandwidth)
    if h <= 0.0:
        raise ValueError("bandwidth must be positive")
    it = int(iterations)
    if it < 0:
        raise ValueError("iterations must be non-negative")
    delta = [1.0] * n
    fit = [0.0] * n
    for _ in range(it + 1):
        for k in range(n):
            x0 = x[k]
            s0 = s1 = s2 = t0 = t1 = 0.0
            for i in range(n):
                d = x[i] - x0
                w = _tricube(d / h) * delta[i]
                if w == 0.0:
                    continue
                s0 += w
                s1 += w * d
                s2 += w * d * d
                t0 += w * y[i]
                t1 += w * d * y[i]
            det = s0 * s2 - s1 * s1
            if s0 <= 0.0:
                fit[k] = y[k]
            elif abs(det) < 1e-12 * (1.0 + abs(s0 * s2)):
                fit[k] = t0 / s0
            else:
                fit[k] = (t0 * s2 - t1 * s1) / det
        r = [y[i] - fit[i] for i in range(n)]
        s = core.median([abs(v) for v in r])
        if s <= 0.0:
            break
        delta = [_bisquare(r[i] / (6.0 * s)) for i in range(n)]
    pk = 0
    for i in range(1, n):
        if fit[i] > fit[pk]:
            pk = i
    return RichResult(payload={
        "estimate": fit[pk],
        "fitted": fit,
        "peak_date": x[pk],
        "peak_value": fit[pk],
        "total": sum(y),
        "n": n,
        "method": "Epidemic curve smoothing",
    })


def cheatsheet():
    return "epicur: Epidemic curve smoothing"
