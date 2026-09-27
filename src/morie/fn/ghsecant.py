"""Generalized secant hyperbolic distribution (Vaughan 2002): density, distribution function, quantile, draws.

Vaughan, D. C. (2002). The generalized secant hyperbolic distribution and its properties.
Communications in Statistics - Theory and Methods 31, 219-238.
"""

import math

from ._distcore import draws, out, vec
from ._richresult import RichResult

__all__ = ["ghsecant"]


def _consts(t):
    if t < 0:
        c2 = math.sqrt((math.pi**2 - t * t) / 3)
        return math.cos(t), c2, math.sin(t) / t * c2
    if t == 0:
        c2 = math.pi / math.sqrt(3)
        return 1.0, c2, c2
    c2 = math.sqrt((math.pi**2 + t * t) / 3)
    return math.cosh(t), c2, math.sinh(t) / t * c2


def ghsecant(x=None, t=0.0, loc=0.0, scale=1.0, p=None, n=0, seed=0):
    r"""Standardised density f(z) = c1 e^{c2 z} / (e^{2 c2 z} + 2 a e^{c2 z} + 1) (mean 0, variance 1), x = loc + scale z.

    a = cos t, c2 = sqrt((pi^2 - t^2)/3), c1 = (sin t / t) c2 for -pi < t < 0;
    a = cosh t, c2 = sqrt((pi^2 + t^2)/3), c1 = (sinh t / t) c2 for t > 0; the
    logistic (c1 = c2 = pi/sqrt 3) at t = 0; t = -pi/2 is the hyperbolic secant.
    With w = e^{c2 z} the distribution function is
    1 + pi/(2t) + arctan((w + cos t)/sin t)/t (t < 0), w/(1 + w) (t = 0) and
    1 + log((w + e^{-t})/(w + e^{t}))/(2t) (t > 0), each inverted in closed form.

    Parameters
    ----------
    x : float or sequence, optional
    t : float
        Shape, t > -pi (kurtosis from 1.8 upwards).
    loc, scale : float
    p : float or sequence, optional
    n : int
    seed : int

    Returns
    -------
    RichResult
        Keys: pdf, logpdf, cdf, quantile, random.

    References
    ----------
    Vaughan, D. C. (2002). Communications in Statistics - Theory and Methods 31, 219-238.

    Examples
    --------
    >>> round(ghsecant(0.0, t=-math.pi / 2)["pdf"], 12)
    0.5
    """
    if not (t > -math.pi and scale > 0):
        raise ValueError("need t > -pi and scale > 0")
    a, c2, c1 = _consts(t)

    def pdf(v):
        y = c2 * (v - loc) / scale
        if y > 0:  # divide through by e^{2y} to avoid overflow
            e = math.exp(-y)
            return c1 * e / (1 + 2 * a * e + e * e) / scale
        e = math.exp(y)
        return c1 * e / (e * e + 2 * a * e + 1) / scale

    def cdf(v):
        y = c2 * (v - loc) / scale
        if t == 0:
            return 1 / (1 + math.exp(-y)) if y >= 0 else math.exp(y) / (1 + math.exp(y))
        if t < 0:
            w = math.exp(min(y, 700.0))
            return 1 + math.pi / (2 * t) + math.atan((w + math.cos(t)) / math.sin(t)) / t
        if y > 0:
            e = math.exp(-y)
            return 1 + math.log((1 + math.exp(-t) * e) / (1 + math.exp(t) * e)) / (2 * t)
        w = math.exp(y)
        return 1 + math.log((w + math.exp(-t)) / (w + math.exp(t))) / (2 * t)

    def qf(u):
        if t == 0:
            y = math.log(u / (1 - u))
        elif t < 0:
            y = math.log(-math.sin(t) / math.tan(t * (u - 1)) - math.cos(t))
        else:
            r = math.exp(2 * t * (u - 1))
            y = math.log((r * math.exp(t) - math.exp(-t)) / (1 - r))
        return loc + scale * y / c2

    payload = {}
    if x is not None:
        xs, sc = vec(x)
        d = [pdf(v) for v in xs]
        payload["pdf"] = out(d, sc)
        payload["logpdf"] = out([math.log(v) if v > 0 else -math.inf for v in d], sc)
        payload["cdf"] = out([cdf(v) for v in xs], sc)
    if p is not None:
        ps, sc = vec(p)
        payload["quantile"] = out([qf(u) for u in ps], sc)
    if n:
        payload["random"] = draws(qf, n, seed)
    return RichResult(title="Generalized secant hyperbolic", summary_lines=[("t", t)], payload=payload)


def cheatsheet():
    return (
        "ghsecant: Vaughan's generalized secant hyperbolic distribution (standardised), closed-form cdf and quantile."
    )
