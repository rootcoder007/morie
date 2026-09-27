"""Dagum distribution: density, distribution function, quantile and random draws.

Dagum, C. (1977). A new model of personal income distribution. Economie Appliquee 30, 413-437.
"""

import math

from ._distcore import draws, out, vec
from ._richresult import RichResult

__all__ = ["dagumdist"]


def dagumdist(x=None, shape1a=None, shape2p=None, scale=1.0, p=None, n=0, seed=0):
    r"""Dagum distribution: F(x) = (1 + (x/scale)^(-a))^(-p), x > 0.

    Returns the density (and its log), the distribution function at ``x``, the
    quantile function at ``p`` and ``n`` draws by inversion of morie's Philox
    uniforms (``seed``; identical in the R arm).

    Parameters
    ----------
    x : float or sequence, optional
        Evaluation points.
    shape1a : float
        Parameter (required).
    shape2p : float
        Parameter (required).
    scale : float
        Parameter.
    p : float or sequence, optional
        Probabilities for the quantile function.
    n : int
        Number of random draws.
    seed : int

    Returns
    -------
    RichResult
        Keys: pdf, logpdf, cdf (when x is given), quantile (when p is given), random.

    References
    ----------
    Dagum, C. (1977). A new model of personal income distribution. Economie Appliquee 30, 413-437.
    Matches ``VGAM::ddagum(x, scale = scale, shape1.a = shape1a, shape2.p = shape2p)``.

    Examples
    --------
    >>> round(dagumdist(1.0, shape1a=2.5, shape2p=0.7, scale=1.2)["cdf"], 10)
    0.5154278621
    """
    if shape1a is None or shape2p is None:
        raise ValueError("shape1a, shape2p are required")
    if not (shape1a > 0 and shape2p > 0 and scale > 0):
        raise ValueError("invalid parameters: need shape1a > 0 and shape2p > 0 and scale > 0")

    def pdf(v):
        return (
            0.0
            if v <= 0
            else shape1a
            * shape2p
            * v ** (shape1a * shape2p - 1)
            / (scale ** (shape1a * shape2p) * (1 + (v / scale) ** shape1a) ** (shape2p + 1))
        )

    def cdf(v):
        return 0.0 if v <= 0 else (1 + (v / scale) ** (-shape1a)) ** (-shape2p)

    def qf(u):
        return scale * (u ** (-1 / shape2p) - 1) ** (-1 / shape1a)

    payload = {}
    if x is not None:
        xs, sc = vec(x)
        dens = [pdf(v) for v in xs]
        payload["pdf"] = out(dens, sc)
        payload["logpdf"] = out([math.log(v) if v > 0 else -math.inf for v in dens], sc)
        payload["cdf"] = out([cdf(v) for v in xs], sc)
    if p is not None:
        ps, sc = vec(p)
        payload["quantile"] = out([qf(u) for u in ps], sc)
    if n:
        payload["random"] = draws(qf, n, seed)
    return RichResult(
        title="Dagum distribution",
        summary_lines=[(k, v) for k, v in payload.items() if not isinstance(v, list)],
        payload=payload,
    )


def cheatsheet():
    return "dagumdist: Dagum distribution; pdf, cdf, quantile, Philox draws."
