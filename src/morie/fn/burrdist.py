"""Burr (type XII) distribution: density, distribution function, quantile and random draws.

Burr, I. W. (1942). Cumulative frequency functions. Annals of Mathematical Statistics 13, 215-232.
"""

import math

from ._distcore import draws, out, vec
from ._richresult import RichResult

__all__ = ["burrdist"]


def burrdist(x=None, shape1=None, shape2=None, scale=1.0, p=None, n=0, seed=0):
    r"""Burr (type XII) distribution: F(x) = 1 - (1 + (x/scale)^shape2)^(-shape1), x > 0.

    Returns the density (and its log), the distribution function at ``x``, the
    quantile function at ``p`` and ``n`` draws by inversion of morie's Philox
    uniforms (``seed``; identical in the R arm).

    Parameters
    ----------
    x : float or sequence, optional
        Evaluation points.
    shape1 : float
        Parameter (required).
    shape2 : float
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
    Burr, I. W. (1942). Cumulative frequency functions. Annals of Mathematical Statistics 13, 215-232.
    Matches ``actuar::dburr(x, shape1, shape2, scale = scale)``.

    Examples
    --------
    >>> round(burrdist(1.0, shape1=2.3, shape2=1.6, scale=1.4)["cdf"], 10)
    0.6526657008
    """
    if shape1 is None or shape2 is None:
        raise ValueError("shape1, shape2 are required")
    if not (shape1 > 0 and shape2 > 0 and scale > 0):
        raise ValueError("invalid parameters: need shape1 > 0 and shape2 > 0 and scale > 0")

    def pdf(v):
        return (
            0.0
            if v <= 0
            else shape1 * shape2 * (v / scale) ** shape2 / (v * (1 + (v / scale) ** shape2) ** (shape1 + 1))
        )

    def cdf(v):
        return 0.0 if v <= 0 else 1 - (1 + (v / scale) ** shape2) ** (-shape1)

    def qf(u):
        return scale * ((1 - u) ** (-1 / shape1) - 1) ** (1 / shape2)

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
        title="Burr (type XII) distribution",
        summary_lines=[(k, v) for k, v in payload.items() if not isinstance(v, list)],
        payload=payload,
    )


def cheatsheet():
    return "burrdist: Burr (type XII) distribution; pdf, cdf, quantile, Philox draws."
