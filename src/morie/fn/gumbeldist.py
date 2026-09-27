"""Gumbel (type I extreme value) distribution: density, distribution function, quantile and random draws.

Gumbel, E. J. (1958). Statistics of Extremes. Columbia University Press.
"""

import math

from ._distcore import draws, out, vec
from ._richresult import RichResult

__all__ = ["gumbeldist"]


def gumbeldist(x=None, loc=0.0, scale=1.0, p=None, n=0, seed=0):
    r"""Gumbel (type I extreme value) distribution: F(x) = exp(-exp(-(x - loc)/scale)).

    Returns the density (and its log), the distribution function at ``x``, the
    quantile function at ``p`` and ``n`` draws by inversion of morie's Philox
    uniforms (``seed``; identical in the R arm).

    Parameters
    ----------
    x : float or sequence, optional
        Evaluation points.
    loc : float
        Parameter.
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
    Gumbel, E. J. (1958). Statistics of Extremes. Columbia University Press.
    Matches ``extraDistr::dgumbel(x, mu = loc, sigma = scale)``.

    Examples
    --------
    >>> round(gumbeldist(1.0, loc=1.0, scale=2.0)["cdf"], 10)
    0.3678794412
    """
    if not (scale > 0):
        raise ValueError("invalid parameters: need scale > 0")

    def pdf(v):
        return math.exp(-(v - loc) / scale - math.exp(-(v - loc) / scale)) / scale

    def cdf(v):
        return math.exp(-math.exp(-(v - loc) / scale))

    def qf(u):
        return loc - scale * math.log(-math.log(u))

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
        title="Gumbel (type I extreme value) distribution",
        summary_lines=[(k, v) for k, v in payload.items() if not isinstance(v, list)],
        payload=payload,
    )


def cheatsheet():
    return "gumbeldist: Gumbel (type I extreme value) distribution; pdf, cdf, quantile, Philox draws."
