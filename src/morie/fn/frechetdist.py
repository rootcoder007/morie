"""Frechet distribution: density, distribution function, quantile and random draws.

Frechet, M. (1927). Sur la loi de probabilite de l'ecart maximum. Ann. Soc. Polon. Math. 6, 93-116.
"""

import math

from ._distcore import draws, out, vec
from ._richresult import RichResult

__all__ = ["frechetdist"]


def frechetdist(x=None, shape=None, loc=0.0, scale=1.0, p=None, n=0, seed=0):
    r"""Frechet distribution: F(x) = exp(-((x - loc)/scale)^(-shape)), x > loc.

    Returns the density (and its log), the distribution function at ``x``, the
    quantile function at ``p`` and ``n`` draws by inversion of morie's Philox
    uniforms (``seed``; identical in the R arm).

    Parameters
    ----------
    x : float or sequence, optional
        Evaluation points.
    shape : float
        Parameter (required).
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
    Frechet, M. (1927). Sur la loi de probabilite de l'ecart maximum. Ann. Soc. Polon. Math. 6, 93-116.
    Matches ``extraDistr::dfrechet(x, lambda = shape, mu = loc, sigma = scale)``.

    Examples
    --------
    >>> round(frechetdist(1.0, shape=2.2, loc=0.3, scale=1.5)["cdf"], 10)
    0.0047580269
    """
    if shape is None:
        raise ValueError("shape is required")
    if not (shape > 0 and scale > 0):
        raise ValueError("invalid parameters: need shape > 0 and scale > 0")

    def pdf(v):
        return (
            0.0
            if v <= loc
            else shape / scale * ((v - loc) / scale) ** (-1 - shape) * math.exp(-(((v - loc) / scale) ** (-shape)))
        )

    def cdf(v):
        return 0.0 if v <= loc else math.exp(-(((v - loc) / scale) ** (-shape)))

    def qf(u):
        return loc + scale * (-math.log(u)) ** (-1 / shape)

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
        title="Frechet distribution",
        summary_lines=[(k, v) for k, v in payload.items() if not isinstance(v, list)],
        payload=payload,
    )


def cheatsheet():
    return "frechetdist: Frechet distribution; pdf, cdf, quantile, Philox draws."
