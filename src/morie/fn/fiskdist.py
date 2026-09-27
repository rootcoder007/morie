"""Fisk (log-logistic) distribution: density, distribution function, quantile and random draws.

Fisk, P. R. (1961). The graduation of income distributions. Econometrica 29, 171-185.
"""

import math

from ._distcore import draws, out, vec
from ._richresult import RichResult

__all__ = ["fiskdist"]


def fiskdist(x=None, shape=None, scale=1.0, p=None, n=0, seed=0):
    r"""Fisk (log-logistic) distribution: F(x) = 1 / (1 + (x/scale)^(-shape)), x > 0.

    Returns the density (and its log), the distribution function at ``x``, the
    quantile function at ``p`` and ``n`` draws by inversion of morie's Philox
    uniforms (``seed``; identical in the R arm).

    Parameters
    ----------
    x : float or sequence, optional
        Evaluation points.
    shape : float
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
    Fisk, P. R. (1961). The graduation of income distributions. Econometrica 29, 171-185.
    Matches ``actuar::dllogis(x, shape = shape, scale = scale)``.

    Examples
    --------
    >>> round(fiskdist(1.0, shape=3.1, scale=0.8)["cdf"], 10)
    0.6663550002
    """
    if shape is None:
        raise ValueError("shape is required")
    if not (shape > 0 and scale > 0):
        raise ValueError("invalid parameters: need shape > 0 and scale > 0")

    def pdf(v):
        return 0.0 if v <= 0 else (shape / scale) * (v / scale) ** (shape - 1) / (1 + (v / scale) ** shape) ** 2

    def cdf(v):
        return 0.0 if v <= 0 else 1 / (1 + (v / scale) ** (-shape))

    def qf(u):
        return scale * (u / (1 - u)) ** (1 / shape)

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
        title="Fisk (log-logistic) distribution",
        summary_lines=[(k, v) for k, v in payload.items() if not isinstance(v, list)],
        payload=payload,
    )


def cheatsheet():
    return "fiskdist: Fisk (log-logistic) distribution; pdf, cdf, quantile, Philox draws."
