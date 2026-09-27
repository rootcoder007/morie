"""Rayleigh distribution: density, distribution function, quantile and random draws.

Rayleigh, Lord (1880). On the resultant of a large number of vibrations. Phil. Mag. 10, 73-78.
"""

import math

from ._distcore import draws, out, vec
from ._richresult import RichResult

__all__ = ["rayleighdist"]


def rayleighdist(x=None, sigma=1.0, p=None, n=0, seed=0):
    r"""Rayleigh distribution: F(x) = 1 - exp(-x^2/(2 sigma^2)), x >= 0.

    Returns the density (and its log), the distribution function at ``x``, the
    quantile function at ``p`` and ``n`` draws by inversion of morie's Philox
    uniforms (``seed``; identical in the R arm).

    Parameters
    ----------
    x : float or sequence, optional
        Evaluation points.
    sigma : float
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
    Rayleigh, Lord (1880). On the resultant of a large number of vibrations. Phil. Mag. 10, 73-78.
    Matches ``extraDistr::drayleigh(x, sigma = sigma)``.

    Examples
    --------
    >>> round(rayleighdist(1.0, sigma=1.7)["cdf"], 10)
    0.1588711166
    """
    if not (sigma > 0):
        raise ValueError("invalid parameters: need sigma > 0")

    def pdf(v):
        return 0.0 if v < 0 else v / sigma**2 * math.exp(-v * v / (2 * sigma**2))

    def cdf(v):
        return 0.0 if v < 0 else 1 - math.exp(-v * v / (2 * sigma**2))

    def qf(u):
        return sigma * math.sqrt(-2 * math.log(1 - u))

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
        title="Rayleigh distribution",
        summary_lines=[(k, v) for k, v in payload.items() if not isinstance(v, list)],
        payload=payload,
    )


def cheatsheet():
    return "rayleighdist: Rayleigh distribution; pdf, cdf, quantile, Philox draws."
