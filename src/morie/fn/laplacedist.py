"""Laplace distribution: density, distribution function, quantile and random draws.

Johnson, N. L., Kotz, S. & Balakrishnan, N. (1995). Continuous Univariate Distributions, Vol. 2, ch. 24.
"""

import math

from ._distcore import draws, out, vec
from ._richresult import RichResult

__all__ = ["laplacedist"]


def laplacedist(x=None, loc=0.0, scale=1.0, p=None, n=0, seed=0):
    r"""Laplace distribution: f(x) = exp(-|x - loc|/scale) / (2 scale).

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
    Johnson, N. L., Kotz, S. & Balakrishnan, N. (1995). Continuous Univariate Distributions, Vol. 2, ch. 24.
    Matches ``extraDistr::dlaplace(x, mu = loc, sigma = scale)``.

    Examples
    --------
    >>> round(laplacedist(0.2, loc=0.5, scale=1.3)["cdf"], 10)
    0.3969613289
    """
    if not (scale > 0):
        raise ValueError("invalid parameters: need scale > 0")

    def pdf(v):
        return math.exp(-abs(v - loc) / scale) / (2 * scale)

    def cdf(v):
        return 0.5 * math.exp((v - loc) / scale) if v < loc else 1 - 0.5 * math.exp(-(v - loc) / scale)

    def qf(u):
        return loc + scale * math.log(2 * u) if u < 0.5 else loc - scale * math.log(2 - 2 * u)

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
        title="Laplace distribution",
        summary_lines=[(k, v) for k, v in payload.items() if not isinstance(v, list)],
        payload=payload,
    )


def cheatsheet():
    return "laplacedist: Laplace distribution; pdf, cdf, quantile, Philox draws."
