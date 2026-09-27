"""Circular uniform distribution: density, distribution function, quantile and random draws.

Mardia, K. V. & Jupp, P. E. (2000). Directional Statistics. Wiley, Sec 3.5.3.
"""

import math

from ._distcore import draws, out, vec
from ._richresult import RichResult

__all__ = ["circunif"]


def circunif(x=None, p=None, n=0, seed=0):
    r"""Circular uniform distribution: f(t) = 1/(2 pi), 0 <= t < 2 pi.

    Returns the density (and its log), the distribution function at ``x``, the
    quantile function at ``p`` and ``n`` draws by inversion of morie's Philox
    uniforms (``seed``; identical in the R arm).

    Parameters
    ----------
    x : float or sequence, optional
        Evaluation points.
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
    Mardia, K. V. & Jupp, P. E. (2000). Directional Statistics. Wiley, Sec 3.5.3.
    Matches ``circular::dcircularuniform(x)``.

    Examples
    --------
    >>> round(circunif(1.0)["cdf"], 10)
    0.1591549431
    """
    if not (True):
        raise ValueError("invalid parameters: need True")

    def pdf(v):
        return 1 / (2 * math.pi) if 0 <= v <= 2 * math.pi else 0.0

    def cdf(v):
        return 0.0 if v <= 0 else 1.0 if v >= 2 * math.pi else v / (2 * math.pi)

    def qf(u):
        return 2 * math.pi * u

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
        title="Circular uniform distribution",
        summary_lines=[(k, v) for k, v in payload.items() if not isinstance(v, list)],
        payload=payload,
    )


def cheatsheet():
    return "circunif: Circular uniform distribution; pdf, cdf, quantile, Philox draws."
