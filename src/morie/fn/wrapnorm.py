"""Wrapped normal distribution: density, distribution function, quantile and random draws.

Mardia, K. V. & Jupp, P. E. (2000). Directional Statistics. Wiley, Sec 3.5.7; rho = exp(-sigma^2/2).
"""

import math

from ._distcore import bisect_q, draws, out, vec
from ._richresult import RichResult
from ._rrng_core import pnorm

__all__ = ["wrapnorm"]


def wrapnorm(x=None, mu=0.0, rho=0.5, p=None, n=0, seed=0):
    r"""Wrapped normal distribution: f(t) = sum_k phi((t - mu + 2 pi k)/sigma)/sigma, sigma^2 = -2 log rho.

    Returns the density (and its log), the distribution function at ``x``, the
    quantile function at ``p`` and ``n`` draws by inversion of morie's Philox
    uniforms (``seed``; identical in the R arm).

    Parameters
    ----------
    x : float or sequence, optional
        Evaluation points.
    mu : float
        Parameter.
    rho : float
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
    Mardia, K. V. & Jupp, P. E. (2000). Directional Statistics. Wiley, Sec 3.5.7; rho = exp(-sigma^2/2).
    Matches ``circular::dwrappednormal(x, mu = circular(mu), rho = rho)``.

    Examples
    --------
    >>> round(wrapnorm(2.0, mu=2.0, rho=0.6)["cdf"], 10)
    0.47608607
    """
    if not (0 < rho < 1):
        raise ValueError("invalid parameters: need 0 < rho < 1")
    sd = math.sqrt(-2 * math.log(rho))
    kmax = int(10 + 6 * sd)

    def pdf(v):
        return (
            sum(math.exp(-0.5 * ((v - mu + 2 * math.pi * k) / sd) ** 2) for k in range(-kmax, kmax + 1))
            / (sd * math.sqrt(2 * math.pi))
            if 0 <= v <= 2 * math.pi
            else 0.0
        )

    def cdf(v):
        return (
            0.0
            if v <= 0
            else 1.0
            if v >= 2 * math.pi
            else sum(
                pnorm((v - mu + 2 * math.pi * k) / sd) - pnorm((-mu + 2 * math.pi * k) / sd)
                for k in range(-kmax, kmax + 1)
            )
        )

    def qf(u):
        return bisect_q(cdf, u, 0.0, 2 * math.pi)

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
        title="Wrapped normal distribution",
        summary_lines=[(k, v) for k, v in payload.items() if not isinstance(v, list)],
        payload=payload,
    )


def cheatsheet():
    return "wrapnorm: Wrapped normal distribution; pdf, cdf, quantile, Philox draws."
