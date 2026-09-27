"""Wrapped Cauchy distribution: density, distribution function, quantile and random draws.

Mardia, K. V. & Jupp, P. E. (2000). Directional Statistics. Wiley, Sec 3.5.7.
"""

import math

from ._distcore import bisect_q, draws, gauss_legendre, out, vec
from ._richresult import RichResult

__all__ = ["wrapcauchy"]


def wrapcauchy(x=None, mu=0.0, rho=0.5, p=None, n=0, seed=0):
    r"""Wrapped Cauchy distribution: f(t) = (1 - rho^2) / (2 pi (1 + rho^2 - 2 rho cos(t - mu))), 0 <= t < 2 pi.

    Returns the density (and its log), the distribution function at ``x``, the
    quantile function at ``p`` and ``n`` draws by inversion of morie's Philox
    uniforms (``seed``; identical in the R arm). The distribution function is
    integrated numerically (composite 16-point Gauss-Legendre) and the quantile found by bisection.

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
    Mardia, K. V. & Jupp, P. E. (2000). Directional Statistics. Wiley, Sec 3.5.7.
    Matches ``circular::dwrappedcauchy(x, mu = circular(mu), rho = rho)``.

    Examples
    --------
    >>> round(wrapcauchy(2.0, mu=2.0, rho=0.6)["cdf"], 10)
    0.4493361064
    """
    if not (0 <= rho < 1):
        raise ValueError("invalid parameters: need 0 <= rho < 1")

    def pdf(v):
        return (
            (1 - rho * rho) / (2 * math.pi * (1 + rho * rho - 2 * rho * math.cos(v - mu)))
            if 0 <= v <= 2 * math.pi
            else 0.0
        )

    def cdf(v):
        if v <= 0.0:
            return 0.0
        if v >= 2 * math.pi:
            return 1.0
        return min(1.0, gauss_legendre(pdf, 0.0, v))

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
        title="Wrapped Cauchy distribution",
        summary_lines=[(k, v) for k, v in payload.items() if not isinstance(v, list)],
        payload=payload,
    )


def cheatsheet():
    return "wrapcauchy: Wrapped Cauchy distribution; pdf, cdf, quantile, Philox draws."
