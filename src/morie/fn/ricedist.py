"""Rice distribution: density, distribution function, quantile and random draws.

Rice, S. O. (1945). Mathematical analysis of random noise. Bell System Technical Journal 24, 46-156.
"""

import math

from ._distcore import bisect_q, draws, gauss_legendre, log_i0, out, vec
from ._richresult import RichResult

__all__ = ["ricedist"]


def ricedist(x=None, sigma=1.0, vee=0.0, p=None, n=0, seed=0):
    r"""Rice distribution: f(x) = (x/sigma^2) exp(-(x^2 + v^2)/(2 sigma^2)) I_0(x v/sigma^2), x >= 0.

    The density is evaluated in log space with log I_0 (series below 50, asymptotic
    expansion above); the distribution function is the integral of the density
    (composite 16-point Gauss-Legendre) and the quantile is found by bisection.
    Draws invert morie's Philox uniforms (identical in the R arm). vee = 0 gives
    the Rayleigh distribution.

    Parameters
    ----------
    x : float or sequence, optional
    sigma : float
        Scale, > 0.
    vee : float
        Non-centrality, >= 0.
    p : float or sequence, optional
    n : int
    seed : int

    Returns
    -------
    RichResult
        Keys: pdf, logpdf, cdf, quantile, random.

    References
    ----------
    Rice, S. O. (1945). Bell System Technical Journal 24, 46-156.
    Matches ``VGAM::drice(x, sigma, vee)``.

    Examples
    --------
    >>> round(ricedist(1.0, sigma=1.0, vee=0.0)["cdf"], 12)
    0.393469340287
    """
    if not (sigma > 0 and vee >= 0):
        raise ValueError("need sigma > 0 and vee >= 0")
    s2 = sigma * sigma

    def logpdf(v):
        if v <= 0:
            return -math.inf
        return math.log(v) - math.log(s2) - (v * v + vee * vee) / (2 * s2) + log_i0(v * vee / s2)

    def pdf(v):
        lp = logpdf(v)
        return math.exp(lp) if lp > -math.inf else 0.0

    def cdf(v):
        if v <= 0:
            return 0.0
        return min(1.0, gauss_legendre(pdf, 0.0, v))

    def qf(u):
        return bisect_q(cdf, u, 0.0, math.inf)

    payload = {}
    if x is not None:
        xs, sc = vec(x)
        payload["pdf"] = out([pdf(v) for v in xs], sc)
        payload["logpdf"] = out([logpdf(v) for v in xs], sc)
        payload["cdf"] = out([cdf(v) for v in xs], sc)
    if p is not None:
        ps, sc = vec(p)
        payload["quantile"] = out([qf(u) for u in ps], sc)
    if n:
        payload["random"] = draws(qf, n, seed)
    return RichResult(
        title="Rice distribution",
        summary_lines=[(k, v) for k, v in payload.items() if not isinstance(v, list)],
        payload=payload,
    )


def cheatsheet():
    return "ricedist: Rice distribution; pdf via log I0, cdf by quadrature, bisection quantile, Philox draws."
