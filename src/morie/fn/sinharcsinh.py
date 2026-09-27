"""Sinh-arcsinh distribution: density, distribution function, quantile and random draws.

Jones, M. C. & Pewsey, A. (2009). Sinh-arcsinh distributions. Biometrika 96, 761-780.
"""

import math

from ._distcore import draws, out, vec
from ._richresult import RichResult
from ._rrng_core import pnorm, qnorm

__all__ = ["sinharcsinh"]


def sinharcsinh(x=None, loc=0.0, scale=1.0, eps=0.0, delta=1.0, p=None, n=0, seed=0):
    r"""Sinh-arcsinh distribution: Z = sinh(delta asinh((x - loc)/scale) - eps) ~ N(0, 1).

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
    eps : float
        Parameter.
    delta : float
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
    Jones, M. C. & Pewsey, A. (2009). Sinh-arcsinh distributions. Biometrika 96, 761-780.
    Matches ``gamlss.dist::dSHASHo(x, mu = loc, sigma = scale, nu = eps, tau = delta)``.

    Examples
    --------
    >>> round(sinharcsinh(0.2, loc=0.1, scale=1.2, eps=0.4, delta=0.8)["cdf"], 10)
    0.3670706826
    """
    if not (scale > 0 and delta > 0):
        raise ValueError("invalid parameters: need scale > 0 and delta > 0")

    def pdf(v):
        return (
            delta
            * math.cosh(delta * math.asinh((v - loc) / scale) - eps)
            / (scale * math.sqrt(2 * math.pi) * math.sqrt(1 + ((v - loc) / scale) ** 2))
            * math.exp(-0.5 * math.sinh(delta * math.asinh((v - loc) / scale) - eps) ** 2)
        )

    def cdf(v):
        return pnorm(math.sinh(delta * math.asinh((v - loc) / scale) - eps))

    def qf(u):
        return loc + scale * math.sinh((math.asinh(qnorm(u)) + eps) / delta)

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
        title="Sinh-arcsinh distribution",
        summary_lines=[(k, v) for k, v in payload.items() if not isinstance(v, list)],
        payload=payload,
    )


def cheatsheet():
    return "sinharcsinh: Sinh-arcsinh distribution; pdf, cdf, quantile, Philox draws."
