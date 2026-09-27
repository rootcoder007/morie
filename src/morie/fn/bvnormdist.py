"""Bivariate normal density and distribution function.

Owen, D. B. (1956). Tables for computing bivariate normal probabilities. Annals of Mathematical Statistics 27, 1075-1090.
"""

import math

from ._distcore import gauss_legendre
from ._richresult import RichResult
from ._rrng_core import pnorm

__all__ = ["bvnormdist"]


def bvnormdist(x, y, mean1=0.0, mean2=0.0, sd1=1.0, sd2=1.0, rho=0.0):
    r"""Density and P(X <= x, Y <= y) of the bivariate normal with correlation rho.

    With h = (x - mu1)/sd1 and k = (y - mu2)/sd2,
    Phi_2(h, k; rho) = integral_{-inf}^{h} phi(t) Phi((k - rho t)/sqrt(1 - rho^2)) dt,
    evaluated by composite 16-point Gauss-Legendre from -9 (the neglected mass
    is below 1e-18); rho = +-1 is handled exactly.

    Parameters
    ----------
    x, y : float
    mean1, mean2, sd1, sd2 : float
    rho : float
        Correlation in [-1, 1].

    Returns
    -------
    RichResult
        Keys: pdf, cdf.

    References
    ----------
    Owen, D. B. (1956). Annals of Mathematical Statistics 27, 1075-1090.
    Matches ``mvtnorm::pmvnorm`` and ``mvtnorm::dmvnorm`` in two dimensions.

    Examples
    --------
    >>> round(bvnormdist(0.0, 0.0, rho=0.5)["cdf"], 12)
    0.333333333333
    """
    if not (sd1 > 0 and sd2 > 0 and -1 <= rho <= 1):
        raise ValueError("need sd1, sd2 > 0 and -1 <= rho <= 1")
    h = (x - mean1) / sd1
    k = (y - mean2) / sd2
    if abs(rho) == 1:
        pdf = 0.0
        cdf = min(pnorm(h), pnorm(k)) if rho == 1 else max(0.0, pnorm(h) + pnorm(k) - 1)
    else:
        s = math.sqrt(1 - rho * rho)
        pdf = math.exp(-(h * h - 2 * rho * h * k + k * k) / (2 * s * s)) / (2 * math.pi * sd1 * sd2 * s)
        lo = -9.0
        if h <= lo:
            cdf = 0.0
        else:
            f = lambda t: math.exp(-t * t / 2) / math.sqrt(2 * math.pi) * pnorm((k - rho * t) / s)  # noqa: E731
            cdf = gauss_legendre(f, lo, h, n=max(64, int(8 * (h - lo))))
    return RichResult(title="Bivariate normal", summary_lines=[("cdf", cdf)], payload={"pdf": pdf, "cdf": cdf})


def cheatsheet():
    return "bvnormdist: bivariate normal density and cdf (Owen's integral by quadrature)."
