# morie.fn -- slice s03 (rootcoder007/morie)
"""The Matern correlation function.

Matern, B. (1960), *Spatial Variation*, Meddelanden fran Statens
Skogsforskningsinstitut 49(5), 1-144 (reprinted as Springer Lecture Notes
in Statistics 36, 1986).  Pages 17 and 18 were rendered at 150 dpi with
pdftoppm and read as images; the scan's text layer is Paper-Capture OCR
and mangles the Bessel subscripts.

Section 2.4, "Examples of correlation functions", builds the family from
the spectral side.  Equation (2.4.5) p. 17 mixes the Gaussian correlation
exp(-a^2 v^2) over a type III distribution for a^2 and gets
(1 + v^2/b^2)^-s; equation (2.4.6) p. 17 gives the matching spectral
density const. w^(s - n/2) K_(s - n/2)(wb).  Fourier-transforming that
density back, eq. (2.4.7) p. 18 states the correlation function itself::

    2 (b v / 2)^nu K_nu(b v) / Gamma(nu)          (2.4.7)

for b, nu >= 0, where K_nu is the modified Bessel function of the second
kind and v is the lag.  Matern cites Watson (1944, p. 80) for the
constant.

Two special cases are printed on the same page and are used here as
anchors, since they are the author's own numbers rather than ours:

    nu = 1/2   exp(-b v)                          (2.4.8) p. 18
    nu = 1     v b K_1(v b)                       (2.4.9) p. 18

Parameterisation.  Modern usage writes the same function as

    sigma^2 (2^(1-nu) / Gamma(nu)) (sqrt(2 nu) d / rho)^nu
        K_nu(sqrt(2 nu) d / rho),

which is exactly (2.4.7) with b v = sqrt(2 nu) d / rho and an added scale
sigma^2, because 2 (z/2)^nu / Gamma(nu) = 2^(1-nu) z^nu / Gamma(nu).  The
rho form is what this module takes, and ``b`` is reported in the payload
so the reader can get back to Matern's own variable.

The value at v = 0 is sigma^2: z^nu K_nu(z) -> 2^(nu-1) Gamma(nu) as
z -> 0, which is the whole point of the constant in (2.4.7).  As
nu -> infinity the function tends to the Gaussian correlation
exp(-d^2 / (2 rho^2)), Matern's own eq. (2.4.2) p. 17.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["matern_kernel"]


def _matern_one(d, nu, rho, sigma2):
    """Equation (2.4.7) p. 18 at a single lag."""
    if d < 0.0:
        raise ValueError("matern_kernel: distances d must be non-negative")
    if d == 0.0:
        return sigma2
    z = math.sqrt(2.0 * nu) * d / rho
    # 2 (z/2)^nu K_nu(z) / Gamma(nu), written in logs so that large nu
    # does not overflow (z/2)^nu before Gamma(nu) divides it out again.
    lead = math.log(2.0) + nu * math.log(z / 2.0) - core.lgamma(nu)
    return sigma2 * math.exp(lead) * core.besselk(nu, z)


def matern_kernel(d, nu, rho, sigma2=1.0):
    """The Matern correlation function, Matern (1960) eq. (2.4.7) p. 18.

    Parameters
    ----------
    d : array-like
        Lags v, all non-negative.
    nu : float
        Smoothness, Matern's nu of (2.4.7); must be positive.
    rho : float
        Range.  Matern's own inverse scale is b = sqrt(2 nu) / rho.
    sigma2 : float, optional
        Variance at lag zero; (2.4.7) itself is the correlation, so the
        default 1.0 reproduces the printed function exactly.

    Returns
    -------
    estimate : the kernel at the first supplied lag
    k        : the kernel at every supplied lag
    nu, rho, sigma2, b : the parameters actually used, b as in (2.4.7)
    """
    v = core.vec(d)
    if len(v) == 0:
        raise ValueError("matern_kernel: no distances supplied")
    nu = float(nu)
    rho = float(rho)
    sigma2 = float(sigma2)
    if nu != nu or not (nu > 0.0):
        raise ValueError("matern_kernel: the smoothness nu must be positive")
    if rho != rho or not (rho > 0.0):
        raise ValueError("matern_kernel: the range rho must be positive")
    if sigma2 != sigma2 or sigma2 < 0.0:
        raise ValueError("matern_kernel: the variance sigma2 must be non-negative")
    k = [_matern_one(x, nu, rho, sigma2) for x in v]
    return RichResult(
        payload={
            "estimate": k[0],
            "k": k,
            "d": v,
            "nu": nu,
            "rho": rho,
            "sigma2": sigma2,
            "b": math.sqrt(2.0 * nu) / rho,
            "n": len(v),
            "method": "Matern (1960) correlation function, eq. (2.4.7) p. 18",
        }
    )


def cheatsheet():
    return "matnK: Matern kernel"
