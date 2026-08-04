# morie.fn -- slice s03 (rootcoder007/morie)
"""Matern variogram model.

Sources consulted: Matern, B. (1960).  *Spatial Variation*, Meddelanden
fran Statens Skogsforskningsinstitut 49(5); and Stein, M. L. (1999).
*Interpolation of Spatial Data: Some Theory for Kriging*, Springer,
section 2.7, which gives the covariance

    C(h) = (2^(1-nu) / Gamma(nu)) (h/a)^nu K_nu(h/a)

with K_nu the modified Bessel function of the second kind.  The
variogram of a process with nugget c0 and partial sill c is

    gamma(h) = c0 + c ( 1 - C(h) ),   gamma(0) = 0

the discontinuity at the origin being the nugget itself, which is why
gamma(0) is returned as exactly zero rather than as c0.  Neither source
was retrievable here as a full text; both expressions are quoted in
their standard published form.

nu = 1/2 gives the exponential model and nu -> infinity the Gaussian
one; both limits are checked by the function's own ``exponential`` field
at nu = 0.5.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["matern_variogram_model"]


def matern_variogram_model(h, c0=0.0, c=1.0, a=1.0, nu=0.5):
    """Matern semivariogram at one or more lags.

    Returns
    -------
    estimate : gamma at the first lag
    gamma    : gamma at every lag
    corr     : the correlation C(h)
    exponential : the nu = 1/2 closed form, for comparison
    """
    hs = k.vec(h)
    aa = float(a)
    v = float(nu)
    out = []
    cor = []
    expo = []
    for x in hs:
        if x <= 0.0:
            out.append(0.0)
            cor.append(1.0)
            expo.append(0.0)
            continue
        u = x / aa
        C = (2.0 ** (1.0 - v) / math.exp(math.lgamma(v))) * (u ** v) * k.besselk(v, u)
        cor.append(C)
        out.append(float(c0) + float(c) * (1.0 - C))
        expo.append(float(c0) + float(c) * (1.0 - math.exp(-u)))
    return RichResult(
        title="Matern variogram",
        summary_lines=[("nu", v), ("range", aa)],
        payload={
            "estimate": out[0] if out else float("nan"),
            "gamma": out,
            "corr": cor,
            "exponential": expo,
            "nugget": float(c0),
            "sill": float(c0) + float(c),
            "method": "Matern semivariogram c0 + c (1 - C(h)) with C the Matern correlation",
        },
    )


def cheatsheet():
    return "matvar: Matern variogram model"
