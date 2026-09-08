# morie.fn -- slice k04 (rootcoder007/morie)
"""Anderson-Darling goodness-of-fit test for a fitted GEV.

The A^2 statistic itself is Anderson and Darling (1952), read from the
corpus PDF as Hedderich, Sachs and Reynarowych, *Applied Statistics:
Methods Using R*, eq (7.33); see :mod:`morie.fn.hedderich7e33`, which
holds the single implementation and records the two extraction defects
in the book's text layer.  It is imported rather than copied.

    A^2 = -n - (1/n) sum_i (2i-1) [ log u_i + log(1 - u_{n+1-i}) ]

Applied here to the probability-integral transform under the
generalized extreme value distribution, Coles (2001), *An Introduction
to Statistical Modeling of Extreme Values*, eq (3.2):

    G(z) = exp( -[1 + xi (z - mu)/sigma]^(-1/xi) )   on 1 + xi(z-mu)/sigma > 0
    G(z) = exp( -exp( -(z - mu)/sigma ) )            when xi = 0

The parameters are supplied by the caller and are NOT fitted here.  That
is deliberate: the package's GEV/GPD fitters use Nelder-Mead, whose
optima agree across language arms only to about 1e-4, and folding one
into this function would make the statistic irreproducible.  Feed it
closed-form probability-weighted-moment or L-moment estimates if you
need fitted parameters.

Stephens (1986) tabulates critical values for A^2, but they depend on
which parameters were estimated and by what method; no p-value is
returned rather than attach a table that may not apply.  The 1986
chapter (in D'Agostino and Stephens, *Goodness-of-Fit Techniques*) was
not obtainable here.

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  Deleted.
"""

from __future__ import annotations

import math

from . import _array_core as np

from ._richresult import RichResult
from .hedderich7e33 import ad_statistic

__all__ = ["evt_gev_anderson_darling"]


def gev_cdf(x, mu=0.0, sigma=1.0, xi=0.0):
    """GEV distribution function, Coles (2001) eq (3.2).

    Returns 0 below and 1 above the support endpoint implied by ``xi``.
    """
    x = np.asarray(x, dtype=float).ravel()
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    xi = float(xi)
    z = (x - float(mu)) / sigma
    out = np.empty(z.size, dtype=float)
    for i in range(z.size):
        zi = float(z[i])
        if xi == 0.0:
            out[i] = math.exp(-math.exp(-zi))
        else:
            t = 1.0 + xi * zi
            if t <= 0.0:
                out[i] = 0.0 if xi > 0.0 else 1.0
            else:
                out[i] = math.exp(-(t ** (-1.0 / xi)))
    return out


def evt_gev_anderson_darling(x, mu=0.0, sigma=1.0, xi=0.0):
    """Anderson-Darling A^2 for a GEV with the given parameters.

    Parameters
    ----------
    x : array-like
        Block maxima.
    mu, sigma, xi : float
        Location, scale and shape.  Supplied, not fitted.

    Returns
    -------
    RichResult
        keys: ``statistic`` (A^2), ``u`` (sorted PIT values), ``mu``,
        ``sigma``, ``xi``, ``n``, ``method``.
    """
    x = np.asarray(x, dtype=float).ravel()
    u = gev_cdf(x, mu, sigma, xi)
    return RichResult(
        payload={
            "statistic": ad_statistic(u),
            "u": np.sort(u),
            "mu": float(mu),
            "sigma": float(sigma),
            "xi": float(xi),
            "n": int(x.size),
            "method": "Anderson-Darling A^2 for a fitted GEV (Anderson and Darling 1952; Coles 2001 eq. 3.2)",
        }
    )


def cheatsheet():
    return "evgofa: Anderson-Darling A^2 for a GEV fit"
