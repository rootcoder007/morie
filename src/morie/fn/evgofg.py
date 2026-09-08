# morie.fn -- slice k04 (rootcoder007/morie)
"""Anderson-Darling goodness-of-fit test for a fitted GPD.

Same A^2 statistic as :mod:`morie.fn.evgofa`, imported from
:mod:`morie.fn.hedderich7e33` where it is defined once from the corpus
PDF (Hedderich, Sachs and Reynarowych eq (7.33); Anderson and Darling
1952).  Only the probability-integral transform changes: here it is the
generalized Pareto distribution, Coles (2001), *An Introduction to
Statistical Modeling of Extreme Values*, eq (4.2):

    H(y) = 1 - (1 + xi y / sigma)^(-1/xi)   on y > 0, 1 + xi y/sigma > 0
    H(y) = 1 - exp(-y / sigma)              when xi = 0

``y`` are threshold excesses, so they must be positive; the caller
subtracts the threshold.  As in ``evgofa`` the parameters are supplied
rather than fitted, to keep the statistic reproducible across language
arms -- the package's GPD fitter is Nelder-Mead and agrees only to
about 1e-4.  No p-value is returned: Stephens (1986) critical values
depend on which parameters were estimated and how, and that chapter was
not obtainable here.

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  Deleted.
"""

from __future__ import annotations

import math

from . import _array_core as np

from ._richresult import RichResult
from .hedderich7e33 import ad_statistic

__all__ = ["evt_gpd_anderson_darling"]


def gpd_cdf(y, sigma=1.0, xi=0.0):
    """GPD distribution function of threshold excesses, Coles (2001) eq (4.2)."""
    y = np.asarray(y, dtype=float).ravel()
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    xi = float(xi)
    out = np.empty(y.size, dtype=float)
    for i in range(y.size):
        yi = float(y[i])
        if yi <= 0.0:
            out[i] = 0.0
        elif xi == 0.0:
            out[i] = 1.0 - math.exp(-yi / sigma)
        else:
            t = 1.0 + xi * yi / sigma
            out[i] = 1.0 if t <= 0.0 else 1.0 - t ** (-1.0 / xi)
    return out


def evt_gpd_anderson_darling(y, sigma=1.0, xi=0.0):
    """Anderson-Darling A^2 for a GPD with the given parameters.

    Parameters
    ----------
    y : array-like
        Threshold excesses; must be positive.
    sigma, xi : float
        Scale and shape.  Supplied, not fitted.

    Returns
    -------
    RichResult
        keys: ``statistic`` (A^2), ``u`` (sorted PIT values), ``sigma``,
        ``xi``, ``n``, ``method``.
    """
    y = np.asarray(y, dtype=float).ravel()
    if bool(np.any(y <= 0.0)):
        raise ValueError("threshold excesses must be positive; subtract the threshold first")
    u = gpd_cdf(y, sigma, xi)
    return RichResult(
        payload={
            "statistic": ad_statistic(u),
            "u": np.sort(u),
            "sigma": float(sigma),
            "xi": float(xi),
            "n": int(y.size),
            "method": "Anderson-Darling A^2 for a fitted GPD (Anderson and Darling 1952; Coles 2001 eq. 4.2)",
        }
    )


def cheatsheet():
    return "evgofg: Anderson-Darling A^2 for a GPD fit"
