# morie.fn -- function file (rootcoder007/morie)
"""Moran's I asymptotic z-test.

Verified against ``spdep::moran.test``; see
:func:`morie.fn._robust_core.morans_i_test` for the two null
variances (randomisation and normality) and which to prefer.
"""

from . import _robust_core as _rc
from ._richresult import RichResult, with_describe_pointer

__all__ = ["morans_i_asymptotic_test"]


def morans_i_asymptotic_test(x, W, cdf=None, randomisation=True,
                            alternative="greater"):
    """z = (I - E[I]) / sqrt(Var[I]) for Moran's I.

    Under the null of no spatial autocorrelation E[I] = -1/(n-1).
    ``randomisation=True`` conditions on the observed values and treats
    only their arrangement as random, which is the honest null for
    non-normal data; ``False`` uses the normality variance.  ``cdf`` is
    accepted for backward compatibility and ignored -- the normal CDF
    is always used. Keys: estimate."""
    r = _rc.morans_i_test(x, W, randomisation=randomisation,
                          alternative=alternative)
    res = RichResult(payload={"estimate": r["statistic"],
                              "statistic": r["statistic"],
                              "moran_i": r["estimate"],
                              "expectation": r["expectation"],
                              "variance": r["variance"],
                              "p_value": r["p_value"],
                              "method": r["method"]})
    return with_describe_pointer(res, "moranI")


def cheatsheet():
    return "moranI: Moran's I asymptotic z-test"
