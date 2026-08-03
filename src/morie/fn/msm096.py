# morie.fn -- function file (rootcoder007/morie)
"""eq. (6.7) p.186, re-exported from :mod:`morie.fn.msm063`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm063 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm063 import mvsml_bayesian_regression_eq_6_7

__all__ = ["mvsml_bayesian_regression_eq_6_7"]


def cheatsheet():
    return "msm096: see msm063"
