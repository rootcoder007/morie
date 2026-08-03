# morie.fn -- function file (rootcoder007/morie)
"""eq. (6.2) p.172, re-exported from :mod:`morie.fn.msm043`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm043 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm043 import mvsml_bayesian_regression_eq_6_2

__all__ = ["mvsml_bayesian_regression_eq_6_2"]


def cheatsheet():
    return "msm077: see msm043"
