# morie.fn -- function file (rootcoder007/morie)
"""eq. (6.11) pp.195-196, re-exported from :mod:`morie.fn.msm076`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm076 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm076 import mvsml_bayesian_regression_eq_6_11

__all__ = ["mvsml_bayesian_regression_eq_6_11"]


def cheatsheet():
    return "msm080: see msm076"
