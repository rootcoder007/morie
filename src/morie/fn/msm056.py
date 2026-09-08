# morie.fn -- function file (rootcoder007/morie)
"""eq. (6.5) p.177, re-exported from :mod:`morie.fn.msm055`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm055 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm055 import mvsml_bayesian_regression_eq_6_5

__all__ = ["mvsml_bayesian_regression_eq_6_5"]


def cheatsheet():
    return "msm056: see msm055"
