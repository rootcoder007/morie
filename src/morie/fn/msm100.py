# morie.fn -- function file (rootcoder007/morie)
"""eq. (7.4) p.220, re-exported from :mod:`morie.fn.msm095`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm095 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm095 import mvsml_bayesian_regression_pt2_eq_7_4

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_4"]


def cheatsheet():
    return "msm100: see msm095"
