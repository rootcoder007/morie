# morie.fn -- function file (rootcoder007/morie)
"""eq. (7.1) pp.210-213, re-exported from :mod:`morie.fn.msm085`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm085 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm085 import mvsml_bayesian_regression_pt2_eq_7_1

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_1"]


def cheatsheet():
    return "msm104: see msm085"
