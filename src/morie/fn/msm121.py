# morie.fn -- function file (rootcoder007/morie)
"""eq. (7.6) p.225, re-exported from :mod:`morie.fn.msm106`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm106 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm106 import mvsml_bayesian_regression_pt2_eq_7_6

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_6"]


def cheatsheet():
    return "msm121: see msm106"
