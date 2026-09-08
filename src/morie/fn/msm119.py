# morie.fn -- function file (rootcoder007/morie)
"""eq. (7.10) p.227, re-exported from :mod:`morie.fn.msm115`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm115 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm115 import mvsml_bayesian_regression_pt2_eq_7_10

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_10"]


def cheatsheet():
    return "msm119: see msm115"
