# morie.fn -- function file (rootcoder007/morie)
"""eq. (7.2) p.214, re-exported from :mod:`morie.fn.msm089`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm089 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm089 import mvsml_bayesian_regression_pt2_eq_7_2

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_2"]


def cheatsheet():
    return "msm105: see msm089"
