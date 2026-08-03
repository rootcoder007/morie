# morie.fn -- function file (rootcoder007/morie)
"""eq. (7.5) p.221, re-exported from :mod:`morie.fn.msm098`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm098 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm098 import mvsml_bayesian_regression_pt2_eq_7_5

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_5"]


def cheatsheet():
    return "msm101: see msm098"
