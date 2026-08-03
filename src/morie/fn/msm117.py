# morie.fn -- function file (rootcoder007/morie)
"""eq. (7.7) p.226, re-exported from :mod:`morie.fn.msm109`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm109 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm109 import mvsml_bayesian_regression_pt2_eq_7_7

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_7"]


def cheatsheet():
    return "msm117: see msm109"
