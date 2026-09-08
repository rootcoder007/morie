# morie.fn -- function file (rootcoder007/morie)
"""eq. (7.3) p.219, re-exported from :mod:`morie.fn.msm092`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm092 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm092 import mvsml_bayesian_regression_pt2_eq_7_3

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_3"]


def cheatsheet():
    return "msm103: see msm092"
