# morie.fn -- function file (rootcoder007/morie)
"""eq. (7.9) p.227, re-exported from :mod:`morie.fn.msm112`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm112 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm112 import mvsml_bayesian_regression_pt2_eq_7_9

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_9"]


def cheatsheet():
    return "msm116: see msm112"
