# morie.fn -- function file (rootcoder007/morie)
"""eq. (6.3) pp.173-175, re-exported from :mod:`morie.fn.msm046`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm046 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm046 import mvsml_bayesian_regression_eq_6_3

__all__ = ["mvsml_bayesian_regression_eq_6_3"]


def cheatsheet():
    return "msm050: see msm046"
