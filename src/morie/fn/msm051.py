# morie.fn -- function file (rootcoder007/morie)
"""eq. (6.4) pp.176-177, re-exported from :mod:`morie.fn.msm049`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm049 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm049 import mvsml_bayesian_regression_eq_6_4

__all__ = ["mvsml_bayesian_regression_eq_6_4"]


def cheatsheet():
    return "msm051: see msm049"
