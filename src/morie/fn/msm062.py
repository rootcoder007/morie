# morie.fn -- function file (rootcoder007/morie)
"""eq. (6.6) p.186, re-exported from :mod:`morie.fn.msm061`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm061 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm061 import mvsml_bayesian_regression_eq_6_6

__all__ = ["mvsml_bayesian_regression_eq_6_6"]


def cheatsheet():
    return "msm062: see msm061"
