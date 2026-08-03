# morie.fn -- function file (rootcoder007/morie)
"""eq. (6.8) p.191, re-exported from :mod:`morie.fn.msm065`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm065 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm065 import mvsml_bayesian_regression_eq_6_8

__all__ = ["mvsml_bayesian_regression_eq_6_8"]


def cheatsheet():
    return "msm066: see msm065"
