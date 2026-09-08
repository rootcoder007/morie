# morie.fn -- function file (rootcoder007/morie)
"""eq. (6.9) pp.191-193, re-exported from :mod:`morie.fn.msm067`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm067 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm067 import mvsml_bayesian_regression_eq_6_9

__all__ = ["mvsml_bayesian_regression_eq_6_9"]


def cheatsheet():
    return "msm081: see msm067"
