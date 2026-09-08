# morie.fn -- function file (rootcoder007/morie)
"""eq. (6.1)-(6.2) pp.172, re-exported from :mod:`morie.fn.msm042`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm042 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm042 import mvsml_bayesian_regression_eq_6_1

__all__ = ["mvsml_bayesian_regression_eq_6_1"]


def cheatsheet():
    return "msm045: see msm042"
