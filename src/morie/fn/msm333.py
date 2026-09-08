# morie.fn -- function file (rootcoder007/morie)
"""eq. (3.1) p.71 with the OLS solution pp.72-73, re-exported from :mod:`morie.fn.msm332`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm332 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm332 import mvsml_elements_lin_reg_eq_3_1

__all__ = ["mvsml_elements_lin_reg_eq_3_1"]


def cheatsheet():
    return "msm333: see msm332"
