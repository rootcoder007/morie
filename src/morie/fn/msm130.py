# morie.fn -- function file (rootcoder007/morie)
"""eq. (8.3) p.254, re-exported from :mod:`morie.fn.msm128`.

The stub generator stamped several extracted page fragments with this
same function name, so the implementation lives once in msm128 and this
module re-exports it.
"""

from .msm128 import mvsml_categorical_count_eq_8_3, mvsml_rkhs_fit

__all__ = ["mvsml_categorical_count_eq_8_3", "mvsml_rkhs_fit"]


def cheatsheet():
    return "msm130: see msm128"
