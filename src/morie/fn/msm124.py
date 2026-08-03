# morie.fn -- function file (rootcoder007/morie)
"""eq. (8.1) p.253, re-exported from :mod:`morie.fn.msm123`.

The stub generator stamped several extracted page fragments with this
same function name, so the implementation lives once in msm123 and this
module re-exports it.
"""

from .msm123 import mvsml_categorical_count_eq_8_1, mvsml_rkhs_objective

__all__ = ["mvsml_categorical_count_eq_8_1", "mvsml_rkhs_objective"]


def cheatsheet():
    return "msm124: see msm123"
