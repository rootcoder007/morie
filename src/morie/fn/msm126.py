# morie.fn -- function file (rootcoder007/morie)
"""eq. (8.2) p.254, re-exported from :mod:`morie.fn.msm125`.

The stub generator stamped several extracted page fragments with this
same function name, so the implementation lives once in msm125 and this
module re-exports it.
"""

from .msm125 import mvsml_categorical_count_eq_8_2, mvsml_rkhs_representer

__all__ = ["mvsml_categorical_count_eq_8_2", "mvsml_rkhs_representer"]


def cheatsheet():
    return "msm126: see msm125"
