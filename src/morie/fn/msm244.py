# morie.fn -- function file (rootcoder007/morie)
"""eq. (2.1) p.36, re-exported from :mod:`morie.fn.msm240`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm240 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm240 import mvsml_preprocessing_eq_2_1

__all__ = ["mvsml_preprocessing_eq_2_1"]


def cheatsheet():
    return "msm244: see msm240"
