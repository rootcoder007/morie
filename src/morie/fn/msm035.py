# morie.fn -- function file (rootcoder007/morie)
"""eq. (5.6) p.155, re-exported from :mod:`morie.fn.msm032`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm032 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm032 import mvsml_linear_mixed_models_eq_5_6

__all__ = ["mvsml_linear_mixed_models_eq_5_6"]


def cheatsheet():
    return "msm035: see msm032"
