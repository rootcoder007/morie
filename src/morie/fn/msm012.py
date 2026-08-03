# morie.fn -- function file (rootcoder007/morie)
"""eq. (5.2) p.142, re-exported from :mod:`morie.fn.msm011`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm011 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm011 import mvsml_linear_mixed_models_eq_5_2

__all__ = ["mvsml_linear_mixed_models_eq_5_2"]


def cheatsheet():
    return "msm012: see msm011"
