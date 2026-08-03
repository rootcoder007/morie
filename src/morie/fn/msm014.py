# morie.fn -- function file (rootcoder007/morie)
"""eq. (5.1) p.142, re-exported from :mod:`morie.fn.msm010`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm010 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm010 import mvsml_linear_mixed_models_eq_5_1

__all__ = ["mvsml_linear_mixed_models_eq_5_1"]


def cheatsheet():
    return "msm014: see msm010"
