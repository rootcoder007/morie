# morie.fn -- function file (rootcoder007/morie)
"""eq. (5.3) p.148, re-exported from :mod:`morie.fn.msm015`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm015 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm015 import mvsml_linear_mixed_models_eq_5_3

__all__ = ["mvsml_linear_mixed_models_eq_5_3"]


def cheatsheet():
    return "msm016: see msm015"
