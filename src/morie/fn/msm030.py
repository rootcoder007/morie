# morie.fn -- function file (rootcoder007/morie)
"""eq. (5.5) p.153, re-exported from :mod:`morie.fn.msm026`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm026 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm026 import mvsml_linear_mixed_models_eq_5_5

__all__ = ["mvsml_linear_mixed_models_eq_5_5"]


def cheatsheet():
    return "msm030: see msm026"
