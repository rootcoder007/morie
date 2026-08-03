# morie.fn -- function file (rootcoder007/morie)
"""eq. (5.4) p.150, re-exported from :mod:`morie.fn.msm018`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm018 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm018 import mvsml_linear_mixed_models_eq_5_4

__all__ = ["mvsml_linear_mixed_models_eq_5_4"]


def cheatsheet():
    return "msm021: see msm018"
