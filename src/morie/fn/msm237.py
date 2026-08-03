# morie.fn -- function file (rootcoder007/morie)
"""eq. (1.3) p.16, re-exported from :mod:`morie.fn.msm003`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm003 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm003 import mvsml_general_eq_1_3

__all__ = ["mvsml_general_eq_1_3"]


def cheatsheet():
    return "msm237: see msm003"
