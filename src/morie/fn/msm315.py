# morie.fn -- function file (rootcoder007/morie)
"""eq. (1.2) p.15, re-exported from :mod:`morie.fn.msm002`.

The stub generator stamped several extracted page
fragments with this same function name, so the
implementation lives once in msm002 and this module re-exports
it.  Calling either path runs the same code.
"""

from .msm002 import mvsml_general_eq_1_2

__all__ = ["mvsml_general_eq_1_2"]


def cheatsheet():
    return "msm315: see msm002"
