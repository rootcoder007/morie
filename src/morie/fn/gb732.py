# morie.fn -- function file (rootcoder007/morie)
"""Moments of a linear rank statistic -- re-export.

The generator emitted more than one module for this method.  The
implementation lives once in ``morie.fn.gb731`` and this module
re-exports it.
"""

from .gb731 import lrankmom

__all__ = ["lrankmom", "gibbons_linrank_mean_var"]

gibbons_linrank_mean_var = lrankmom


def cheatsheet():
    return "gb732: Moments of a linear rank statistic -- re-export (see gb731)"
