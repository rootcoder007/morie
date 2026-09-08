# morie.fn -- function file (rootcoder007/morie)
"""LinUCB arm scores -- re-export.

The generator emitted more than one module for this method.  The
implementation lives once in ``morie.fn.linucb`` and this module
re-exports it.
"""

from .linucb import linucb

__all__ = ["linucb", "contextual_bandit_rec"]

contextual_bandit_rec = linucb


def cheatsheet():
    return "banditRS: LinUCB arm scores -- re-export (see linucb)"
