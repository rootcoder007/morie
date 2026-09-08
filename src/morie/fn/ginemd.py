# morie.fn -- function file (rootcoder007/morie)
"""Graph isomorphism network aggregation -- re-export.

The generator emitted more than one module for this method.  The
implementation lives once in ``morie.fn.gin`` and this module
re-exports it.
"""

from .gin import ginagg

__all__ = ["ginagg", "graph_isomorphism_net"]

graph_isomorphism_net = ginagg


def cheatsheet():
    return "ginemd: Graph isomorphism network aggregation -- re-export (see gin)"
