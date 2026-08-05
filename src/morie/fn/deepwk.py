# morie.fn -- function file (rootcoder007/morie)
"""DeepWalk node embeddings -- alias of :mod:`morie.fn.deepw`.

`deepwk` and `deepw` are the SAME method: truncated uniform random walks
fed to skip-gram, Perozzi, Al-Rfou & Skiena (2014).  Two module names
for one method is exactly the duplicate this campaign is trying not to
create, so this file re-exports the implementation rather than repeating
it -- a second copy would agree with the first at 1e-9 forever and tell
nobody anything.
"""

from .deepw import adjacency_lists, deepwalk, skipgram, uniform_walk

__all__ = ["deepwalk"]


def cheatsheet():
    return "deepwk: DeepWalk node embeddings (alias of deepw)"
