# morie.fn -- function file (rootcoder007/morie)
"""Renyi entropy of order alpha.

This module is a re-export.  ``renyi_entropy`` is implemented in
:mod:`morie.fn.renent` and ``_lazy_map.json`` already resolves the
public name there; the stub this module replaced was a duplicate of
that name under a second module, so it re-exports rather than shipping
a second copy that could drift.

Renyi, A. (1961).  On measures of entropy and information.  Proc. 4th
Berkeley Symp. Math. Statist. Prob. 1:547-561.
"""

from .renent import renyi_entropy

__all__ = ["renyi_entropy"]


def cheatsheet():
    return "reniyd: Renyi entropy (re-export of renent)"


# compact alias per ledger/NAMING.md
renyidiv = renyi_entropy
