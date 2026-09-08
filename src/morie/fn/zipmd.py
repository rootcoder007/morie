# morie.fn -- function file (rootcoder007/morie)
"""Zero-inflated Poisson model for excess-zero count data.

This module is a re-export.  ``zero_inflated_poisson`` already has a
full EM implementation in :mod:`morie.fn.zinfl` -- Lambert, D. (1992),
Zero-inflated Poisson regression, with an application to defects in
manufacturing, Technometrics 34(1):1-14 -- and ``_lazy_map.json``
already resolves the public name to ``zinfl``.  The stub this module
replaced was a duplicate of that name, so it re-exports the canonical
function rather than fitting a second, weaker model.

MVSML (2022) ch.7 discusses count responses with an excess of zeros;
the mixture itself is Lambert's.
"""

from .zinfl import zero_inflated_poisson

__all__ = ["zero_inflated_poisson"]


def cheatsheet():
    return "zipmd: Zero-inflated Poisson (re-export of zinfl)"


# compact alias per ledger/NAMING.md
zipmodel = zero_inflated_poisson
