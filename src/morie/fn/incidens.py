# morie.fn -- function file (rootcoder007/morie)
"""Person-time incidence rate.

This module is a re-export.  ``incidence_rate`` already has a full
implementation with an exact Poisson confidence interval in
:mod:`morie.fn.cdinc`, and ``_lazy_map.json`` already resolves the
public name there.  The stub this module replaced was a duplicate of
that name.

Rothman, K.J., Greenland, S. & Lash, T.L., Modern Epidemiology:
IR = new cases / person-time at risk.
"""

from .cdinc import incidence_rate

__all__ = ["incidence_rate"]


def cheatsheet():
    return "incidens: Person-time incidence rate (re-export of cdinc)"


# compact alias per ledger/NAMING.md
incrate = incidence_rate
