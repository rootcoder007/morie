# morie.fn -- function file (rootcoder007/morie)
"""VAR(p) multivariate autoregression -- alias of :mod:`morie.fn.varest`.

DUPLICATE, resolved by aliasing.  This module and ``morie.fn.varest``
name the same estimator: the reduced-form VAR(p)

    y_t = nu + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t

fitted by multivariate least squares.  Sims, C.A. (1980),
"Macroeconomics and Reality", *Econometrica* 48(1):1-48,
doi:10.2307/1912017, is the paper that put the reduced-form VAR into
macroeconometrics; Lutkepohl (2005), doi:10.1007/978-3-540-27752-1, is
the textbook statement of the estimator, and that is what ``varest``
implements.  The two differ in provenance, not in arithmetic.

Shipping a second copy would double the surface under a name that reads
right and would pass parity forever, so this module re-exports the one
implementation instead.  (This pair was NOT in the wave-2 DUPMAP; it was
found during implementation and is reported as such.)
"""

from __future__ import annotations

from .varest import vector_autoregression

__all__ = ["vector_autoregression"]


def cheatsheet():
    return "varF: VAR(p) multivariate AR -- alias of varest (Sims 1980; Lutkepohl 2005)"
