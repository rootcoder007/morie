# morie.fn -- function file (rootcoder007/morie)
"""VAR impulse response function -- alias of :mod:`morie.fn.irfun`.

DUPLICATE, resolved by aliasing (wave-2 DUPMAP: varimp -> irfun).  Both
names denote the orthogonalised impulse response of a VAR(p): the MA
recursion Phi_0 = I, Phi_h = sum_{j=1..min(h,p)} A_j Phi_{h-j}, with the
shocks orthogonalised by the lower Cholesky factor P of Sigma_u, so that
Theta_h = Phi_h P and column k of Theta_h is the response to a
one-standard-deviation shock in variable k (Lutkepohl 2005 Ch. 2.3,
doi:10.1007/978-3-540-27752-1; Sims 1980, doi:10.2307/1912017).

``morie.fn.irfun`` already implements it; this module re-exports it
rather than shipping a second copy.  Note the name is a trap: ``varimp``
reads as "variable importance", but the wave-2 categorisation and the
stub docstring both give VAR impulse response, which is what is aliased.
"""

from __future__ import annotations

from .irfun import impulse_response as var_impulse_response

__all__ = ["var_impulse_response"]


def cheatsheet():
    return "varimp: VAR impulse response -- alias of irfun (Lutkepohl 2005 Ch. 2.3)"
