# morie.fn -- slice s03 (rootcoder007/morie)
"""Stick-breaking weights of a Dirichlet process.

Source consulted: Sethuraman, J. (1994).  A constructive definition of
Dirichlet priors.  *Statistica Sinica* 4(2), 639-650, whose
representation is

    V_k ~ Beta(1, alpha) independently,
    pi_k = V_k prod_(j<k) (1 - V_j)

so that sum_k pi_k = 1 almost surely.  The 1994 paper is free but was
not retrievable here; the construction is quoted in its standard
published form and is reproduced identically wherever it is used (e.g.
Teh et al. 2006, *JASA* 101, 1566-1581, equations 5-6, which WAS
fetched).

DETERMINISM.  A Beta(1, alpha) draw is not taken from a generator: its
quantile function is available in closed form, F^(-1)(u) = 1 - (1 - u)^
(1/alpha), and it is evaluated at van der Corput points.  The result has
the right marginal law and is identical in both arms.

The truncation leaves a remainder prod_k (1 - V_k), which is returned as
``remainder`` rather than absorbed silently -- it is the exact
truncation error of the representation.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["stick_breaking_weights"]


def stick_breaking_weights(alpha=1.0, truncation=10, V=None, base=2):
    """Truncated stick-breaking weights for DP(alpha).

    Returns
    -------
    estimate : pi_1
    pi       : the truncated weights
    V        : the stick fractions
    remainder: the unallocated mass prod_k (1 - V_k)
    """
    a = float(alpha)
    K = int(truncation)
    if V is not None:
        Vs = k.vec(V)
    else:
        Vs = [1.0 - (1.0 - k.vdc(i, int(base))) ** (1.0 / a) for i in range(K)]
    pi = []
    rest = 1.0
    for i in range(len(Vs)):
        pi.append(Vs[i] * rest)
        rest *= (1.0 - Vs[i])
    tot = 0.0
    for x in pi:
        tot += x
    return RichResult(
        title="DP stick-breaking weights",
        summary_lines=[("alpha", a), ("truncation", K)],
        payload={
            "estimate": pi[0] if pi else float("nan"),
            "pi": pi,
            "V": Vs,
            "remainder": rest,
            "mass": tot,
            "method": "Sethuraman (1994) stick-breaking, Beta quantiles at low-discrepancy points",
        },
    )


def cheatsheet():
    return "dpsbw: Stick-breaking weights for DP"
