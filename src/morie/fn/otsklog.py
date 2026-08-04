# morie.fn -- slice s03 (rootcoder007/morie)
"""Log-domain Sinkhorn for entropic optimal transport.

Sources consulted: Cuturi, M. (2013).  Sinkhorn distances: lightspeed
computation of optimal transport.  *NIPS* 26, 2292-2300
(arXiv:1306.0895), for the entropic problem itself; and Schmitzer, B.
(2019).  Stabilized sparse scaling algorithms for entropy regularized
transport problems.  *SIAM Journal on Scientific Computing* 41(3),
A1443-A1481 (arXiv:1610.06519 -- FETCHED), for the log-domain
stabilisation that keeps the iteration finite at small epsilon.

The iteration, in the dual potentials, is

    f_i <- eps log a_i - eps logsumexp_j( (g_j - C_ij) / eps )
    g_j <- eps log b_j - eps logsumexp_i( (f_i - C_ij) / eps )
    T_ij = exp( (f_i + g_j - C_ij) / eps )

which is Sinkhorn's scaling written with u = exp(f/eps), v = exp(g/eps)
substituted out.  The point of the log form is that u and v underflow
for small eps while f and g do not, so the two are mathematically
identical and numerically are not.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["ot_sinkhorn_log"]


def ot_sinkhorn_log(a, b, C, epsilon=0.1, max_iter=200, tol=1e-13,
                    f0=None, g0=None):
    """Entropic OT plan by the log-domain Sinkhorn iteration.

    Returns
    -------
    RichResult with payload:
        T        : the transport plan
        cost     : <T, C>
        f, g     : the dual potentials
        estimate : the transport cost
        err      : final marginal violation
        n_iter
    """
    av = k.vec(a)
    bv = k.vec(b)
    Cm = k.mat(C)
    n = len(av)
    m = len(bv)
    e = float(epsilon)
    f = k.vec(f0) if f0 is not None else [0.0] * n
    g = k.vec(g0) if g0 is not None else [0.0] * m
    la = [math.log(x) if x > 0.0 else -1e300 for x in av]
    lb = [math.log(x) if x > 0.0 else -1e300 for x in bv]
    it = 0
    err = float("nan")
    for it in range(1, int(max_iter) + 1):
        for i in range(n):
            f[i] = e * la[i] - e * k.logsumexp(
                [(g[j] - Cm[i][j]) / e for j in range(m)])
        for j in range(m):
            g[j] = e * lb[j] - e * k.logsumexp(
                [(f[i] - Cm[i][j]) / e for i in range(n)])
        err = 0.0
        for i in range(n):
            s = 0.0
            for j in range(m):
                s += math.exp((f[i] + g[j] - Cm[i][j]) / e)
            err += abs(s - av[i])
        if err < tol:
            break
    T = [[math.exp((f[i] + g[j] - Cm[i][j]) / e) for j in range(m)]
         for i in range(n)]
    cost = 0.0
    for i in range(n):
        for j in range(m):
            cost += T[i][j] * Cm[i][j]
    return RichResult(
        title="Log-domain Sinkhorn",
        summary_lines=[("cost", cost), ("iterations", it)],
        payload={
            "T": T,
            "cost": cost,
            "f": f,
            "g": g,
            "estimate": cost,
            "err": err,
            "n_iter": it,
            "method": "Log-domain Sinkhorn for entropic OT (Cuturi 2013; Schmitzer 2019)",
        },
    )


def cheatsheet():
    return "otsklog: Log-domain Sinkhorn for numerical stability"


otsinkhornlog = ot_sinkhorn_log
