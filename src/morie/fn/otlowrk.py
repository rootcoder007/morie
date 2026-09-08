# morie.fn -- slice s03 (rootcoder007/morie)
"""Low-rank Sinkhorn.

Source consulted (FETCHED): Scetbon, M., Cuturi, M. and Peyre, G.
(2021).  Low-rank Sinkhorn factorization.  *ICML* 139, 9344-9354
(arXiv:2103.04737).  The paper restricts the transport plan to the set

    Pi_(a,g,b) = { P = Q diag(1/g) R^T : Q in Pi_(a,g), R in Pi_(b,g) }

-- a product of two sub-couplings with a common right marginal g, so
that rank_+(P) <= r by construction.  The optimisation is mirror
descent: at each outer step the linearisation of the objective is
projected back onto Pi_(a,g) and Pi_(b,g) by two Sinkhorn solves.

DETERMINISM.  The paper initialises Q, R and g at random; here they are
initialised at the rank-one product a g^T, b g^T with g uniform, which
is feasible, canonical, and identical in both arms.  No generator is
consulted.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

from .otsklog import ot_sinkhorn_log

__all__ = ["ot_low_rank_sinkhorn"]


def ot_low_rank_sinkhorn(a, b, C, rank=2, epsilon=0.1, max_iter=20,
                         inner=50, gamma=1.0):
    """Rank-constrained entropic OT by mirror descent.

    Returns
    -------
    RichResult with payload:
        U, V     : the sub-couplings Q and R
        T        : the low-rank plan Q diag(1/g) R^T
        g        : the common inner marginal
        estimate : <T, C>
    """
    av = k.vec(a)
    bv = k.vec(b)
    Cm = k.mat(C)
    n = len(av)
    m = len(bv)
    r = int(rank)
    g = [1.0 / r] * r
    Q = [[av[i] * g[t] for t in range(r)] for i in range(n)]
    R = [[bv[j] * g[t] for t in range(r)] for j in range(m)]
    for _ in range(int(max_iter)):
        # gradient of <Q diag(1/g) R', C> with respect to Q and R
        CR = [[0.0] * r for _ in range(n)]
        for i in range(n):
            for t in range(r):
                s = 0.0
                for j in range(m):
                    s += Cm[i][j] * R[j][t]
                CR[i][t] = s / g[t] if g[t] > 0.0 else 0.0
        CQ = [[0.0] * r for _ in range(m)]
        for j in range(m):
            for t in range(r):
                s = 0.0
                for i in range(n):
                    s += Cm[i][j] * Q[i][t]
                CQ[j][t] = s / g[t] if g[t] > 0.0 else 0.0
        Q = ot_sinkhorn_log(av, g, CR, float(epsilon) / float(gamma), inner)["T"]
        R = ot_sinkhorn_log(bv, g, CQ, float(epsilon) / float(gamma), inner)["T"]
        ng = [0.0] * r
        for t in range(r):
            s = 0.0
            for i in range(n):
                s += Q[i][t]
            ng[t] = s
        tot = 0.0
        for x in ng:
            tot += x
        g = [x / tot if tot > 0.0 else 1.0 / r for x in ng]
    T = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for t in range(r):
                if g[t] > 0.0:
                    s += Q[i][t] * R[j][t] / g[t]
            T[i][j] = s
    cost = 0.0
    for i in range(n):
        for j in range(m):
            cost += T[i][j] * Cm[i][j]
    return RichResult(
        title="Low-rank Sinkhorn",
        summary_lines=[("rank", r), ("cost", cost)],
        payload={
            "U": Q,
            "V": R,
            "T": T,
            "g": g,
            "estimate": cost,
            "cost": cost,
            "rank": r,
            "method": "Low-rank Sinkhorn factorisation P = Q diag(1/g) R' (Scetbon et al. 2021)",
        },
    )


def cheatsheet():
    return "otlowrk: Low-rank Sinkhorn approximating T = U V^T"
