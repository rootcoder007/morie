# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Interior-point linear programming: the logarithmic barrier.

Karmarkar (1984), "A new polynomial-time algorithm for linear
programming", Combinatorica 4(4):373-395, doi:10.1007/BF02579150,
opened the interior-point line; the barrier form used here is the
standard one (Boyd and Vandenberghe, *Convex Optimization*, CUP 2004,
sect. 11.2):

    minimise  c'x - tau [ sum_i log(b_i - a_i'x) + sum_j log x_j ]

solved by damped Newton steps from a strictly feasible x0.  The
central-path point x(tau) satisfies

    c'x(tau) - p*  <=  (m + n) tau,

which is a bound on the answer that does not run through this code and
is what the tests check against the simplex optimum.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["interior_point_lp"]


def interior_point_lp(c, A, b, x0, tau=0.01, iters=60):
    """Central-path point x(tau) of the barrier problem."""
    cv = core.vec(c)
    M = core.mat(A)
    bv = core.vec(b)
    x = core.vec(x0)
    m = len(M)
    n = len(cv)
    if m == 0 or n == 0:
        raise ValueError("interior_point_lp: empty problem")
    if len(bv) != m:
        raise ValueError("interior_point_lp: A and b have different row counts")
    if len(x) != n:
        raise ValueError("interior_point_lp: x0 has the wrong length")
    t = float(tau)
    if t <= 0:
        raise ValueError("interior_point_lp: tau must be positive")
    slack = [bv[i] - sum(M[i][j] * x[j] for j in range(n)) for i in range(m)]
    if min(slack) <= 0 or min(x) <= 0:
        raise ValueError("interior_point_lp: x0 must be strictly feasible")
    lam = float("inf")
    for _ in range(int(iters)):
        s = [bv[i] - sum(M[i][j] * x[j] for j in range(n)) for i in range(m)]
        g = [cv[j] for j in range(n)]
        for j in range(n):
            for i in range(m):
                g[j] += t * M[i][j] / s[i]
            g[j] -= t / x[j]
        H = [[0.0] * n for _ in range(n)]
        for j in range(n):
            for k in range(n):
                acc = 0.0
                for i in range(m):
                    acc += M[i][j] * M[i][k] / (s[i] * s[i])
                H[j][k] = t * acc
            H[j][j] += t / (x[j] * x[j])
        step = core.cholsolve(H, [-v for v in g])
        lam = 0.0
        for j in range(n):
            lam += -g[j] * step[j]
        if lam / 2.0 <= 1e-14:
            break
        a = 1.0
        for _ in range(80):
            xn = [x[j] + a * step[j] for j in range(n)]
            sn = [bv[i] - sum(M[i][j] * xn[j] for j in range(n)) for i in range(m)]
            if min(xn) > 0 and min(sn) > 0:
                break
            a *= 0.5
        x = [x[j] + a * step[j] for j in range(n)]
    obj = 0.0
    for j in range(n):
        obj += cv[j] * x[j]
    return RichResult(
        title="Interior-point LP (barrier)",
        summary_lines=[("variables", n), ("constraints", m), ("tau", t)],
        payload={
            "estimate": obj,
            "x": x,
            "objective": obj,
            "duality_bound": (m + n) * t,
            "newton_decrement": lam,
            "tau": t,
            "n": n,
            "method": "log-barrier Newton central path, Karmarkar (1984); Boyd & Vandenberghe sect. 11.2",
        },
    )


def cheatsheet():
    return "intlpa: interior-point linear programming"
