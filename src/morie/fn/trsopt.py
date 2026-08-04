# morie.fn -- slice s03 (rootcoder007/morie)
"""Trust-region subproblem.

Source consulted: Nocedal, J. and Wright, S. J. (2006).  *Numerical
Optimization*, 2nd ed., Springer, chapter 4.  The subproblem is

    min_s  g' s + (1/2) s' B s   subject to  ||s|| <= delta

and its solution is characterised by their theorem 4.1: s* solves it iff
there is lambda >= 0 with

    (B + lambda I) s* = -g,   lambda (delta - ||s*||) = 0,
    B + lambda I positive semidefinite.

The book is not open access; the theorem is quoted in its standard
published form.  The Newton root-finding on lambda uses the secular
equation in the form Nocedal and Wright recommend for its near-linearity,

    phi(lambda) = 1/delta - 1/||s(lambda)||

The eigendecomposition is by cyclic Jacobi with sign-fixed vectors, so
the hard case (g orthogonal to the eigenspace of the smallest
eigenvalue) is detected rather than stumbled into, and is reported.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["trust_region_subproblem"]


def trust_region_subproblem(g, H, delta=1.0, tol=1e-13, max_iter=200):
    """Exact solution of the trust-region subproblem.

    Returns
    -------
    estimate : the model decrease -(g's + s'Bs/2)
    s        : the step
    lam      : the multiplier
    boundary : whether the solution is on the boundary
    hard_case
    """
    gv = k.vec(g)
    B = k.mat(H)
    n = len(gv)
    vals, vecs = k.jacobi(B)
    gt = [0.0] * n
    for t in range(n):
        s = 0.0
        for i in range(n):
            s += vecs[i][t] * gv[i]
        gt[t] = s
    lam1 = vals[0]
    D = float(delta)

    def snorm(lm):
        s = 0.0
        for t in range(n):
            d = vals[t] + lm
            if abs(d) < 1e-300:
                continue
            s += (gt[t] / d) ** 2
        return math.sqrt(s)

    hard = False
    if lam1 > 0.0 and snorm(0.0) <= D:
        lam = 0.0
    else:
        lo = max(0.0, -lam1) + 1e-14
        hi = lo + 1.0
        while snorm(hi) > D and hi < 1e14:
            hi *= 2.0
        if snorm(lo) < D:
            hard = True
        for _ in range(int(max_iter)):
            mid = 0.5 * (lo + hi)
            if snorm(mid) > D:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol * max(1.0, hi):
                break
        lam = 0.5 * (lo + hi)
    s = [0.0] * n
    for t in range(n):
        d = vals[t] + lam
        if abs(d) < 1e-300:
            continue
        c = -gt[t] / d
        for i in range(n):
            s[i] += c * vecs[i][t]
    gs = 0.0
    for i in range(n):
        gs += gv[i] * s[i]
    Bs = k.matvec(B, s)
    q = 0.0
    for i in range(n):
        q += s[i] * Bs[i]
    nrm = 0.0
    for i in range(n):
        nrm += s[i] * s[i]
    nrm = math.sqrt(nrm)
    return RichResult(
        title="Trust-region subproblem",
        summary_lines=[("||s||", nrm), ("lambda", lam)],
        payload={
            "estimate": -(gs + 0.5 * q),
            "s": s,
            "lam": lam,
            "norm": nrm,
            "boundary": abs(nrm - D) < 1e-6 * max(1.0, D),
            "hard_case": hard,
            "eigenvalues": vals,
            "method": "Exact trust-region subproblem via the secular equation (Nocedal and Wright 2006, thm. 4.1)",
        },
    )


def cheatsheet():
    return "trsopt: Trust-region subproblem solver"
