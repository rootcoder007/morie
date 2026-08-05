# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Basic reproduction number under heterogeneous mixing.

Diekmann and Heesterbeek (2000), *Mathematical Epidemiology of
Infectious Diseases*, Wiley, chapter 5, and Diekmann, Heesterbeek and
Metz (1990), J. Math. Biol. 28(4):365-382, doi:10.1007/BF00178324:
R0 is the spectral radius of the next generation matrix,

    K = C D,   D = diag(1 / gamma),

with C the contact (transmission) matrix and gamma the per-group
removal rates.  The dominant eigenvalue is found by power iteration,
which converges for a non-negative primitive K by Perron-Frobenius;
the same theorem is why the dominant eigenvector -- the stable
distribution of infections across groups -- can be reported with a
positive sign.  Homogeneous mixing collapses to R0 = c n / gamma,
which is the closed form the tests check.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["heterogeneous_mixing"]


def heterogeneous_mixing(contact_matrix, gamma, iters=2000, tol=1e-14):
    """R0 and the stable group distribution of the next generation matrix."""
    C = core.mat(contact_matrix)
    n = len(C)
    if n == 0:
        raise ValueError("heterogeneous_mixing: contact matrix is empty")
    for r in C:
        if len(r) != n:
            raise ValueError("heterogeneous_mixing: contact matrix must be square")
        for v in r:
            if v < 0:
                raise ValueError("heterogeneous_mixing: contact rates must be non-negative")
    g = core.vec(gamma)
    if len(g) == 1 and n > 1:
        g = [g[0]] * n
    if len(g) != n:
        raise ValueError("heterogeneous_mixing: gamma must have one rate per group")
    for v in g:
        if v <= 0:
            raise ValueError("heterogeneous_mixing: removal rates must be positive")
    K = [[C[i][j] / g[j] for j in range(n)] for i in range(n)]
    x = [1.0 / n] * n
    lam = 0.0
    it = 0
    for _ in range(int(iters)):
        y = [sum(K[i][j] * x[j] for j in range(n)) for i in range(n)]
        nrm = math.sqrt(sum(v * v for v in y))
        if nrm == 0.0:
            lam = 0.0
            break
        y = [v / nrm for v in y]
        new = sum(y[i] * sum(K[i][j] * y[j] for j in range(n)) for i in range(n))
        it += 1
        if abs(new - lam) <= tol:
            lam = new
            x = y
            break
        lam = new
        x = y
    s = sum(x)
    if s != 0:
        x = [v / s for v in x]
    return RichResult(
        title="R0 under heterogeneous mixing",
        summary_lines=[("groups", n), ("R0", lam)],
        payload={
            "estimate": lam,
            "R0": lam,
            "stable_distribution": x,
            "iterations": it,
            "epidemic": 1 if lam > 1.0 else 0,
            "n": n,
            "method": "spectral radius of K = C diag(1/gamma), Diekmann & Heesterbeek (2000) ch. 5",
        },
    )


def cheatsheet():
    return "hetmix: R0 under heterogeneous mixing"
