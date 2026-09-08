# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Intrinsic conditional autoregressive prior.

Besag (1974), "Spatial interaction and the statistical analysis of
lattice systems", JRSS B 36(2):192-236,
doi:10.1111/j.2517-6161.1974.tb00999.x, with the intrinsic form of
Besag, York and Mollie (1991), Ann. Inst. Statist. Math. 43(1):1-20,
doi:10.1007/BF00116466.  The full conditionals are

    u_i | u_{-i} ~ N( mean of the n_i neighbours, tau^2 / n_i ),

which correspond to the joint precision Q = (D - W)/tau^2 with D the
diagonal of neighbour counts.  Q has the constant vector in its null
space, so the prior is improper (rank n - 1) and needs a sum-to-zero
constraint; the smallest eigenvalue being exactly zero is the check
the tests apply.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["icar_prior"]


def icar_prior(adjacency, tau=1.0, u=None):
    """Precision matrix, conditional moments and log density of an ICAR field."""
    W = core.mat(adjacency)
    n = len(W)
    if n == 0:
        raise ValueError("icar_prior: adjacency is empty")
    for r in W:
        if len(r) != n:
            raise ValueError("icar_prior: adjacency must be square")
    for i in range(n):
        for j in range(n):
            if W[i][j] != W[j][i]:
                raise ValueError("icar_prior: adjacency must be symmetric")
        if W[i][i] != 0:
            raise ValueError("icar_prior: adjacency must have a zero diagonal")
    t = float(tau)
    if t <= 0:
        raise ValueError("icar_prior: tau must be positive")
    deg = [sum(W[i]) for i in range(n)]
    for v in deg:
        if v <= 0:
            raise ValueError("icar_prior: every unit needs at least one neighbour")
    Q = [[((deg[i] if i == j else 0.0) - W[i][j]) / (t * t) for j in range(n)] for i in range(n)]
    cvar = [t * t / deg[i] for i in range(n)]
    if u is None:
        cmean = [0.0] * n
        quad = float("nan")
        centred = float("nan")
    else:
        uv = core.vec(u)
        if len(uv) != n:
            raise ValueError("icar_prior: u and adjacency have different lengths")
        cmean = [sum(W[i][j] * uv[j] for j in range(n)) / deg[i] for i in range(n)]
        quad = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if W[i][j] != 0:
                    quad += W[i][j] * (uv[i] - uv[j]) ** 2
        quad = quad / (t * t)
        centred = sum(uv) / n
    vals, _ = core.jacobi(Q)
    return RichResult(
        title="Intrinsic CAR prior",
        summary_lines=[("units", n), ("tau", t)],
        payload={
            "estimate": vals[0],
            "precision": Q,
            "conditional_mean": cmean,
            "conditional_var": cvar,
            "pairwise_quadratic": quad,
            "log_density_kernel": (float("nan") if quad != quad else -0.5 * quad),
            "smallest_eigenvalue": vals[0],
            "mean_u": centred,
            "n": n,
            "method": "u_i | u_-i ~ N(mean of neighbours, tau^2/n_i); Q = (D - W)/tau^2, Besag (1974)",
        },
    )


def cheatsheet():
    return "icarbm: intrinsic CAR prior"


# compact alias per ledger/NAMING.md
icarprior = icar_prior
