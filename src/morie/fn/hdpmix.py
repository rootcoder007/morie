# morie.fn -- slice s03 (rootcoder007/morie)
"""Hierarchical Dirichlet process.

Source consulted (FETCHED, PDF from the author's page): Teh, Y. W.,
Jordan, M. I., Beal, M. J. and Blei, D. M. (2006).  Hierarchical
Dirichlet processes.  *Journal of the American Statistical Association*
101(476), 1566-1581.  Their equation (2) is the model,

    G_0 | gamma, H  ~ DP(gamma, H)
    G_j | alpha_0, G_0 ~ DP(alpha_0, G_0)   for each j

and equation (19) gives the equivalent stick-breaking form,

    beta | gamma      ~ GEM(gamma)
    pi_j | alpha_0, beta ~ DP(alpha_0, beta)
    z_ji | pi_j       ~ pi_j

with GEM the stick-breaking law of their equations (5)-(6),
pi_k' ~ Beta(1, alpha_0), pi_k = pi_k' prod_(l<k) (1 - pi_l').  The point
the paper makes, and which this implementation preserves, is that G_0
being *discrete* is what lets the groups share atoms at all.

DETERMINISM.  beta comes from the exact Beta quantile at low-discrepancy
points; pi_j comes from the group's Dirichlet posterior mean given beta,
Dir(alpha_0 beta + n_j), not from a draw.  No generator is consulted.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

from .dpsbw import stick_breaking_weights

__all__ = ["hierarchical_dp"]


def hierarchical_dp(y, groups=None, gamma=1.0, alpha=1.0, truncation=6):
    """Global and group-level atom weights of an HDP.

    Parameters
    ----------
    y : array-like
        Atom index (zero-based) of each observation.
    groups : array-like
        Group label of each observation.
    gamma : float
        Top-level concentration.
    alpha : float
        Group-level concentration.
    truncation : int
        Number of atoms retained.

    Returns
    -------
    estimate : beta_1
    beta     : the global weights
    pi       : group weights, one row per group
    counts   : atom counts per group
    shared   : atoms used by more than one group
    """
    z = [int(x) for x in k.vec(y)]
    g = [str(x) for x in (groups if groups is not None else [0] * len(z))]
    ids = []
    for c in g:
        if c not in ids:
            ids.append(c)
    K = int(truncation)
    beta = stick_breaking_weights(gamma, K)["pi"]
    tot = 0.0
    for x in beta:
        tot += x
    beta = [x / tot if tot > 0.0 else 1.0 / K for x in beta]
    counts = []
    pi = []
    for c in ids:
        row = [0.0] * K
        for i in range(len(z)):
            if g[i] == c and 0 <= z[i] < K:
                row[z[i]] += 1.0
        counts.append(row)
        nj = 0.0
        for x in row:
            nj += x
        pi.append([(float(alpha) * beta[t] + row[t]) / (float(alpha) + nj)
                   for t in range(K)])
    shared = 0
    for t in range(K):
        used = 0
        for row in counts:
            if row[t] > 0.0:
                used += 1
        if used > 1:
            shared += 1
    return RichResult(
        title="Hierarchical Dirichlet process",
        summary_lines=[("groups", len(ids)), ("shared atoms", shared)],
        payload={
            "estimate": beta[0] if beta else float("nan"),
            "beta": beta,
            "pi": pi,
            "counts": counts,
            "shared": shared,
            "group_ids": ids,
            "method": "HDP: beta ~ GEM(gamma), pi_j ~ DP(alpha_0, beta) at its posterior mean (Teh et al. 2006, eqs. 2 and 19)",
        },
    )


def cheatsheet():
    return "hdpmix: Hierarchical Dirichlet Process for shared mixture components"


hierarchicaldp = hierarchical_dp
