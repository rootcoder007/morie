# morie.fn -- slice s03 (rootcoder007/morie)
"""Hierarchical Dirichlet process Gaussian mixture.

Source consulted (FETCHED): Teh, Y. W., Jordan, M. I., Beal, M. J. and
Blei, D. M. (2006).  Hierarchical Dirichlet processes.  *JASA* 101(476),
1566-1581, equations (2) and (19).  The mixture is the HDP applied to a
Gaussian likelihood:

    beta | gamma ~ GEM(gamma),  pi_j | alpha_0, beta ~ DP(alpha_0, beta)
    z_ji | pi_j  ~ pi_j,        x_ji | z_ji, phi ~ N(phi_(z_ji))

so every group has its own mixing proportions but the *component
locations* phi_k are shared -- which is the entire reason for the
hierarchy, as the paper's section 3 argues at length.

DETERMINISM.  The component locations are fitted by EM with the HDP
weights entering as pseudo-counts; the stick-breaking prior comes from
the exact Beta quantile at low-discrepancy points.  No draws.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

from .dpsbw import stick_breaking_weights

__all__ = ["hdp_gaussian_mixture"]


def hdp_gaussian_mixture(y, groups=None, gamma=1.0, alpha=1.0, truncation=4,
                         max_iter=200, tol=1e-13):
    """HDP mixture of univariate normals with shared component locations.

    Returns
    -------
    estimate : the log likelihood
    mu, sigma : the shared component locations and scales
    pi       : group mixing proportions, one row per group
    beta     : the global weights
    """
    v = k.vec(y)
    n = len(v)
    g = [str(x) for x in (groups if groups is not None else [0] * n)]
    ids = []
    for c in g:
        if c not in ids:
            ids.append(c)
    J = len(ids)
    gi = [ids.index(c) for c in g]
    K = int(truncation)
    beta = stick_breaking_weights(gamma, K)["pi"]
    tot = 0.0
    for x in beta:
        tot += x
    beta = [x / tot if tot > 0.0 else 1.0 / K for x in beta]
    lo = min(v)
    hi = max(v)
    mu = [lo + (hi - lo) * (t + 0.5) / K for t in range(K)]
    sd = [max((hi - lo) / K, 1e-6)] * K
    pi = [[beta[t] for t in range(K)] for _ in range(J)]
    ll = float("-inf")
    for _ in range(int(max_iter)):
        R = [[0.0] * K for _ in range(n)]
        newll = 0.0
        for i in range(n):
            lp = []
            for t in range(K):
                w = pi[gi[i]][t]
                z = (v[i] - mu[t]) / sd[t]
                lp.append(math.log(w if w > 1e-300 else 1e-300)
                          - 0.5 * z * z - math.log(sd[t])
                          - 0.5 * math.log(2.0 * math.pi))
            m = k.logsumexp(lp)
            newll += m
            for t in range(K):
                R[i][t] = math.exp(lp[t] - m)
        for j in range(J):
            nj = 0.0
            row = [0.0] * K
            for i in range(n):
                if gi[i] == j:
                    for t in range(K):
                        row[t] += R[i][t]
                    nj += 1.0
            pi[j] = [(float(alpha) * beta[t] + row[t]) / (float(alpha) + nj)
                     for t in range(K)]
        for t in range(K):
            nk = 0.0
            s = 0.0
            for i in range(n):
                nk += R[i][t]
                s += R[i][t] * v[i]
            eff = nk + float(gamma) * beta[t]
            mu[t] = s / nk if nk > 1e-12 else mu[t]
            q = 0.0
            for i in range(n):
                q += R[i][t] * (v[i] - mu[t]) ** 2
            sd[t] = math.sqrt(q / eff) if eff > 0.0 else sd[t]
            if sd[t] < 1e-8:
                sd[t] = 1e-8
        if abs(newll - ll) < tol:
            ll = newll
            break
        ll = newll
    return RichResult(
        title="HDP Gaussian mixture",
        summary_lines=[("groups", J), ("components", K), ("log-lik", ll)],
        payload={
            "estimate": ll,
            "loglik": ll,
            "mu": mu,
            "sigma": sd,
            "pi": pi,
            "beta": beta,
            "n": n,
            "method": "HDP mixture with shared component locations (Teh et al. 2006, eqs. 2 and 19)",
        },
    )


def cheatsheet():
    return "hdpgmm: Hierarchical DP Gaussian mixture"
