# morie.fn -- function file (rootcoder007/morie)
"""Neal's algorithm 8 for a Dirichlet process mixture."""

import math

from . import _array_core as np
from . import _s03core as core
from ._richresult import RichResult
from .crpcol import norm_logpdf

__all__ = ["crp_gibbs"]


def crp_gibbs(y, alpha=1.0, n_iter=50, m=3, mu0=0.0, tau2=10.0, sigma2=1.0,
              seed=42):
    """
    CRP Gibbs sampler with auxiliary parameters

    Formula: P(z_i | z_{-i}) proportional to n_k^{-i} f(y_i | theta_k),
    and to (alpha/m) f(y_i | theta_aux) for each of m auxiliary draws

    Neal's algorithm 8: the cluster parameters are kept explicit rather
    than integrated out, and m fresh draws from the base measure stand
    in for the infinitely many empty tables.  That makes the sampler
    valid for non-conjugate base measures, unlike the collapsed
    algorithm 3; here the conjugate normal case is used so the two can
    be compared.

    Parameters
    ----------
    y : array-like
        Observations.
    alpha : float
        Concentration, strictly positive.
    n_iter : int
        Number of sweeps.
    m : int
        Number of auxiliary components.
    mu0, tau2 : float
        Base measure N(mu0, tau2).
    sigma2 : float
        Known within-cluster variance.
    seed : int
        Seed of the deterministic stream.

    Returns
    -------
    result : dict
        Keys: estimate (number of clusters), z, counts, theta,
        n_clusters, loglik, n.

    References
    ----------
    Neal (2000), J. Comput. Graph. Statist. 9(2):249-265, algorithm 8.
    """
    y = core.vec(y)
    n = len(y)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    if not (alpha > 0.0):
        raise ValueError("alpha must be strictly positive")
    m = int(m)
    if m < 1:
        raise ValueError("m must be at least 1")
    if not (tau2 > 0.0 and sigma2 > 0.0):
        raise ValueError("tau2 and sigma2 must be strictly positive")
    rng = np.random.default_rng(seed)
    z = [0] * n
    counts = [n]
    theta = [sum(y) / n]
    for _ in range(int(n_iter)):
        for i in range(n):
            k = z[i]
            counts[k] -= 1
            aux = [float(rng.normal(mu0, math.sqrt(tau2))) for _ in range(m)]
            if counts[k] == 0:
                aux[0] = theta[k]
            w = []
            for c in range(len(counts)):
                w.append(counts[c] * math.exp(norm_logpdf(y[i], theta[c], sigma2))
                         if counts[c] > 0 else 0.0)
            for j in range(m):
                w.append(alpha / m * math.exp(norm_logpdf(y[i], aux[j], sigma2)))
            tot = sum(w)
            u = float(rng.uniform(0.0, 1.0)) * tot
            acc = 0.0
            pick = len(w) - 1
            for c in range(len(w)):
                acc += w[c]
                if u <= acc:
                    pick = c
                    break
            if pick >= len(counts):
                theta.append(aux[pick - len(counts)])
                counts.append(0)
                pick = len(counts) - 1
            z[i] = pick
            counts[pick] += 1
        keep = [c for c in range(len(counts)) if counts[c] > 0]
        remap = dict((c, j) for j, c in enumerate(keep))
        counts = [counts[c] for c in keep]
        theta = [theta[c] for c in keep]
        z = [remap[v] for v in z]
        # conjugate update of the surviving cluster parameters
        for c in range(len(counts)):
            s = sum(y[i] for i in range(n) if z[i] == c)
            prec = 1.0 / tau2 + counts[c] / sigma2
            mpost = (mu0 / tau2 + s / sigma2) / prec
            theta[c] = float(rng.normal(mpost, math.sqrt(1.0 / prec)))
    ll = sum(norm_logpdf(y[i], theta[z[i]], sigma2) for i in range(n))
    return RichResult(payload={
        "estimate": len(counts),
        "z": z,
        "counts": counts,
        "theta": theta,
        "n_clusters": len(counts),
        "loglik": ll,
        "n": n,
        "method": "CRP Gibbs sampler, Neal algorithm 8",
    })


def cheatsheet():
    return "crpgib: CRP Gibbs sampler (Neal algorithm 8)"


# compact alias per ledger/NAMING.md
crpgibbs = crp_gibbs
