# morie.fn -- function file (rootcoder007/morie)
"""Collapsed Gibbs sampler for a Dirichlet process mixture."""

import math

from . import _array_core as np
from . import _s03core as core
from ._richresult import RichResult

__all__ = ["crp_collapsed"]


def norm_logpdf(x, mu, var):
    return -0.5 * (math.log(2.0 * math.pi * var) + (x - mu) ** 2 / var)


def collapsed_sweep(y, alpha, n_iter, mu0, tau2, sigma2, seed):
    """Neal's algorithm 3 for a conjugate normal DP mixture.

    theta is integrated out, so a customer joins table k with weight
    n_k f(y_i | y_k) and opens a new one with weight alpha f(y_i), both
    of which are available in closed form for the normal-normal pair.
    The sweep order and the uniform draws are fixed, so the two language
    arms visit identical states.
    """
    n = len(y)
    z = [0] * n
    counts = [n]
    sums = [sum(y)]
    rng = np.random.default_rng(seed)
    for _ in range(n_iter):
        for i in range(n):
            k = z[i]
            counts[k] -= 1
            sums[k] -= y[i]
            w = []
            for c in range(len(counts)):
                if counts[c] == 0:
                    w.append(0.0)
                    continue
                prec = 1.0 / tau2 + counts[c] / sigma2
                m = (mu0 / tau2 + sums[c] / sigma2) / prec
                w.append(counts[c] * math.exp(
                    norm_logpdf(y[i], m, sigma2 + 1.0 / prec)))
            w.append(alpha * math.exp(norm_logpdf(y[i], mu0, sigma2 + tau2)))
            tot = sum(w)
            u = float(rng.uniform(0.0, 1.0)) * tot
            acc = 0.0
            pick = len(w) - 1
            for c in range(len(w)):
                acc += w[c]
                if u <= acc:
                    pick = c
                    break
            if pick == len(counts):
                counts.append(0)
                sums.append(0.0)
            z[i] = pick
            counts[pick] += 1
            sums[pick] += y[i]
        # drop empty tables, keeping the surviving order
        keep = [c for c in range(len(counts)) if counts[c] > 0]
        remap = dict((c, j) for j, c in enumerate(keep))
        counts = [counts[c] for c in keep]
        sums = [sums[c] for c in keep]
        z = [remap[v] for v in z]
    return z, counts, sums


def crp_collapsed(y, alpha=1.0, n_iter=50, mu0=0.0, tau2=10.0, sigma2=1.0,
                  seed=42):
    """
    Collapsed Gibbs sampler for a CRP mixture

    Formula: P(z_i | z_{-i}, y) with theta marginalized

    P(z_i = k | .) proportional to n_k^{-i} f(y_i | y_{-i,k}) and to
    alpha f(y_i) for a new table, with the normal-normal predictive
    density in closed form.  Marginalising theta is what collapses the
    sampler: no cluster parameters are ever stored.

    Parameters
    ----------
    y : array-like
        Observations.
    alpha : float
        Concentration, strictly positive.
    n_iter : int
        Number of full sweeps.
    mu0, tau2 : float
        Base measure N(mu0, tau2) for the cluster means.
    sigma2 : float
        Known within-cluster variance.
    seed : int
        Seed of the deterministic stream.

    Returns
    -------
    result : dict
        Keys: estimate (number of clusters), z, counts, cluster_mean,
        n_clusters, loglik, n.

    References
    ----------
    MacEachern (1994), Commun. Statist. B 23(3):727-741.
    Neal (2000), J. Comput. Graph. Statist. 9(2):249-265, algorithm 3.
    """
    y = core.vec(y)
    n = len(y)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    if not (alpha > 0.0):
        raise ValueError("alpha must be strictly positive")
    if not (tau2 > 0.0 and sigma2 > 0.0):
        raise ValueError("tau2 and sigma2 must be strictly positive")
    n_iter = int(n_iter)
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    z, counts, sums = collapsed_sweep(y, alpha, n_iter, mu0, tau2, sigma2, seed)
    means = []
    ll = 0.0
    for c in range(len(counts)):
        prec = 1.0 / tau2 + counts[c] / sigma2
        m = (mu0 / tau2 + sums[c] / sigma2) / prec
        means.append(m)
    for i in range(n):
        ll += norm_logpdf(y[i], means[z[i]], sigma2)
    return RichResult(payload={
        "estimate": len(counts),
        "z": z,
        "counts": counts,
        "cluster_mean": means,
        "n_clusters": len(counts),
        "loglik": ll,
        "n": n,
        "method": "collapsed Gibbs sampler for a CRP mixture",
    })


def cheatsheet():
    return "crpcol: collapsed Gibbs sampler for a CRP mixture"


# compact alias per ledger/NAMING.md
crpcollapsed = crp_collapsed
