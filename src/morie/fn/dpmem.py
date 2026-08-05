# morie.fn -- function file (rootcoder007/morie)
"""Dirichlet process mixture model."""

import math

from . import _s03core as core
from ._richresult import RichResult
from .crpcol import collapsed_sweep, norm_logpdf

__all__ = ["dirichlet_process_mixture"]


def dirichlet_process_mixture(y, alpha=1.0, base_distribution=None, n_iter=50,
                              sigma2=1.0, seed=42):
    """
    Dirichlet process mixture model

    Formula: G ~ DP(alpha, G_0); theta_i | G ~ G; y_i ~ f(theta_i)

    Fitted by the collapsed Gibbs sampler with a normal base measure and
    a known within-cluster variance.  Because theta is integrated out,
    the single-cluster case reduces exactly to the conjugate normal
    posterior, mean (mu0/tau2 + sum y/sigma2) / (1/tau2 + n/sigma2),
    which is the degenerate check the fit has to reproduce.

    Parameters
    ----------
    y : array-like
        Observations.
    alpha : float
        Concentration, strictly positive.
    base_distribution : sequence or None
        (mu0, tau2) of the normal base measure; None uses (0, 10).
    n_iter : int
        Number of sweeps.
    sigma2 : float
        Known within-cluster variance.
    seed : int
        Seed of the deterministic stream.

    Returns
    -------
    result : dict
        Keys: estimate (posterior predictive mean), n_clusters, z,
        counts, cluster_mean, weights, loglik, n.

    References
    ----------
    Antoniak (1974), Ann. Statist. 2(6):1152-1174.
    Escobar & West (1995), JASA 90(430):577-588.
    Neal (2000), J. Comput. Graph. Statist. 9(2):249-265.
    """
    y = core.vec(y)
    n = len(y)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    if not (alpha > 0.0):
        raise ValueError("alpha must be strictly positive")
    if base_distribution is None:
        mu0, tau2 = 0.0, 10.0
    else:
        b = core.vec(base_distribution)
        if len(b) != 2:
            raise ValueError("base_distribution must be (mu0, tau2)")
        mu0, tau2 = b[0], b[1]
    if not (tau2 > 0.0 and sigma2 > 0.0):
        raise ValueError("tau2 and sigma2 must be strictly positive")
    z, counts, sums = collapsed_sweep(y, alpha, int(n_iter), mu0, tau2,
                                      sigma2, seed)
    means, w = [], []
    for c in range(len(counts)):
        prec = 1.0 / tau2 + counts[c] / sigma2
        means.append((mu0 / tau2 + sums[c] / sigma2) / prec)
        w.append(counts[c] / (n + alpha))
    w_new = alpha / (n + alpha)
    pred = sum(w[c] * means[c] for c in range(len(counts))) + w_new * mu0
    ll = sum(norm_logpdf(y[i], means[z[i]], sigma2) for i in range(n))
    return RichResult(payload={
        "estimate": pred,
        "n_clusters": len(counts),
        "z": z,
        "counts": counts,
        "cluster_mean": means,
        "weights": w,
        "w_new": w_new,
        "loglik": ll,
        "n": n,
        "method": "Dirichlet process mixture, collapsed Gibbs",
    })


def cheatsheet():
    return "dpmem: Dirichlet process mixture model"


# compact alias per ledger/NAMING.md
dirichletprocessmixture = dirichlet_process_mixture
