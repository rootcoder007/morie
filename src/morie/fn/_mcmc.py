# morie.fn -- shared helpers (rootcoder007/morie)
"""Shared MCMC diagnostic machinery.

Effective sample size and R-hat as specified in Vehtari et al. (2021), which
is the modern replacement for the classic Gelman-Rubin diagnostic: it uses
rank normalisation, folds the chains to detect scale differences, and splits
each chain in half so that a chain which has not mixed *within itself* is
caught rather than passing on between-chain agreement alone.
"""

from __future__ import annotations

import numpy as np

__all__ = ["autocov", "ess_from_chains", "split_rhat", "rank_normalize"]


def autocov(x, max_lag=None):
    """Autocovariance of a 1-D series by FFT, biased (divided by n)."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if max_lag is None:
        max_lag = n - 1
    xc = x - x.mean()
    size = 1
    while size < 2 * n:
        size *= 2
    f = np.fft.rfft(xc, size)
    acov = np.fft.irfft(f * np.conjugate(f), size)[: max_lag + 1].real / n
    return acov


def ess_from_chains(chains):
    r"""Effective sample size across chains (Vehtari et al. 2021, eq. 10-13).

    Combines the within-chain autocorrelations with the between-chain
    variance, and truncates the sum at the first negative pair -- Geyer's
    initial positive sequence -- which is what keeps the estimate from being
    dominated by noise in the long-lag tail.
    """
    C = np.atleast_2d(np.asarray(chains, dtype=float))
    m, n = C.shape
    if n < 4:
        return float("nan")
    acovs = np.array([autocov(C[j], n - 1) for j in range(m)])
    chain_var = acovs[:, 0] * n / max(n - 1, 1)
    W = float(chain_var.mean())
    if m > 1:
        B = n * float(np.var(C.mean(axis=1), ddof=1))
        var_hat = ((n - 1) * W + B) / n
    else:
        var_hat = W
    if var_hat <= 0:
        return float("nan")
    rho = 1.0 - (W - acovs[:, 1:].mean(axis=0)) / var_hat
    # Geyer initial positive sequence: stop at the first negative pair sum.
    t = 0
    total = 0.0
    while t + 1 < rho.size:
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        total += pair
        t += 2
    tau = -1.0 + 2.0 * total
    return float(m * n / max(tau, 1e-12)) if tau > 0 else float(m * n)


def rank_normalize(C):
    """Rank-normalise pooled draws, then map back to the normal scale."""
    from scipy.stats import norm

    flat = C.ravel()
    ranks = np.argsort(np.argsort(flat, kind="stable"), kind="stable") + 1
    z = norm.ppf((ranks - 0.375) / (flat.size + 0.25))
    return z.reshape(C.shape)


def split_rhat(chains, rank_normalized=True):
    r"""Split-R-hat: each chain halved before comparison.

    Splitting is what catches a chain that is drifting: two halves of a
    non-stationary chain disagree even when several chains happen to agree
    with each other.
    """
    C = np.atleast_2d(np.asarray(chains, dtype=float))
    m, n = C.shape
    if n < 4:
        return float("nan")
    half = n // 2
    S = np.vstack([C[:, :half], C[:, n - half:]])
    if rank_normalized:
        S = rank_normalize(S)
    m2, n2 = S.shape
    W = float(np.mean(np.var(S, axis=1, ddof=1)))
    B = n2 * float(np.var(S.mean(axis=1), ddof=1)) if m2 > 1 else 0.0
    if W <= 0:
        return float("nan")
    var_hat = ((n2 - 1) * W + B) / n2
    return float(np.sqrt(var_hat / W))
