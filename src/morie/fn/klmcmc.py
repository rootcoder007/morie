# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Kullback-Leibler diagnostic between an MCMC chain and its target.

Brooks and Gelman (1998), "General methods for monitoring convergence
of iterative simulations", J. Computational and Graphical Statistics
7(4):434-455, doi:10.1080/10618600.1998.10474787, section 3, treat
convergence as the agreement between the sampled distribution and the
target; the Kullback-Leibler divergence

    D(p || q) = sum_k p_k log(p_k / q_k)

over a common binning is the sharpest such summary.  Bins with no
chain mass contribute nothing (0 log 0 = 0); a bin with chain mass but
no target mass makes the divergence infinite, which is a genuine
failure and is reported as such rather than smoothed away.  D is zero
if and only if the two agree, which anchors the diagnostic.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["kl_mcmc_diagnostic"]


def kl_mcmc_diagnostic(chain, target, bins=20, lo=None, hi=None):
    """Binned KL divergence of a chain from a target density.

    Parameters
    ----------
    chain : array-like
        Draws.
    target : callable or array-like
        A density evaluated at the bin midpoints, or the bin
        probabilities themselves when its length equals bins.
    bins : int
        Number of equal-width bins.
    lo, hi : float, optional
        Binning range; the chain's own range by default.
    """
    x = core.vec(chain)
    n = len(x)
    if n < 2:
        raise ValueError("kl_mcmc_diagnostic: chain needs at least two draws")
    B = int(bins)
    if B < 2:
        raise ValueError("kl_mcmc_diagnostic: need at least two bins")
    a = min(x) if lo is None else float(lo)
    b = max(x) if hi is None else float(hi)
    if not b > a:
        raise ValueError("kl_mcmc_diagnostic: the binning range is degenerate")
    w = (b - a) / B
    cnt = [0.0] * B
    for v in x:
        k = int((v - a) / w)
        if k < 0:
            k = 0
        if k >= B:
            k = B - 1
        cnt[k] += 1.0
    p = [c / n for c in cnt]
    mid = [a + (k + 0.5) * w for k in range(B)]
    if callable(target):
        q = [float(target(m)) for m in mid]
    else:
        q = core.vec(target)
        if len(q) != B:
            raise ValueError("kl_mcmc_diagnostic: target must give one value per bin")
    for v in q:
        if v < 0:
            raise ValueError("kl_mcmc_diagnostic: target must be non-negative")
    sq = sum(q)
    if sq <= 0:
        raise ValueError("kl_mcmc_diagnostic: target has no mass on the binning range")
    q = [v / sq for v in q]
    kl = 0.0
    empty = 0
    for k in range(B):
        if p[k] <= 0:
            continue
        if q[k] <= 0:
            kl = float("inf")
            empty += 1
            continue
        kl += p[k] * math.log(p[k] / q[k])
    return RichResult(
        title="MCMC Kullback-Leibler diagnostic",
        summary_lines=[("draws", n), ("bins", B)],
        payload={
            "estimate": kl,
            "kl": kl,
            "p": p,
            "q": q,
            "unsupported_bins": empty,
            "n": n,
            "method": "binned D(p_chain || p_target), Brooks & Gelman (1998) sect. 3",
        },
    )


def cheatsheet():
    return "klmcmc: KL diagnostic for an MCMC chain"
