# morie.fn -- wave 2 slice x_0_01 (rootcoder007/morie)
"""Circular block bootstrap: wrap the series, then resample blocks.

Politis, D. N. and Romano, J. P. (1992), "A circular block-resampling
procedure for stationary data", in R. LePage and L. Billard (eds),
*Exploring the Limits of Bootstrap*, Wiley, 263-270.  The block-length
theory used alongside it is Politis, D. N. and White, H. (2004),
"Automatic Block-Length Selection for the Dependent Bootstrap",
*Econometric Reviews* 23(1), 53-70, whose Lemma 3.1 treats the circular
and moving-block cases together (see ``btblen``).

The moving-block bootstrap has a defect that is easy to miss: with
n - ell + 1 starting positions, the observations at the two ends of the
series appear in fewer blocks than those in the middle, so the resample
mean is a WEIGHTED mean of the data and the bootstrap is biased.  Wrapping
the series into a circle gives all n starting positions equal footing,
every observation appears in exactly ell blocks, and that bias vanishes:
E*[xbar*] = xbar exactly.

That exactness is the anchor, and it is a genuine discriminator -- the
same check applied to the moving-block bootstrap fails, which is the
entire reason this variant exists.  ``ebar_star`` reports the exact
conditional mean of the resample mean, computed combinatorially rather
than by simulation.

Anchor 2: ell = n makes every circular block a full rotation of the
series, so for any permutation-invariant statistic every replicate equals
the statistic on the whole sample and the spread is exactly zero.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult
from .btmbb import block_reps

__all__ = ["boot_circular_block"]


def boot_circular_block(x, block_len=None, stat=None, B=200, seed=1, alpha=0.05):
    """Circular block bootstrap replicates.

    Parameters
    ----------
    x : array-like
        The series, in time order.
    block_len : int, optional
        Block length ell.  Defaults to ``max(floor(n^(1/3)), 1)``.
    stat : callable, optional
        Statistic of a series.  Defaults to the mean.
    B : int
        Replicates.
    seed : int
        Seed for the shared deterministic stream.
    alpha : float
        Two-sided error rate.

    Returns
    -------
    RichResult
        ``theta_b``, ``estimate``, ``se``, ``lo``/``hi``, ``block_len``,
        ``n_blocks``, ``ebar_star`` (exact conditional mean of the
        resample mean, equal to xbar), ``n``, ``B``.
    """
    xx = core.vec(x)
    n = len(xx)
    if n < 2:
        raise ValueError("boot_circular_block: need at least two observations")
    if block_len is None:
        block_len = max(int(n ** (1.0 / 3.0)), 1)
    ell = int(block_len)
    if not 1 <= ell <= n:
        raise ValueError("boot_circular_block: block_len must lie in 1..n")
    if int(B) < 2:
        raise ValueError("boot_circular_block: need at least two replicates")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_circular_block: alpha must lie strictly between 0 and 1")
    f = core.mean if stat is None else stat
    theta, k = block_reps(xx, ell, f, B, seed, True)
    # Exact conditional mean of a circular block mean: every observation
    # appears in exactly ell of the n blocks, so the block means average
    # to xbar with no edge weighting at all.
    tot = 0.0
    for s in range(n):
        for t in range(ell):
            tot += xx[(s + t) % n]
    ebar = tot / (n * ell)
    return RichResult(
        title="Circular block bootstrap (Politis and Romano 1992)",
        summary_lines=[("n", n), ("block", ell), ("B", int(B))],
        payload={
            "theta_b": theta,
            "estimate": float(f(xx)),
            "se": core.sd(theta, 1),
            "lo": core.quantile7(theta, a / 2.0),
            "hi": core.quantile7(theta, 1.0 - a / 2.0),
            "block_len": ell,
            "n_blocks": k,
            "ebar_star": ebar,
            "n": n,
            "B": int(B),
            "method": "Politis and Romano (1992) in Exploring the Limits of Bootstrap, 263-270",
        },
    )


def cheatsheet():
    return "btcbb: wrapping the series equalises how often each point is used; E*[xbar*] = xbar EXACTLY"
