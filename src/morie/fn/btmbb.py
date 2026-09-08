# morie.fn -- wave 2 slice x_0_01 (rootcoder007/morie)
"""Moving-block bootstrap for a stationary series (Kuensch).

Kuensch, H. R. (1989), "The Jackknife and the Bootstrap for General
Stationary Observations", *The Annals of Statistics* 17(3), 1217-1241,
doi:10.1214/aos/1176347265 (verified against Crossref).

The iid bootstrap destroys the dependence that defines a time series and
so understates every standard error.  Kuensch's fix is to resample
OVERLAPPING blocks of ell consecutive observations: there are
n - ell + 1 possible starting positions, k = ceil(n / ell) blocks are
drawn uniformly from them with replacement and concatenated, and the
result is truncated back to n.

Overlapping blocks are the point of the moving version.  The
non-overlapping variant has only floor(n / ell) resampling units, and
the overlapping one has n - ell + 1, which is why it is the more
efficient of the two -- at the cost that neighbouring blocks share
observations and are therefore not independent.

The last block is truncated whenever ell does not divide n, so the
observations near the end of a drawn block are used slightly less often
than those near its start.  ``n_blocks`` and ``truncated`` report this
rather than leaving it implicit.

Anchor: ell = 1 makes the moving-block bootstrap the ordinary iid
bootstrap exactly, and the ordinary bootstrap variance of the mean is
the closed form sum (x - xbar)^2 / n^2, which ``var_iid`` reports.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["boot_moving_block"]


def block_reps(x, ell, stat, B, seed, circular):
    """Shared block-resampling engine; ``circular`` wraps the series."""
    from . import _tail1core as C

    n = len(x)
    k = int(math.ceil(n / float(ell)))
    starts = n if circular else n - ell + 1
    g = C.Lcg(seed)
    out = []
    for _ in range(int(B)):
        smp = []
        for _b in range(k):
            s = int(g.unif() * starts)
            if s >= starts:
                s = starts - 1
            for t in range(ell):
                smp.append(x[(s + t) % n] if circular else x[s + t])
        out.append(float(stat(smp[:n])))
    return out, k


def boot_moving_block(x, block_len=None, stat=None, B=200, seed=1, alpha=0.05):
    """Moving-block bootstrap replicates.

    Parameters
    ----------
    x : array-like
        The series, in time order.
    block_len : int, optional
        Block length ell.  Defaults to ``max(floor(n^(1/3)), 1)``, the
        standard rate for variance estimation.
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
        ``n_blocks``, ``n_starts``, ``truncated``, ``var_iid``, ``n``,
        ``B``.
    """
    xx = core.vec(x)
    n = len(xx)
    if n < 2:
        raise ValueError("boot_moving_block: need at least two observations")
    if block_len is None:
        block_len = max(int(n ** (1.0 / 3.0)), 1)
    ell = int(block_len)
    if not 1 <= ell <= n:
        raise ValueError("boot_moving_block: block_len must lie in 1..n")
    if int(B) < 2:
        raise ValueError("boot_moving_block: need at least two replicates")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_moving_block: alpha must lie strictly between 0 and 1")
    f = core.mean if stat is None else stat
    theta, k = block_reps(xx, ell, f, B, seed, False)
    xb = core.mean(xx)
    return RichResult(
        title="Moving-block bootstrap (Kuensch 1989)",
        summary_lines=[("n", n), ("block", ell), ("B", int(B))],
        payload={
            "theta_b": theta,
            "estimate": float(f(xx)),
            "se": core.sd(theta, 1),
            "lo": core.quantile7(theta, a / 2.0),
            "hi": core.quantile7(theta, 1.0 - a / 2.0),
            "block_len": ell,
            "n_blocks": k,
            "n_starts": n - ell + 1,
            "truncated": k * ell - n,
            "var_iid": sum((u - xb) ** 2 for u in xx) / (n * n),
            "n": n,
            "B": int(B),
            "method": "Kuensch (1989) Ann. Statist. 17(3):1217-1241",
        },
    )


def cheatsheet():
    return "btmbb: overlapping blocks, n-ell+1 starts; more efficient than non-overlapping, blocks correlated"
