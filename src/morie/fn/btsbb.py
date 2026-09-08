# morie.fn -- wave 2 slice x_0_01 (rootcoder007/morie)
"""Stationary bootstrap: geometric block lengths on a wrapped series.

Politis, D. N. and Romano, J. P. (1994), "The Stationary Bootstrap",
*Journal of the American Statistical Association* 89(428), 1303-1313,
doi:10.1080/01621459.1994.10476870 (verified against Crossref).

Fixed-length blocks make the resampled series NON-stationary: whether
observation t* is followed by its true successor depends on the position
of t* within its block.  Politis and Romano remove that by randomising
the block length: start at a uniform position on the wrapped series and
at each subsequent step continue to the next observation with
probability 1 - p, or jump to a fresh uniform position with probability
p.  Block lengths are then iid Geometric(p) with mean 1/p, and the
resampled series is genuinely stationary conditionally on the data --
which is what the paper's title claims and what fixed blocks cannot
deliver.

The expected block length 1/p plays the role that ell plays in the
moving-block bootstrap, and ``btblen`` estimates the optimal one.

Anchors, both exact and neither an asymptotic claim: p = 1 makes every
block length 1, so the procedure collapses to the ordinary iid
bootstrap; and the number of block starts is exactly
1 + Binomial(n - 1, p) per replicate, because the first step always
starts a block and each of the remaining n - 1 steps restarts
independently with probability p.  So E[n_runs] = B (1 + (n-1) p)
exactly -- a closed form free of the censoring that makes the realised
block LENGTHS a biased sample of the Geometric law (a run still in
progress when the series ends is truncated, and short runs are
over-represented among those that finish).  ``mean_block`` is reported
as n B / n_runs and is stated to be that ratio, not a Geometric mean.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["boot_stationary_block"]


def boot_stationary_block(x, p=None, stat=None, B=200, seed=1, alpha=0.05):
    """Stationary bootstrap replicates.

    Parameters
    ----------
    x : array-like
        The series, in time order.
    p : float, optional
        Restart probability, ``0 < p <= 1``; the mean block length is
        ``1/p``.  Defaults to ``n^(-1/3)``, matching the standard block
        rate.
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
        ``theta_b``, ``estimate``, ``se``, ``lo``/``hi``, ``p``,
        ``n_runs`` (total block starts), ``exp_runs`` (its exact mean,
        B (1 + (n-1) p)), ``mean_block`` (= n B / n_runs),
        ``exp_block`` (= 1/p), ``n``, ``B``.
    """
    from . import _tail1core as C

    xx = core.vec(x)
    n = len(xx)
    if n < 2:
        raise ValueError("boot_stationary_block: need at least two observations")
    if p is None:
        p = n ** (-1.0 / 3.0)
    p = float(p)
    if not (0.0 < p <= 1.0):
        raise ValueError("boot_stationary_block: p must lie in (0, 1]")
    if int(B) < 2:
        raise ValueError("boot_stationary_block: need at least two replicates")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_stationary_block: alpha must lie strictly between 0 and 1")
    f = core.mean if stat is None else stat
    g = C.Lcg(seed)
    theta = []
    runs = 0
    for _ in range(int(B)):
        smp = []
        j = 0
        for t in range(n):
            if t == 0 or g.unif() < p:
                runs += 1
                j = int(g.unif() * n)
                if j >= n:
                    j = n - 1
            else:
                j = (j + 1) % n
            smp.append(xx[j])
        theta.append(float(f(smp)))
    mb = (n * int(B)) / float(runs)
    return RichResult(
        title="Stationary bootstrap (Politis and Romano 1994)",
        summary_lines=[("n", n), ("p", p), ("B", int(B))],
        payload={
            "theta_b": theta,
            "estimate": float(f(xx)),
            "se": core.sd(theta, 1),
            "lo": core.quantile7(theta, a / 2.0),
            "hi": core.quantile7(theta, 1.0 - a / 2.0),
            "p": p,
            "exp_block": 1.0 / p,
            "mean_block": mb,
            "exp_runs": int(B) * (1.0 + (n - 1.0) * p),
            "n_runs": runs,
            "n": n,
            "B": int(B),
            "method": "Politis and Romano (1994) JASA 89(428):1303-1313",
        },
    )


def cheatsheet():
    return "btsbb: geometric block lengths make the RESAMPLED series stationary; fixed blocks do not"
