# morie.fn -- wave 2 slice x_0_01 (rootcoder007/morie)
"""Subsampling: size-m subsets drawn WITHOUT replacement.

Politis, D. N., Romano, J. P. and Wolf, M. (1999), *Subsampling*,
Springer Series in Statistics.

Subsampling is not a bootstrap.  It needs only that the normalised
statistic have SOME limit law; it never assumes that law is estimable by
resampling with replacement, so it stays consistent where both Efron's
bootstrap and the m-out-of-n bootstrap fail.  The construction is: draw
subsets of size m without replacement, form the roots

    L_m(t) = #{ b : tau_m (theta_hat_{m,b} - theta_hat) <= t } / B,

with tau_m = sqrt(m) the default rate, and read the interval off the
quantiles of L_m rescaled by tau_n = sqrt(n):

    [ theta_hat - q_{1-alpha/2}/tau_n , theta_hat - q_{alpha/2}/tau_n ].

Note the inversion: the UPPER quantile of the root sets the LOWER
endpoint.  Getting this backwards is the classical sign error here and
it is invisible on a symmetric fixture, so the module is anchored on an
asymmetric one.

Anchors, neither of which runs through the resampling loop: with m = n
every subsample is the whole sample, so for any permutation-invariant
statistic every replicate equals theta_hat and the root distribution
collapses to a point mass at zero; and for the mean the exact
without-replacement variance of a subsample mean is
(s^2/m)(n-m)/(n-1), which ``var_closed`` reports.

Draws come from the package's shared Lehmer stream via a partial
Fisher-Yates shuffle, so the two arms select the same subsets.
"""

from __future__ import annotations

import math

from . import _s03core as core
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["boot_subsampling"]


def subset_idx(g, n, m):
    """m distinct indices from 0..n-1 by a partial Fisher-Yates shuffle."""
    p = list(range(n))
    for i in range(m):
        k = i + int(g.unif() * (n - i))
        if k > n - 1:
            k = n - 1
        p[i], p[k] = p[k], p[i]
    return p[:m]


def boot_subsampling(x, m=None, stat=None, B=200, seed=1, alpha=0.05):
    """Subsampling replicates and the subsampling interval.

    Parameters
    ----------
    x : array-like
        The observed sample.
    m : int, optional
        Subsample size.  Defaults to ``floor(sqrt(n))``.
    stat : callable, optional
        Statistic of a sample.  Defaults to the mean.
    B : int
        Number of subsamples.
    seed : int
        Seed for the shared deterministic stream.
    alpha : float
        Two-sided error rate.

    Returns
    -------
    RichResult
        ``theta_b``, ``theta_hat``, ``roots`` (the scaled roots),
        ``lo``/``hi``, ``se`` (implied, = sd(roots)/sqrt(n)),
        ``var_closed`` (exact without-replacement variance of the
        subsample mean; NaN for a custom statistic), ``m``, ``n``, ``B``.
    """
    xx = core.vec(x)
    n = len(xx)
    if n < 2:
        raise ValueError("boot_subsampling: need at least two observations")
    if m is None:
        m = int(math.floor(math.sqrt(n)))
    m = int(m)
    if not 1 <= m <= n:
        raise ValueError("boot_subsampling: m must lie in 1..n")
    if int(B) < 2:
        raise ValueError("boot_subsampling: need at least two subsamples")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_subsampling: alpha must lie strictly between 0 and 1")
    f = core.mean if stat is None else stat
    th = float(f(xx))
    g = C.Lcg(seed)
    theta = []
    for _ in range(int(B)):
        idx = subset_idx(g, n, m)
        theta.append(float(f([xx[j] for j in idx])))
    tm = math.sqrt(m)
    tn = math.sqrt(n)
    roots = [tm * (u - th) for u in theta]
    qlo = core.quantile7(roots, a / 2.0)
    qhi = core.quantile7(roots, 1.0 - a / 2.0)
    if stat is None and n > 1 and m > 0:
        s2 = core.variance(xx, 1)
        vc = (s2 / m) * (n - m) / (n - 1.0)
    else:
        vc = float("nan")
    return RichResult(
        title="Subsampling (Politis, Romano and Wolf 1999)",
        summary_lines=[("n", n), ("m", m), ("B", int(B)), ("theta_hat", th)],
        payload={
            "theta_b": theta,
            "roots": roots,
            "theta_hat": th,
            "lo": th - qhi / tn,
            "hi": th - qlo / tn,
            "se": (core.sd(roots, 1) / tn) if len(roots) > 1 else float("nan"),
            "var_closed": vc,
            "m": m,
            "n": n,
            "B": int(B),
            "estimate": th,
            "method": "Politis, Romano and Wolf (1999) Subsampling, Springer",
        },
    )


def cheatsheet():
    return "btsubs: size-m subsets without replacement; the UPPER root quantile gives the LOWER endpoint"
