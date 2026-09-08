# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Bootstrap confidence interval for a Pearson correlation, on Fisher's z.

Source: Efron, B. and Tibshirani, R. J. (1993), *An Introduction to the
Bootstrap*, Chapman and Hall, chapters 12-13 (the law-school correlation
example), and Davison and Hinkley (1997), *Bootstrap Methods and their
Application*, Section 5.2 for the percentile machinery.

Pairs are resampled -- rows of (x, y) together, never the two vectors
independently, which would destroy the very dependence being estimated.
Each resample gives r*_b; the interval is built on the variance-
stabilising transform z*_b = atanh(r*_b) and mapped back with tanh:

    [ tanh( z*_{(alpha/2)} ),  tanh( z*_{(1-alpha/2)} ) ].

Because tanh is monotone the transform does not move the *percentile*
endpoints at all -- tanh(quantile) = quantile(tanh) -- so on its own it
buys nothing.  What it buys is a symmetric scale on which the normal and
basic intervals are worth building, and those are returned alongside as
``lo_normal``/``hi_normal``.  The z-scale standard error is reported so
the difference is visible.

Resampling is deterministic, from the Park-Miller generator
s <- 16807 s mod (2^31 - 1), so both language arms draw identical pairs.

This module previously carried a body that computed a Kolmogorov-Smirnov
statistic and a Spearman correlation, matching neither its name nor its
documentation; that body has been discarded.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["boot_ci_correlation"]

_M = 2147483647


def boot_ci_correlation(x, y, B=999, alpha=0.05, seed=1):
    """Fisher-z percentile interval for Pearson's r.

    Returns
    -------
    lo, hi : the endpoints on the r scale
    r_hat : the correlation on the original data
    z_se : the standard deviation of the replicate z values
    lo_normal, hi_normal : the normal interval built on the z scale
    """
    xv = core.vec(x)
    yv = core.vec(y)
    n = len(xv)
    if n == 0:
        raise ValueError("boot_ci_correlation: x is empty")
    if len(yv) != n:
        raise ValueError("boot_ci_correlation: x and y have different lengths")
    if n < 3:
        raise ValueError("boot_ci_correlation: need at least three pairs")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_ci_correlation: alpha must lie strictly between 0 and 1")
    Bn = int(B)
    if Bn < 1:
        raise ValueError("boot_ci_correlation: B must be at least one")
    r_hat = core.corr(xv, yv)
    s = int(seed) % _M
    if s <= 0:
        s += _M - 1
    zs = []
    rs = []
    for _ in range(Bn):
        xa = []
        ya = []
        for _ in range(n):
            s = (16807 * s) % _M
            u = (s - 1.0) / (_M - 1.0)
            j = int(u * n)
            if j >= n:
                j = n - 1
            xa.append(xv[j])
            ya.append(yv[j])
        r = core.corr(xa, ya)
        if r != r:
            continue
        if r >= 1.0:
            r = 1.0 - 1e-15
        if r <= -1.0:
            r = -1.0 + 1e-15
        rs.append(r)
        zs.append(0.5 * math.log((1.0 + r) / (1.0 - r)))
    if not zs:
        raise ValueError("boot_ci_correlation: every resample was degenerate")
    lo_z = core.quantile7(zs, a / 2.0)
    hi_z = core.quantile7(zs, 1.0 - a / 2.0)
    zse = core.sd(zs, 1) if len(zs) > 1 else float("nan")
    zc = 0.5 * math.log((1.0 + r_hat) / (1.0 - r_hat)) if abs(r_hat) < 1.0 else float("nan")
    q = core.qnorm(1.0 - a / 2.0)
    return RichResult(
        title="Bootstrap interval for a correlation",
        summary_lines=[("r", r_hat), ("lo", math.tanh(lo_z)), ("hi", math.tanh(hi_z))],
        payload={
            "lo": math.tanh(lo_z),
            "hi": math.tanh(hi_z),
            "estimate": math.tanh(hi_z) - math.tanh(lo_z),
            "r_hat": r_hat,
            "z_hat": zc,
            "z_se": zse,
            "lo_z": lo_z,
            "hi_z": hi_z,
            "lo_normal": math.tanh(zc - q * zse),
            "hi_normal": math.tanh(zc + q * zse),
            "B_used": len(zs),
            "B": Bn,
            "n": n,
            "method": "pairs bootstrap, percentile interval on atanh(r), mapped back by tanh",
        },
    )


def cheatsheet():
    return "btcicor: Bootstrap CI for Pearson correlation via Fisher z"


# compact alias per ledger/NAMING.md
bootcicorrelation = boot_ci_correlation
