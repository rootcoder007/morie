"""Bootstrap percentile confidence interval for the median."""

from __future__ import annotations

from . import _s03core as core

from ._richresult import RichResult
from .btmult import _counts

__all__ = ["boot_ci_median"]


def _median(sorted_vals):
    m = len(sorted_vals)
    h = m // 2
    if m % 2 == 1:
        return sorted_vals[h]
    return 0.5 * (sorted_vals[h - 1] + sorted_vals[h])


def boot_ci_median(x, B=200, alpha=0.05, rng=2, exhaustive=False):
    """Percentile interval for the median, the tau = 0.5 bootstrap quantile.

    Efron, B. (1979), "Bootstrap methods: another look at the
    jackknife", *The Annals of Statistics* 7(1), 1-26,
    doi:10.1214/aos/1176344552, p. 3, steps 1-3, read from the Project
    Euclid PDF rendered as page images; Section 3 of that paper is the
    sample-median example, the case where the ordinary jackknife fails
    and the bootstrap does not.  Here R(X*, F-hat) is the median of the
    bootstrap sample, and the interval is the alpha/2 and 1 - alpha/2
    quantiles of its distribution, taken with the type-7 rule that R's
    own quantile() uses by default.

    With ``exhaustive`` the median's bootstrap distribution is computed
    completely rather than sampled.  For n = 3 that distribution is
    exactly 7/27, 13/27, 7/27 on the three order statistics -- a
    resample of three draws has its median at x_(1) exactly when two or
    more draws are index 1, which happens in 3*2 + 1 = 7 of the 27
    ordered triples, and symmetrically at x_(3) -- and that count is
    the anchor for this module.

    Resampling is deterministic; see ``boot_multinomial_weights``.

    Parameters
    ----------
    x : array-like
        The sample.
    B : int
        Replications when not enumerating.
    alpha : float
        Two-sided level; the interval is (alpha/2, 1 - alpha/2).
    rng : int
        Base of the van der Corput sequence.
    exhaustive : bool
        Enumerate all n^n resamples (n <= 6).

    Returns
    -------
    estimate : the median of the data itself
    lo, hi   : the percentile interval
    medians  : the B bootstrap medians
    """
    v = core.vec(x)
    n = len(v)
    if n == 0:
        raise ValueError("boot_ci_median: x is empty")
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError("boot_ci_median: alpha must lie strictly between 0 and 1")
    B = int(B)
    if not exhaustive and B < 1:
        raise ValueError("boot_ci_median: B must be at least 1")
    rng = int(rng)
    if rng < 2:
        raise ValueError("boot_ci_median: rng must be a base of at least 2")
    cs = _counts(n, B, rng, bool(exhaustive))
    order = sorted(range(n), key=lambda i: v[i])
    meds = []
    for row in cs:
        rs = []
        for i in order:
            rs.extend([v[i]] * row[i])
        meds.append(_median(rs))
    sm = sorted(meds)
    lo = core.quantile7(sm, a / 2.0)
    hi = core.quantile7(sm, 1.0 - a / 2.0)
    return RichResult(payload={
        "estimate": _median(sorted(v)),
        "lo": lo,
        "hi": hi,
        "medians": meds,
        "alpha": a,
        "B": len(meds),
        "n": n,
        "exhaustive": bool(exhaustive),
        "method": "Bootstrap CI for the median",
    })


def cheatsheet():
    return "btcimed: Bootstrap CI for the median"
