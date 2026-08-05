# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Bootstrap percentile confidence interval for a quantile.

Source: Davison, A. C. and Hinkley, D. V. (1997), *Bootstrap Methods and
their Application*, Cambridge University Press, Sections 2.3 and 5.3.
Resample the data with replacement, recompute the tau-quantile on each
resample, and read the alpha/2 and 1 - alpha/2 percentiles off the B
replicates.

Two things are deliberate.  First, the resampling is deterministic: the
indices come from the Park-Miller multiplicative congruential generator
s <- 16807 s mod (2^31 - 1), so the Python and R arms draw the *same*
resamples and land on the same numbers, not merely the same
distribution.  The arithmetic stays exact in a double, since
16807 * (2^31 - 2) < 2^53.

Second, the sample quantile of a bootstrap resample is a lattice-valued
statistic -- it can only take values that appear in the data -- so its
bootstrap distribution is discrete and the interval endpoints are
themselves data values.  For extreme tau and small n the interval
degenerates to a point.  ``n_distinct`` is returned so that degeneracy is
visible rather than mistaken for precision.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["boot_ci_quantile"]

_M = 2147483647
_A = 16807


def lcg_seed(seed):
    """Park-Miller state, forced into 1..2^31-2."""
    s = int(seed) % _M
    if s <= 0:
        s += _M - 1
    return s


def lcg_next(s):
    """One step; returns (new state, uniform in (0,1))."""
    s = (_A * s) % _M
    return s, (s - 1.0) / (_M - 1.0)


def boot_ci_quantile(x, tau=0.5, B=999, alpha=0.05, seed=1):
    """Percentile interval for the tau-quantile.

    Returns
    -------
    lo, hi : the endpoints
    q_hat : the quantile on the original data
    theta_b : the B replicate quantiles
    """
    v = core.vec(x)
    n = len(v)
    if n == 0:
        raise ValueError("boot_ci_quantile: x is empty")
    t = float(tau)
    if not (0.0 <= t <= 1.0):
        raise ValueError("boot_ci_quantile: tau must lie in [0, 1]")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_ci_quantile: alpha must lie strictly between 0 and 1")
    Bn = int(B)
    if Bn < 1:
        raise ValueError("boot_ci_quantile: B must be at least one")
    s = lcg_seed(seed)
    reps = []
    for _ in range(Bn):
        samp = []
        for _ in range(n):
            s, u = lcg_next(s)
            j = int(u * n)
            if j >= n:
                j = n - 1
            samp.append(v[j])
        reps.append(core.quantile7(samp, t))
    lo = core.quantile7(reps, a / 2.0)
    hi = core.quantile7(reps, 1.0 - a / 2.0)
    return RichResult(
        title="Bootstrap percentile interval for a quantile",
        summary_lines=[("lo", lo), ("hi", hi)],
        payload={
            "lo": lo,
            "hi": hi,
            "estimate": hi - lo,
            "q_hat": core.quantile7(v, t),
            "theta_b": reps,
            "n_distinct": len(set(reps)),
            "B": Bn,
            "n": n,
            "tau": t,
            "method": "percentile interval on B deterministic resample quantiles (Park-Miller indices)",
        },
    )


def cheatsheet():
    return "btciqua: Bootstrap percentile CI for an arbitrary quantile"


# compact alias per ledger/NAMING.md
bootciquantile = boot_ci_quantile
