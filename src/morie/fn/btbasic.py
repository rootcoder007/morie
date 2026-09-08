# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Basic (reverse-percentile) bootstrap confidence interval.

Source: Davison, A. C. and Hinkley, D. V. (1997), *Bootstrap Methods and
their Application*, Cambridge University Press, Section 5.2, limits
(2.10)/(5.6).  The interval inverts the bootstrap distribution of
T - theta rather than reading quantiles of T* directly:

    [ 2 t - t*_{(1-alpha/2)} ,  2 t - t*_{(alpha/2)} ].

The reversal is the whole point.  If the replicates are skewed to the
right, the percentile interval leans right and the basic interval leans
*left*, because the quantity being inverted is the error t* - t, not the
estimate.  Getting the direction wrong silently produces an interval of
the correct width on the wrong side, which no width check would catch;
the anchor therefore pins the reflection about t, not just the length.

Quantiles are R's type 7 in both language arms, computed by the shared
helper, so the two agree to the last bit.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["boot_basic_ci"]


def boot_basic_ci(theta_hat, theta_b, alpha=0.05):
    """Basic bootstrap interval.

    Parameters
    ----------
    theta_hat : float
        The estimate on the original data.
    theta_b : array-like
        The bootstrap replicates.
    alpha : float
        Two-sided error rate; the interval has nominal coverage
        1 - alpha.

    Returns
    -------
    lo, hi : the interval endpoints
    q_lo, q_hi : the replicate quantiles that were reflected
    """
    v = core.vec(theta_b)
    n = len(v)
    if n == 0:
        raise ValueError("boot_basic_ci: no bootstrap replicates")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_basic_ci: alpha must lie strictly between 0 and 1")
    t = float(theta_hat)
    qlo = core.quantile7(v, a / 2.0)
    qhi = core.quantile7(v, 1.0 - a / 2.0)
    lo = 2.0 * t - qhi
    hi = 2.0 * t - qlo
    return RichResult(
        title="Basic bootstrap interval",
        summary_lines=[("lo", lo), ("hi", hi)],
        payload={
            "lo": lo,
            "hi": hi,
            "estimate": hi - lo,
            "q_lo": qlo,
            "q_hi": qhi,
            "theta_hat": t,
            "B": n,
            "method": "Davison and Hinkley (1997) basic bootstrap limits, 2t - t*_(1-a/2), 2t - t*_(a/2)",
        },
    )


def cheatsheet():
    return "btbasic: Basic (reverse-percentile) bootstrap CI"


# compact alias per ledger/NAMING.md
bootbasicci = boot_basic_ci
