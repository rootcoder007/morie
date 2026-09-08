# morie.fn -- wave 2 slice x_0_01 (rootcoder007/morie)
"""m-out-of-n bootstrap: resample m < n points with replacement.

Bickel, P. J., Goetze, F. and van Zwet, W. R. (1997), "Resampling fewer
than n observations: gains, losses, and remedies for losses",
*Statistica Sinica* 7(1), 1-31.

The ordinary bootstrap fails when the statistic is not smooth in the
empirical measure -- boundary parameters, extremes, shrinkage at a
kink.  Drawing m << n points with replacement restores consistency
because the resample no longer resolves the non-smooth feature, at the
cost of a slower rate.  The rescaling that makes the replicates usable
is the one the paper is built on: it is the law of

    sqrt(m) (theta*_m - theta_hat)

that approximates the law of sqrt(n) (theta_hat - theta), so the
implied standard error for theta_hat is sqrt(m/n) times the standard
deviation of the replicates, and the interval is centred on theta_hat
with half-widths shrunk by the same factor.

Both are returned: ``se_raw`` is the raw replicate spread and ``se`` is
the rescaled one.  m = n makes the two identical, which is the
degenerate anchor.

Draws come from the package's shared Lehmer stream so the Python and R
arms index the same observations.
"""

from __future__ import annotations

import math

from . import _s03core as core
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["boot_m_out_of_n"]


def resample_idx(g, n, m):
    """m indices drawn with replacement from 0..n-1.  Shared with btsubrho."""
    out = []
    for _ in range(m):
        j = int(g.unif() * n)
        if j >= n:
            j = n - 1
        out.append(j)
    return out


def boot_m_out_of_n(x, m=None, stat=None, B=200, seed=1, alpha=0.05):
    """Replicates of ``stat`` on m-out-of-n resamples.

    Parameters
    ----------
    x : array-like
        The observed sample.
    m : int, optional
        Resample size.  Defaults to ``floor(sqrt(n))``, the canonical
        choice when no rate is known.
    stat : callable, optional
        Statistic of a sample.  Defaults to the mean.
    B : int
        Replicates.
    seed : int
        Seed for the shared deterministic stream.
    alpha : float
        Two-sided error rate for the interval.

    Returns
    -------
    RichResult
        ``theta_b``, ``theta_hat``, ``se_raw``, ``se`` (rescaled by
        sqrt(m/n)), ``lo``/``hi`` (rescaled percentile interval about
        ``theta_hat``), ``m``, ``n``, ``B``.
    """
    xx = core.vec(x)
    n = len(xx)
    if n < 2:
        raise ValueError("boot_m_out_of_n: need at least two observations")
    if m is None:
        m = int(math.floor(math.sqrt(n)))
    m = int(m)
    if not 1 <= m <= n:
        raise ValueError("boot_m_out_of_n: m must lie in 1..n")
    if int(B) < 2:
        raise ValueError("boot_m_out_of_n: need at least two replicates")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_m_out_of_n: alpha must lie strictly between 0 and 1")
    f = core.mean if stat is None else stat
    th = float(f(xx))
    g = C.Lcg(seed)
    theta = []
    for _ in range(int(B)):
        idx = resample_idx(g, n, m)
        theta.append(float(f([xx[j] for j in idx])))
    raw = core.sd(theta, 1)
    r = math.sqrt(m / float(n))
    qlo = core.quantile7(theta, a / 2.0)
    qhi = core.quantile7(theta, 1.0 - a / 2.0)
    return RichResult(
        title="m-out-of-n bootstrap",
        summary_lines=[("n", n), ("m", m), ("B", int(B)), ("se", r * raw)],
        payload={
            "theta_b": theta,
            "theta_hat": th,
            "se_raw": raw,
            "se": r * raw,
            "lo": th + r * (qlo - th),
            "hi": th + r * (qhi - th),
            "m": m,
            "n": n,
            "B": int(B),
            "estimate": th,
            "method": "Bickel, Goetze and van Zwet (1997) Statist. Sinica 7(1):1-31",
        },
    )


def cheatsheet():
    return "btmoutn: draw m<n with replacement; sqrt(m)(theta*-theta_hat) mimics sqrt(n)(theta_hat-theta)"


# compact alias per ledger/NAMING.md
bootmoutofn = boot_m_out_of_n
