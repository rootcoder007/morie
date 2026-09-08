# morie.fn -- function file (rootcoder007/morie)
"""Consistency of a test: power tends to 1 as the sample size grows."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['consist', 'gibbons_consistency']


def consist(nvals, effect, alpha=0.05):
    """Power along a sequence of sample sizes, and the consistency verdict.

    Section 1.2.10 (book p. 23).  A test is consistent against an
    alternative when its power tends to 1 as n grows without bound.
    For a statistic whose standardised form shifts by sqrt(n) * d under
    the alternative,

    .. math:: Pw(n) = 1 - \\Phi(z_\\alpha - \\sqrt{n}\\, d),

    which increases monotonically to 1 for every d > 0 -- that is the
    consistency statement, and the returned ``consistent`` flag reports
    whether the supplied effect satisfies it.

    Parameters
    ----------
    nvals : sequence of int
        Sample sizes, increasing.
    effect : float
        Standardised effect d per root observation.
    alpha : float, optional
        One-sided size (default 0.05).

    Returns
    -------
    RichResult
        keys ``power`` (one per n), ``consistent`` (1/0),
        ``monotone`` (1/0), ``limit`` (power at the largest n),
        ``effect``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 1.2.10, p. 23.
    """
    ns = [int(v) for v in nvals]
    if len(ns) < 1:
        raise ValueError("nvals must be non-empty.")
    if any(v < 1 for v in ns):
        raise ValueError("sample sizes must be at least 1.")
    effect = float(effect)
    alpha = float(alpha)
    za = stats.norm.ppf(1.0 - alpha)
    pw = [1.0 - stats.norm.cdf(za - math.sqrt(v) * effect) for v in ns]
    mono = all(pw[i] <= pw[i + 1] + 1e-15 for i in range(len(pw) - 1))
    return RichResult(
        payload={
            "power": [float(v) for v in pw],
            "consistent": int(effect > 0.0),
            "monotone": int(mono),
            "limit": float(pw[-1]),
            "effect": effect,
            "method": "consistency: Pw(n) = 1 - Phi(z_alpha - sqrt(n) d)",
        }
    )


gibbons_consistency = consist
