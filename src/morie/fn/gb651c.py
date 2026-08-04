# morie.fn -- function file (rootcoder007/morie)
"""Curtailed sampling form of the control median test -- eq. (6.5.2)."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['ctrlmedcur', 'gibbons_ctrl_median_curtail']


def ctrlmedcur(m, n, alpha=0.05):
    """Stopping index d for the curtailed control median test.

    Section 6.5.1 (book p. 258), eq. (6.5.2): testing H0: q = 0.5
    against H1: q < 0.5, the normal approximation rejects when
    V <= d with

    .. math:: d = \\frac{m}{2} - z_\\alpha
        \\left[\\frac{m(m+n)}{4n}\\right]^{1/2},

    rounded down, and the rejection is equivalent to
    Y_(r+1) <= X_(d).  So the experiment can stop as soon as either the
    Y median or the d-th X order statistic is observed, whichever comes
    first -- the decision is the same as with the complete data.  The
    exact d from the null distribution of eq. (6.5.1) is returned
    alongside.

    Parameters
    ----------
    m : int
        Size of the X (treatment) sample.
    n : int
        Size of the Y (control) sample, odd.
    alpha : float, optional
        One-sided size (default 0.05).

    Returns
    -------
    RichResult
        keys ``d`` (rounded down), ``d_raw``, ``d_exact``,
        ``alpha_exact``, ``z_alpha``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 6.5.1, eq. (6.5.2), p. 258.
    """
    m = int(m)
    n = int(n)
    alpha = float(alpha)
    if m < 1 or n < 1:
        raise ValueError("m and n must be at least 1.")
    if n % 2 == 0:
        raise ValueError("the control sample size n must be odd (n = 2r+1).")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1).")
    za = stats.norm.ppf(1.0 - alpha)
    draw = m / 2.0 - za * math.sqrt(m * (m + n) / (4.0 * n))
    r = (n - 1) // 2
    den = math.comb(m + 2 * r + 1, m)
    pmf = [
        math.comb(m + r - j, m - j) * math.comb(j + r, j) / den
        for j in range(m + 1)
    ]
    dex = float("nan")
    aex = 0.0
    acc = 0.0
    for j in range(m + 1):
        acc += pmf[j]
        if acc <= alpha:
            dex = float(j)
            aex = acc
        else:
            break
    return RichResult(
        payload={
            "d": float(math.floor(draw)),
            "d_raw": float(draw),
            "d_exact": dex,
            "alpha_exact": float(aex),
            "z_alpha": float(za),
            "m": m,
            "n": n,
            "method": "curtailed control median test, eq. (6.5.2)",
        }
    )


gibbons_ctrl_median_curtail = ctrlmedcur
