# morie.fn -- function file (rootcoder007/morie)
"""Confidence interval for the scale ratio -- eqs. (9.8.1)-(9.8.2)."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['scaleci', 'gibbons_scale_ci']


def scaleci(x, y, alpha=0.05, k=None):
    """Interval for theta from the ordered positive ratios x_i / y_j.

    Section 9.8 (book p. 328).  In the pure scale model
    F_Y(x) = F_X(theta x) with a common median of zero, the Sukhatme
    criterion counts the positive pairs with x_i / y_j < theta, so
    inverting it gives

    .. math:: \\left(\\frac{x_i}{y_j}\\right)_{(k)} < \\theta
        < \\left(\\frac{x_i}{y_j}\\right)_{(k')} \\qquad (9.8.1),

    over the array of POSITIVE ratios only.  For m, n > 10 the book
    takes

    .. math:: k = \\frac{mn}{4} + 0.5
        - z_{\\alpha/2}\\sqrt{\\frac{mn(N+7)}{48}}
        \\qquad (9.8.2),

    rounded down, with k' = mn/2 - k + 1.  Pass ``k`` explicitly to use
    a small-sample value from Laubscher and Odeh (1976) instead.

    Parameters
    ----------
    x, y : sequence of float
        The two samples, centred at a common median of zero.
    alpha : float, optional
        Two-sided level (default 0.05).
    k : int, optional
        Override the index from eq. (9.8.2).

    Returns
    -------
    RichResult
        keys ``lower``, ``upper``, ``k``, ``kprime``, ``k_raw``,
        ``npos`` (number of positive ratios), ``estimate`` (median
        positive ratio), ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 9.8, eqs. (9.8.1)-(9.8.2),
    p. 328 (Laubscher and Odeh, 1976).
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    alpha = float(alpha)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1).")
    ratios = sorted(
        xi / yj for xi in xs for yj in ys if yj != 0.0 and xi / yj > 0.0
    )
    npos = len(ratios)
    if npos < 2:
        raise ValueError("need at least 2 positive ratios x_i / y_j.")
    nn = m + n
    za = stats.norm.ppf(1.0 - alpha / 2.0)
    kraw = m * n / 4.0 + 0.5 - za * math.sqrt(m * n * (nn + 7.0) / 48.0)
    kk = int(math.floor(kraw)) if k is None else int(k)
    kk = max(1, min(npos, kk))
    kp = int(m * n / 2.0 - kk + 1)
    kp = max(1, min(npos, kp))
    lo, hi = sorted((kk, kp))
    mid = npos // 2
    est = (
        ratios[mid] if npos % 2 else (ratios[mid - 1] + ratios[mid]) / 2.0
    )
    return RichResult(
        payload={
            "lower": float(ratios[lo - 1]),
            "upper": float(ratios[hi - 1]),
            "k": int(kk),
            "kprime": int(kp),
            "k_raw": float(kraw),
            "npos": int(npos),
            "estimate": float(est),
            "m": m,
            "n": n,
            "method": "scale-ratio CI from Sukhatme, eqs. (9.8.1)-(9.8.2)",
        }
    )


gibbons_scale_ci = scaleci
