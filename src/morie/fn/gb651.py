# morie.fn -- function file (rootcoder007/morie)
"""Control median test based on V, the placement of the Y median."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['ctrlmed', 'gibbons_ctrl_median']


def ctrlmed(x, y, alternative="two-sided"):
    """Mathisen's control median test, eq. (6.5.1).

    Section 6.5 (book p. 256).  With the Y sample of odd size
    n = 2r + 1 as the control, V is the number of X observations
    preceding the Y median, i.e. the placement P_(r+1).  Under H0

    .. math:: P[V = j] = \\frac{\\binom{m+r-j}{m-j}\\binom{j+r}{j}}
        {\\binom{m+2r+1}{m}}, \\qquad j = 0,\\dots,m,

    with E[V] = m/2 and, asymptotically,

    .. math:: Z = \\frac{V - m/2}{\\sqrt{m(m+n)/4n}}
             = \\frac{\\sqrt{n}(2V - m)}{\\sqrt{m(m+n)}}.

    Parameters
    ----------
    x, y : sequence of float
        Treatment and control samples; ``y`` must have odd size.
    alternative : str, optional
        ``"two-sided"``, ``"greater"`` (M_Y > M_X) or ``"less"``.

    Returns
    -------
    RichResult
        keys ``statistic`` (V), ``p_value``, ``z``, ``mean``, ``var``,
        ``pmf``, ``r``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 6.5, eq. (6.5.1), p. 256
    (Mathisen, 1943; Gastwirth, 1968).
    """
    xs = sorted(float(v) for v in x)
    ys = sorted(float(v) for v in y)
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    if n % 2 == 0:
        raise ValueError("the control sample size n must be odd (n = 2r+1).")
    r = (n - 1) // 2
    my = ys[r]
    v = sum(1 for t in xs if t <= my)
    den = math.comb(m + 2 * r + 1, m)
    pmf = [
        math.comb(m + r - j, m - j) * math.comb(j + r, j) / den
        for j in range(m + 1)
    ]
    lower = sum(pmf[: v + 1])
    upper = sum(pmf[v:])
    if alternative == "greater":
        pv = upper
    elif alternative == "less":
        pv = lower
    elif alternative == "two-sided":
        pv = min(1.0, 2.0 * min(lower, upper))
    else:
        raise ValueError("alternative must be two-sided, greater or less.")
    var = m * (m + n) / (4.0 * n)
    z = (v - m / 2.0) / math.sqrt(var)
    return RichResult(
        payload={
            "statistic": int(v),
            "p_value": float(pv),
            "z": float(z),
            "mean": m / 2.0,
            "var": float(var),
            "pmf": pmf,
            "r": int(r),
            "m": m,
            "n": n,
            "method": "control median test, eq. (6.5.1)",
        }
    )


gibbons_ctrl_median = ctrlmed
