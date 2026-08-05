# morie.fn -- wave 4 slice b_3 (rootcoder007/morie)
"""The MAD-median outlier detection rule.

Source: Wilcox, R. R. (2017), *Modern Statistics for the Social and
Behavioral Sciences: A Practical Introduction*, 2nd edn, CRC Press,
section 2.5.2, equation (2.14), p.33.  Declare X an outlier if

    |X - M| / MADN > 2.24,                                     (2.14)

where M is the sample median, MADN = MAD / 0.6745 and MAD is the
unscaled median absolute deviation median |X_i - M|.  Both M and MADN
have a breakdown point of 0.5, which is what lets the rule survive the
masking that defeats the mean/SD rule of equation (2.13).

The constant 2.24 is from Rousseeuw and van Zomeren (1990), "Unmasking
multivariate outliers and leverage points", *Journal of the American
Statistical Association* 85(411), 633-639.  The book notes in a
footnote that equation (2.14) is the Hampel identifier, for which
Hampel used 3.5 rather than 2.24; ``crit`` is exposed for that reason.

The divisor is the book's 0.6745, not R's stats::mad constant of
1.4826 (= 1/0.674489...), so MADN here is stats::mad(x, constant =
1/0.6745); the two agree to about seven significant digits.

Anchor (printed worked example, p.33): for 2, 2, 3, 3, 3, 4, 4, 4,
100000, 100000 the book gets M = 3.5, MADN = MAD/0.6745 = 0.7413 and
(100000 - 3.5)/0.7413 = 134893.4, so both copies of 100000 are declared
outliers -- while the mean/SD rule (2.13) declares neither.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k  # noqa: F401

from ._richresult import RichResult

__all__ = ["outmad"]

_METHOD = "Wilcox (2017) MAD-median outlier rule, eq. (2.14)"


def outmad(x, crit=2.24):
    """Flag outliers with the MAD-median rule (2.14).

    Parameters
    ----------
    x : array-like
        The sample; at least two observations.  More than half the
        values tied at the median gives MAD = 0 and no usable scale.
    crit : float, default 2.24
        The cut-off of equation (2.14).  Must be positive.  Use 3.5 for
        Hampel's original identifier.

    Returns
    -------
    result : RichResult
        Keys: flag (0/1 per observation), which (1-based positions of
        the outliers), out_val, n_out, center, scale, dis, crit, n,
        estimate (= n_out), method.
    """
    xs = [float(v) for v in x]
    n = len(xs)
    if n < 2:
        raise ValueError("outmad: need at least 2 observations")
    for v in xs:
        if v != v:
            raise ValueError("outmad: x contains a missing value")
    crit = float(crit)
    if not (crit > 0.0):
        raise ValueError("outmad: crit must be positive")
    center = k.median(xs)
    mad = k.median([abs(v - center) for v in xs])
    scale = mad / 0.6745
    if not (scale > 0.0):
        raise ValueError("outmad: the median absolute deviation is zero")
    dis = [abs(v - center) / scale for v in xs]
    flag = [1 if d > crit else 0 for d in dis]
    which = [i + 1 for i in range(n) if flag[i] == 1]
    out_val = [xs[i - 1] for i in which]
    return RichResult(
        title="MAD-median outlier rule",
        summary_lines=[("n", n), ("n_out", len(which))],
        payload={
            "flag": flag,
            "which": which,
            "out_val": out_val,
            "n_out": len(which),
            "center": center,
            "scale": scale,
            "dis": dis,
            "crit": crit,
            "n": n,
            "estimate": float(len(which)),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "outmad: Wilcox MAD-median outlier rule (eq. 2.14)"
