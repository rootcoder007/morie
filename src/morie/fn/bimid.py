# morie.fn -- wave 4 slice b_3 (rootcoder007/morie)
"""Biweight midvariance, a robust measure of dispersion.

Source: Wilcox, R. R. (2017), *Modern Statistics for the Social and
Behavioral Sciences: A Practical Introduction*, 2nd edn, CRC Press,
Box 2.1, equation (2.12), p.31.

With M the sample median and MAD the (unscaled) median absolute
deviation, MAD = median |X_i - M|, set

    Y_i = (X_i - M) / (9 MAD),
    a_i = 1 if |Y_i| < 1, else 0,

in which case

                      sqrt(n) sqrt( sum a_i (X_i - M)^2 (1 - Y_i^2)^4 )
    zeta_bimid  =    ------------------------------------------------- .   (2.12)
                        | sum a_i (1 - Y_i^2)(1 - 5 Y_i^2) |

The biweight midvariance is zeta_bimid^2.  Note that the MAD entering
Y_i is the raw median absolute deviation, *not* the rescaled MADN of
equation (2.14); the 9 in the denominator is the tuning constant that
sets which points get weight a_i = 0.

Anchor (closed form, evaluated by hand off the code path): for the
three values -1, 0, 1 we have M = 0 and MAD = 1, so Y = (-1/9, 0, 1/9),
every a_i = 1, the numerator is sqrt(3) sqrt(2) (80/81)^2 and the
denominator is 1 + 2 (80/81)(76/81).
"""

from __future__ import annotations

from math import sqrt

from . import _array_core as np  # noqa: F401
from . import _s03core as k  # noqa: F401

from ._richresult import RichResult

__all__ = ["bimid"]

_METHOD = "Wilcox (2017) biweight midvariance, eq. (2.12)"


def bimid(x):
    """Biweight midvariance of a sample.

    Parameters
    ----------
    x : array-like
        The sample; at least two observations, with a non-zero median
        absolute deviation (a sample more than half of whose values are
        tied at the median has MAD = 0 and no biweight midvariance).

    Returns
    -------
    result : RichResult
        Keys: estimate (= zeta^2), zeta, med, mad, n_used, n, method.
    """
    xs = [float(v) for v in x]
    n = len(xs)
    if n < 2:
        raise ValueError("bimid: need at least 2 observations")
    for v in xs:
        if v != v:
            raise ValueError("bimid: x contains a missing value")
    med = k.median(xs)
    mad = k.median([abs(v - med) for v in xs])
    if not (mad > 0.0):
        raise ValueError("bimid: the median absolute deviation is zero")
    num = 0.0
    den = 0.0
    n_used = 0
    for v in xs:
        y = (v - med) / (9.0 * mad)
        if abs(y) < 1.0:
            n_used += 1
            y2 = y * y
            num += (v - med) ** 2 * (1.0 - y2) ** 4
            den += (1.0 - y2) * (1.0 - 5.0 * y2)
    den = abs(den)
    if not (den > 0.0):
        raise ValueError("bimid: the biweight denominator vanished")
    zeta = sqrt(float(n)) * sqrt(num) / den
    return RichResult(
        title="Biweight midvariance",
        summary_lines=[("n", n), ("estimate", zeta * zeta)],
        payload={
            "estimate": zeta * zeta,
            "zeta": zeta,
            "med": med,
            "mad": mad,
            "n_used": n_used,
            "n": n,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "bimid: Wilcox biweight midvariance (eq. 2.12)"
