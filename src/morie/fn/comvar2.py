"""Modified percentile bootstrap for the difference between two variances.

Wilcox (2017), Modern Statistics for the Social and Behavioral Sciences, eq (7.39).
"""

import math

from ._richresult import RichResult
from ._rng import random_uniform

__all__ = ["comvar2"]


def _var(v):
    m = sum(v) / len(v)
    return sum((a - m) ** 2 for a in v) / (len(v) - 1)


def comvar2(x, y, seed=0):
    """0.95 CI for sigma_1^2 - sigma_2^2 (Wilcox 2002).

    With n_m = min(n_1, n_2), each of B = 599 bootstrap rounds draws n_m values
    with replacement from each group and records D* = s*_1^2 - s*_2^2; the
    interval is (D*_(l), D*_(u)) (7.39) with (l, u) = (7, 593) for n_m < 40,
    (8, 592) below 80, (11, 588) below 180, (14, 585) below 250 and (15, 584)
    otherwise. Resampling uses morie's Philox stream (``seed``; stream 0 for
    x, 1 for y), identical in the R arm.

    Parameters
    ----------
    x, y : sequences of float
    seed : int

    Returns
    -------
    RichResult
        Keys: ci, estimate (s_1^2 - s_2^2), l, u, boot (sorted D*), reject (0 outside the CI).

    References
    ----------
    Wilcox, R. R. (2002). Comparing the variances of two independent groups.
    British Journal of Mathematical and Statistical Psychology 55, 169-175.
    Wilcox, R. R. (2017). Modern Statistics for the Social and Behavioral
    Sciences (2nd ed.). CRC Press. Eq (7.39).

    Examples
    --------
    >>> comvar2([1, 2, 3, 4, 5, 6, 7, 8], [1, 3, 5, 7, 9, 11, 13, 15])["l"]
    7
    """
    x = [float(v) for v in x]
    y = [float(v) for v in y]
    if len(x) < 2 or len(y) < 2:
        raise ValueError("need at least 2 observations per group")
    B = 599
    nm = min(len(x), len(y))
    ux = random_uniform(B * nm, seed=seed, stream=0)
    uy = random_uniform(B * nm, seed=seed, stream=1)
    d = []
    for b in range(B):
        xs = [x[math.floor(ux[b * nm + i] * len(x))] for i in range(nm)]
        ys = [y[math.floor(uy[b * nm + i] * len(y))] for i in range(nm)]
        d.append(_var(xs) - _var(ys))
    d.sort()
    lo, up = next(
        (a, b)
        for lim, a, b in ((40, 7, 593), (80, 8, 592), (180, 11, 588), (250, 14, 585), (math.inf, 15, 584))
        if nm < lim
    )
    ci = (d[lo - 1], d[up - 1])
    return RichResult(
        title="Difference between two variances",
        summary_lines=[("estimate", _var(x) - _var(y)), ("ci", ci)],
        payload={
            "ci": ci,
            "estimate": _var(x) - _var(y),
            "l": lo,
            "u": up,
            "boot": d,
            "reject": not ci[0] <= 0 <= ci[1],
        },
    )


def cheatsheet():
    return "comvar2: modified percentile bootstrap CI for sigma1^2 - sigma2^2, B = 599. Wilcox (2017) eq (7.39)."
