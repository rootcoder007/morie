"""Cochran-Armitage test for a linear trend in proportions."""

import math

from ._richresult import RichResult
from ._stats_core import chi2, norm

__all__ = ["cochran_armitage_test"]


def cochran_armitage_test(successes, totals, scores=None):
    r"""Cochran-Armitage decomposition of the k x 2 chi-square into trend and deviation.

    Hedderich, Sachs & Reynarowych (2023, Sec. 7.7.9, eqs 7.340-7.344):
    with proportions :math:`p_i = y_i/n_i`, pooled :math:`p` and scores
    :math:`x_i`, the least-squares slope
    :math:`b = \sum n_i(p_i-p)(x_i-\bar x)/\sum n_i(x_i-\bar x)^2`
    splits :math:`\chi^2 = \sum n_i(p_i-p)^2/(p(1-p))` into
    :math:`\chi^2_{trend} = b^2\sum n_i(x_i-\bar x)^2/(p(1-p))` on 1 df
    (``prop.trend.test``) and the deviation from linearity on k - 2 df.
    Also returned is the unpooled statistic (7.344),
    :math:`\hat z = \sum x_ip_i/\sqrt{\sum x_i^2 p_i(1-p_i)/n_i}`, meaningful
    for scores centred at zero.

    Parameters
    ----------
    successes, totals : sequence of int
    scores : sequence of float, optional
        Default ``1, 2, ..., k``.

    Returns
    -------
    RichResult
        ``chi2_trend``, ``p_trend`` (two-sided), ``chi2_deviation``,
        ``df_deviation``, ``p_deviation``, ``chi2_total``, ``slope``,
        ``z_unpooled``.

    References
    ----------
    Cochran, W. G. (1954). Some methods for strengthening the common chi2
    tests. Biometrics 10, 417-451. Armitage, P. (1955). Tests for linear
    trends in proportions and frequencies. Biometrics 11, 375-386.
    """
    y = [float(v) for v in successes]
    n = [float(v) for v in totals]
    k = len(y)
    if k < 2 or len(n) != k or any(a < 0 or a > b for a, b in zip(y, n)):
        raise ValueError("need k >= 2 groups with 0 <= successes <= totals")
    x = [float(v) for v in (scores if scores is not None else range(1, k + 1))]
    N = sum(n)
    p = sum(y) / N
    pi = [a / b for a, b in zip(y, n)]
    xb = sum(a * b for a, b in zip(n, x)) / N
    sxx = sum(a * (b - xb) ** 2 for a, b in zip(n, x))
    b = sum(ni * (q - p) * (xi - xb) for ni, q, xi in zip(n, pi, x)) / sxx
    denom = p * (1 - p)
    total = sum(ni * (q - p) ** 2 for ni, q in zip(n, pi)) / denom
    trend = b * b * sxx / denom
    dev = total - trend
    zu_den = sum(xi * xi * q * (1 - q) / ni for xi, q, ni in zip(x, pi, n))
    zu = sum(xi * q for xi, q in zip(x, pi)) / math.sqrt(zu_den) if zu_den > 0 else float("nan")
    return RichResult(
        title="Cochran-Armitage trend test",
        summary_lines=[("chi2 trend", trend), ("chi2 deviation", dev)],
        payload={
            "chi2_trend": trend,
            "p_trend": float(chi2.sf(trend, 1)),
            "chi2_deviation": dev,
            "df_deviation": k - 2,
            "p_deviation": float(chi2.sf(dev, k - 2)) if k > 2 else float("nan"),
            "chi2_total": total,
            "slope": b,
            "z_unpooled": zu,
            "p_z_unpooled": float(2 * norm.sf(abs(zu))) if zu == zu else float("nan"),
        },
    )


def cheatsheet():
    return "catrnd: Cochran-Armitage chi2 = trend (1 df) + deviation (k - 2 df)"
