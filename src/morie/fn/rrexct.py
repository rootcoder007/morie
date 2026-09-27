"""Exact conditional confidence interval and test for the ratio of two Poisson rates."""

import math

from ._richresult import RichResult
from ._stats_core import beta, binom

__all__ = ["rate_ratio_exact_ci"]


def _binom_two_sided(x, m, p):
    # binom.test: sum the probabilities no larger than P(X = x)
    d = float(binom.pmf(x, m, p)) * (1 + 1e-7)
    return min(1.0, sum(float(binom.pmf(i, m, p)) for i in range(m + 1) if float(binom.pmf(i, m, p)) <= d))


def rate_ratio_exact_ci(k1, n1, k2, n2, conf_level=0.95):
    r"""Exact CI and test for :math:`\lambda_1/\lambda_2` from counts over exposures.

    Given :math:`m = k_1 + k_2`, :math:`k_1` is binomial with
    :math:`\pi = n_1\lambda_1/(n_1\lambda_1 + n_2\lambda_2)`
    (Hedderich, Sachs & Reynarowych 2023, eq 6.59; Price & Bonett 2000). The
    Clopper-Pearson limits :math:`p_l, p_u` for :math:`\pi` give

    .. math::  \left[\frac{n_2 p_l}{n_1(1-p_l)},\ \frac{n_2 p_u}{n_1(1-p_u)}\right],

    and the p-value of :math:`\lambda_1 = \lambda_2` is the two-sided exact
    binomial test of :math:`\pi = n_1/(n_1+n_2)` (``poisson.test``).

    Parameters
    ----------
    k1, k2 : int
        Event counts.
    n1, n2 : float
        Exposures (> 0).
    conf_level : float

    Returns
    -------
    RichResult
        ``ratio``, ``lower``, ``upper``, ``p_value``.

    References
    ----------
    Price, R. M. & Bonett, D. G. (2000). Estimating the ratio of two Poisson
    rates. Computational Statistics & Data Analysis 34, 345-356.
    """
    k1, k2 = int(k1), int(k2)
    n1, n2 = float(n1), float(n2)
    if k1 < 0 or k2 < 0 or k1 + k2 == 0 or n1 <= 0 or n2 <= 0:
        raise ValueError("need non-negative counts with k1 + k2 > 0 and positive exposures")
    m = k1 + k2
    a = (1 - conf_level) / 2
    pl = 0.0 if k1 == 0 else float(beta.ppf(a, k1, m - k1 + 1))
    pu = 1.0 if k1 == m else float(beta.ppf(1 - a, k1 + 1, m - k1))
    lo = n2 * pl / (n1 * (1 - pl))
    hi = math.inf if pu == 1 else n2 * pu / (n1 * (1 - pu))
    ratio = math.inf if k2 == 0 else (k1 / n1) / (k2 / n2)
    p = _binom_two_sided(k1, m, n1 / (n1 + n2))
    return RichResult(
        title="Exact rate ratio CI",
        summary_lines=[("ratio", ratio), ("lower", lo), ("upper", hi), ("p_value", p)],
        payload={"ratio": ratio, "lower": lo, "upper": hi, "p_value": p},
    )


def cheatsheet():
    return "rrexct: Clopper-Pearson [p_l, p_u] for k1 of k1 + k2, mapped by n2 p / (n1 (1 - p))"
