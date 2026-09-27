"""Prevalence from inverse binomial sampling: Haldane's estimate and exact limits."""

from ._richresult import RichResult
from ._stats_core import beta

__all__ = ["inverse_binomial_prevalence"]


def inverse_binomial_prevalence(k, x, conf_level=0.95):
    r"""Prevalence when sampling stops at the k-th case after x non-cases.

    Haldane's unbiased estimate (Hedderich, Sachs & Reynarowych 2023, eq 6.30)
    :math:`\hat p = (k-1)/(k+x-1)` and the exact limits of eq (6.31) as the
    book computes them, :math:`[B(\alpha/2; k, x+1),\ B(1-\alpha/2; k, x)]`
    (beta quantiles; the printed formula swaps the second shape parameters,
    its R code and example [0.105, 0.238] use these).

    Parameters
    ----------
    k : int
        Cases observed (>= 2).
    x : int
        Non-cases before the k-th case (>= 0).
    conf_level : float

    Returns
    -------
    RichResult
        ``estimate``, ``lower``, ``upper``.

    References
    ----------
    Haldane, J. B. S. (1945). On a method of estimating frequencies.
    Biometrika 33, 222-225.
    """
    k, x = int(k), int(x)
    if k < 2 or x < 0:
        raise ValueError("need k >= 2 and x >= 0")
    a = (1 - conf_level) / 2
    est = (k - 1) / (k + x - 1)
    lo = float(beta.ppf(a, k, x + 1))
    hi = 1.0 if x == 0 else float(beta.ppf(1 - a, k, x))
    return RichResult(
        title="Inverse binomial prevalence",
        summary_lines=[("estimate", est), ("lower", lo), ("upper", hi)],
        payload={"estimate": est, "lower": lo, "upper": hi},
    )


def cheatsheet():
    return "invbpr: (k - 1)/(k + x - 1), [qbeta(a/2, k, x + 1), qbeta(1 - a/2, k, x)]"
