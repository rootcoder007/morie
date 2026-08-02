# morie.fn -- function file (rootcoder007/morie)
"""Binomial tail = incomplete beta identity."""

from math import comb

from . import _array_core as np
from scipy import special

from ._richresult import RichResult

__all__ = ["gibbons_binomial_beta_link"]


def gibbons_binomial_beta_link(t, r, n):
    r"""Corollary 2.4.3.1: the identity behind order-statistic CDFs,

    .. math:: \sum_{i=r}^{n} \binom{n}{i} t^i (1 - t)^{n-i}
              = I_t(r,\; n - r + 1),

    linking the binomial upper tail to the regularised incomplete
    beta function -- i.e. :math:`P(X_{(r)} \le x) = I_{F(x)}(r,
    n - r + 1)`. Both sides are computed and returned.

    Parameters
    ----------
    t : float in [0, 1]
        The probability argument.
    r : int
        Lower summation index / first beta parameter, 1 <= r <= n.
    n : int
        Number of trials.

    Returns
    -------
    RichResult
        keys: ``binomial_tail``, ``incomplete_beta``, ``agree``,
        ``t``, ``r``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Corollary 2.4.3.1.
    """
    t = float(t)
    if not 0 <= t <= 1:
        raise ValueError(f"t must lie in [0, 1], got {t}.")
    r, n = int(r), int(n)
    if not 1 <= r <= n:
        raise ValueError(f"need 1 <= r <= n, got r={r}, n={n}.")
    tail = float(sum(comb(n, i) * t**i * (1 - t) ** (n - i) for i in range(r, n + 1)))
    ib = float(special.betainc(r, n - r + 1, t))
    return RichResult(
        payload={
            "binomial_tail": tail, "incomplete_beta": ib,
            "agree": bool(abs(tail - ib) < 1e-12), "t": t, "r": r, "n": n,
            "method": "sum_{i>=r} C(n,i)t^i(1-t)^(n-i) = I_t(r, n-r+1)",
        }
    )


def cheatsheet():
    return "gb2431: binomial tail = I_t(r, n-r+1); order-statistic CDF engine"
