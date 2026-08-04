# morie.fn -- function file (rootcoder007/morie)
"""Jonckheere-Terpstra test against ordered alternatives."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['jtstat', 'gibbons_jonckheere']


def jtstat(samples, alternative="greater"):
    """B = sum of the pairwise Mann-Whitney counts, Sec. 10.6.

    Book p. 365.  For the simple-order alternative
    theta_1 <= ... <= theta_k, the JT statistic adds the Mann-Whitney
    statistics of every ordered pair,

    .. math:: B = \\sum_{1 \\le i < j \\le k} U_{ij}, \\qquad
        U_{ij} = \\#\\{(a,b): X_{ia} < X_{jb}\\},

    with null moments eqs. (10.6.2)-(10.6.3),

    .. math:: E_0[B] = \\frac{N^2 - \\sum n_i^2}{4}, \\qquad
        Var_0[B] = \\frac{N^2(2N+3)
            - \\sum n_i^2(2n_i+3)}{72},

    and H0 rejected for large B.  Ties contribute 1/2 each, the
    modification the book describes for U*_ij.

    Parameters
    ----------
    samples : sequence of sequence of float
        The k samples, in the hypothesised increasing order.
    alternative : str, optional
        ``"greater"`` (the simple order, default), ``"less"`` or
        ``"two-sided"``.

    Returns
    -------
    RichResult
        keys ``statistic`` (B), ``mean``, ``var``, ``z``,
        ``p_value``, ``k``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 10.6, eqs. (10.6.2)-(10.6.3),
    pp. 365-366 (Terpstra, 1952; Jonckheere, 1954).
    """
    ss = [[float(v) for v in s] for s in samples]
    k = len(ss)
    if k < 2:
        raise ValueError("need at least 2 samples.")
    if any(len(s) < 1 for s in ss):
        raise ValueError("every sample must be non-empty.")
    b = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            for a in ss[i]:
                for c in ss[j]:
                    if a < c:
                        b += 1.0
                    elif a == c:
                        b += 0.5
    ns = [len(s) for s in ss]
    nn = sum(ns)
    mean = (float(nn) ** 2 - sum(float(v) ** 2 for v in ns)) / 4.0
    var = (
        float(nn) ** 2 * (2.0 * nn + 3.0)
        - sum(float(v) ** 2 * (2.0 * v + 3.0) for v in ns)
    ) / 72.0
    z = (b - mean) / math.sqrt(var)
    if alternative == "greater":
        pv = 1.0 - stats.norm.cdf(z)
    elif alternative == "less":
        pv = stats.norm.cdf(z)
    elif alternative == "two-sided":
        pv = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    else:
        raise ValueError("alternative must be greater, less or two-sided.")
    return RichResult(
        payload={
            "statistic": float(b),
            "mean": float(mean),
            "var": float(var),
            "z": float(z),
            "p_value": float(min(1.0, pv)),
            "k": int(k),
            "n": int(nn),
            "method": "Jonckheere-Terpstra B (Sec. 10.6)",
        }
    )


gibbons_jonckheere = jtstat
